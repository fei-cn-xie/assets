# vLLM 生命周期 · 读图指南（配套 `vLLM生命周期-端到端.drawio`）

本目录的 drawio 文件是一个 **9 页**的端到端流程图集，回答三个问题以及它们的实现细节：

- **模型加载的生命周期**（vllm 拉起时，权重是怎么一层层装进来的）
- **KV cache 的生命周期**（显存画像 → 块池 → 请求级分配 → PD 跨机交接 → 释放）
- **一个请求的生命周期**（客户端 → 边 → PD Proxy → 云 P → 云 D → 边 → 客户端）

分析对象为两个代码仓的 `feat/pd-sep-lwd-v0.30.0` 分支，部署形态取自
`assets/c_home/bash/pds/30rc_1124/sidecar/`（`edge.sh` / `cloud_p.sh` / `cloud_d.sh` /
`start_proxy.sh` / `pd_host_topology.yaml`）。

---

## 1. 页面清单与阅读顺序

| # | 页面 | 回答什么问题 | 适合先看 |
| --- | --- | --- | --- |
| 01 | 部署与进程全景 | 几台机、几个进程、几条通道、谁监听谁 | ★ 第一张 |
| 02 | 启动时序 | 从 `bash edge.sh` 到 ready，进程树怎么长出来 | ★ 第二张 |
| 03 | 模型加载·三分侧 | 同一段加载代码在 edge / P / D 上跑出什么不同结果 | ★ 主图 |
| 04 | 模型加载·调用链 | 逐函数调用栈 + 每步为什么存在 | 对照 03 |
| 05 | KV Cache 生命周期 | 规格 → 画像 → 块池 → 分配/写入 → 释放 → PD 交接 | ★ 主图 |
| 06 | PD·KV 交接时序 | P 导出 pin、D 块级 pull、KV_READY、解 pin | 对照 05 |
| 07 | 单请求端到端 | 一次请求的八段时序，38 个关键步 | ★ 主图 |
| 08 | 控制面协议与状态机 | 11 类消息 + 4 类对象的状态迁移 | 读 07 时穿插 |
| 09 | vLLM v1 Step 循环 | 与 LWD 无关的引擎地基：一个 step 里发生什么 | 想补基础时看 |

建议路线：**01 → 02 → 09 → 03 → 04 → 05 → 06 → 07（配合 08）**。
先有"进程与循环"的骨架，再看三条生命周期，最省力。

图上每个节点都带两类小字：**文件:行号**（灰色）与**补充说明**（棕色）。
行号以本文写作时的分支为准，代码改动后可能小幅漂移，用同名的类/函数名检索即可。

---

## 2. 三分侧总对照（一张表记住全部差异）

| 维度 | edge（.194，NPU 4，TP=1） | cloud-P（.195，NPU 0,1，TP=2） | cloud-D（.195，NPU 2,3，TP=2） |
| --- | --- | --- | --- |
| role / instance | `edge / 0` | `cloud / 0`，`stage_role=prefill` | `cloud / 1`，`stage_role=decode` |
| 执行器 | `LwdHostUniProcExecutor`（worker 与引擎同进程） | `MultiprocExecutor`（2 个 worker 子进程） | 同 P |
| 引擎类 | `LwdEdgeEngineCore` | `LwdCloudEngineCore` | `LwdCloudEngineCore` |
| worker / runner | `LwdEdgeWorker` / `LwdEdgeModelRunner` | `LwdCloudWorker` / `LwdCloudModelRunner` | 同 P |
| 建的模块 | embed_tokens + lm_head + visual（**0 层**，无 norm，无 MTP） | 全部 layers + norm + MTP（无 visual） | 同 P |
| KV cache | **无**（`get_kv_cache_spec()→{}`，num_blocks=1 空壳） | 真实块池，Mooncake **producer** 14579 | 真实块池，Mooncake **consumer** 14580 |
| 图模式 | `--enforce-eager`，且整段跳过 warmup/capture | `--enforce-eager` | `FULL_DECODE_ONLY` cudagraph |
| 是否采样 | 否（lm_head + rank 回放） | 是，但**抑制**用户输出（内部 token 不进历史） | 是，用户可见输出 |
| 数据面方向 | 只 connect：UP send / DOWN recv（唯一阻塞侧，605s/条） | 监听：UP recv / DOWN send | 同 P |
| 控制面 | DEALER connect 到云 `ctrl_port` 6100/6101 | ROUTER bind 0.0.0.0:6100 | ROUTER bind 0.0.0.0:6101 |
| coordination | 只作为客户端访问 `pd_proxy_url`(8102) | bind HTTP coord_port 8100 | bind HTTP coord_port 8101 |

---

## 3. 三条生命周期的主线（一句话版）

1. **模型加载**：`vllm serve` → `create_engine_config`（解析 lwd_config + 拓扑 + digest）→
   `worker_cls` 二分（platform.py:1370）→ 进程/线程 → `init_device`（NPU 绑定 + 并行组 +
   数据面建链 + sidecar daemon）→ `load_model`（构造期按 `LWD_MODULE_OWNERSHIP` 建/不建，
   装载期按 role 跳过读盘）→ 装载后处理 → `_initialize_kv_caches`（此时才有 KV）。
2. **KV cache**：`get_kv_cache_spec`（边端为空）→ `determine_available_memory`
   （整卡 × 0.90 再减非 KV）→ `num_blocks = available // bytes_per_block` →
   `BlockPool` + 张量分配（2MiB 对齐，供 RDMA）→ `allocate_slots` / `slot_mapping` /
   写 `kv_cache[slot]` → `request_finished` 释放（PD 时先 pin）→ D 读完回执才真正释放。
3. **一个请求**：边侧持原文（tokenize / detokenize / 文本 stop / K 计数），
   发一个"控制信封"给 PD Proxy → Proxy 先 P 后 D → 边↔云用 ZMQ 协商绑定与窗口、
   用 UCX 送 embedding / 收 hidden → P 算完导出 KV 给 D 并 pin 块 → D 导入 KV + 收 1 行
   tail 后生成 → C2e 下行 → 边 UNEMBED 恢复 token → 边冻结 K → D 形成 usage。

---

## 4. 关键常数速查（都来自脚本与拓扑，不是代码默认值）

| 项 | 值 | 出处 |
| --- | --- | --- |
| 端口 | 边 8510；P 8600 / coord 8100 / ctrl 6100 / data 7200；D 8601 / 8101 / 6101 / 7201；Proxy 8102；Mooncake 14579 / 14580 | 三份 `.sh` + `pd_host_topology.yaml` |
| 窗口与并发 | `max_num_seqs 128`、`max_num_batched_tokens 8192`、`max-model-len 65536`、`pd.max_inflight_windows 4` | 同上 |
| host 缓冲 | `up_buffer_size 256` MiB、`down_buffer_size 32` MiB（**与 KV 无关**，是 UCX 帧块） | topology `deployment.data_plane_host` |
| 超时 | `request_ttl_s 1800`、`p/d_bind_timeout_s 30`、`notify_claim 60`、`export_hold 600`、`result_hold 600`、链路 `wait_ready 605s/条` | topology + `lwd_host_channels.py:391` |
| 显存 | `gpu-memory-utilization 0.90`；投机解码 MTP 3 tokens | 三份 `.sh` |

---

## 5. 自己复核的几条命令

```bash
# 角色分派（两处唯一的"接缝"）
grep -n "LwdEdgeEngineCore\|LwdCloudEngineCore" vllm/vllm/v1/engine/core.py
grep -n "LwdEdgeWorker\|LwdCloudWorker"            vllm-ascend/vllm_ascend/platform.py

# 边端为什么没有 KV
grep -n "def get_kv_cache_spec" -A3 vllm-ascend/vllm_ascend/worker/lwd_edge_model/lwd_edge_model_runner.py

# 模块归属表（谁建哪些模块）
sed -n '860,900p' vllm/vllm/model_executor/models/utils.py
grep -n "LWD_MODULE_OWNERSHIP" -r vllm/vllm | head

# 权重跳过（读盘前丢弃）
grep -n "lwd_should_skip_weight" -r vllm/vllm

# PD 的 KV pin/解 pin
grep -n "delay_free_blocks" -r vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py | head
grep -n "KV_READY\|kv_ready" vllm/vllm/v1/lwd_control/pd/cloud_scheduler.py

# 控制面消息全集
grep -n "^class Lwd" vllm/vllm/v1/lwd_control/lwd_comm/lwd_messages.py
```

日志锚点（图 02/07 里也标了）：
`[LWD][config] resolved=…`（三端应一致）、`[LWD] host sidecar deployment: listeners owned by
the transport daemon`、`[LWD][handshake] host link not ready: …`、`[edge] request received req=…`、
`[Lwd][cloud-model] loaded layers=…`、`KV cache usage: …%`。

---

## 6. 读图时容易踩的坑（本次走读中确认的）

1. 旧文档里的 `lwd_serve_guard` / `lwd_resolve_engine_cls` **不存在**；
   真实接缝是 `lwd_dispatch_attach_frontend`（本形态是 no-op）、引擎类分派、worker_cls 分派。
2. `vllm/v1/lwd_sidecar/` 目录只剩 `__pycache__`，sidecar 的实现在
   `vllm/v1/lwd_comm/data_comm/`（`ucx_daemon.py` / `ucx_port.py` / `ucx_shm.py`）。
3. `up_buffer_size` / `down_buffer_size` 是 host 数据面的帧块，**不是** KV 容量；
   KV 走 Mooncake 的 `kv_port`。
4. Mooncake 的 KV 交接是**块级 pull**，三个 layerwise 钩子全是 `pass`，别按 layerwise 理解。
5. `pd/` 与 `control_*_scheduler/` 是两代代码共存：本 PD 形态下 `LwdPdEdgeRuntime`
   接管请求（不再走 prefix probe 那条老路）；`LwdPrefixReadyNotify` / `LwdRefillNotify`
   在本路径上不使用。
6. 边侧是**唯一阻塞等链路**的一侧（605s/条，最坏 4 条），所以部署顺序要求边端最后起。
