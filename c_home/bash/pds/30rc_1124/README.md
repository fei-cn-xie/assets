# v0.30.0 边云 PD 分离拉起脚本（Qwen3.6-27B）

目标分支：`feat/pd-sep-lwd-v0.30.0`（`vllm` 与 `vllm-ascend` 两个仓，**必须同分支配套**）。
基线参考：`dev/pd-sep-comm` 的 `private_home/vllm_script/pds_comm_1117/{tcp,rdma}`。
产出：`tcp/` 与 `rdma/` 两套自包含目录，各 4 个入口脚本 + 拓扑 YAML + proxy 配置。

两份代码里，v0.30.0 把"边云 layerwise 分离"从**外部 sidecar 进程 + 目录 JSON**
改写成**引擎进程内的 host 数据面 + 拓扑 YAML**，因此拉起脚本不是改参数，而是换结构。
下面第 1 节给结论，第 2 节给完整参数清单（这是本次分析的主体），第 3 节起给部署与验证。

---

## 1. 与 `dev/pd-sep-comm` 参考脚本的结构性差异

| 维度 | `dev/pd-sep-comm`（pds_comm_1117） | `feat/pd-sep-lwd-v0.30.0` | 后果 |
| --- | --- | --- | --- |
| 进程数 | 6 个：`edge`、`cloud_p`、`cloud_d`、`sidecar_p`、`sidecar_d`、`proxy` | 4 个：`edge`、`cloud_p`、`cloud_d`、`proxy` | 少 2 个 sidecar |
| 传输承载 | 独立进程 `vllm/tools/lwd_sidecar.py` + `lwd_sidecar_directory.json` | 引擎进程内（`data_plane_host.deployment: in_process`） | 目录 JSON 整个作废 |
| TCP / RDMA 选择 | 目录 JSON 的 `backend: "tcp" \| "rdma"` | **拓扑里没有这个字段**，由 `UCX_TLS` / `UCX_NET_DEVICES` 决定 | tcp 与 rdma 只差环境变量 |
| 传输实现 | tcp = 自研 socket；rdma = UCX + UHC1 | 两者**都是 UCX**（ctypes 调 libucp），tcp 只是 `UCX_TLS=tcp,self` | **tcp 变体也必须装 UCX/libucp** |
| `--additional-config` | 3 段：`lwd_config`（含 `enabled/mode/edge_head_tail_layers/transport.directory_path`）+ `lwd_coordination` + `lwd_pd_config` | 1 段：`{"lwd_config":{"path","role","instance_id"}}`，其余键 `extra="forbid"` 直接报错 | 参考脚本的 additional-config 全部要重写 |
| 协调面 | `lwd_coordination.listen_port/control_url/tenant_key_file/consumer_id` | 移到拓扑每个 instance 的 `coordination`；端口改为云 DP 的 `coord_port` | 旧 `listen_port`/`control_url` 被显式拒绝 |
| PD 开关 | `lwd_pd_config: {enabled, tenant_id}` | 拓扑 `deployment.pd` | `lwd_pd_config` 传了也不报错、被静默忽略 |
| 层次切分 | `edge_head_tail_layers: [0,0]` + `mode: prefill_only` | 字段删除；边端 0 层由代码按 role 决定，权重切片用 `split_lwd_checkpoint.py`（可选） | 层号不再可配 |
| 端口模型 | `control_endpoint`/`data_endpoint` + `ucx_port_base` | 云 DP 三个端口：`ctrl_port`（控制面）/ `data_port`（UCX listener）/ `coord_port`（PD 协调 HTTP） | 一端口拆成三端口 |
| 缓冲/槽位 | `limits.slot_bytes=33554432` + `rx/tx_slots` | `up_buffer_size`（MiB，PD **必填**）/ `down_buffer_size`（MiB），块数固定 4 | 单位与语义都变 |
| PD 代理 | `proxy.py --prefill-url ... --decode-url ...` | `python -m vllm.v1.lwd_control.pd.proxy --config pd_proxy.json` | 目标表改成 JSON 数组 |
| 边端并行度 | TP=1、DP=1 | TP=1、DP=1（**硬约束**，TP>1 首次 UP 提交即 missing-IO） | 一致 |
| 云 DP 校验 | 无 | 云 `--data-parallel-size` 必须等于 `len(clouds[i].dp)` | 新增启动期失败点 |

参考脚本里**保留不变**的部分：NPU 卡位、网卡名、`VLLM_HOST_IP`、HCCL/GLOO/TP socket 环境、
jemalloc 预载、`--max-model-len/--max-num-seqs/--max-num-batched-tokens/--gpu-memory-utilization`、
Mooncake `--kv-transfer-config` 结构、`--speculative-config`、D 侧图模式、`--override-generation-config`。

---

## 2. v0.30.0 边云 PD 拉起需要的参数清单

### 2.1 拓扑 YAML（唯一事实源，两台机器必须同一份内容）

以前散在 `--additional-config` 和目录 JSON 里的东西，现在都在这里；代码里**没有任何环境变量可以覆盖**它
（`vllm_ascend/distributed/lwd_data_plane/host/lwd_host_channels.py:69-85` 明确 "The YAML is the single
source of truth — there is no env override"）。文件：`lwd_topology.yaml`。

| 块 / 字段 | 取值（本脚本） | 约束与来源 |
| --- | --- | --- |
| `deployment.mode` | `0` | 只接受 0（`topology.py:139`） |
| `deployment.scene` | `lwd_cluster` | PD 不能用 `edge_share`；`single_instance` 只允许 1 云，P/D 需要 2 云（`topology.py:442-443,629-634`） |
| `deployment.edges_num` / `clouds_num` | `1` / `2` | 必须等于 `edges`/`clouds` 长度（`topology.py:387-393`） |
| `deployment.data_plane_comm` | `host` | **PD 强制 host**（`topology.py:433-434`） |
| `deployment.hccl_world_size` | 不写 | host 模式下出现即报错（`topology.py:474-480`） |
| `deployment.data_plane_host.up_buffer_size` | `256` | **PD 必须显式给**，两侧一致；覆盖最大完整帧（`topology.py:435-441`） |
| `deployment.data_plane_host.down_buffer_size` | `32` | 默认 32，两侧一致 |
| `deployment.data_plane_host.deployment` | `in_process` | 或用 `sidecar`（每机一个 transport daemon）；两侧必须一致（`topology.py:109-113`） |
| `deployment.pd.enabled` | `true` | 取代旧 `lwd_pd_config.enabled`（`topology.py:119-134`） |
| `deployment.pd.tenant_id` | `edge-0` | 也可留空用边端 `consumer_id`（`topology.py:457-458`） |
| `deployment.pd.max_inflight_windows` / `request_ttl_s` | `4` / `1800.0` | 默认值，通常不用改 |
| `feature_ctrl` | `{enable_early_recv: true}` | **块本身必填**；本分支只被解析、无运行期读取点 |
| `edges[].id` | `0` | 从 0 连续（`topology.py:375-383`） |
| `edges[].dp[].dp_idx` | `0` | 从 0 连续（`topology.py:274-280`） |
| `edges[].dp[].addr` | `76.76.26.194` | 必须是单播 IP（`topology.py:34-38`） |
| `edges[].dp[].ranks` | **不写** | host 模式写 `ranks` 即报错（`topology.py:467-473`） |
| `edges[].dp[].ctrl_port`/`data_port` | **不写** | 边端只连出不监听；出现 ctrl_port/data_port 即报错（`topology.py:401-407,481-485`） |
| `edges[].coordination.enabled` | `true` | PD 要求**每个** instance 都开（`topology.py:446-448`） |
| `edges[].coordination.pd_proxy_url` | `http://76.76.26.189:8500/v1/chat/completions` | 必须是带 host 的 http(s)（`topology.py:454-456`） |
| `edges[].coordination.tenant_key_file` | `/vllm-workspace/lwd_tenant.key` | PD 必须给，且 `placement_only` 不被接受（`topology.py:449-453`） |
| `edges[].coordination.consumer_id` | `edge-0` | 与 tenant_id 二选一即可 |
| `clouds[].id` / `stage_role` | `0`/`prefill`、`1`/`decode` | 云的 `stage_role` 集合必须正好是 `{prefill, decode}`（`topology.py:444-445`） |
| `clouds[].dp[].ctrl_port` | `19180` / `19182` | 云侧**必填**；同机不同 DP 不许重复（`topology.py:586-591`） |
| `clouds[].dp[].data_port` | `19600` / `19700` | 云侧**必填**；同 addr 不许重复（`topology.py:487-498`） |
| `clouds[].dp[].coord_port` | `8660` / `8661` | 云 coordination 开启时必填；不许撞同 addr 的 ctrl/data 端口（`topology.py:500-509,610-623`） |
| `clouds[].dp[].coord_url` | 不写 | 省略时由 `addr:coord_port` 派生（`topology.py:254-261`） |
| `clouds[].coordination.enabled` | `true` | 同上；`listen_host` 默认 `0.0.0.0` |
| `instance_links` | `[{edge:0,cloud:0},{edge:0,cloud:1}]` | 非 edge_share 场景必须全互联，否则报错（`topology.py:661-667`） |
| `qos` | 注释掉 | RDMA 专属：`data_plane.traffic_class`(0-255)/`sl`(0-15)，走配置不走 env（`topology.py:152-164`） |

校验派生值：`LwdTopology` 会算拓扑 digest（sha256 前 16 hex），每进程启动打一条
`[LWD][config] resolved=...`。**两侧 digest 不一致 = 文件不同步**，这是最省时的自检点。

### 2.2 `vllm serve` 参数（按角色）

| 参数 | edge | cloud P | cloud D | 说明 |
| --- | --- | --- | --- | --- |
| `--model` | `/weight/Qwen3.6-27B/` | 同 | 同 | 两侧都可用完整模型；切片目录可选 |
| `--port` | `9060` | `8560` | `8561` | 各自 OpenAI API；同机 P/D 必须错开 |
| `--host` | `0.0.0.0` | `0.0.0.0` | `0.0.0.0` | |
| `--tensor-parallel-size` | **`1`** | `2` | `2` | 边端 TP>1 在 host 模式不可用（`lwd_host_data_plane.md:51,59`） |
| `--data-parallel-size` | `1` | `1` | `1` | 云侧必须 `== len(clouds[i].dp)`；边端必须 1（`lwd_control_plane_factory.py:48-69`） |
| `--data-parallel-size-local` | `1` | `1` | `1` | 单机单 DP |
| `--distributed-executor-backend` | `mp` | `mp` | `mp` | |
| `--additional-config` | `{"lwd_config":{"path":<yaml>,"role":"edge","instance_id":0}}` | `...role:"cloud",instance_id:0` | `...role:"cloud",instance_id:1` | **必须内联 JSON**，传文件路径会被静默忽略 |
| `--enable-prefix-caching` | **`--no-enable-prefix-caching`** | 开 | 开 | P 需要前缀快照做 handoff |
| `--enforce-eager` | 开（非必需，边端 0 层自动跳过 warmup/capture） | 开 | 不开 | D 走图模式 |
| `--async-scheduling` | 开 | 开 | 开 | 边端未验证，见 §6 |
| `--compilation-config` | — | — | `{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[...]}` | |
| `--speculative-config` | **不传** | `qwen3_5_mtp,3` | 同 | drafter 只在云；边端不建 |
| `--kv-transfer-config` | 不传 | `kv_producer,kv_rank=0,kv_port=24679` | `kv_consumer,kv_rank=1,kv_port=24680` | `kv_connector_extra_config.prefill/decode` 的 `tp_size/dp_size` 必须与真实并行度一致 |
| `--max-model-len` / `--max-num-seqs` / `--max-num-batched-tokens` | 65536 / 128 / 8192 | 同 | 同 | 与 `up_buffer_size` 要匹配 |
| `--gpu-memory-utilization` | 0.90 | 0.90 | 0.90 | 参考 tcp 曾是 0.95，这里取更保守的 0.90 |
| `--trust-remote-code` | 开 | 开 | 开 | |
| `--override-generation-config` | `{"temperature": 0, "topk": 1}` | 同 | 同 | `topk` 沿用参考脚本写法，HF 侧有效键是 `top_k`，见 §6 |

### 2.3 环境变量

| 变量 | 值（tcp） | 值（rdma） | 作用 |
| --- | --- | --- | --- |
| `VLLM_HOST_IP` | `76.76.26.194`(edge) / `76.76.26.189`(cloud) | 同 | 本机业务 IP |
| `ASCEND_RT_VISIBLE_DEVICES` | `4` / `0,1` / `4,5` | 同 | 卡位 |
| `GLOO_SOCKET_IFNAME` / `TP_SOCKET_IFNAME` / `HCCL_SOCKET_IFNAME` | `enp189s0f0` | 同 | 业务网 |
| `UCX_TLS` | **`tcp,self`** | **`rc_verbs,tcp,self`** | **tcp/rdma 的唯一区别** |
| `UCX_NET_DEVICES` | `enp189s0f0` | `rocep67s0f0:1,enp189s0f0` | 建链网卡 |
| `UCX_PREFIX` | `/vllm-workspace/ucx` | 同 | 补 `bin/lib/lib64`；显式置空则沿用既有库路径 |
| `LD_LIBRARY_PATH` | mooncake site-packages 前置，UCX lib 保留 | 同 | 两者都要 |
| `LD_PRELOAD` | `libjemalloc.so.2` | 同 | 仅云侧脚本设置 |
| `HCCL_OP_EXPANSION_MODE`/`HCCL_BUFFSIZE`/`OMP_*`/`TASK_QUEUE_ENABLE`/`PYTORCH_NPU_ALLOC_CONF`/`VLLM_WORKER_MULTIPROC_METHOD` | 同参考 | 同 | 云侧脚本设置 |
| `VLLM_LOGGING_LEVEL` | `ERROR`（可覆盖） | 同 | 首轮建议 `INFO` |
| `VLLM_LWD_LOG_LEVEL` | 未设（默认 `DISABLE`） | 同 | `[lwd]` 独立日志流，排查时置 `INFO` |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | 注释掉 | 同 | 参考脚本即注释状态 |

### 2.4 外部依赖

- **UCX 运行库**（两种变体都要）：host 数据面是 ctypes 直接调 `libucp`，缺库时报
  "set LD_LIBRARY_PATH to the UCX install lib directory"。`/vllm-workspace/ucx` 由 `UCX_PREFIX` 引入。
- **Mooncake**：P/D 的 KV 传输仍用原生气泡 `MooncakeConnectorV1`，需要 `mooncake` 包与
  `LD_LIBRARY_PATH` 指向其 site-packages。
- **边端租户密钥**：`/vllm-workspace/lwd_tenant.key`（PD 下 `tenant_key_file` 不可省略）。
- **代码目录**：`/vllm-workspace/{vllm,vllm-ascend}`，两仓切到同一 `feat/pd-sep-lwd-v0.30.0`。
- **模型**：`/weight/Qwen3.6-27B/`。

---

## 3. 端口与卡位规划

| 用途 | edge A `76.76.26.194` | cloud B `76.76.26.189` |
| --- | --- | --- |
| NPU | `4` | P `0,1`；D `4,5` |
| OpenAI API | `9060` | P `8560`；D `8561` |
| PD 协调 `coord_port`（proxy 目标） | — | P `8660`；D `8661` |
| 控制面 `ctrl_port` | —（边端不监听） | P `19180`；D `19182` |
| 数据面 `data_port`（UCX listener） | — | P `19600`；D `19700` |
| PD proxy | — | `8500` |
| Mooncake `kv_port` | — | P `24679`；D `24680` |
| UCX 网卡 | `enp189s0f0` | RDMA `rocep67s0f0:1` + 业务 `enp189s0f0` |

沿用参考脚本的编号，便于和旧部署对照；**切换前先停掉原 sidecar 部署**，两套共用 NPU 与业务端口。

---

## 4. 目录内容与启动

```
30rc_1124/
├── README.md                 # 本文件：参数分析 + 部署 + 验证
├── check_configs.py          # 拓扑/代理配置自检（标准库，无第三方依赖）
├── tcp/
│   ├── env.sh                # 公共环境（UCX_TLS=tcp,self）、--dry-run 支持
│   ├── lwd_topology.yaml     # 拓扑（与 rdma 解析内容相同）
│   ├── pd_proxy.json         # PD proxy 的 P/D 目标表
│   ├── edge.sh               # 边端 A
│   ├── cloud_p.sh            # 云 B：prefill
│   ├── cloud_d.sh            # 云 B：decode
│   ├── proxy.sh              # 云 B：PD proxy
│   └── README.md
└── rdma/                     # 同结构；差异只在 env.sh 的 UCX_TLS/UCX_NET_DEVICES
```

把选中的**整个目录**拷到两台机器（例如 `/bash_dir/pds/30rc_1124/tcp`）。
脚本按自身所在目录解析 `lwd_topology.yaml`，可以从任意工作目录启动，但
**两端必须用同一份 YAML**（改地址时两边一起改）。

改完地址/端口后先跑一次配置自检（纯标准库，不需要 vllm 依赖）：

```bash
python3 check_configs.py          # 校验 tcp/ 与 rdma/ 两份拓扑与 proxy 配置
python3 check_configs.py rdma     # 只校验一个变体
```

启动顺序（每条一个终端，保持运行）：

```bash
# 云端 B
bash cloud_p.sh
bash cloud_d.sh
bash proxy.sh
# 边端 A（等云端三条都起来后再启）
bash edge.sh
```

host 数据面**没有跨侧 barrier**：云端只报 listener 已绑定、异步接纳边端；
边端会等自己的链路（握手超时 + 5s）。所以边端必须最后起。

请求入口是**边端 `9060`**；边端把 PD 信封投给 proxy `8500`，proxy 按
`pd_proxy.json` 分别打到 P `8660` / D `8661`。

---

## 5. 自检

先跑配置自检（改完地址端口必跑一次）：

```bash
python3 check_configs.py
```

它按 `vllm/config/lwd/topology.py` 与 `pd/proxy.py` 的规则逐条复核，包括
`extra="forbid"`、host 布局（边端不许有 `ctrl_port`/`data_port`/`ranks`）、PD 八条硬约束、
同址端口冲突、host 内存预算，以及 proxy 的 `cloud_id` 不相交。

四个入口都支持 `--dry-run`，只打印命令与关键环境变量，不加载模型、不建链：

```bash
bash cloud_p.sh --dry-run
bash edge.sh --dry-run
```

上线前建议按顺序确认：

1. `check_configs.py` 全绿；
2. `--dry-run` 里 `--additional-config` 是**内联 JSON**，且 `path` 是绝对路径；
3. 两侧启动日志的 `[LWD][config] ... digest=...` 一致（`VLLM_LOGGING_LEVEL=INFO`）；
4. P/D 日志出现 `[LWD][ucx-block] ... up_buffer_size=256`，且 `source` 不是"低于推导值"的告警；
5. 云侧 `data_port` 已监听：`ss -lntp | grep -E '19600|19700'`；
6. 边端日志出现 UP/DOWN 链路 ready；
7. 发一次请求确认**同时**经过 P 和 D（P 有 prefix 命中、D 有解码输出）。

---

## 6. 已知风险与首轮排查顺序

按"最可能先炸"排序：

1. **`--async-scheduling` 在边端未验证**。v0.30.0 的 LWD 代码对 `async_scheduling` 零处理，
   而边端 `execute_model` 是劫持实现。若边端起不来或首个请求挂住，**先去掉边端的
   `--async-scheduling`**。
2. **`kv_port` 落在临时端口段**。v0.30.0 的 AscendDirectTransport 会在
   `[20000, 20000+1000*npu)` 内随机取 KV 端口；8 卡机器上 `24679/24680` 正好落在这个区间。
   若 Mooncake 握手偶发失败，把两侧改成 `28679/28680`（`--kv-transfer-config` 与
   `kv_connector_extra_config` 一起改）。
3. **`topk` 是参考脚本的笔误**。HF `GenerationConfig` 的有效键是 `top_k`，`topk` 会被当作
   未使用属性静默忽略。此处保留参考写法以保持行为一致；要真正生效请改成 `"top_k": 1`。
4. **D 侧 `FULL_DECODE_ONLY` 与 LWD cloud 的数值一致性无测试覆盖**。若精度异常，
   先换 `PIECEWISE` 或去掉 `--compilation-config` 复测。
5. **端到端无重连恢复**：对端重启后 seqno 不复位，必须整套重启（`lwd_host_data_plane.md:128-132`）。
6. **P 必须 `prefill_tp >= decode_tp`**，且 `decode` 侧 `pp=1`；本脚本两侧 TP=2 满足。
7. **PD 下 Engram 模型被明确拒绝**（Qwen3.6-27B 不涉及）；PCP/DCP 必须为 1，PP 必须为 1。
8. **将来要多机多 DP 时**，LWD 会把 `--data-parallel-address` 的默认值从 `--master-addr`
   改成回环地址，跨机 DP 必须显式指定。当前每实例单机 DP=1，不受影响。
9. **`lwd_config` 与 vllm-ascend 版本必须配套**：vllm-ascend 会把 `lwd_config` 从
   additional-config 中剔除（交由 vLLM 解析）。两仓分支不一致时 LWD 可能静默失效。
10. Mooncake 若报找不到库，按 v0.30.0 文档把 `/usr/local/lib`、`/usr/local/lib64` 也加进
    `LD_LIBRARY_PATH`（本脚本保持参考脚本的最前置写法，未改动系统库搜索顺序）。

---

## 7. 本目录做过的验证

| 验证项 | 方法 | 结果 |
| --- | --- | --- |
| shell 语法 | `bash -n`（Git Bash，10 个脚本） | 全通过 |
| 入口可执行性 | 四个入口 `--dry-run` 实际跑通，核对拼出的命令与环境 | 全通过 |
| tcp/rdma 差异 | dry-run 输出对比 | 仅 `UCX_TLS` 与 `UCX_NET_DEVICES` 不同 |
| 拓扑 YAML | 按 `vllm/config/lwd/topology.py` 规则逐条复核（含 `extra="forbid"`、host 布局、PD 规则、端口冲突、内存预算） | 通过；host 内存估算 2304 MiB/rank（预算 12288） |
| proxy JSON | 按 `vllm/v1/lwd_control/pd/proxy.py` 规则复核（cloud_id 不相交、url 合法性等） | 通过 |
| 校验器有效性 | 14 个故意注入的故障（缺 `up_buffer_size`、边端带 `ctrl_port`、`ranks` 残留、双 P、缺 tenant key、端口撞车、重复 YAML key、proxy cloud_id 重叠等） | 全部被拒 |
| 交付的 `check_configs.py` | 在交付目录内以相对路径运行 | 两份变体均 PASS |
| 未验证（需真机） | UCX 实际是否走 RoCE、NPU 数值一致性、流式/事件时序、吞吐与 TPOT | 待实测 |

分析依据的代码位置：`vllm/config/lwd/{entry,topology,api,coordination}.py`、
`vllm/v1/lwd_control/pd/{proxy.py,edge_runtime.py}`、`vllm/v1/lwd_comm/data_comm/ucx_channel.py`、
`vllm/docs/design/lwd_host_data_plane.md`、`vllm/docs/design/lwd_pd_host.md`、`vllm/examples/lwd/`、
`vllm-ascend/vllm_ascend/distributed/lwd_data_plane/host/lwd_host_channels.py`、
`vllm_ascend/worker/lwd_{edge_model,cloud_model}/`。
