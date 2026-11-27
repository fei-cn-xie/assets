# mooncake_pd2_tp1_195：集中式 PD 分离，两侧都是 2 DP × 1 TP

**集中式 = 没有边端、没有 LWD 控制面**：prefill 与 decode 是两个独立的 `vllm serve`，
KV 通过 Mooncake 连接器点对点传输，前面挂上游的负载均衡 proxy。

```
client ──9060──▶ LB proxy (195, CPU)
                    ├── prefill 9160 (195 NPU 0,1)  DP=2 × TP=1  kv_producer base 30000
                    └── decode  9161 (195 NPU 2,3)  DP=2 × TP=1  kv_consumer base 30100
```

目录名 = `{连接器}_pd{每侧DP数}_tp{每副本TP数}_{主机}`。

## 引擎参数与边云 DP=2 包一一对齐

本包的引擎 flag 与**已确认可用的** `../dp2_tp1_194_195`（边云式 DP=2）逐项相同，
这样两个部署之间的任何行为差异都只能来自 LWD 本身，而不是我漏配了某个参数。
相对那份脚本，只有集中式 PD 固有的差异：

| 项 | 边云包 | 本包 | 说明 |
| --- | --- | --- | --- |
| `--port` | 8600 / 8601 | 9160 / 9161 | 部署不同 |
| `--master-port` | 29525 / 29526 | 29535 / 29536 | 集中式另取一对，避免与边云部署冲突 |
| `--additional-config` | 有（LWD 拓扑） | **无** | 集中式没有 LWD |
| `--kv-transfer-config` | kv_rank + kv_port，无 engine_id | 额外带 `engine_id` 0/1 | 集中式 Mooncake 参考需要它 |
| 其余全部（TP/DP/DP rpc/长度/并发/内存/调度/图模式/MTP） | — | **完全相同** | — |

## 卡与端口

| 角色 | 卡 | 并行 | API | master | DP rpc | KV 基址（实际监听） |
| --- | --- | --- | --- | --- | --- | --- |
| proxy | — | — | **9060** | — | — | — |
| prefill | NPU 0,1 | **DP=2 × TP=1** | 9160 | 29535 | 29600 | 30000（→ 30000 / 30001） |
| decode | NPU 2,3 | **DP=2 × TP=1** | 9161 | 29536 | 29601 | 30100（consumer 只连不绑） |

`--data-parallel-size 2` 时 vLLM 默认让 `api_server_count = data_parallel_size`，
即每个引擎有两个 API server 进程共享端口、内部做 LB；proxy 每侧只需一个端点。

## 文件

| 文件 | 作用 |
| --- | --- |
| `env.sh` | 运行时环境与全部常量（端口 / 内存 / MTP 开关），支持 `--dry-run` |
| `prefill.sh` | P 引擎（`kv_producer`） |
| `decode.sh` | D 引擎（`kv_consumer` + `FULL_DECODE_ONLY`） |
| `start_proxy.sh` | 调用 vllm-ascend 自带 `examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py`，仓内无 Python 胶水 |
| `check_configs.py` | 静态自检（见文末），拉起**前**跑 |
| `verify_dp.sh` | 只读自检（见文末），拉起**后**跑：进程数 / 每 rank worker / KV 端口归属 / 端点 / 一次代理请求 |

## 内存：TP=1 的硬约束与一个实测差异

TP=1 意味着每个副本把完整 64 层模型装进一张卡（50.2 GB ≈ 46.8 GiB）。
卡总内存 60.96 GiB，启动时实际空闲 60.57 GiB；当前默认 `PD_GPU_MEM_UTIL=0.98`
→ 预算 59.74 GiB，扣掉权重与激活后剩 2.13 GiB 给 KV。

同机实测（util 0.95、`max_model_len 32768`、MTP 开）：

| 场景 | Available KV |
| --- | --- |
| 边云部署（MTP 开） | 引擎正常拉起 |
| **集中式部署（MTP 开）** | **0.30 GiB** → 起不来（32768 需要 3.02 GiB） |
| 集中式部署若关掉 MTP（预估） | ≈6.8 GiB → 够用 |

同样是 64 GB 卡、同样 TP=1、同样的 flag，边云能起而集中式起不来，原因在**代码结构**而不是脚本：

```python
# vllm/model_executor/models/utils.py:864-875
LWD_MODULE_OWNERSHIP = {
    "model.embed_tokens": both,          # 边、云都加载
    "lm_head":            both,
    "model.layers":       cloud only,    # 语言模型层只在云侧
    "model.norm":         cloud only,
    "visual":             edge only,     # ← 视觉塔只有边端 owns
    "model.mtp":          cloud only,
}
```

这张表有三处消费者，所以边端与云端的权重是**物理不同**的两份：

| 消费者 | 行为 |
| --- | --- |
| 建模型 | `make_layers` 的 LWD 分支 + `lwd_skips_module()` 守卫，云进程**根本不构造** `visual.*` |
| 加载权重 | `model_loader/lwd_weight_filter.py` 由同一张表推导 skip keys，checkpoint 里的 `visual.*` 张量**不读**（日志 `[LWD] role=cloud skips loading tensors of visual`，`default_loader.py:367`） |
| 离线切分 | `scripts/lwd/split_lwd_checkpoint.py` 用同一张表 |

所以**边云部署的视觉能力在边端**：edge 拥有 `visual`，编码后经 UP 把 embeddings 送给云；云侧只跑语言模型 + MTP。
你用边云包拉起来的服务当然有视觉能力——那部分由边端承担，云端的卡上从来没有视觉塔。

集中式没有角色切分：**每个 DP 副本都要装视觉塔 + 语言模型 + MTP draft**，而且它的 profiling 还要跑一遍
编码器（`MultimodalConfig.skip_mm_profiling` 默认 False，`arg_utils.py:622/1427`）。这就是 6.76 GiB 与 0.25 GiB 的差距来源。

最新一次实拉（util 0.98、`max_model_len 512`、MTP 开）的 worker 输出：

```
Desired GPU memory utilization is (0.98, 59.74 GiB). Actual usage: 54.25 GiB for weights,
3.27 GiB for peak activation, 0.09 GiB for non-torch memory, 0.58 GiB for NPU graph memory.
Current KV cache memory: 2.13 GiB   ->   NPU KV cache size: 1,152 tokens (512 上下文下 2.25x 并发)
```

即：这个包在 util 0.98 下真正能用的 KV 池是 **2.13 GiB**，而 32768 上下文需要 3.02 GiB ——
所以默认给的是 512（冒烟值），要放大必须同时腾出显存（关 MTP / 关视觉 profiling / 提 util / 改 TP=2）。

单 token KV ≈ 96.6 KiB（3.02 GiB / 32768），据此估算：可用 KV 1 GiB ≈ 10.8k tokens。

模型含 GDN(Mamba) 层，每个 decode 序列占一个 Mamba block；`PD_MAX_NUM_SEQS=32` 是实测可用值
（更早的 TP=1 配置在 64 时被 Mamba block 数挡住）。

调参阶梯（每次改完看启动日志的 `Available KV cache memory` 再决定下一步）：

| 想要 | 做法 |
| --- | --- |
| 先能起来 | `PD_ENABLE_MTP=0`，必要时再加 `PD_SKIP_MM_PROFILING=1` |
| 更长上下文 | 确认 KV 余量后 `PD_MAX_MODEL_LEN=65536`（需 6.05 GiB） |
| 更高并发 | `PD_MAX_NUM_SEQS=64`，但需 `Available Mamba cache blocks ≥ 64` |
| 再挤显存 | `PD_GPU_MEM_UTIL=0.97`（+1.2 GiB），注意碎片风险 |

## 集中式的三个内存开关

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `PD_ENABLE_MTP` | 1 | 关掉可回收 draft 权重（边云包同款开关） |
| `PD_SKIP_MM_PROFILING` | 0 | 置 1 时加 `--skip-mm-profiling`：不做视觉编码器的 profiling 预留，省显存但图像请求的预留会退化 |
| `PD_MAX_MODEL_LEN` | 512 | 当前冒烟值；每 token ≈96.6 KiB，32768 需 3.02 GiB，正式服务按需调大 |
| `PD_MAX_NUM_SEQS` | 32 | 受 Mamba block 数限制，先看日志里的 `Available Mamba cache blocks` |
| `PD_MAX_NUM_BATCHED_TOKENS` | 8192 | 单步批窗口，两侧必须一致（校验器会拦） |
| `PD_GPU_MEM_UTIL` | 0.98 | 每 +0.01 ≈ +0.61 GiB KV；0.99 以上没给运行时留余量 |
| `PD_API_SERVER_COUNT` | 空（= DP 数 2） | 置 1 时只起一个前端托管两个 DP 引擎；排查启动握手时可用来排除多前端因素 |

边云包没有后两个的必要：云端不装视觉塔，draft 也是它唯一要担心的部分。

## 启动

```bash
python3 check_configs.py        # 静态自检

bash prefill.sh                 # 先 P
bash decode.sh                  # 再 D
bash start_proxy.sh             # 最后 proxy
```

两个引擎都打印 `Application startup complete` 后再打请求：

```bash
curl -s http://76.76.26.195:9060/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6","messages":[{"role":"user","content":"你好"}],"max_tokens":32}'
```

## 两侧都是 DP=2 时，KV 怎么对上 rank

这是本包最容易被误判的一点："P 的 rank0 做了 prefill、proxy 又把 decode 发给 D 的 rank1，KV 还对得上吗？"

对得上，依据在 `mooncake_connector.py`：

| 环节 | 代码 | 说明 |
| --- | --- | --- |
| 每个 rank 的 `engine_id` 唯一 | `vllm/v1/engine/core.py:1311-1316`，`engine_id += f"_dp{dp_rank}"` | 日志里能看到 `1-<uuid>_dp0` / `_dp1` |
| 每个 rank 各自广播握手端口 | `mooncake_connector.py:2109-2114` | 生产者里是 `kv_port + dp_rank`（本包 30000 / 30001） |
| 生产者在回包里带上自己的三元组 | `mooncake_connector.py:127-130`（`remote_host` / `remote_port` / `remote_engine_id`） | 谁做的 prefill，就回填谁自己的地址 |
| 消费者优先用回包里的值 | `mooncake_connector.py:3979-3990` | 映射表只是修正；查不到时**直接**用回包里的 `host` / `engine_id` |

所以 D 的任意 rank 都能按回包找到**真正做过 prefill 的那个 P rank**，不要求两侧 rank 编号相同。

另外 `kv_connector_extra_config` 里的 `prefill/decode: {dp_size, tp_size}` **不参与路由**：
`_prefill_dp_size` / `_decode_dp_size` 在 `mooncake_connector.py:2237/2245` 赋值后全文件再无引用，
它们只用于配置一致性校验（也就是第一处修复 `check_kv_extra_config` 比对的那两个值）。

## 三处修复与本包的对照

本包在集中式路径上会依次暴露三个问题，`feat/pd-sep-lwd-v0.30.0` 全有，
`fix/lwd-multidp-dense-pd` 全部修掉：

| 顺序 | 失败现象 | 根因 | 修复（vllm / vllm-ascend） |
| --- | --- | --- | --- |
| 1 | `KV transfer 'prefill' config has a conflicting data parallel size. Expected 1, but got 2` | dense 引擎把 DP 折叠成 1，二次校验时对不上 | `data_parallel_size_global` + `check_kv_extra_config` |
| 2 | `ZMQError: Address already in use (…:30000)`，或握手后同样的 `Expected 1, but got 2` | 生产者各 rank 共用同一个 KV 端口 | 端口改用 `data_parallel_index` |
| 3 | 某一个 DP rank 5 分钟拿不到 init（`Did not receive response from front-end process within 5 minutes`），随后 `DP Coordinator received unexpected message while waiting for engines to subscribe` | 非 LWD 的 dense DP 路径没把 `dp_rank` 作为 `engine_index` 传下去，两个 rank 的 ZMQ 身份都是 `\x00\x00`；ROUTER 对同一身份只保留最后一条连接 | `core.py:1399` 传 `engine_index=dp_rank` |

第 3 条**已在 195 上验证修复**（2026-11-27）：两个 `EngineCore_DP0` / `EngineCore_DP1` 都输出了 `All engine subscriptions received by DP coordinator`，服务正常拉起。

第 3 个问题**只影响非 LWD 的 DP>1 部署**：边云包正常，是因为 LWD 分支本来就传了
`engine_index=dp_rank`（`core.py:1379`），而 MoE 的 `DPEngineCoreProc` 会自己按
`data_parallel_rank` 取值——只有 dense 的非 LWD 路径既没传、又把 rank 折叠成了 0。

同一个 run 里到底是 DP0 还是 DP1 报错会**随机互换**，这是身份冲突最典型的特征：
谁先连上前端、谁的路由被后来的连接顶掉，取决于进程启动顺序。

## 如何确认 DP=2 真的生效

拉起后直接跑：

```bash
bash verify_dp.sh          # 全量
bash verify_dp.sh --quick  # 跳过最后那次代理请求
```

它只读、不启停任何进程，失败时会指名哪一项不达标。手工核对的话看下表。

`curl` 能通并不能说明是 DP=2（`DP=1 × TP=2` 也一样能应答），看这些**两者必然不同**的量：

| 观测量 | DP=2 × TP=1 | DP=1 × TP=2 |
| --- | --- | --- |
| EngineCore 进程 | **2 个**（`EngineCore_DP0/DP1`） | 1 个（`EngineCore`） |
| 每 rank `world_size` | **1** | **2** |
| 每卡权重 | ≈46.8 GiB | ≈23.4 GiB |
| KV 端口归属 | 30000、30001 分属两个 PID | 同属一个 PID |

⚠️ 不要用引擎配置行的 `data_parallel_size` 判断——真·DP=2 的 dense 部署那里也打印 1。

## `check_configs.py` 检查什么

- 两侧都是 DP=2 × TP=1，参数取自 `env.sh`（不是各写一份字面量）
- **prefill 与 decode 的引擎 flag 一一相同**（只允许端口、KV 配置、图模式不同）——这条把"配置对齐"变成可校验的
- 两张卡的集合不重叠、总数 = DP × TP = 4
- 所有监听端口（proxy / 两侧 API / 两侧 master / 两侧 DP rpc / KV 区间）互不冲突
- KV 连接器角色、`engine_id` 不同、`extra_config` 的 dp/tp 与并行度一致
- 两侧共享的三个 knobs 都被引用；`max_num_seqs ≤ 32`

反向验证过：卡重叠、KV 基址相邻冲突、TP 改 2、两侧 knob 漂移、两侧 flag 不一致——都会被拦下并指名文件与值。
