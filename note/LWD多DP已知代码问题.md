# `feat/pd-sep-lwd-v0.30.0` 多 DP 的 PD 分离：已知代码问题

本文记录"用 LWD 边云 PD 跑多 DP"过程中撞到的**全部阻断点**，每条都给到代码行号与日志证据，
供后续排查/上游修复参考。结论先行：

> **该分支不支持 dense 模型的多 DP PD。**
> 多 DP 的引擎核心路径（`LwdCloudDPEngineCore`）只为 **MoE** 模型接通；
> dense 模型会被 `reconfigure_for_independent_dp_rank()` 把 DP 抹成 1，
> 随后与 PD 的 KV 语义产生三处冲突，其中 I-2 与 I-3 无法用配置绕过。

复现环境：Qwen3.6-27B（dense，`Qwen3_5ForConditionalGeneration`，含 GDN/Mamba 层），
194 边端 + 195 云端共 5 卡，分支 `feat/pd-sep-lwd-v0.30.0`，vLLM v0.30.0。

---

## 0. 先看结论：LWD 的多 DP 只为 MoE 接通

```python
# vllm/v1/engine/core.py:1355-1390
if lwd.role == "cloud" and vllm_config.model_config.is_moe and (lwd_dp_size > 1):
    parallel_config.data_parallel_rank = dp_rank          # MoE：保留 rank
    engine_core = LwdCloudDPEngineCore(*args, lwd_dp_size=lwd_dp_size, **kwargs)
else:
    parallel_config.reconfigure_for_independent_dp_rank()  # 非 MoE：DP 抹成 1
    engine_core = LwdCloudEngineCore(*args, engine_index=dp_rank, lwd_dp_size=lwd_dp_size, **kwargs)
```

```python
# vllm/config/parallel.py:1104-1113
def reconfigure_for_independent_dp_rank(self) -> None:
    """Reconfigure for a single independent non-MoE DP rank."""
    nnodes = self.nnodes_within_dp
    node_rank = self.node_rank_within_dp
    self.data_parallel_size = 1        # ← DP 被抹平
    self.data_parallel_size_local = 1
    self.data_parallel_rank = 0        # ← rank 被清零
    self.nnodes = nnodes
    self.node_rank = node_rank
```

LWD 自己知道这件事——它把**真实**的 DP 通过显式参数（`lwd_dp_size` / `engine_index`）带进去，
并在注释里点名不要依赖那两个字段：

```python
# vllm/v1/lwd_control/lwd_control_plane.py:102-103
# Global DP size. Read it before non-MoE engine processes reset
# ParallelConfig.data_parallel_size to 1. Not `data_parallel_rank`, which
# non-MoE engine processes reset to 0.
```

问题在于：**它自己避开了，但 Pillar 之外的组件（KV 连接器、平台校验）没有避开。**

---

## I-1 MTP 的 draft 配置继承了 `kv_transfer_config`，触发 DP 校验冲突

**现象**

```
ValueError: KV transfer 'prefill' config has a conflicting data parallel size.
Expected 1, but got 2.
```

**调用链**

```
worker.load_model()                             vllm_ascend/worker/worker.py:865
└─ model_runner.load_model()                    vllm_ascend/worker/model_runner_v1.py:4416
   └─ self.drafter.load_model(self.model)       ← 只有开 MTP 才走这里
      └─ _create_draft_vllm_config()            vllm_ascend/spec_decode/llm_base_proposer.py:178
         └─ base = replace(base, attention_config=...)   vllm/v1/spec_decode/llm_base_proposer.py:1292
            └─ return cls(**dataclass_dict)     vllm/config/utils.py:119-127
               └─ VllmConfig.__init__ → __post_init__    vllm/config/vllm.py:1887
                  └─ current_platform.check_and_update_config(self)
                     └─ check_kv_extra_config()   vllm_ascend/platform.py:820
                        └─ raise ValueError       vllm_ascend/utils.py:1592-1599
```

**根因**：`replace()` 会重建 `VllmConfig` 并重跑全部校验；draft 那份配置的
`parallel_config.data_parallel_size` 是 1，而继承来的 `kv_connector_extra_config.dp_size` 是 2。

**影响**：`DP>1` + MTP 不可用。

**配置层规避**：去掉 `--speculative-config`（本文所有包的做法）。

**建议改法**：`_create_draft_vllm_config()` 里让 draft 不带 KV 传输语义：
`replace(base, kv_transfer_config=None, attention_config=...)`。

---

## I-2 引擎核心握手后重跑 `__post_init__`，DP 校验必然失败 ⚠️ 最关键

**现象**（第 4 次拉起，decode 侧，已过 KV 分配与图捕获）

```
LwdCloudEngineCore.__init__                                 lwd_cloud_engine.py:158
└─ lwd_base_engine.py:95 → core.py:1089 → core.py:1059     with self._perform_handshakes(...)
   └─ core.py:1206-1207                                    # Update config which may have changed
                                                           vllm_config.__post_init__()
      └─ vllm/config/vllm.py:1887                          current_platform.check_and_update_config
         └─ vllm_ascend/platform.py:515                    _fix_incompatible_config
            └─ vllm_ascend/platform.py:820                 check_kv_extra_config(vllm_config)
               └─ vllm_ascend/utils.py:1604                _check("decode", ...)
ValueError: KV transfer 'decode' config has a conflicting data parallel size.
Expected 1, but got 2.
```

**根因**：`core.py:1377` 先调 `reconfigure_for_independent_dp_rank()` 把 DP 抹成 1，
再构造引擎核心；引擎核心在握手结束时**有意**重跑 `__post_init__`（注释写着
*"Update config which may have changed from the handshake"*），于是
`check_kv_extra_config` 拿"已抹平的 DP=1"去比"按真实布局写的 dp_size=2"。

**影响**：**dense 模型下任何 DP>1 的 PD 引擎（producer 与 consumer 都一样）必挂**，
且发生在 KV 分配、图捕获成功之后——前期看起来一切正常，最后一步才炸。

**和 I-1 的关系**：I-1 是"draft 会重建配置"，I-2 是"引擎核心握手后也会重建配置"。
即使关掉 MTP，I-2 依然存在。

**佐证**：同一段 `platform.py` 对 `engine_id` 做了幂等保护，说明作者知道
`__post_init__` 会被多次执行；但 KV 校验没有做同样的归一化：

```python
# vllm_ascend/platform.py:818-823
check_kv_extra_config(vllm_config)
if not getattr(vllm_config.kv_transfer_config, "_engine_id_patched", False):
    vllm_config.kv_transfer_config.engine_id = f"{...}-{uuid4().hex}"
    vllm_config.kv_transfer_config._engine_id_patched = True
```

**配置层规避**：无。`dp_size` 写 2 → 第二次校验失败；写 1 → 第一次校验
（API server 侧，DP=2）失败。

**建议改法**（任选其一）：
1. `reconfigure_for_independent_dp_rank()` 里同步把
   `kv_connector_extra_config.{prefill,decode}.dp_size` 归一化为 1；
2. `check_kv_extra_config` 跳过"已被 LWD 重置"的配置（例如比对 `data_parallel_index`，
   或加一个 `_lwd_dp_normalized` 标记做幂等）；
3. `_perform_handshakes` 重跑前先恢复 DP 字段。

---

## I-3 producer 的 KV 握手端口用 `data_parallel_rank` 派生，DP>1 撞端口

**现象**

```
(Worker pid=39916) Mooncake KVCacheSendingThread encountered exception.
  Thread: tp_rank=0, pp_rank=0, listening_path=tcp://76.76.26.195:14570
  Error: Address already in use (addr='tcp://76.76.26.195:14570')
RuntimeError: KV Cache sending/receiving thread failed to start.
```

两个 DP rank 打印**同一个** `listening_path`。

**位置**

```python
# vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py:1762-1768
self.side_channel_port = (
    vllm_config.kv_transfer_config.kv_port
    + vllm_config.parallel_config.data_parallel_rank      # ← 被清零的字段
    * tensor_parallel_size * pipeline_parallel_size * pcp_size
)
# :319 发送线程是 ROUTER → bind=True（:4126 make_zmq_socket(bind=socket_type == zmq.ROUTER)）
```

**影响**：producer（prefill）DP>1 必挂。consumer（decode）不 bind
（`:1546-1551` 用 `zmq.REQ, bind=False`），端口层面本来是安全的——但仍受 I-2 阻断。

**建议改法**：改用 `parallel_config.data_parallel_index`（LWD 自己就是这么做的）。

---

## I-4 「P 单副本 + D 多 DP」的非对称方案同样不可行

基于 I-3 的推论（只有 producer 会 bind）曾设计过非对称布局
（prefill `DP=1×TP=2`、decode `DP=2×TP=1`）。拓扑与参数都能通过校验
（`check_kv_extra_config` 只校验各自角色那一块，`utils.py:1601-1604`），
但 decode 侧仍在 I-2 处失败。

**结论**：端口不是唯一障碍，I-2 与角色无关，非对称救不回来。

---

## I-5 `qwen3_5_mtp` 会被静默改写成 `mtp`

```python
# vllm/config/speculative.py:1121-1125
if self.method in get_args(MTPModelTypes) and self.method != "mtp":
    logger.warning("method `%s` is deprecated and replaced with mtp.", self.method)
    self.method = "mtp"
```

**影响**：`--speculative-config '{"method":"qwen3_5_mtp",...}'` 与 `"mtp"` 等价，
换方法名**不能**绕过 I-1；旧脚本里的 `qwen3_5_mtp` 写法仍可用（只有一条 warning）。

---

## I-6 云端加载的是完整模型，TP=1 时显存极紧（非 bug，但决定可用性）

```
[Lwd][cloud-model] loaded layers=? first=...layers.0 last=...layers.63 — full load should cover every layer from 0
Loading model weights took 50.2267 GB
```

每个 cloud rank 都持有 **完整 64 层**（50.2 GB ≈ 46.8 GiB），不是层切。64 GB 卡上：

| gpu-memory-utilization | 权重 | 激活/图捕获 | 剩给 KV |
| --- | --- | --- | --- |
| 0.90 | 46.8 GiB | ≈3.5 GiB | ≈3.3 GiB |
| 0.95 | 46.8 GiB | ≈3.5 GiB | ≈6.7 GiB |

`max seq len 65536` 需要 4.38 GiB → 0.90 时直接失败（首次拉起的报错）。

**含义**：dense 27B 在单卡 64 GB 上做 TP=1 副本，KV 只有 3~6 GiB，多 DP 即使能跑，
并发能力也很有限。**dense + TP=1 本身就不划算。**

---

## I-7 Mamba(GDN) block 数上限约束 `max_num_seqs`

```
ValueError: max_num_seqs (128) exceeds available Mamba cache blocks (123).
Each decode sequence requires one Mamba cache block, so CUDA graph capture cannot proceed.
```

**根因**：模型含 GDN 线性注意力层，每个 decode 序列占一个 mamba block；
KV 池 6.7 GiB 时约 123 个 block。

**规避**：`--max-num-seqs` ≤ blocks（本仓包用 64）。**每次调大 KV 池都要重看这个数。**

---

## I-8 环境/脚本层的两个小问题（已修）

| 项 | 说明 |
| --- | --- |
| `LD_PRELOAD` 无条件设置 | 目标机没有 `libjemalloc.so.2`，每进程刷一条 `ld.so: object ... ignored` 的 ERROR，淹没真错误。改为"文件存在才设置" |
| `causal_conv1d_update` 缺失 | NPU Triton 版不可用，回退 PyTorch 实现并 *"stalls ACL graph capture at decode-FULL"*：decode 启动比 prefill 多约 60 s |

---

## 还想在多 DP 上推进时的选择

| 路径 | 可行性 |
| --- | --- |
| dense 模型 + LWD + DP>1 | ❌ 需要修掉 I-2（必须）+ I-3（producer 需要） |
| **MoE 模型** + LWD + DP>1 | ✅ 走 `LwdCloudDPEngineCore`，DP rank 被保留，I-2/I-3 都不会触发 |
| dense 模型 + 单 DP、TP 分摊 | ✅ 见 `assets/c_home/bash/pds/30rc_1124/sidecar`（DP=1、TP=2，已验证） |
| dense 模型 + 非 LWD 的 PD（Mooncake） | DP 语义同样是 `reconfigure_for_independent_dp_rank` 路径，**先验证再说**，不要假定可用 |

## 快速自检：这份配置能不能跑多 DP

1. 模型是 MoE 吗？——`model_config.is_moe` 为真才有多 DP 引擎核心路径。
2. 拉起日志里有没有 `LwdCloudDPEngineCore`？——没有就说明走的是抹平分支。
3. 失败是否发生在"KV 分配/图捕获之后"？——是则大概率是 I-2，而不是你的参数。
4. 报错里出现 `Expected 1, but got 2`？——那就是 I-1/I-2 家族，改参数无解。
5. 报错里出现 `Address already in use` 且端口相同？——I-3，只在 producer 侧。

## 证据来源

| 日志 | 阶段 | 结果 |
| --- | --- | --- |
| `1126-1112` | dp2 × tp1，带 MTP，util 0.90 | I-1（两侧都挂） |
| `1126-1147` | 去 MTP，util 0.90 → 0.95 | prefill: I-3；decode: I-7 |
| `1126-1413` | 降 `max_num_seqs` 到 64 | prefill: I-3；decode: KV 3.71 GiB 下 I-7 系（参数未全生效） |
| `1126-1433` | 非对称 P(DP1×TP2)/D(DP2×TP1) | decode: **I-2**（KV 6.76 GiB、图捕获均通过） |
