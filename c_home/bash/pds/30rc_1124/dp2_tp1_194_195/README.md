# dp2_tp1_194_195：LWD 边云 PD，每个 cloud 2 个 DP × TP=1

在 [../sidecar](../sidecar/README.md)（P/D 各 1 个 TP=2 副本）的基础上改成**多 DP**：
195 的两对卡各自跑 **2 个 data-parallel 副本**，而不是 1 个 TP=2 副本。
edge 仍占 194 的第 4 号卡。

> 目录名 = `dp{每云DP数}_tp{每副本TP数}_{边端主机}_{云主机}`。
> 后续要别的规模（如 dp4、tp2）时，按本目录复制并改名，改法见文末「改成其它 DP 数」。

```
client ──8510──▶ edge (194, NPU 4, TP=1 DP=1)
                    │  pd_proxy_url
                    ▼
              proxy (195:8102)
                    ├── cloud 0 prefill (195 NPU 0,1)  DP=2 TP=1  → coord 8100 / 8101
                    └── cloud 1 decode  (195 NPU 2,3)  DP=2 TP=1  → coord 8110 / 8111
```

## 什么是多DP

- **DP（data parallel）**：把整个模型复制 N 份，每份独立处理不同请求；与 TP（切一层到多卡）正交，
  总卡数 = DP × TP。DP 提升吞吐（尤其 decode 阶段），不改善单请求延迟。
- **本栈的落地方式**：每个 cloud 实例在拓扑里有一份 `dp:` 列表，**每个 `dp_idx` 一套
  `ctrl_port` / `data_port` / `coord_port`**；proxy 用 `pd_proxy.json` 的 `target_dp` 指定打到哪个副本。
  官方同款示例：`vllm/examples/lwd/pd_host_2p4d.yaml`（P 2 DP、D 4 DP）。
- **允许 P/D 的 DP 数不同**：`topology.py:411-417` 的"DP 数必须一致"只在 **PD 关闭**时校验；
  PD 开启时不做此限制。**edge 必须保持 DP=1 / TP=1**（`lwd_control_plane.py:102-106`）。

## ⛔ 状态：Qwen3.6-27B（dense）下不可运行

三次拉起分别撞到三个阻断点，其中第 3 个**没有配置层的绕法**：

| # | 现象 | 根因 | 能否配置解决 |
| --- | --- | --- | --- |
| 1 | `KV transfer 'prefill' config has a conflicting data parallel size. Expected 1, but got 2` | MTP drafter 以 DP=1 重建 VllmConfig，与 `kv_connector_extra_config.dp_size=2` 冲突 | 去掉 MTP（已做） |
| 2 | `max_num_seqs (128) exceeds available Mamba cache blocks (123)` | TP=1 下 KV 池只够 123 个 mamba block | 降 `max_num_seqs`（已做，默认 64） |
| 3 | **`ZMQError: Address already in use (tcp://76.76.26.195:14570)`** | 连接器用 `parallel_config.data_parallel_rank` 算握手端口，而 LWD 的非 MoE 分支会把它清零，两个 DP rank 绑同一端口 | **不能** |

第 3 点的代码依据：

```python
# vllm_ascend/.../mooncake_connector.py:1762-1768
self.side_channel_port = (
    vllm_config.kv_transfer_config.kv_port
    + vllm_config.parallel_config.data_parallel_rank * tp * pp * pcp   # ← 被清零的就是它
)

# vllm/config/parallel.py:1109-1111（LWD 非 MoE 分支会调用）
def reconfigure_for_independent_dp_rank(self):
    self.data_parallel_size = 1
    self.data_parallel_size_local = 1
    self.data_parallel_rank = 0        # ← 所有 rank 都变成 0
```

而 LWD 只为 **MoE** 模型保留了 DP rank：

```python
# vllm/v1/engine/core.py:1355-1390
if lwd.role == "cloud" and vllm_config.model_config.is_moe and (lwd_dp_size > 1):
    parallel_config.data_parallel_rank = dp_rank      # MoE：保留
    engine_core = LwdCloudDPEngineCore(...)
else:
    parallel_config.reconfigure_for_independent_dp_rank()   # 非 MoE：清零
    engine_core = LwdCloudEngineCore(...)
```

Qwen3.6-27B 是 dense（`Qwen3_5ForConditionalGeneration`，MoE 变体另有
`Qwen3_5MoeForConditionalGeneration`），走 else 分支。**结论：dense 模型 + LWD + DP>1
在本版本不可用**，需要上游把连接器的 `data_parallel_rank` 换成 `data_parallel_index`
（LWD 自己就这么做，见 `lwd_control_plane.py:102-103` 的注释）。

所以多 DP 目前只有两条可行路径：

| 想要多 DP | 做法 |
| --- | --- |
| 换 **MoE** 模型 | LWD 的 `LwdCloudDPEngineCore` 分支会保留 DP rank，本目录的拓扑可直接复用 |
| 保持 dense 模型 | 用 [../sidecar](../sidecar/README.md)（DP=1、TP=2），或等上游修复 |

本目录保留下来作为**拓扑模板**：多 DP 的拓扑/proxy/端口分配/脚本骨架都是对的（`dp_size=2`
的控制面、4 条 host 数据面链路都实际建立成功过），只差连接器这一处代码限制。

## 已知限制：DP>1 不能开 MTP（第一次拉起失败的原因）

首次按本目录拉起时，两个 cloud 都在 `load_model` 阶段挂掉，报：

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for VllmConfig
  Value error, KV transfer 'prefill' config has a conflicting data parallel size.
  Expected 1, but got 2.
```

调用链与原因（可复核）：

```
vllm_ascend/spec_decode/llm_base_proposer.py:178  _create_draft_vllm_config()
  → vllm/v1/spec_decode/llm_base_proposer.py:1292  replace(vllm_config, ...)   # draft 以 DP=1 重建
    → vllm_ascend/utils.py:1580-1604               check_kv_extra_config()
        config_dp = kv_connector_extra_config["prefill"]["dp_size"]   # 我们写的是 2
        vllm_dp   = parallel_config.data_parallel_size                # draft 里是 1
        → raise "Expected 1, but got 2"
```

也就是说：`kv_connector_extra_config` 的 `dp_size` 必须等于**当前这份 VllmConfig** 的 DP，
而 MTP 的 draft 固定按 DP=1 重建，于是 DP>1 与 MTP 在这版代码里**互斥**。

因此本包的两个引擎都**不带 `--speculative-config`**。想要 MTP 就用
[../sidecar](../sidecar/README.md)（DP=1、TP=2）——那里 extra_config 的 `dp_size` 是 1，校验自然通过。

> 顺带确认：DP 本身没问题。失败前两侧都已经走到
> `[LWD] cloud control HTTP ready on 0.0.0.0 ports=[8100, 8101] cloud_id=0 dp_size=2`
> 和 `host data-plane links created ... dp_rank=0/1`，4 个 DP rank 的链路都建好了。

## 已知限制：TP=1 每卡显存极紧（第二次拉起失败的原因）

去掉 MTP 后，两个 cloud 都通过了配置校验、加载完权重，但死在 KV cache 分配：

```
[Lwd][cloud-model] loaded layers=? first=...layers.0 last=...layers.63 (pp=1 tp=1)
Loading model weights took 50.2267 GB
Available KV cache memory: 3.34 GiB          # prefill
ValueError: To serve at least one request with the model's max seq len (65536),
  (4.38 GiB KV cache is needed, which is larger than the available KV cache
   memory (3.34 GiB). ... estimated maximum model length is 49280
```

**关键事实：LWD 的 cloud 侧加载的是完整 64 层模型**（日志里 `layers.0 … layers.63`，
50.2 GB ≈ 46.8 GiB）。sidecar 用 TP=2 时权重被两卡分摊，所以 65536 从容；本包用 TP=1，
46.8 GiB 权重全压在一张 64 GB 卡上，KV 就只剩 3 GiB 出头。

算账（64 GB 卡）：

| gpu-memory-utilization | 预算 | 权重 | 激活/图捕获 | 剩给 KV |
| --- | --- | --- | --- | --- |
| 0.90 | 53.6 GiB | 46.8 GiB | ≈3.5 GiB | **≈3.3 GiB**（不够 4.38） |
| **0.95** | **56.6 GiB** | 46.8 GiB | ≈3.5 GiB | **≈6.3 GiB**（够） |

因此本包的两个引擎改成：

```bash
--max-model-len "${LWD_MAX_MODEL_LEN:-32768}"       # 32768 只需要 ≈2.2 GiB KV
--gpu-memory-utilization "${LWD_GPU_MEM_UTIL:-0.95}"  # 把 KV 池从 3.3 提到 ≈6.3 GiB
```

两个值都可以用环境变量覆盖，不必改脚本：

```bash
LWD_MAX_MODEL_LEN=65536 bash cloud_p.sh     # 0.95 下也够（≈6.3 GiB > 4.38 GiB）
LWD_GPU_MEM_UTIL=0.90  bash cloud_p.sh      # 但此时必须把 max-model-len 降到 ≤49152
```

### 根本上的取舍

`dp2 × tp1` 意味着**每个 rank 独占一整份权重**。27B 在单卡 64 GB 上只剩 3~6 GiB 给 KV，
并发大上下文的能力很有限。如果更看重长上下文/高并发而不是 DP 副本数：

| 目标 | 选择 |
| --- | --- |
| 2 个 DP 副本、短请求为主（本包） | `dp2_tp1_194_195` + `max-model-len 32768` |
| 长上下文 + 充足 KV | [../sidecar](../sidecar/README.md)（TP=2，权重分摊，65536 已验证） |

## 卡与端口（当前值）

| 角色 | 主机 / 卡 | 并行 | API | kv_port（基址） | LWD ctrl / data / coord |
| --- | --- | --- | --- | --- | --- |
| edge | 194 / NPU 4 | TP=1 DP=1 | 8510 | — | 不监听 |
| cloud 0 prefill | 195 / NPU 0,1 | **DP=2 × TP=1** | 8600 | 14570 | 6100/7200/8100、6101/7201/8101 |
| cloud 1 decode | 195 / NPU 2,3 | **DP=2 × TP=1** | 8601 | 14580 | 6110/7210/8110、6111/7211/8111 |
| proxy | 195（纯 CPU） | — | 8102 | — | — |

`kv_port` 是**基址**：连接器按 `kv_port + dp_rank × tp × pp` 派生握手端口
（`mooncake_connector.py:1762-1768, 2199-2200`），所以 P 实际用 14570/14571、D 用 14580/14581。
两组基址之间留了 10 的间隔，避免同机撞端口。

## 文件

与 sidecar 相同的 8 个文件；本目录只改了两个引擎脚本、拓扑和 proxy 目标表。

| 文件 | 与 sidecar 的关系 |
| --- | --- |
| `env.sh` | **逐字相同** |
| `edge.sh` | **逐字相同**（194 / NPU 4 / 8510 / TP=1） |
| `start_proxy.sh` | **逐字相同**（8102） |
| `check_configs.py` | 相同，但内存估算按 **cloud DP 数**计 |
| `pd_host_topology.yaml` | 每个 cloud 从 1 个 dp 变成 2 个 dp |
| `pd_proxy.json` | 从 2 个目标变成 4 个（2 prefill + 2 decode） |
| `cloud_p.sh` / `cloud_d.sh` | TP 2→1，新增 `--data-parallel-*`，kv_port 改基址，**去掉 `--speculative-config`**（见上文限制） |

## 启动

```bash
python3 check_configs.py        # 拓扑 + proxy 自检（含每个 DP 的端口冲突检查）

# 195：先 P 后 D
bash cloud_p.sh
bash cloud_d.sh
bash start_proxy.sh

# 194：最后起（只有边端会阻塞等链路）
bash edge.sh
```

自测（打边端）：

```bash
curl -s http://76.76.26.194:8510/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6","messages":[{"role":"user","content":"你好"}],"max_tokens":32}'
```

## 内存账（多 DP 会线性放大）

edge 拥有**每条 edge→cloud DP 链路**一个连接池（`lwd_host_channels.py:262-266`）：

```
links = 2 × (Np + Nd) 个 cloud DP = 2 × 4 = 8
need  = (8/2) × 4 blocks × (256 + 32 MiB) = 4608 MiB   （预算 12288 MiB）
```

sidecar 单 DP 时是 2304 MiB。`check_configs.py` 已按 DP 数计算并在超出时报错。

## 改成其它 DP 数

以「每个 cloud N 个 DP、每副本 TP=M」为例，需要同步改 **6 处**：

| # | 文件 | 改什么 |
| --- | --- | --- |
| 1 | `pd_host_topology.yaml` | 每个 cloud 的 `dp:` 写 N 条，`dp_idx` 从 0 连续；每条给**唯一的** `ctrl_port`/`data_port`/`coord_port` |
| 2 | `pd_proxy.json` | `prefill`/`decode` 各写 N 条，`target_dp` 从 0 到 N-1，URL 指向对应 `coord_port` |
| 3 | `cloud_p.sh` / `cloud_d.sh` | `--data-parallel-size N`、`--tensor-parallel-size M`、`ASCEND_RT_VISIBLE_DEVICES` 给够 N×M 张卡、`--data-parallel-rpc-port` 两侧必须不同 |
| 4 | 同上 | `kv_connector_extra_config` 的 `{"dp_size": N, "tp_size": M}`（P、D 两侧都写） |
| 5 | 同上 | `kv_port` 基址：每侧要留出 **N×M** 个连续端口（握手端口 = 基址 + dp_rank×tp×pp），两侧基址之间留间隔 |
| 6 | 目录名 | `dp{N}_tp{M}_{边端主机}_{云主机}` |

约束检查：

- **卡数守恒**：edge 1 张 + P(Np×Mp) + D(Nd×Md) ≤ 可用卡；P 与 D 不能共用卡。
  本机现状：194 只有 NPU 4 可用，195 有 NPU 0-3。
- **LWD 端口预算**：每个 cloud DP 占 3 个端口（ctrl/data/coord），共 3×(Np+Nd) 个；
  改完先跑 `python3 check_configs.py`，它会检查 listener 冲突与 DP 内存预算。
- **edge 的 host buffer**：3×(Np+Nd) 增加后 4608 MiB 会继续涨，上限 12288 MiB；
  不够就降 `up_buffer_size`/`down_buffer_size`。

## 风险与回退

| 风险 | 说明 | 回退 |
| --- | --- | --- |
| **每卡显存** | TP 从 2 降到 1，同一份层切片的权重压到单卡 | 报 KV 不足就降 `--max-model-len`（如 32768） |
| **没有 MTP** | DP>1 与 MTP 互斥（见上文限制），本包不带 draft | 需要 MTP 就用 [../sidecar](../sidecar/README.md)（DP=1、TP=2） |
| **DP 端口** | 每个 DP rank 的 ctrl/data/coord 都不同 | `check_configs.py` 会报冲突；改拓扑里 `dp[].*_port` |
| **再往上加 DP** | 每加 2 个 DP，edge 的 host buffer 多 4 条链路（+2304 MiB） | 见上文「改成其它 DP 数」，注意 12288 MiB 上限 |

## 本目录做过的校验（均实跑）

| 检查 | 方法 | 结果 |
| --- | --- | --- |
| 拓扑 + proxy | `check_configs.py` | PASS；edge 内存估算 4608 MiB（4 条 cloud DP 链路） |
| 脚本语法 | `bash -n` × 5 | 通过 |
| 拉起参数 | 四个入口 `--dry-run` | DP=2 / TP=1 / 卡号 / kv 基址 / lwd_config instance_id 全部符合预期 |
| 与 sidecar 的差异受控 | 逐参数比对 | 仅 TP 2→1、新增两个 `--data-parallel-*`、kv_port 基址、拓扑 DP 数 |
| 端口唯一性 | `check_configs.py` 的 listener 冲突检查 | 8 个 cloud DP 端口 + 2 个 API 端口互不冲突 |
