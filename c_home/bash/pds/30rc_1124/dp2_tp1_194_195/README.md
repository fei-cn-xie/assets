# dp2_tp1_194_195：两侧都是 2 DP × 1 TP 的 LWD 边云 PD

目录名 = `dp{每云DP数}_tp{每副本TP数}_{边端主机}_{云主机}`。

```
client ──8510──▶ edge (194, NPU 4, TP=1 DP=1)
                    │ pd_proxy_url
                    ▼
              proxy (195:8102)
                    ├── cloud 0 prefill (195 NPU 0,1)  DP=2 × TP=1  → coord 8100 / 8101
                    └── cloud 1 decode  (195 NPU 2,3)  DP=2 × TP=1  → coord 8110 / 8111
```

## ⚠️ 需要配套的代码修复分支

dense 模型下每个 DP rank 进程会把 `ParallelConfig` 的 DP 抹成 1，导致两处必然失败。
本包依赖 `fix/lwd-multidp-dense-pd`（vllm 与 vllm-ascend 各一个）：

| 仓 | 修改 | 解决的现象 |
| --- | --- | --- |
| vllm | `ParallelConfig.data_parallel_size_global`：在 DP 坍缩时保留部署级 DP（并加入 `compute_hash` 的 `ignored_factors`） | 引擎核心握手结束后重跑 `VllmConfig.__post_init__()` 时，KV extra-config 的 DP 校验拿"已抹平的 1"去比"布局里的 2" |
| vllm-ascend | `check_kv_extra_config` 改用部署级 DP（而非 `data_parallel_size`）；`MooncakeConnector` 的 `side_channel_port` 改用 `data_parallel_index`（DP 坍缩不会清零它） | ① 上述校验；② producer 两个 DP rank 抢同一个 KV 握手端口（`Address already in use`） |

`data_parallel_index` 的原有注释就写着 *"not overridden for dense models"*，正是为此预留。

## 卡与端口

| 角色 | 主机 / 卡 | 并行 | API | kv_port 基址 | LWD ctrl / data / coord |
| --- | --- | --- | --- | --- | --- |
| edge | 194 / NPU 4 | TP=1 DP=1 | 8510 | — | 不监听 |
| cloud 0 prefill | 195 / NPU 0,1 | **DP=2 × TP=1** | 8600 | 14570（rank0→14570, rank1→14571） | 6100/7200/8100、6101/7201/8101 |
| cloud 1 decode | 195 / NPU 2,3 | **DP=2 × TP=1** | 8601 | 14580（consumer 只连不绑） | 6110/7210/8110、6111/7211/8111 |
| proxy | 195（纯 CPU） | — | 8102 | — | — |

DP 协调端口：prefill `29600`、decode `29601`；master 端口 29525 / 29526。

`--kv-transfer-config` 两侧形状都要写，各引擎只校验自己角色那一块
（`vllm_ascend/utils.py:1601-1604`）：

```json
"kv_connector_extra_config": {
  "prefill": {"dp_size": 2, "tp_size": 1},
  "decode":  {"dp_size": 2, "tp_size": 1}
}
```

## 关键参数与理由

| 参数 | 值 | 理由 |
| --- | --- | --- |
| `--max-model-len` | 32768 | TP=1 时 KV 只有 ≈6.7 GiB，32768 需 ≈2.2 GiB |
| `--gpu-memory-utilization` | 0.95 | 0.90 时 KV 只有 3.3 GiB，65536 上下文直接失败 |
| `--max-num-seqs` | 64 | 模型含 GDN(Mamba) 层，每 decode 序列占一个 mamba block；6.7 GiB 只有 ≈123 个 |
| `--speculative-config` | MTP，3 tokens | 修好 DP 校验后 DP>1 也能用 MTP；若显存吃紧可去掉，能换来约 1 GiB KV |
| `--data-parallel-size 2` | 两侧都是 2 | 每个 DP rank 独占一张卡（TP=1，权重不切分） |

三个值都能用环境变量覆盖，不必改脚本：

```bash
LWD_MAX_MODEL_LEN=65536 bash cloud_d.sh   # 需 Available KV > 4.4 GiB
LWD_MAX_NUM_SEQS=96     bash cloud_d.sh   # 上限由启动日志的 mamba block 数决定
LWD_GPU_MEM_UTIL=0.96   bash cloud_d.sh
```

## 启动

```bash
python3 check_configs.py        # 拓扑 + proxy 自检

# 195：先 P 后 D
bash cloud_p.sh
bash cloud_d.sh
bash start_proxy.sh

# 194：最后起
bash edge.sh
```

自测：

```bash
curl -s http://76.76.26.194:8510/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6","messages":[{"role":"user","content":"你好"}],"max_tokens":32}'
```

## 启动成功时应看到

| 位置 | 期望 |
| --- | --- |
| prefill / decode 日志 | `[LWD] cloud control HTTP ready on 0.0.0.0 ports=[8100, 8101] cloud_id=0 dp_size=2`（decode 为 8110/8111、cloud_id=1） |
| worker 日志 | 每个云各 2 条 `host data-plane links created ... dp_rank=0/1` |
| 资源 | `Available Mamba cache blocks` ≥ 64；`host memory plan ... links=4 need=4608MiB` |
| 无 | `Address already in use`、`conflicting data parallel size` |

## 本目录做过的校验

| 检查 | 方法 | 结果 |
| --- | --- | --- |
| 拓扑 + proxy | `check_configs.py` | PASS，edge 内存估算 4608 MiB（4 条 cloud DP 链路） |
| 脚本语法 | `bash -n` × 5 | 通过 |
| 拉起参数 | 四个入口 `--dry-run` | 两侧 DP=2/TP=1、MTP、64、0.95、14570/14580 |
| 拓扑与 proxy 一致性 | 结构断言 | 两侧各 2 个 DP，coord 8100/8101/8110/8111，proxy 2+2 个 target |

## 排错对照

| 现象 | 含义 | 处置 |
| --- | --- | --- |
| `Address already in use (...:14570)` | 用的是未修复的 vllm-ascend（端口仍按 `data_parallel_rank` 派生） | 确认 `PYTHONPATH` 指向 `fix/lwd-multidp-dense-pd` 分支 |
| `conflicting data parallel size. Expected 1, but got 2` | 用的是未修复的 vllm（没有 `data_parallel_size_global`） | 同上，vllm 也要切到修复分支 |
| `max_num_seqs exceeds available Mamba cache blocks` | KV 池变小（例如加了 MTP 或降了 util） | 降 `LWD_MAX_NUM_SEQS` 或提 `LWD_GPU_MEM_UTIL` |
| `KV cache is needed ... larger than available` | 上下文太长 | 降 `LWD_MAX_MODEL_LEN` |
