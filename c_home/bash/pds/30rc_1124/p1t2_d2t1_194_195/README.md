# p1t2_d2t1_194_195：非对称多 DP 的 LWD 边云 PD

目录名 = `p{P的DP}×t{P的TP}_d{D的DP}×t{D的TP}_{边端主机}_{云主机}`。

```
client ──8510──▶ edge (194, NPU 4, TP=1 DP=1)
                    │ pd_proxy_url
                    ▼
              proxy (195:8102)
                    ├── cloud 0 prefill (195 NPU 0,1)  DP=1 × TP=2  → coord 8100
                    └── cloud 1 decode  (195 NPU 2,3)  DP=2 × TP=1  → coord 8110 / 8111
```

为什么非对称，见下文「三次失败与最终形态」——**producer 侧不能 DP>1，consumer 侧可以**，
而 decode 正是吞吐瓶颈，所以把多 DP 放在 decode 上。

## 卡与端口

| 角色 | 主机 / 卡 | 并行 | API | kv_port 基址 | LWD ctrl / data / coord |
| --- | --- | --- | --- | --- | --- |
| edge | 194 / NPU 4 | TP=1 DP=1 | 8510 | — | 不监听 |
| cloud 0 prefill | 195 / NPU 0,1 | **DP=1 × TP=2** | 8600 | 14570（+tp_rank → 14570/14571） | 6100 / 7200 / 8100 |
| cloud 1 decode | 195 / NPU 2,3 | **DP=2 × TP=1** | 8601 | 14580（每 rank 计算但**不 bind**） | 6110/7210/8110、6111/7211/8111 |
| proxy | 195（纯 CPU） | — | 8102 | — | — |

`--kv-transfer-config` 里两侧形状都要写，且各引擎只校验自己角色的那一块
（`vllm_ascend/utils.py:1601-1604`）：

```json
"kv_connector_extra_config": {
  "prefill": {"dp_size": 1, "tp_size": 2},
  "decode":  {"dp_size": 2, "tp_size": 1}
}
```

## 三次失败与最终形态（代码级原因）

| # | 报错 | 根因 | 处置 |
| --- | --- | --- | --- |
| 1 | `KV transfer 'prefill' config has a conflicting data parallel size. Expected 1, but got 2` | MTP drafter 以 DP=1 重建 `VllmConfig`（`vllm/v1/spec_decode/llm_base_proposer.py:1292` → `vllm/config/utils.py:127` → 重跑 `__post_init__`），与 `dp_size=2` 冲突 | decode 侧去掉 MTP（DP=1 的 prefill 仍可保留 MTP） |
| 2 | `max_num_seqs (128) exceeds available Mamba cache blocks (123)` | 模型含 GDN(Mamba) 层，每个 decode 序列占一个 mamba block；TP=1 时 KV 池只够 123 个 | decode `--max-num-seqs 64` |
| 3 | **`ZMQError: Address already in use (tcp://…:14570)`** | producer 的 `KVCacheSendingThread` 是 `zmq.ROUTER` 且 `bind=True`（`mooncake_connector.py:319`），端口 = `kv_port + data_parallel_rank*tp*pp*pcp + device_index`；LWD 非 MoE 分支把 `data_parallel_rank` 清零（`vllm/v1/engine/core.py:1377` → `parallel.py:1111`），两个 DP rank 算出同一端口 | **prefill 不做 DP**（DP=1、TP=2） |

第 3 点的豁免条件：consumer 侧建的是 `zmq.REQ` 且 `bind=False`
（`mooncake_connector.py:1546-1551`）——**只连不绑**，所以 decode 可以 DP>1。

## 其它已排查 / 已规避的问题

| 项 | 结论 |
| --- | --- |
| TP=1 每卡显存 | cloud 侧加载**完整 64 层**模型（50.2 GB ≈ 46.8 GiB）。decode 用 0.95 利用率 → KV ≈ 6.7 GiB；prefill 用 TP=2 分摊，0.90 足够 |
| `max_model_len` | 两侧都设 32768（可用 `LWD_MAX_MODEL_LEN` 覆盖）。decode 在 6.7 GiB 下 65536 也够（需 4.38 GiB），但两者必须一致，否则超长请求会在 decode 侧失败 |
| Mamba block 上限 | 每次调大 KV 池后，重启时看 `Available Mamba cache blocks`，`max_num_seqs` 不得超过它 |
| `LD_PRELOAD` | 已改成"文件存在才设置"，消除每进程一条 `ld.so: … ignored` 的 ERROR 噪音 |
| `causal_conv1d_update` 回退 | WARNING 提示 NPU Triton 版不可用，decode-FULL 的图捕获会变慢（本次 decode 启动比 prefill 多约 60 s），属预期 |
| `torch.compile` 缓存 | 首次各编译 ~55 s 并落盘 `/root/.cache/vllm`，重启复用 |
| edge 的 host buffer | 3 条 cloud DP 链路 → `3 × 4 × (256+32) = 3456 MiB`，`check_configs.py` 已按 DP 数计算 |

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

可覆盖的调参项：

```bash
LWD_MAX_MODEL_LEN=65536 bash cloud_d.sh   # 需 Available KV > 4.4 GiB
LWD_MAX_NUM_SEQS=96  bash cloud_d.sh      # 上限由 mamba block 数决定
LWD_GPU_MEM_UTIL=0.96 bash cloud_d.sh
```

## 改成其它规模

| 想加的副本 | 能不能 | 怎么做 |
| --- | --- | --- |
| decode 再加一个 DP（d3） | **可以** | 拓扑给 cloud 1 加第 3 条 `dp:`（ctrl/data/coord 各新端口）、`pd_proxy.json` 加第 3 个 decode target、`--data-parallel-size 3`、卡要够 3 张 |
| prefill 加 DP | **不能** | producer 会撞端口（见上表第 3 点），只能加 TP |
| 两边都用 DP | 需要上游修复 | 把 `side_channel_port` 的 `data_parallel_rank` 换成 `data_parallel_index`（LWD 自己就用后者，见 `lwd_control_plane.py:102-103`） |

## 本目录做过的校验（均实跑）

| 检查 | 方法 | 结果 |
| --- | --- | --- |
| 拓扑 + proxy | `check_configs.py` | PASS，edge 内存估算 3456 MiB（3 条 cloud DP 链路） |
| 脚本语法 | `bash -n` × 5 | 通过 |
| 拉起参数 | 四个入口 `--dry-run` | P：TP=2/无 DP/MTP 保留/14570；D：DP=2/TP=1/无 MTP/64/0.95/14580 |
| 拓扑与 proxy 一致性 | 结构断言 | prefill 1 个 DP、decode 2 个 DP，coord 8100/8110/8111，proxy 1+2 个 target |
| 目录卫生 | — | 除 `check_configs.py` 外无脚本 |

⚠️ 仍未实机验证：decode 侧 DP=2 在"consumer 只连不绑"的前提下是否真的能同时起两个 rank（代码依据充分，但这是第一次真机跑）。首次启动请看 decode 日志里是否出现两条
`host data-plane links created ... dp_rank=0/1`，以及是否还报端口占用。
