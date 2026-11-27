# mooncake_pd2_tp4_231：集中式 PD 分离，两侧都是 2 DP × 4 TP（单机 16 卡）

**集中式 = 没有边端、没有 LWD 控制面**：prefill 与 decode 是两个独立的 `vllm serve`，
KV 通过 Mooncake 连接器点对点传输，前面挂上游的负载均衡 proxy。

```
client ──9060──▶ LB proxy (231, CPU)
                    ├── prefill 9160 (231 NPU 0-7)   DP=2 × TP=4  kv_producer 30000-30007
                    └── decode  9161 (231 NPU 8-15)  DP=2 × TP=4  kv_consumer 30100 起
```

目录名 = `{连接器}_pd{每侧DP数}_tp{每副本TP数}_{主机}`。同目录的 `mooncake_pd2_tp1_195`
是同一形态的 2DP×1TP 版本；本包只是并行度、主机与"性能对比固定档位"不同。

## 与边云包的对比关系（本包存在的理由）

两套部署跑同一个模型、同一组固定 knob，用来比较**集中式 PD** 与**边云式 PD** 的性能：

| 项 | 本包（集中式） | `../lwd_dp2_tp4_17_231`（边云式） |
| --- | --- | --- |
| prefill | 231 NPU 0-7，2DP×4TP | 231 NPU 0-7，2DP×4TP |
| decode | 231 NPU 8-15，2DP×4TP | 231 NPU 8-15，2DP×4TP |
| 边端 | 无 | 17 NPU 0，TP=1 |
| 显存利用率 | **0.95** | **0.95**（边端同值） |
| `--max-model-len` | **65536** | **65536** |
| `--max-num-seqs` | **128** | **128** |
| `--max-num-batched-tokens` | **8192** | **8192** |
| MTP | **开** | **开** |
| 调度 | 自带 internal LB（每侧 2 个前端） | 同左 + LWD 边云控制面 |

这五个 knob 由两边的 `check_configs.py` **交叉校验**：本包的校验器读
`../lwd_dp2_tp4_17_231/env.sh`，任何一项漂移都会 FAIL。改任何一边都要同时改另一边。

## 卡与端口

| 角色 | 卡 | 并行 | API | master | DP rpc | KV 握手端口 |
| --- | --- | --- | --- | --- | --- | --- |
| proxy | — | — | **9060** | — | — | — |
| prefill | NPU 0-7 | **2DP × 4TP** | 9160 | 29545 | 29610 | 30000–30007（每个 dp×tp 一个） |
| decode | NPU 8-15 | **2DP × 4TP** | 9161 | 29546 | 29611 | 30100 起（consumer 只连不绑） |

DP rank 0 拿到第一组 4 张卡、rank 1 拿到第二组；`ASCEND_RT_VISIBLE_DEVICES` 在
`env.sh` 里分成 `PD_PREFILL_DEVICES` / `PD_DECODE_DEVICES`，校验器会断言两者不重叠
且合起来正好是 0-15。

端口刻意与边云包（proxy 8102 / 边端 8510 / 云 8600-8601 / KV 14570 起）错开，
这样两套脚本留在同一台机器上也不会互相抢端口。

## 内存账（TP=4 与 TP=1 的差别）

TP=4 把权重和 KV 都摊到 4 张卡上，所以 65536 上下文 + 128 并发才放得下：

| 量 | TP=1（参考包） | 本包 TP=4 |
| --- | --- | --- |
| 每卡权重 | 46.8 GiB | ≈11.7 GiB |
| KV / token（全模型） | 96.6 KiB | 96.6 KiB |
| KV / token（每卡） | 96.6 KiB | ≈24.2 KiB |
| util 0.95 时的预算 | 57.9 GiB | 57.9 GiB |

剩下的预算全部进 KV 池，因此 65536 的单个请求（每卡约 1.55 GiB）绰绰有余；
`--max-num-seqs 128` 是**上限**而不是预留，同时驻留的 token 总量仍受 KV 池约束。

模型含 GDN(Mamba) 层，每个 decode 序列占一个 Mamba block，所以启动日志里必须看到
`Available Mamba cache blocks ≥ 128`，否则 128 并发会被挡住（校验器会提示这一行）。

## 文件

| 文件 | 作用 |
| --- | --- |
| `env.sh` | 全部常量与运行时环境；五个对比 knob 就在这里 |
| `prefill.sh` / `decode.sh` | 两侧引擎（KV producer / consumer），除端口、KV 角色、图模式外逐项相同 |
| `start_proxy.sh` | 调用 vllm-ascend 自带 `examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py`，仓内无 Python 胶水 |
| `check_configs.py` | 静态自检（拉起**前**）：并行度、卡集合、端口、KV 配置、两侧参数一致、与边云包的 knob 一致性、CRLF |
| `verify_dp.sh` | 只读自检（拉起**后**）：进程与 worker 数、KV 端口分属两个 DP rank、端点、一次真实请求 |

## 启动

```bash
python3 check_configs.py     # 必须先过；会同时校验与边云包的一致性

bash prefill.sh              # 先 P（等它打印 Application startup complete）
bash decode.sh               # 再 D
bash start_proxy.sh          # 最后 proxy

bash verify_dp.sh            # 拉起后核对 DP=2×TP=4 真的生效
```

两个引擎都起来后再打请求：

```bash
curl -s http://76.76.26.231:9060/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6","messages":[{"role":"user","content":"你好"}],"max_tokens":32}'
```

## CPU 绑定：两套默认一致，无需额外配置

`enable_cpu_binding` 在 `AscendConfig` 里的默认值就是 **True**
（`vllm_ascend/ascend_config.py:565`），worker 在 `worker.py:962` 据此做 topo-affinity 绑定。
边云包在 `--additional-config` 里显式写了 `true`，等价于默认值，因此：

- 集中式（不传 `--additional-config`）与边云式（传 `true`）**CPU 亲和性相同**
- 不需要为了公平对比在集中式包上补这个键
- `platform.py:890-901` 只在用户传 `--numa-bind` 时才会额外注入该键（Ascend 不支持 NUMA 探测）

在 a3（231）上，绑定的 board/chip 解析由 `npu-smi info -m` 的 `Chip ID`/`Chip Logic ID` 列完成
（`cpu_binding.py:87-129`），`ASCEND_RT_VISIBLE_DEVICES` 依然按 **chip logic id** 计数，
所以本包的 `0-7` / `8-15` 写法在 a2 与 a3 上都成立。

## 性能对比时要注意的三件事

1. **必须同一个代码分支**：dense 模型的 DP>1 依赖 `fix/lwd-multidp-dense-pd`
   （DP 折叠、KV 端口、引擎身份三处修复），两套部署与 231/17 两台机器都要在这个分支上。
2. **同一组 knob 与同一份输入**：五个 knob 已由校验器锁定；benchmark 的输入长度、
   输出长度、并发档位也要一致，否则比较没有意义。
3. **`--api-server-count` 留空**：置 1 会只剩一个前端做 LB，绝对吞吐明显下降。
   它只是排查启动编排问题的开关，不该出现在性能数据里。

## 与 2DP×1TP 包的三处代码前提

`../mooncake_pd2_tp1_195/README.md` 里那张表列了 dense 模型 DP>1 必须的三个修复
（DP 折叠、KV 握手端口、引擎身份）。本包沿用同一套修复，不再重复。
