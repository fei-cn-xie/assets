# lwd_dp2_tp4_17_231：边云式 PD 分离，云侧 2 DP × 4 TP（+17 边端 1 卡）

**边云式 = 有 LWD 控制面**：边端（17）负责输入/采样与加密封装，两个云端实例（prefill / decode）
各跑 2 DP × 4 TP，KV 走 LWD host 数据面 + Mooncake 连接器。

```
client ──8102──▶ LB proxy (231, CPU)
                    ├── cloud 0 prefill 8600 (231 NPU 0-7)   DP=2 × TP=4  kv_producer 14570-14577
                    └── cloud 1 decode  8601 (231 NPU 8-15)  DP=2 × TP=4  kv_consumer 14580 起

edge 8510 (17 NPU 0, TP=1 DP=1) ──LWD host 数据面(UCX)──▶ 两个 cloud
```

边端把请求发给 proxy（拓扑里 `edges[0].coordination.pd_proxy_url`），proxy 不感知 LWD，
照常做 prefill → decode 两跳。

目录名 = `{方案}_dp{每云DP数}_tp{每副本TP数}_{边端主机}_{云端主机}`。

## 与集中式包的对比关系（本包存在的理由）

| 项 | `../mooncake_pd2_tp4_231`（集中式） | 本包（边云式） |
| --- | --- | --- |
| prefill | 231 NPU 0-7，2DP×4TP | 231 NPU 0-7，2DP×4TP |
| decode | 231 NPU 8-15，2DP×4TP | 231 NPU 8-15，2DP×4TP |
| 边端 | 无 | 17 NPU 0，TP=1 |
| 显存利用率 | **0.95** | **0.95**（含边端） |
| `--max-model-len` | **65536** | **65536** |
| `--max-num-seqs` | **128** | **128** |
| `--max-num-batched-tokens` | **8192** | **8192** |
| MTP | **开** | **开** |

这五个 knob 由两边的 `check_configs.py` **交叉校验**（本包读 `../mooncake_pd2_tp4_231/env.sh`），
任何一边漂移都会 FAIL。改一个必须同时改另一个。

## 卡与端口

| 角色 | 主机 | 卡 | 并行 | API | master | DP rpc | KV 握手端口 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| proxy | 231 | — | — | **8102** | — | — | — |
| cloud prefill | 231 | NPU 0-7 | **2DP × 4TP** | 8600 | 29525 | 29620 | 14570–14577 |
| cloud decode | 231 | NPU 8-15 | **2DP × 4TP** | 8601 | 29526 | 29621 | 14580 起（只连不绑） |
| edge | 17 | NPU 0 | TP=1 / DP=1 | 8510 | 29527 | — | — |

拓扑里每个云 DP 各有一组 `ctrl_port / data_port / coord_port`（6100/7200/8100、6101/7201/8101、
6110/7210/8110、6111/7211/8111）。**TP 不写进拓扑**：它来自 `--tensor-parallel-size`，
拓扑只按 DP 计数（每条边→云的链路按 DP 算，缓冲区预算 4 链路 × 4 × (256+32) = 4608 MiB，低于 12288 MiB 上限）。

端口与集中式包（9060/9160/9161/29545/29546/29610/29611 与 KV 30000-30007/30100-30107）
完全错开，两套脚本可以同时留在这些机器上。

## UCX 安装位置（两台主机相同）

```
/home/extra/ucx/lib        # libucp.so / libuct.so ... 在此
```

`env.sh` 里 `LWD_UCX_LIB_DIR` 的默认值就是这个路径（**不**从 `CODEBASE_DIR` 推导——
UCX 不在 vllm-workspace 里）。脚本启动时会把它加到 `LD_LIBRARY_PATH`，并在拉起前确认
`libucp.so*` 存在；缺失时会在**几秒内**直接报错退出，而不是等到 LWD 握手 605s 超时：

```
[lwd][preflight] ERROR: libucp not found under /home/extra/ucx/lib
```

需要换路径时覆盖即可：`export LWD_UCX_LIB_DIR=/other/ucx/lib`。

## ⚠️ 待补：231（a3）的 RoCE 设备名

UCX 走 RoCE 需要**每台主机自己的 IB 设备名**。17 是 a2，沿用 194/195 验证过的
`rocep67s0f0:1,enp67s0f0np0`；**231 是 a3，设备名不同，必须先确认再补上**，否则：

```
[lwd][warn] LWD_UCX_NET_DEVICES_231 is unset: falling back to UCX_TLS=tcp
```

回退到 TCP 能跑通，但**带宽和延迟不可比**——做性能对比前务必补齐：

```bash
# 在 231 上查设备名与对应网卡
ibdev2netdev
ls /sys/class/infiniband

# 然后在两个 cloud 脚本的运行环境里导出（建议写进 /etc/profile.d 或直接前置）
export LWD_UCX_NET_DEVICES_231=rocepXXs0f0:1,enpXXs0f0np0
# 如 gid index 不是 5，一并覆盖
export UCX_IB_GID_INDEX=5
```

补齐后 `check_configs.py` 的 note 会显示 `UCX: 231 will use RoCE device ...`。

## 文件

| 文件 | 作用 |
| --- | --- |
| `env.sh` | 全部常量、运行时环境、按角色（edge/cloud）选择 IP/网卡/UCX 设备 |
| `pd_host_topology.yaml` | LWD 拓扑（边=17 一卡，云=231 两个实例各 2 DP）；两台主机必须用同一份 |
| `edge.sh` | 边端（17，NPU 0）；**最后启动**，它会阻塞等待链路（每个 IO 最多 605s） |
| `cloud_p.sh` / `cloud_d.sh` | 云端 prefill / decode（231，NPU 0-7 / 8-15） |
| `start_proxy.sh` | 调用 vllm-ascend 自带 `examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py` |
| `check_configs.py` | 静态自检（拉起**前**）：拓扑形状/端口/链路、三方 knob 一致、与集中式包交叉校验、CRLF |

## 启动顺序

```bash
python3 check_configs.py        # 先过静态校验（会同时校验与集中式包的一致性）

# 231（云侧）
bash cloud_p.sh
bash cloud_d.sh
bash start_proxy.sh

# 17（边端，最后）
bash edge.sh
```

边端日志里出现 `[LWD][config] ... digest=<16 hex>` 时，确认两台机器打印的 digest 相同
（不同说明拓扑文件不是同一份）。边端会等链路就绪，通常需要几十秒到数分钟。

请求从 proxy 进：

```bash
curl -s http://76.76.26.231:8102/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6","messages":[{"role":"user","content":"你好"}],"max_tokens":32}'
```

## 性能对比时要注意的四件事

1. **两套必须在同一个代码分支**：dense 模型 DP>1 依赖 `fix/lwd-multidp-dense-pd`，
   17 与 231 两台都要它（否则会重现 DP 折叠 / KV 端口 / 引擎身份三类问题）。
2. **先补齐 231 的 RoCE 设备名**，否则本包跑在 TCP 上，比较的是网络不是方案。
3. **边端也要 0.95 利用率、同样的 65536/128/8192**：边端虽只有一卡，但它决定输入处理与采样开销，
   配置漂移会让对比失真（校验器会拦）。
4. **两套不要同时跑在同一批卡上**：端口虽然错开，但 231 的 16 张卡是共享的，
   同时拉两套会互相抢显存/带宽，性能数据无效。
