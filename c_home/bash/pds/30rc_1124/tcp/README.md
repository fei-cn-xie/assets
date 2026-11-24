# TCP 变体的边云 PD 分离（v0.30.0）

v0.30.0 的 host 数据面**固定是 UCX**，所以这里的 "tcp" 不是"不用 UCX"，
而是把 UCX 的传输层钉到 TCP：`UCX_TLS=tcp,self`、`UCX_NET_DEVICES=<业务网卡>`。
和 `rdma/` 的唯一区别就在 `env.sh` 的这两行；四个入口脚本与 `lwd_topology.yaml`
与 rdma 目录逐字节相同（拓扑解析内容完全一样）。

完整参数分析、端口规划、风险清单见上级 [README.md](../README.md)。

## 文件

| 文件 | 作用 |
| --- | --- |
| `env.sh` | 公共环境：`CODEBASE_DIR`/`LWD_PYTHON`/`PYTHONPATH`、拓扑路径、`lwd_transport_env`、`lwd_run`、`--dry-run` |
| `lwd_topology.yaml` | 拓扑（边 A + 云 P + 云 D，host 数据面，`up_buffer_size=256`） |
| `pd_proxy.json` | PD proxy 目标表（P→`8660`，D→`8661`） |
| `cloud_p.sh` | 云 B / prefill：API `8560`，TP=2，`--enforce-eager`，Mooncake producer |
| `cloud_d.sh` | 云 B / decode：API `8561`，TP=2，图模式，Mooncake consumer |
| `edge.sh` | 边 A：API `9060`，TP=1、DP=1，无投机解码 |
| `proxy.sh` | 云 B：`python -m vllm.v1.lwd_control.pd.proxy --config pd_proxy.json --port 8500` |

## 部署

```bash
# 两台机器都要有这一整份目录（拓扑内容必须一致）
scp -r tcp/ user@76.76.26.189:/bash_dir/pds/30rc_1124/
scp -r tcp/ user@76.76.26.194:/bash_dir/pds/30rc_1124/
```

云端 B 依次启动（各占一个终端）：

```bash
bash /bash_dir/pds/30rc_1124/tcp/cloud_p.sh
bash /bash_dir/pds/30rc_1124/tcp/cloud_d.sh
bash /bash_dir/pds/30rc_1124/tcp/proxy.sh
```

边端 A：

```bash
bash /bash_dir/pds/30rc_1124/tcp/edge.sh
```

## 先干跑一遍

```bash
bash cloud_p.sh --dry-run     # 打印 vllm serve 命令与 UCX / HCCL 环境变量
bash edge.sh --dry-run
```

检查点：
`UCX_TLS=tcp,self`、`UCX_NET_DEVICES=enp189s0f0`、`--additional-config` 是内联 JSON 且
`path` 指向本目录的 `lwd_topology.yaml`。

## 边端启动会静默等云侧（最长 605s/条链路）

边端是唯一阻塞等待的一方，单条链路超时 605s，且逐条 IO 串行（1 边 × 2 云 = 4 条，
最坏约 40 分钟才报错）。等待期的进展日志是 WARNING 级，所以脚本默认
`VLLM_LOGGING_LEVEL=INFO`；压测时再覆盖成 `ERROR`。判断与排查见上级
[README.md](../README.md) 的 §5。

## 可覆盖项

```bash
CODEBASE_DIR=/vllm-workspace \
LWD_PYTHON=/usr/bin/python3 \
LWD_TOPOLOGY=/abs/path/lwd_topology.yaml \
UCX_TLS=tcp,self UCX_NET_DEVICES=enp189s0f0 \
UCX_PREFIX=/vllm-workspace/ucx \
VLLM_LOGGING_LEVEL=ERROR VLLM_LWD_LOG_LEVEL=INFO \
bash cloud_p.sh
```

`UCX_PREFIX` 显式置空（`UCX_PREFIX= `）表示沿用已有库搜索路径。

## 确认真的走 TCP

`UCX_TLS=tcp,self` 已经排除 `rc_*`，但 UCX 仍可能在多网卡间自选。上线后用
`UCX_LOG_LEVEL=info` 观察 transport 选择，或直接看 `enp189s0f0` 的收发包计数增长。
