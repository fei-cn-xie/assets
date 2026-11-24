# RDMA 变体的边云 PD 分离（v0.30.0）

host 数据面走 UCX over RoCE：`UCX_TLS=rc_verbs,tcp,self`、
`UCX_NET_DEVICES=rocep67s0f0:1,enp189s0f0`。`tcp` 仍在列表里，因为链路握手用的是
TCP socket（UHC1 全双工，握手方向与数据方向无关）。

和 `tcp/` 的唯一区别就在 `env.sh` 的这两行；四个入口脚本与 `lwd_topology.yaml`
与 tcp 目录逐字节相同。**拓扑里没有"传输后端"字段**——v0.30.0 把
`dev/pd-sep-comm` 目录 JSON 的 `backend: tcp|rdma` 删掉了。

完整参数分析、端口规划、风险清单见上级 [README.md](../README.md)。

## 与 TCP 变体的差异

| 项 | tcp | rdma |
| --- | --- | --- |
| `UCX_TLS` | `tcp,self` | `rc_verbs,tcp,self` |
| `UCX_NET_DEVICES`（cloud） | `enp189s0f0` | `rocep67s0f0:1,enp189s0f0` |
| `UCX_NET_DEVICES`（edge） | `enp189s0f0` | `rocep67s0f0:1,enp189s0f0` |
| `qos.data_plane` | 不适用 | 可在拓扑里打开，见下 |
| 角色专用覆盖 | `LWD_{CLOUD,EDGE}_UCX_NET_DEVICES` | 同 |

角色专用变量优先于通用 `UCX_NET_DEVICES`（与 `dev/pd-sep-comm` 参考脚本一致）。

## QoS 走配置，不走环境变量

v0.30.0 把 RoCE QoS 做成了拓扑字段（注入 UCX 的 UCT 配置）：

```yaml
qos:
  data_plane: {traffic_class: 106, sl: 5}   # -> IB_TRAFFIC_CLASS / IB_SL，TCP 握手上再打 IP_TOS
  ctrl_plane: {dscp: 26}                    # -> ZMQ socket TOS
```

默认在 `lwd_topology.yaml` 里是注释状态（UCX 对 RoCEv2 会自动取 106 / sl 0）。
打开时**两端必须同值**，且需要交换机侧匹配的 PFC/ETS 策略，否则只会把包标错优先级。

## 部署

同 tcp：把整个 `rdma/` 目录拷到两台机器（拓扑必须一致），云端依次
`cloud_p.sh` → `cloud_d.sh` → `proxy.sh`，再在边端起 `edge.sh`。

## 确认真的走 RDMA

**链路建起来 ≠ 走了 RDMA**（旧参考脚本的 RDMA.md 也强调这点）：`UCX_TLS` 里含 `tcp`，
UCX 可能为某条链路回落到 TCP。逐条核对：

1. `UCX_LOG_LEVEL=info` 启动，日志里看实际选中的 transport 是否为 `rc_verbs`/`rc_mlx5`；
2. `ucx_info -d | grep -A3 rocep67s0f0` 确认设备与 GID 可见；
3. 请求压起来后看 RDMA 网卡的计数增长：
   `cat /sys/class/infiniband/rocep67s0f0/ports/1/counters/port_xmit_data`；
4. `data_port`（P `19600` / D `19700`）只是 TCP 握手入口，别用它判断数据面走哪条路。

## 常见失败

- `libucp` 找不到：`UCX_PREFIX` 指到 UCX 安装前缀，或显式设置 `LD_LIBRARY_PATH`。
- 握手超时：检查 `data_port` 在云端是否监听、边端到云端的 TCP 连通性、
  `rocep67s0f0:1` 的 GID 索引是否与对端网卡规划一致。
- 两侧 digest 不一致：两份 `lwd_topology.yaml` 不同步，直接比对 `sha256sum`。
