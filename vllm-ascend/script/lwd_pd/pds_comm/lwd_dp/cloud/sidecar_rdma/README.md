# RDMA P/D 分离配置

以 `../sidecar_tcp` 已验证的 P/D 配置为基准，参考
`../../sidecar/rdma_validation` 的 UCX 环境生成。
保留一边、独立 P/D 的拓扑，以及模型、MTP、图模式、Mooncake KV 和批量参数。
本目录的 RDMA P/D 组合尚需在两台服务器上验证。

## 部署拓扑

| 进程 | 服务器 / 业务 IP | NPU | 入口 |
| --- | --- | --- | --- |
| Edge 0，TP=1 | A / `76.76.26.194` | 4 | `edge.sh` |
| Prefill，TP=2 | B / `76.76.26.189` | 4、5 | `cloud_p.sh` |
| Decode，TP=2 | B / `76.76.26.189` | 6、7 | `cloud_d.sh` |
| P / D sidecar | B / `76.76.26.189` | 无 | `sidecar_p.sh` / `sidecar_d.sh` |
| P/D proxy | B / `76.76.26.189` | 无 | `proxy.sh` |

将整个目录复制到两台服务器，例如 `/home/z00853488/sidecar_rdma`。
两端 `lwd_sidecar_directory.json` 的内容必须一致；脚本自动读取同目录下的配置，
可以从任意工作目录启动。默认代码根目录为 `/vllm-workspace`，其中包含
迁移后的 `vllm/` 和 `vllm-ascend/`；模型仍为 `/weight/Qwen3.6-27B/`。
使用已验证 TCP 配置时的 Python/vLLM 环境，并保留边端的
`/vllm-workspace/lwd_tenant.key`。

## RDMA 设置

- directory 的 `backend` 为 `rdma`；模型的 `transport.mode` 仍为 `sidecar`。
- P / D 的 UCX 握手端口基数分别为 `19600` / `19700`。
  当前只有 edge 0，实际端口也是 `19600` / `19700`，避免同机 P/D 端口冲突。
- P / D 的 IPC 分别为 `/tmp/lwd-rdma-p-zy/ipc.sock` 和
  `/tmp/lwd-rdma-d-zy/ipc.sock`，启动 sidecar 时自动创建父目录。
- 保留 TCP 配置中的 `slot_bytes=33554432`（32 MiB）和
  `max-num-batched-tokens=2048`。UCX 接收块容量由 `slot_bytes` 推导。
  未套用两边一云配置的 TP=4、8192 tokens、256 MiB 槽位；调整批量时需同步核对
  main + aux 的总载荷是否能放入一个槽位。
- UCX 仅在两个云端 sidecar 和边端 Worker 中配置；云端模型 Worker 使用本机 IPC。
  P→D 的 KV 传输继续使用原 Mooncake 配置。

默认 UCX 设置与 RDMA 验证目录一致：

```bash
export UCX_PREFIX=/vllm-workspace/ucx
export UCX_TLS=rc_verbs,tcp,self
export LWD_CLOUD_UCX_NET_DEVICES=rocep67s0f0:1,enp189s0f0
export LWD_EDGE_UCX_NET_DEVICES=rocep67s0f0:1,enp189s0f0
```

两端 RDMA 设备为 `rocep67s0f0:1`，对应接口 `enp67s0f0np0`；
A / B 的 RDMA IP 分别为 `194.32.1.2` / `189.32.1.2`。
保留 `enp189s0f0` 供 UCX 辅助建连，GLOO/TP/HCCL socket 也继续使用此业务网接口。
`control_endpoint`、`data_endpoint` 和 UCX 的 TCP 握手仍使用 `76.76.26.189`；
RDMA 设备由 `UCX_NET_DEVICES` 选择。

角色专用 `LWD_*_UCX_NET_DEVICES` 优先于通用 `UCX_NET_DEVICES`，未设置时使用上述默认值。
`CODEBASE_DIR`、`LWD_PYTHON`、`LWD_DIRECTORY` 也可在启动前导出覆盖；
`LWD_DIRECTORY` 建议使用绝对路径。
`UCX_PREFIX` 会补充 `bin`、`lib`、`lib64` 搜索路径；显式设为空可沿用已有运行库路径。
云端 sidecar 和边端环境须能加载参考配置使用的 UCX 及 RDMA 设备。

## 检查和启动

六个入口均支持 `--dry-run`，输出实际命令参数和关键环境变量，
不加载模型、UCX，不创建 IPC 目录或监听端口。例如：

```bash
bash /home/z00853488/sidecar_rdma/sidecar_p.sh --dry-run
bash /home/z00853488/sidecar_rdma/edge.sh --dry-run
```

云端 B 按以下顺序启动，每条命令使用独立终端并保持运行：

```bash
bash /home/z00853488/sidecar_rdma/sidecar_p.sh
bash /home/z00853488/sidecar_rdma/sidecar_d.sh
bash /home/z00853488/sidecar_rdma/cloud_p.sh
bash /home/z00853488/sidecar_rdma/cloud_d.sh
bash /home/z00853488/sidecar_rdma/proxy.sh
```

边端 A 启动：

```bash
bash /home/z00853488/sidecar_rdma/edge.sh
```

模型和协调接口仍沿用 TCP 配置：边端 API `8060`，P/D API `8560/8561`，
P/D 协调接口 `8660/8661`，proxy `8500`，Mooncake `kv_port=24679/24680`。
sidecar TCP 元数据端口仍为 P `19180/19181`、D `19182/19183`；
另需允许边端到云端的 UCX TCP 握手端口 `19600/19700`。
两套配置共用 NPU 和业务端口，切换时先停止原 TCP 服务。

复用原 TCP P/D 验证请求访问边端 API。两个 sidecar 均应建立 edge 0 的通道，
并确认一次请求经过 P 和 D。由于 `UCX_TLS` 也允许 TCP，
通道建立本身不能证明载荷使用 RDMA；应结合 UCX transport 日志和 RDMA 网卡计数验证。
