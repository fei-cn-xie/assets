# mooncake_pd_195：集中式（同机）PD 分离

在 **76.76.26.195** 上用三个进程实现标准的 prefill/decode 分离：

```
客户端 ──9060──▶ proxy ──┬─▶ prefill 9160 (NPU 0,1, TP=2, kv_producer)
                         └─▶ decode  9161 (NPU 3,4, TP=2, kv_consumer)
                                   ▲
                          MooncakeConnectorV1 搬运 KV
```

三个进程按角色命名：**`prefill.sh` / `decode.sh` / `start_proxy.sh`**——本部署只有
prefill、decode、proxy 三种角色，没有 edge/cloud 概念。

**没有 LWD 控制面/拓扑/UCX**：proxy 直接对客户端提供 OpenAI 接口（9060），自己完成
"先 P 后 D"的两跳编排。这与 [../sidecar](../sidecar/README.md)（LWD 边云式，需要额外的 edge 进程）
是两条不同的路。

本目录**只有 shell 拉起脚本，没有任何 Python 代码**：proxy 直接用 vllm-ascend 仓自带的示例。

## 端口与卡

| 角色 | 端口 | NPU | 说明 |
| --- | --- | --- | --- |
| proxy | **9060** | 无 | 客户端入口；`/v1/chat/completions`、`/v1/completions`、`/v1/models`、`/healthcheck` |
| prefill | 9160 | 0,1 | TP=2，`kv_role=kv_producer`，`kv_port=14589`，`--enforce-eager` |
| decode | 9161 | 3,4 | TP=2，`kv_role=kv_consumer`，`kv_port=14590`，图模式 + MTP |
| master | 29535 / 29536 | — | 两个引擎各自的 torch 分布式端口 |

端口由 `env.sh` 统一定义（`PD_PROXY_PORT` / `PD_PREFILL_PORT` / …），改一处即可。

## 文件

| 文件 | 作用 |
| --- | --- |
| `env.sh` | 运行时环境（NPU/HCCL/OMP/jemalloc）、端口与路径常量、预检、`--dry-run` |
| `prefill.sh` | prefill 引擎 |
| `decode.sh` | decode 引擎 |
| `start_proxy.sh` | 启动 vllm-ascend 自带的 proxy（9060） |

proxy 代码不在本仓，路径为：

```
$CODEBASE_DIR/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py
```

`start_proxy.sh` 启动前会检查该文件是否存在，缺失时给出明确报错（`exit 5`）。

## 启动

```bash
# 顺序：P → D → proxy
bash prefill.sh
bash decode.sh
bash start_proxy.sh
```

三个入口都支持 `--dry-run`（只打印命令行与环境变量，不启动）。

启动前会做端口预检：若 9060/9160/9161 或 kv/master 端口已被占用（例如旧的 LWD 部署还在跑），
直接报错并 `exit 4`。

## 验证

```bash
curl -s http://76.76.26.195:9060/healthcheck
curl -s http://76.76.26.195:9060/v1/models
curl -s http://76.76.26.195:9060/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6","messages":[{"role":"user","content":"你好"}],"max_tokens":64}'
```

- 健康检查端点是 **`/healthcheck`**（上游示例的命名）。
- 若 proxy 报 P 侧没有 `kv_transfer_params`，说明 P 的 `kv_role` 或 KV 通道没建起来，见下节。

## 为什么用 vllm-ascend 的这个 proxy

它的 `handle_completions_impl` 正是标准两跳：

| 步骤 | 代码位置（`load_balance_proxy_server_example.py`） |
| --- | --- |
| 给 P 设 `kv_transfer_params={"do_remote_decode": true, "do_remote_prefill": false}` | `:821-823` |
| 从 P 的响应取 `kv_transfer_params` 并转发给 D | `:973-975` |
| 路由：`/v1/completions`、`/v1/chat/completions`、`/v1/models`、`/healthcheck` | `:1292-1364` |

参数形态与 `examples/disaggregated_prefill_v1/mooncake_connector_deployment_guide.md` 一致，
`--kv-transfer-config` 里的 `kv_buffer_device="npu"`、`kv_parallel_size=1`、字符串形式的
`kv_port`、`kv_rank` 0/1 也都取自该指南。

**不要用** `vllm/examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py`：它给 P 只改
`max_tokens=1`，既不设 `kv_transfer_params`，也不把 P 的返回值转给 D（该文件 `:250-278`、`:286-317`），
配 `MooncakeConnectorV1` 会"跑起来但根本不搬 KV"。

## 与 ../sidecar（LWD 边云）的差异

| 项 | sidecar（LWD） | 本目录 |
| --- | --- | --- |
| edge | 有（194，承担模板渲染/unembed/采样） | **无**：本部署不需要 edge 进程 |
| LWD 控制面 | `--additional-config lwd_config` + 拓扑文件 | **无** |
| host 数据面 | `data_plane_comm: host` + UCX + sidecar daemon | **无** |
| KV 传输 | MooncakeConnectorV1 | MooncakeConnectorV1（同连接器，参数按官方指南补全） |
| 客户端入口 | edge 的 8510 | proxy 的 9060 |
| P 的 MTP | 有（sidecar 的 `cloud_p.sh` 里带 `--speculative-config`） | **去掉**：P 只回答 `max_tokens=1` 的探测，draft 纯属开销 |
| 其余 vLLM 参数 | — | 与 sidecar 逐字一致（`max-model-len/max-num-seqs/max-num-batched-tokens/gpu-memory-utilization/async-scheduling/served-model-name/trust-remote-code/pp`） |

## 调参与排查

- **KV 没搬起来**：先看 P 日志里是否出现 `do_remote_decode` 相关记录、D 是否报找不到 `remote_block_ids`；
  再确认 `kv_connector_extra_config` 里两侧的 `tp_size` 与实际 TP 一致（本目录都是 2），
  以及两个 `kv_port`（14589/14590）在 195 上确实空着。
- **显存**：两个引擎各 TP=2、`--gpu-memory-utilization 0.90`。27B 权重约 51.7 GiB（bf16），
  按 TP=2 切分后每卡约 26 GiB。TP=2 + `--max-model-len 65536` 这套组合已由
  [../sidecar](../sidecar/README.md) 的 cloud_p/cloud_d（同为 TP=2）在本机跑通过，可以先用默认值；
  若仍在启动时报 KV block 不足，再降 `--max-model-len`（如 32768）或提高 `--gpu-memory-utilization`。
- **只想起一个引擎调试**：直接 `bash prefill.sh`，然后手工按两跳顺序 curl P 与 D，
  观察 P 的响应里是否带 `kv_transfer_params`。
- **改并发**：proxy 的并发上限由 `--max-retries` 等参数控制，见上游脚本的 argparse。

## 本目录做过的校验（均实跑）

| 检查 | 方法 | 结果 |
| --- | --- | --- |
| 脚本语法 | `bash -n` × 4 | 通过 |
| 拉起参数 | 三个入口 `--dry-run` 逐项断言 | 端口/TP/卡号/kv_role/kv_rank/kv_port/kv_buffer_device/kv_parallel_size/图模式/MTP 全部符合预期 |
| proxy 指向 | `start_proxy.sh --dry-run` | 指向 vllm-ascend 示例脚本，参数 `--port 9060`、`--prefiller-hosts 76.76.26.195`、`--prefiller-ports 9160`、`--decoder-hosts/ports` 对应 9161 |
| 上游文件存在 | 本地 vllm-ascend 检出的该路径 | 存在，且 kv 握手代码行号与上表一致 |
| 与 LWD 参考一致 | 与 `sidecar/cloud_p.sh` 的 argv 比对（sidecar 按边云命名） | 8 个共享参数逐字相同 |
| 无 LWD 残留 / 无本地 Python | 只查可执行行 + 目录清单 | `lwd_config`/`LWD_TOPOLOGY`/`UCX_`/`libucp` 均无；目录内没有 `*.py` |

⚠️ 仍未在真机上验证的部分：**纯 Mooncake PD（无 LWD）这条链路本身**在本集群没有跑通过，
`MooncakeConnectorV1` 的 `kv_port`/handshake 在"两个引擎同机、各自 TP=2"下是否正常建立，
需要实机首次启动时确认。若 D 报找不到远端 block / 握手超时，优先怀疑 `kv_rank`、`kv_port`
与 `kv_connector_extra_config` 的组合，把两侧日志发我。
