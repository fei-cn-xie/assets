# mooncake_pd_195：集中式（同机）PD 分离，无 edge

在 **76.76.26.195** 上用三个进程实现标准的 prefill/decode 分离：

```
客户端 ──9060──▶ proxy ──┬─▶ prefill 引擎 9160 (NPU 0-3, TP=4, kv_producer)
                         └─▶ decode  引擎 9161 (NPU 4-7, TP=4, kv_consumer)
                                   ▲
                          MooncakeConnectorV1 搬运 KV
```

**没有 edge，也没有 LWD 控制面/拓扑/UCX**：proxy 直接对客户端提供 OpenAI 接口，自己完成
"先 P 后 D"的两跳编排。这与 [../sidecar](../sidecar/README.md)（edge + P + D 的边云式 LWD）是两条不同的路。

## 端口与卡

| 角色 | 端口 | NPU | 说明 |
| --- | --- | --- | --- |
| proxy | **9060** | 无 | 客户端入口；`/v1/chat/completions`、`/v1/completions`、`/v1/models`、`/health` |
| prefill | 9160 | 0,1,2,3 | `kv_role=kv_producer`，`kv_port=14589`，`--enforce-eager` |
| decode | 9161 | 4,5,6,7 | `kv_role=kv_consumer`，`kv_port=14590`，图模式 + MTP |
| master | 29535 / 29536 | — | 两个引擎各自的 torch 分布式端口 |

端口由 `env.sh` 统一定义（`PD_PROXY_PORT` / `PD_PREFILL_PORT` / …），改一处即可。

## 文件

| 文件 | 作用 |
| --- | --- |
| `env.sh` | 运行时环境（NPU/HCCL/OMP/jemalloc）、端口常量、预检、`--dry-run` |
| `cloud_p.sh` | prefill 引擎 |
| `cloud_d.sh` | decode 引擎 |
| `start_proxy.sh` | 启动 `pd_proxy.py`（9060） |
| `pd_proxy.py` | 最小标准 PD proxy：P 探测取 `kv_transfer_params` → 带参打 D → 回放答案 |

## 启动

```bash
# 顺序：P → D → proxy（proxy 先起也能跑，只是前两个就绪前请求会 502）
bash cloud_p.sh
bash cloud_d.sh
bash start_proxy.sh
```

三个入口都支持 `--dry-run`（只打印命令行与环境变量，不启动）。

启动前会做端口预检：若 9060/9160/9161 或 kv/master 端口已被占用（例如旧的 LWD 部署还在跑），
直接报错并 `exit 4`。

## 验证

```bash
curl -s http://76.76.26.195:9060/health
curl -s http://76.76.26.195:9060/v1/models
curl -s http://76.76.26.195:9060/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6","messages":[{"role":"user","content":"你好"}],"max_tokens":64}'
```

- `/health` 返回 `{"status":"ok","stats":{"ok":N,"failed":M}}`。
- 若返回 `502 ... prefill response carries no kv_transfer_params`，说明 P 侧没按 producer 起，
  或 KV 传输没建立（见下节排查）。

## 为什么自己写 pd_proxy.py

仓里的示例 `examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py` **不能用于这套连接器**：
它把请求发给 P 时只改 `max_tokens=1`，**既不设 `kv_transfer_params.do_remote_decode`，也不把 P 返回的
`kv_transfer_params` 转发给 D**（该文件 `:250-278`、`:286-317`）。而 Ascend 的
`MooncakeConnectorV1` 只有看到这些参数才会搬运 KV
（`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py:1911-2049`）。

`pd_proxy.py` 的结构照搬**已经在本集群验证过的** `vllm/v1/lwd_control/pd/proxy.py`，
只去掉 LWD 的 envelope 要求：

1. 给 P：`max_tokens=1`、`stream=false`、`kv_transfer_params={"do_remote_decode":true,"do_remote_prefill":false}`；
2. 从 P 的响应里取 `kv_transfer_params`；
3. 原样带上它打 D（`stream` 跟随客户端），SSE 直接回放。

只用标准库，不依赖 fastapi/httpx。

## 与 ../sidecar（LWD 边云）的差异

| 项 | sidecar（LWD） | 本目录 |
| --- | --- | --- |
| edge | 有（194，承担模板渲染/unembed/采样） | **无** |
| LWD 控制面 | `--additional-config lwd_config` + 拓扑文件 | **无** |
| host 数据面 | `data_plane_comm: host` + UCX + sidecar daemon | **无** |
| KV 传输 | MooncakeConnectorV1 | MooncakeConnectorV1（同） |
| 客户端入口 | edge 的 8510 | proxy 的 9060 |
| P 的 MTP | 有（`cloud_p.sh` 里带 `--speculative-config`） | **去掉**：P 只回答 `max_tokens=1` 的探测，draft 纯属开销 |
| 其余 vLLM 参数 | — | 与 sidecar 逐字一致（`max-model-len/max-num-seqs/max-num-batched-tokens/gpu-memory-utilization/async-scheduling/served-model-name/trust-remote-code/pp`） |

## 调参与排查

- **KV 没搬起来**：先看 P 的日志是否有 `do_remote_decode` 相关记录、D 是否报找不到 `remote_block_ids`；
  再确认 `kv_connector_extra_config` 里两侧的 `tp_size` 与实际 TP 一致（本目录都是 4）。
- **显存不够**：两个引擎各 `--gpu-memory-utilization 0.90`、TP=4，27B bf16 权重每卡约 14 GB；
  若 KV 不足，降 `--max-model-len`（如 32768）或降低 `--max-num-seqs`。
- **只想起一个引擎调试**：可直接 `bash cloud_p.sh`，然后手工按两跳顺序 curl P 和 D，
  观察 `kv_transfer_params` 是否出现在 P 的响应里。
- **proxy 需要更高并发**：`ThreadingHTTPServer` 每请求一线程；压测到极高 QPS 时再考虑换 uvicorn 实现。

## 本目录做过的校验（均实跑）

| 检查 | 方法 | 结果 |
| --- | --- | --- |
| 脚本语法 | `bash -n` × 4 | 通过 |
| proxy 逻辑 | 桩 P/D 服务 + 真实启动 `pd_proxy.py` | 17 项断言全过：P 收到 `max_tokens=1`/`do_remote_decode`；D 收到原始 `max_tokens` 与 P 的 `kv_transfer_params`；SSE 与 `[DONE]` 正常回放；P 缺 KV 参数时返回 502 并点名 `kv_transfer_params`；`/health`、`/v1/models`、`/v1/completions` 正常 |
| 拉起参数 | 三个入口 `--dry-run` 逐项断言 | 端口/TP/卡号/kv_role/kv_port/图模式/MTP 全部符合预期 |
| 与 LWD 参考一致 | 与 `sidecar/cloud_p.sh` 的 argv 比对 | 8 个共享参数逐字相同 |
| 无 LWD 残留 | 只查可执行行 | `lwd_config`/`LWD_TOPOLOGY`/`UCX_`/`libucp` 均无 |

⚠️ 仍未在真机上验证的部分：**纯 Mooncake PD（无 LWD）这条链路本身**在本集群没有跑通过，
`MooncakeConnectorV1` 的 `kv_port`/handshake 是否在"两个引擎同机、各自 TP=4"下正常建立，
需要实机首次启动时确认。若 D 报找不到远端 block / 握手超时，优先怀疑 `kv_rank`、`kv_port`
与 `kv_connector_extra_config` 的组合，把两侧日志发我。
