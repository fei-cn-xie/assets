# mooncake_pd_195：集中式（同机）PD 分离

在 **76.76.26.195** 上用三个进程实现标准的 prefill/decode 分离：

```
客户端 ──9060──▶ proxy ──┬─▶ prefill 9160 (NPU 0,1, TP=2, kv_producer)
                         └─▶ decode  9161 (NPU 2,3, TP=2, kv_consumer)
                                   ▲
                          MooncakeConnectorV1 搬运 KV
```

三个进程按角色命名：**`prefill.sh` / `decode.sh` / `start_proxy.sh`**——本部署只有
prefill、decode、proxy 三种角色，没有 edge/cloud 概念。

**没有 LWD 控制面/拓扑/UCX**：proxy 直接对客户端提供 OpenAI 接口（9060），自己完成
"先 P 后 D"的两跳编排。这与 [../sidecar](../sidecar/README.md)（LWD 边云式，需要额外的 edge 进程）
是两条不同的路。

本目录**只有 shell 拉起脚本，没有任何 Python 代码**：proxy 直接用 vllm-ascend 仓自带的示例。

## 权威参考

引擎参数**逐项对齐**已确认可用的

```
private_home/vllm_script/pds/centralized/{prefill.sh,decode.sh,proxy.sh}
```

本目录与它只有两处差异：内部端口（9160/9161 而非 8060/8061）、以及把公共环境抽到 `env.sh`
（便于两处同时改）。`--kv-port` 与参考保持一致（30000 / 30100）。

## 端口与卡

| 角色 | 端口 | NPU | 说明 |
| --- | --- | --- | --- |
| proxy | **9060** | 无 | 客户端入口；`/v1/chat/completions`、`/v1/completions`、`/v1/models`、`/healthcheck` |
| prefill | 9160 | 0,1 | TP=2，`kv_role=kv_producer`，`kv_port=30000`，`engine_id="0"`，`--enforce-eager` |
| decode | 9161 | 2,3 | TP=2，`kv_role=kv_consumer`，`kv_port=30100`，`engine_id="1"`，图模式 + MTP |

端口由 `env.sh` 统一定义（`PD_PROXY_PORT` / `PD_PREFILL_PORT` / `PD_*_KV_PORT` / …），改一处即可。

## 文件

| 文件 | 作用 |
| --- | --- |
| `env.sh` | 运行时环境（网卡/mooncake 库/NPU/HCCL/OMP/jemalloc）、端口常量、预检、`--dry-run` |
| `prefill.sh` | prefill 引擎 |
| `decode.sh` | decode 引擎 |
| `start_proxy.sh` | 启动 vllm-ascend 自带的 proxy（9060） |

proxy 代码不在本仓：

```
$CODEBASE_DIR/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py
```

`start_proxy.sh` 启动前会检查该文件是否存在，缺失时报错并 `exit 5`。

## 启动

```bash
# 顺序：P → D → proxy
bash prefill.sh
bash decode.sh
bash start_proxy.sh
```

三个入口都支持 `--dry-run`（只打印命令行与环境变量，不启动）。启动前做端口预检，
冲突时报错并 `exit 4`。

## 验证

```bash
curl -s http://76.76.26.195:9060/healthcheck
curl -s http://76.76.26.195:9060/v1/models
curl -s http://76.76.26.195:9060/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6","messages":[{"role":"user","content":"你好"}],"max_tokens":64}'
```

健康检查端点是 **`/healthcheck`**（上游示例的命名）。

## 相对上一版的关键修正

上一版（自己拼参数）拉起能成功、bench 请求会崩。对照已验证脚本后修正如下：

| 项 | 上一版（错） | 现在（对齐参考） |
| --- | --- | --- |
| mooncake 库路径 | 未设置 | `LD_LIBRARY_PATH` 加 `…/ascend-toolkit/latest/python/site-packages/mooncake` |
| KV 身份 | `kv_rank` + `kv_buffer_device` + `kv_parallel_size` | **`engine_id`: `"0"` / `"1"`**（参考里的写法，其余字段都不要） |
| MTP 方法 | `"method":"mtp"` | **`"method":"qwen3_5_mtp"`**（模型专用 proposer） |
| 通信网卡 | `GLOO/TP/HCCL_SOCKET_IFNAME=lo` | **真实网卡 `enp189s0f0`** + `HCCL_IF_IP` |
| 执行后端 | 未指定 | **`--distributed-executor-backend mp`** |
| decode 图捕获 | `FULL_DECODE_ONLY` + 显式档位（4…100） | **相同**（两边都用 25 档显式列表） |
| 显存占比 | 0.95 | **0.95**（两边一致） |
| 前缀缓存 | 依赖默认 | **显式 `--enable-prefix-caching`** |
| P 的 MTP | 去掉了 | **恢复**（参考在 P 上也带 drafts） |
| 多余参数 | `--master-addr/--master-port`、`--additional-config`、`--pipeline-parallel-size`、HCCL 超时 | 全部去掉（参考里都没有） |

## 与 ../sidecar（LWD 边云）的差异

| 项 | sidecar（LWD） | 本目录 |
| --- | --- | --- |
| edge | 有（194，承担模板渲染/unembed/采样） | **无**：本部署不需要 edge 进程 |
| LWD 控制面 | `--additional-config lwd_config` + 拓扑文件 | **无** |
| host 数据面 | `data_plane_comm: host` + UCX + sidecar daemon | **无** |
| KV 传输 | MooncakeConnectorV1 | MooncakeConnectorV1（同连接器，字段按已验证脚本） |
| 客户端入口 | edge 的 8510 | proxy 的 9060 |

## 排查

- **bench 请求崩、拉起正常**：先看两个引擎日志里 mooncake 相关报错。历史上最常见的就是
  `LD_LIBRARY_PATH` 没带 mooncake 目录（引擎能起，第一次 KV 传输才失败）。
- **KV 没搬起来**：确认 P 日志里出现 `do_remote_decode`、D 是否能拿到 `remote_block_ids`；
  `kv_port`（30000/30100）必须在本机空闲，且两侧 `kv_connector_extra_config` 的 `tp_size` 都是 2。
- **只想起一个引擎调试**：直接 `bash prefill.sh`，然后按两跳顺序手工 curl P 与 D，
  观察 P 的响应里是否带 `kv_transfer_params`。
- **网卡名不对**：`env.sh` 里 `PD_NIC_NAME` 默认 `enp189s0f0`（195 的网卡），
  换机器时用 `ip -br link` 确认后覆盖。

## 本目录做过的校验（均实跑）

| 检查 | 方法 | 结果 |
| --- | --- | --- |
| 脚本语法 | `bash -n` × 4 | 通过 |
| 与已验证参考逐项对齐 | 三个入口 `--dry-run` 后与 `pds/centralized/*.sh` 的 argv 逐项比对 | 除内部端口外**完全一致**（TP/前缀缓存/显存/async/mp/enforce-eager/图捕获列表/MTP/kv 字段） |
| 端口与卡 | 同上 | proxy 9060、P 9160（0,1）、D 9161（2,3），kv 30000/30100 |
| 环境变量 | 同上 | `HCCL_IF_IP`、`GLOO/TP/HCCL_SOCKET_IFNAME=enp189s0f0`、`LD_LIBRARY_PATH` 含 mooncake |
| proxy 指向 | `start_proxy.sh --dry-run` | 指向 vllm-ascend 示例，参数 `--port 9060`、prefiller 9160、decoder 9161 |
| 无 LWD 残留 / 无本地 Python | 只查可执行行 + 目录清单 | `lwd_config`/`LWD_TOPOLOGY`/`UCX_`/`libucp` 均无；目录内没有 `*.py` |

## profiling 采集（本目录与基准包的唯一区别）

本目录是 `bash/pds/30rc_1124/mooncake_pd_195` 的**拷贝**，只多了一段采集配置：
`env.sh` 顶层导出两个变量，所有进程（引擎 + proxy）都会继承。

```bash
export SERVICE_PROF_CONFIG_PATH=/workspace/assets/c_home/profiling/ms_profile_config.json
export PROFILING_SYMBOLS_PATH=/workspace/assets/c_home/profiling/proc_dir/ms_config.yaml
```

### 采集前必须做的一步

上游文档 `vllm-ascend/docs/source/developer_guide/performance_and_debug/service_profiling_guide.md` §2
要求把 config 里的 `enable` 从 0 改成 1（仓内 `ms_profile_config.json` 默认是 0）：

```bash
sed -i 's/"enable":\s*0/"enable": 1/' /workspace/assets/c_home/profiling/ms_profile_config.json
```

采集完成后记得改回 0，否则后续基准测试也会带着 profiler 开销。

### 结果落在哪里

`prof_dir` 取自该 config（当前 `/bash_dir/profiling/prof_dir/`），每次启动自动建一个
**时间戳子目录**；A/B 两套共用它，靠时间戳区分即可。解析：

```bash
cd /bash_dir/profiling/prof_dir/<时间戳目录>
msserviceprofiler parse --input-path=./ --output-path output
# output/ 下有 chrome_tracing.json、profiler.db、request.csv、kvcache.csv、batch.csv
```

### A/B 采集流程

1. 停掉另一套部署（两套共用 195 的 NPU 0-3，边云式另用 194 的 NPU 4）；
2. 拉起本套（命令与基准包相同，见上）；
3. 注入**同一份**负载（输入/输出长度、并发档位、请求数都要与 A 侧一致）；
4. 记录本次时间戳子目录，`msserviceprofiler parse` 导出；
5. 换另一套重复 2-4；
6. 对比 `request.csv` 里的 TTFT/队列等待，以及 `batch.csv` / `kvcache.csv` 的调度与搬运时间。

### 注意

- profiler 会引入额外开销，**采集到的 TTFT/TPOT 不能当作性能数据**，只用于定位差异来源；
- 采集期间日志级别仍是 ERROR（这两个包继承了基准包的设置），排查问题时可
  `VLLM_LOGGING_LEVEL=INFO VLLM_LWD_LOG_LEVEL=INFO` 临时提级；
- `PROFILING_SYMBOLS_PATH` 指向全量符号表；若只想看模型执行与采样，仓里还有精简版
  `proc_dir/ms_config_pds.yaml`，覆盖同名变量即可。
