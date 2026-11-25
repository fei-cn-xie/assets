# centralized_195：集中式（非 PD 分离）Qwen3.6-27B 服务

单实例承载完整模型，**没有** edge/cloud 拆分、没有 LWD 控制面、没有 Mooncake KV 搬运、没有 PD proxy。
同一张卡组上跑完整的前处理 + prefill + decode，客户端直接打这一个端口。

参照的已验证配置是 [../../pds/30rc_1124/sidecar](../../pds/30rc_1124/sidecar)（PD 三实例），
本目录只保留其中与 PD 无关的运行时部分。

## 部署参数

| 项 | 值 | 来源 |
| --- | --- | --- |
| 主机 | `76.76.26.195` | 你指定 |
| NPU | `0,1,2,3`（`ASCEND_RT_VISIBLE_DEVICES`） | 你指定 |
| TP | `4` | 你指定 |
| 端口 | `9060` | 你指定 |
| 模型路径 | `/home/weight/Qwen3.6-27B/` | 195 上的实际路径（与 `cloud_p.sh`/`cloud_d.sh` 一致；194 上是 `/weight/...`） |
| MTP | `method=mtp, num_speculative_tokens=3, enforce_eager=true` | 与 `cloud_d.sh` 一致 |
| 图模式 | `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`（不加 `--enforce-eager`） | 与 `cloud_d.sh` 一致 |

## 文件

| 文件 | 作用 |
| --- | --- |
| `env.sh` | 通用运行时环境 + 预检 + `--dry-run` 辅助；被 `serve.sh` source |
| `serve.sh` | 唯一入口：一条 `vllm serve` 命令 |

## 启动

```bash
# 0) 先确认 195 上的 PD 引擎已停 —— 它们占用的正是 NPU 0-3！
#    cloud_p.sh 用 0,1；cloud_d.sh 用 2,3；三者不能共存。
#    （serve.sh 的预检会在 8600/8601 仍在监听时给出 WARNING）

cd /path/to/centralized_195
bash serve.sh
```

`--dry-run` 只打印将要执行的命令行与关键环境变量，不真正启动：

```bash
bash serve.sh --dry-run
```

启动成功后，日志里应能看到 `Uvicorn running on http://0.0.0.0:9060`。

## 验证

```bash
# 模型名与 PD 部署一致，都是 qwen3.6
curl -s http://76.76.26.195:9060/v1/models

curl -s http://76.76.26.195:9060/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6","messages":[{"role":"user","content":"你好"}],"max_tokens":32}'
```

压测脚本（`c_home/bench_test/`）默认打 `194:8510`，改 `host_ip`/`host_port` 为 `76.76.26.195`/`9060` 即可。

## 保留了什么、去掉了什么

相对 `pds/30rc_1124/sidecar/env.sh`：

| 项 | 处理 | 原因 |
| --- | --- | --- |
| `PYTHONPATH`（vllm-ascend + vllm） | 保留 | 与 PD 同一套代码树 |
| `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` | 保留 | 显存碎片优化，通用 |
| `HCCL_OP_EXPANSION_MODE=AIV`、`HCCL_BUFFSIZE=1024` | 保留 | 本硬件验证过的通信配置 |
| `GLOO/TP/HCCL_SOCKET_IFNAME=lo` | 保留 | 单机，bootstrap 走回环；PD 时同机 TP=2 也这么用 |
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | 保留 | 多进程 worker，通用 |
| `OMP_*`、`TASK_QUEUE_ENABLE` | 保留 | 绑核/队列优化，通用 |
| `LD_PRELOAD=libjemalloc` | 保留 | 内存分配器，通用（不需要可自行删） |
| `VLLM_LOGGING_LEVEL`、`VLLM_ENGINE_READY_TIMEOUT_S=1800` | 保留 | 27B 加载 + 图捕获耗时数分钟 |
| `UCX_*`、`LWD_UCX_LIB_DIR` | **去掉** | 只服务于 LWD host 数据面（跨主机 hidden/H 传输） |
| `LWD_TOPOLOGY*`、`--additional-config lwd_config` | **去掉** | 单实例没有边云拓扑 |
| `lwd_require_ucx_runtime`（libucp 预检） | **去掉** | 不加载 libucp |
| `VLLM_LWD_LOG_LEVEL` | **去掉** | 没有 `[lwd]` 日志流 |
| `--kv-transfer-config`（Mooncake） | **去掉** | 没有 P→D 的 KV 搬运 |
| `--master-addr/--master-port` | **去掉** | 单机默认端口足够；若与其它任务抢 29500 再加 |

`--additional-config '{"enable_cpu_binding":true}'` 保留但只带这一个键：它是 vllm-ascend 的通用开关
（`vllm_ascend/ascend_config.py` 里默认即为 `true`），与 LWD 无关；写出来是为了和 PD 的启动参数对齐。

## 调参提示

- **TP 与显存**：27B bf16 约 54 GB 权重。TP=4 时每卡约 14 GB 权重 + KV cache，`--max-model-len 65536`
  在该配置下可行；若把 TP 降到 2，先把 `--max-model-len` 降到 32768 左右再试。
- **关 MTP**：删掉 `--speculative-config` 那一行即可（同时可保留 `--compilation-config`）。
  MTP 的 `enforce_eager:true` 只作用于 draft 模型，主模型仍是图模式。
- **改 eager**：删掉 `--compilation-config`，加 `--enforce-eager`（启动更快、显存更省、吞吐略低）。
- **端口冲突**：预检会直接 `exit 4`；改端口时同步改 `base_preflight` 的入参（`serve.sh` 里那一行）。

## 本目录做过的校验

| 检查 | 方法 | 结果 |
| --- | --- | --- |
| 语法 | `bash -n` × 2 | 通过 |
| 入口可执行性 | `bash serve.sh --dry-run` | 输出完整命令行与 20 个环境变量 |
| 与原 PD 参考的差异受控 | 逐参数比对 `cloud_d.sh` 的 argv | 差异仅限本文档列出的项（端口/TP/卡号/去掉的 LWD+KV+master） |
| 无 LWD 残留 | grep `UCX|LWD|kv-transfer|additional-config` | 仅剩 `enable_cpu_binding` 一处 |
