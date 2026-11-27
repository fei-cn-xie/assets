# 1513：feat/pd-sep-lwd-v0.30.0 边云 PD 拉起（194-195 环境）

这一套不是重新设计参数，而是把 **194-195 上已用 curl 验证过的 `v30_rc` 配置**
（`private_home/vllm_script/pds_comm_1117/v30_rc`，分支 `debug/pd-empty-response-trace`）
**逐字搬到 `feat/pd-sep-lwd-v0.30.0`**，只翻译两者之间唯一不兼容的那一层：拓扑的协调字段。

| 定位 | 说明 |
| --- | --- |
| 目标环境 | 边端 `76.76.26.194`、云端 `76.76.26.195`（即 v30_rc 那套） |
| 目标代码 | 两仓 `feat/pd-sep-lwd-v0.30.0`（已确认） |
| 参数来源 | `v30_rc` 的 5 个脚本 + 拓扑 + `pd_proxy.json` |
| 不改的东西 | 命令行参数、环境变量、端口、卡位、UCX 调优、sidecar 模式、模型路径 |

## 1. 部署拓扑（全部沿用 v30_rc）

| 角色 | 机器 | NPU | 入口端口 | 模型路径 |
| --- | --- | --- | --- | --- |
| edge（instance 0） | 76.76.26.194 | `4` | `8510` | `/weight/Qwen3.6-27B/` |
| cloud 0 / prefill | 76.76.26.195 | `0,1` | `8600`（master 29525） | `/home/weight/Qwen3.6-27B/` |
| cloud 1 / decode | 76.76.26.195 | `2,3` | `8601`（master 29526） | `/home/weight/Qwen3.6-27B/` |
| PD proxy | 76.76.26.195 | — | `8102` | — |

> **注意模型路径逐机不同**：edge 用 `/weight/...`，cloud 用 `/home/weight/...`。
> 这是 v30_rc 的原始布局（两台机器的文件系统不共享），不是笔误，脚本里按字面写死。

端口（都在拓扑里）：P `ctrl_port 6100` / `data_port 7200` / `coord_port 8100`；
D `ctrl_port 6101` / `data_port 7201` / `coord_port 8101`；Mooncake `kv_port 14579/14580`。

## 2. 与 v30_rc 的唯一差异：拓扑协调字段

两个分支的数据面代码**完全相同**（`ucx_channel.py` / `ucx_link.py` / `ucx_io.py` 逐字节一致），
但配置 schema 分叉了：feat 做了一次 "DP 直连协调" 重构，**删除**了 `control_url` /
`control_url_map` / `listen_port`，改成每个云 DP 一个 `coord_port`。
直接把 v30_rc 的拓扑拷过来会被拒（我用校验器实测，见 §5）。

| v30_rc（debug 分支） | 1513（feat 分支） |
| --- | --- |
| `coordination.control_url: http://76.76.26.195:8102/v1/chat/completions`（edge） | `coordination.pd_proxy_url:` 同一个 URL |
| `coordination.listen_port: 8100/8101`（cloud） | `dp[].coord_port: 8100/8101` |
| `control_url_map` | 无对应（未使用） |
| 其余字段 | 完全不变 |

除此之外，`--additional-config` 里的拓扑路径从写死的 `/vllm-workspace/pd_host_topology.yaml`
改为**脚本目录**下的 `pd_host_topology.yaml`（自动解析，可整体搬迁）。`enable_cpu_binding: true`
保留在配置里（feat 的 `ascend_config.py:565` 有这个字段，默认本就是 true）。

## 3. 文件

| 文件 | 作用 |
| --- | --- |
| `env.sh` | 公共环境：照抄 v30_rc 的运行时变量（HCCL/UCX/日志/libucp 路径）、`--dry-run`、libucp 预检 |
| `pd_host_topology.yaml` | 拓扑（上面的 schema 翻译），三端共用同一份 |
| `pd_proxy.json` | proxy 目标表，与 v30_rc 那份语义完全一致 |
| `edge.sh` | 边端，`python3 -m vllm.entrypoints.cli.main serve` |
| `cloud_p.sh` | 云端 P，Mooncake producer |
| `cloud_d.sh` | 云端 D：Mooncake consumer + `FULL_DECODE_ONLY` + 显式 `cudagraph_capture_sizes`（4…100，与集中式包一致） |
| `start_proxy.sh` | PD proxy（沿用 v30_rc 的命名） |
| `check_configs.py` | 拓扑/proxy 自检（纯标准库，按 feat 规则逐条校验） |

## 4. 部署与启动

把整个目录拷到两台机器的同一路径（例如 `/bash_dir/assets/c_home/bash/pds/30rc_1124/1513`），
**两台必须用同一份 `pd_host_topology.yaml`**。

```bash
python3 check_configs.py        # 改过地址/端口后先自检（可选，纯标准库）

# 云端 195：按顺序，各占一个终端
bash cloud_p.sh
bash cloud_d.sh
bash start_proxy.sh

# 边端 194：最后启动（只有边端会阻塞等链路，605s/条 IO，最坏 4 条）
bash edge.sh
```

curl 自测（与 v30_rc 相同的方式，打到边端 8510）：

```bash
curl -s http://76.76.26.194:8510/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6","messages":[{"role":"user","content":"你好"}],"max_tokens":32}'
```

## 5. 这次做了哪些校验（可复现）

| 检查 | 方法 | 结果 |
| --- | --- | --- |
| 脚本语法 | `bash -n` × 5 | 全通过 |
| 配置合法 | `check_configs.py`（按 feat 的 `topology.py` / `proxy.py` 规则） | PASS；host 内存估算 2304 MiB/rank（预算 12288） |
| **命令行逐项对齐** | 从 dry-run 取实际 argv，与 v30_rc 的命令对比 | edge 32 项、cloud_p 36 项、cloud_d 37 项**全等**；proxy 参数全等（仅配置文件路径归一化） |
| **环境变量逐项对齐** | dry-run 的环境快照 vs v30_rc 的 `export` | 三个角色各 24 个共有变量**全等** |
| 拓扑语义 | 逐字段对比 | `deployment`（含 sidecar/缓冲区/pd 块）、`feature_ctrl`、`instance_links`、edge dp、cloud 的 addr/ctrl_port/data_port **全等**；仅协调字段按 §2 翻译 |
| proxy 配置 | 与 v30_rc 的 `pd_proxy.json` 做 JSON 语义比较 | 完全一致 |
| **翻译是否必要** | 用 feat 规则校验 v30_rc 的原始拓扑 | **被拒**：`edges[0].coordination: unknown keys ['control_url']` |
| 校验器是否可用 | 10 个故意注入的故障（回填 control_url/listen_port、删 coord_port、边端加 ctrl_port/data_port、双 P、删 up_buffer_size、删 tenant key 等） | 10/10 全部被拒 |

## 6. 排障要点（都是前几轮实际踩过的）

1. **边端启动会静默等链路**：单条 `wait_ready` 超时 `605s`，逐条串行，最坏约 40 分钟；
   等待期的进展日志是 WARNING 级。脚本已把 `VLLM_LOGGING_LEVEL` / `VLLM_LWD_LOG_LEVEL`
   默认设为 `INFO`（与 v30_rc 一致），不要改成 ERROR 起首轮。
2. **缺 libucp 会伪装成网络问题**：`libucp.so.0 not found` 会让连接线程每 0.2→5s 重试，
   最后只报 `[LWD][handshake] host link not ready`。脚本启动前会预检，缺库直接 `exit 3`。
   本环境 libucp 由 `/vllm-workspace/ucx/lib` 提供（v30_rc 的写法）；换路径用
   `LWD_UCX_LIB_DIR=<dir> bash edge.sh`。
3. **两份拓扑必须一致**：每端启动日志里的 `[LWD][config] ... digest=<16 hex>` 要三端相同，
   不一致就是文件没同步（`sha256sum` 对一下）。
4. **sidecar 模式的前置条件**：每机一个 transport daemon，由首个引擎进程自动拉起，
   依赖 `memfd_create` 与可写的 `/tmp`（ctl socket `/tmp/lwd-host-comm-<uid>.sock`）。
   容器若禁 memfd，把拓扑里 `deployment: sidecar` 改成 `in_process`（两端同改）。
5. **日志级别与 digest 之外，先看云端的 listener 是否就绪**：
   `[LWD] host listener bound for cloud=0 dp_idx=0 on 76.76.26.195:7200`，
   `ss -lntp | grep -E '7200|7201'` 也应能看到。
