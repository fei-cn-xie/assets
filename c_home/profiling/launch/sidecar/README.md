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

## 日志级别（基准测试用 ERROR）

`env.sh` 把 `VLLM_LOGGING_LEVEL` 与 `VLLM_LWD_LOG_LEVEL` 都默认设为 **ERROR**：
日志在每一步的路径上，INFO 在压测时会主导日志量。

代价是拉起阶段的可见性：LWD 链路/握手进展是 **WARNING** 级、KV 与 Mamba cache 统计是 **INFO** 级，
ERROR 下这些都看不到，启动卡住会表现为"静默无输出"。所以：

- **基准测试**：用默认 ERROR
- **首轮拉起 / 排查**：`VLLM_LOGGING_LEVEL=INFO VLLM_LWD_LOG_LEVEL=INFO bash cloud_p.sh`

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

## profiling 采集（本目录与基准包的唯一区别）

本目录是 `bash/pds/30rc_1124/sidecar` 的**拷贝**，只多了一段采集配置：
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
# 用本仓的包装脚本（原因见下）
python3 /workspace/assets/c_home/profiling/parse_profiling.py \
    --input-path /bash_dir/profiling/prof_dir/<时间戳目录> \
    --output-path /bash_dir/profiling/prof_dir/output/<A 或 B>

# 产物：request.csv / request_status.csv / batch.csv / kvcache.csv /
#       chrome_tracing.json / profiler.db
```

> **为什么不用 `msserviceprofiler parse` 原命令**：CANN 9.2.1 自带的
> `ms_service_profiler` 里，任务注册表 `regist_map` 只在 `@Task.register` 被 import
> 时填充，而它的 parse 入口没有 import `data_source` / `pipeline` / `plugins` 三个包，
> 结果是 `python3 -m ms_service_profiler.parse ...` 秒退、只产出一个空的
> `profiler.db`（无 csv、无 chrome_tracing.json、且不报错）。
> `parse_profiling.py` 先 import 这三个包补齐注册表，再走完全相同的解析路径。
> 先用 `--check-only` 可以只验证环境：
>
> ```bash
> python3 /workspace/assets/c_home/profiling/parse_profiling.py \
>     --input-path /bash_dir/profiling/prof_dir/<时间戳目录> --check-only
> ```

### A/B 采集流程

1. 停掉另一套部署（两套共用 195 的 NPU 0-3，边云式另用 194 的 NPU 4）；
2. 拉起本套（命令与基准包相同，见上）；
3. 注入**同一份**负载（输入/输出长度、并发档位、请求数都要与 A 侧一致）；
4. 记录本次时间戳子目录，用 `parse_profiling.py` 导出；
5. 换另一套重复 2-4；
6. 对比 `request.csv` 里的 TTFT/队列等待，以及 `batch.csv` / `kvcache.csv` 的调度与搬运时间。

### 注意

- profiler 会引入额外开销，**采集到的 TTFT/TPOT 不能当作性能数据**，只用于定位差异来源；
- 采集期间日志级别仍是 ERROR（这两个包继承了基准包的设置），排查问题时可
  `VLLM_LOGGING_LEVEL=INFO VLLM_LWD_LOG_LEVEL=INFO` 临时提级；
- `PROFILING_SYMBOLS_PATH` 指向 `proc_dir/ms_config.yaml`，该文件已按当前 vllm / vllm-ascend
  代码重写：
  - **54 条埋点全部静态校验过**（模块文件 + 类 + 方法在 `fix/lwd-multidp-dense-pd` 上存在），
    删掉了 16 条 v0.23 时代的失效埋点（`pd_separated_scheduler`、`_execute_model_edge_*` 等）；
  - 不写 `min_version` / `max_version`：仓里没有 tag，写死版本号会让埋点被静默跳过；
  - 分组：[1] 请求生命周期与调度 [2] 模型执行/采样/MTP [3] PD 的 KV 传输（A/B 共有）
    [4] 边端 [5] 云端 [6] host 数据面 UCX [7] LWD proxy（[4]-[7] 只有 A 侧有）；
  - 只有通用 span 带 `handler`（batch/kvcache/model/request/mtp），LWD 专有埋点一律用默认包装器，
    只记录起止时间戳——TTFT 分解只需要时间戳，这样不依赖 `msserviceprofiler` 的版本细节；
  - 若只想看模型执行与采样，仓里还有精简版 `proc_dir/ms_config_pds.yaml`，覆盖同名变量即可。
