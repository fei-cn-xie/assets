# vLLM 进程与线程 · 读图指南

配套文件：`vLLM进程与线程.drawio`（10 页）。
主题：**进程间通信、进程生命周期、进程/线程个数与启动配置的关系**。
锚点按本仓当前检出分支写（`vllm` / `vllm-ascend` 均为 `fix/lwd-multidp-dense-pd`）。

---

## 1. 页面清单与阅读顺序

| 页 | 讲什么 | 一句话结论 |
| --- | --- | --- |
| ① | 进程/线程全景 | 进程边界＝谁能碰 NPU；线程边界＝谁不能阻塞主循环 |
| ② | 进程个数怎么算 | 进程数 = API(跟 DP) + EngineCore(跟 DP) + Worker(跟 TP×PP) + 协调/daemon/proxy |
| ③ | 三种 executor | world_size=1 → uni（worker 同进程）；多卡 → mp；跨机 → ray |
| ④ | EngineCore 内部 | 1 个不能停的忙循环 + 2 个 IO 线程 + 1 个 worker 监控线程 |
| ⑤ | 三条 ZMQ 通道 | 握手 / 下行 / 上行；前端统一 bind、引擎统一 connect |
| ⑥ | 启动时序 | 预检 → 拉子进程 → 握手(HELLO→地址表→READY) → 就绪；云端不等，边端逐条等 |
| ⑦ | 死亡检测 | 四条独立路径（协议帧 / 进程句柄 / worker 回调 / 前端看门狗）+ 超时表 |
| ⑧ | worker 里还有什么线程 | 一条 RPC 循环 + 三类后台线程；ModelRunner 自己没有线程 |
| ⑨ | 本仓 LWD 真实进程树 | 194 起 3 个、云端每个实例 8 个、proxy 1、daemon 共享 1；附端口表 |
| ⑩ | 排障地图 | 四条命令看进程/线程/端口，日志关键词，常见死法对照 |
| ⑪ | Scheduler 是什么 | 四个容器 + 状态机；每步只产出一份「执行计划」，不碰模型不碰 KV 数据 |
| ⑫ | 一次 schedule() 的流程 | 先跑老的 → 要块 → 块不够就抢占 → 再收新的 → 产出计划 |
| ⑬ | dp=2×tp=4 的 PD 拓扑 | 单实例 12 进程（1 API + 1 协调 + 2 引擎 + 8 worker），全链路 ≈27 |
| ⑭ | 一个请求的完整旅程 | edge 发号 → Proxy 轮询挑 DP → P 注册 → 4 卡算 → 交接单 → D 拉 KV 算 tail |
| ⑮ | 模型加载 | 读 config → NPU 上搭空骨架 → 逐张量读盘切片拷入 → 格式后处理 → profile → 切 KV |
| ⑯ | 权重怎么参与计算 | 加载写的 `layer.weight` 就是 forward 里 GEMM 读的那份，中间没有第二次搬运 |

三条路径：

- **只想搞懂架构**：① → ② → ③ → ⑤ → ⑪
- **要排障/看现场**：⑨ → ⑦ → ⑩ → ⑧
- **要调启动参数/大配置**：② → ③ → ⑥ → ⑬ → ⑭
- **想懂模型加载与计算**：⑮ → ⑯（并可回到 ⑤ 看 SchedulerOutput 怎么来的）

---

## 2. 一页纸答案（进程数）

```
world_size           = PP × TP × PCP                 # 一个引擎用几张卡（跨 DP 之前）
world_size_across_dp = world_size × DP               # 用来决定后端：>1 才需要 mp/ray
本机 EngineCore 个数  = data_parallel_size_local      # 单机默认 = DP
每个 EngineCore 的 worker 数 = local_world_size      # = TP×PP×PCP（单机）
API server 进程数     = --api-server-count           # 不给时内部 LB 默认 = DP
额外进程              = DP 协调(DP>1 且 rank0，1 个) + LWD ucx_daemon(每机 ≤1) + PD proxy
```

本仓四种真实形态（页 ⑨ 已画）：

| 实例 | API | EngineCore | Worker | 协调 | daemon | 合计 |
| --- | --- | --- | --- | --- | --- | --- |
| edge（TP=1,DP=1 → uni） | 1 | 1 | 0（同进程） | 0 | 1 | **3** |
| sidecar 云端（TP=2,DP=1） | 1 | 1 | 2 | 0 | 共享 | **4** |
| dp2_tp1 云端（TP=1,DP=2） | 1 主 + 2 子 | 2 | 2 | 1 | 共享 | **8** |
| PD proxy | 1 | 0 | 0 | 0 | 0 | **1** |

三个边界条件（页 ② 底部）：`PCP` 也乘进 worker 数；`--enable-expert-parallel`
**不改变**进程数；`VLLM_ENABLE_V1_MULTIPROCESSING=0` 可把 EngineCore 也塞回前端进程
（但 AsyncLLM 不允许）。

---

## 3. 端口表（dp2_tp1_194_195）

`8510` 边端 API · `8600/8601` P/D API（DP>1 时各 2 个 SO_REUSEPORT 监听）·
`8100/8101/8110/8111` coord · `6100/6101/6110/6111` 控制面 ctrl ·
`7200/7201/7210/7211` 数据面 · `14570+dp / 14580+dp` Mooncake KV 握手 ·
`8102` PD proxy · `29600/29601` DP 引擎握手 · `/tmp/lwd-host-comm-<uid>.sock` daemon ctl。

---

## 4. 六个坑

1. **`VLLM_ENGINE_ITERATION_TIMEOUT_S` 在本分支没人读**：只是 env 定义，不是生效的看门狗。
   真正兜底的是 `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=300` 与各 manager 的 shutdown 超时。
2. **没有请求级超时**：前端 `await` 输出是无限期。引擎"卡住但没死"时日志里没有报错，
   只能 `py-spy dump` 看线程栈——这是最难查的一类故障。
3. **ZMQ 不背压**：`SNDHWM/RCVHWM = 0`（无上限）；限流在前端准入
   （`max_num_queued_reqs` / `max_num_queued_tokens`，超了 503）。
4. **`ApiServer_i` 是进程、`output_handler` 是 Task、`MPClientEngineMonitor` 是真线程**：
   画图/排障时别把三者混成同一个东西。
5. **ascend 把 worker 的 `daemon` 改成了 False**（为让 worker 能再拉子进程），
   代价：父进程崩溃时 worker 不会自动回收，残留要靠 terminate/kill 链清。
6. **边端启动慢不是 bug**：4 条链路串行等待、每条 605 s，最坏 ~40 分钟；
   云端只报 "waiting for N edge link(s) asynchronously" 就继续。

---

## 5. 新增六页（⑪–⑯）的要点

### 调度器（⑪ ⑫）

- 四个容器：`requests`（所有未完成）/ `waiting`（FCFS 队列）/ `skipped_waiting`（本步被跳过）/ `running`（**list，末尾是抢占受害者**）。
- 每步两个预算：token 预算（能发多少 token）与 input 预算（能喂多少 input token）；chunked prefill 就是被它切出来的。
- 分配失败 → 抢占：**这个版本只有「重算」一种模式**（没有 SWAP / 换出到 CPU）。被抢占 = 释放块 +
  `num_computed_tokens` 归零 + 插回 waiting 队头，下次全部重算。
- **一旦本步发生抢占，这一步就不再收新请求**（代码里 `if not preempted_reqs` 的前提）。
- 前缀缓存：只有 `num_computed_tokens == 0` 时查；最多命中到 `num_tokens − 1`（最后一个 token 必须重算才能出 logits）。
- PD 挂起：connector 说要异步拉 KV → 请求置 `WAITING_FOR_REMOTE_KVS`、只分块不算 token；
  worker 报 `finished_recving` 后才放行，且要求 `num_computed_tokens == tail_start`。
- TP 相关：`SchedulerOutput` **不含 slot_mapping**，4 个 worker 拿同一份 block_ids、各自现场算 slot。

### dp=2×tp=4 的 PD 场景（⑬ ⑭）

- 进程：单实例 = 1 API + 1 DP 协调 + **2 个独立 EngineCore** + **8 个 worker** = 12；edge 3；Proxy 1；全链路 ≈ **27**。
- **DP 由 PD Proxy 轮询挑**（两个独立计数器选 `dp_P` 与 `dp_D`），不是 edge 打分、也不是 coordinator；
  只有非 PD 路径才走 edge 的 `4×prefill + 1×decode` 打分。
- **dense 塌缩后没有 wave 锁步**：每个 DP rank 是独立 `EngineCoreProc`；`DPCoordinator` 进程仍会起，
  但 `enable_wave_coordination = is_moe`，dense 收到波次消息只会打 "Unrecognized input request type"。
- TP=4 协同：QKV（列并行、不通信）→ attention → o_proj（行并行、**每层一次 all_reduce**）→
  只有 `output_rank`（组内 rank0）回采样结果；logits 出词表时 all_gather。
- P→D 的 KV：**P/D 的 TP 相等时是一对一恒等映射**（D worker r ⇄ P worker r），不需要拼装；
  DP 维是 `(dp_P, dp_D)` 自由组合，不要求两边相等。

### 模型加载（⑮ ⑯）

- 八步：读 `config.json` → 按 index 只留该用的分片 → 选 loader（默认 safetensors）→
  **在 NPU 上搭空骨架（参数先分配）** → 逐张量「读盘 → 按 TP narrow 切片 → `param.data.copy_`」→
  后处理（量化 / ND→NZ）→ profile 激活峰值 → 剩余显存切成 KV 块池。
- 所以不是「把权重搬到 NPU」，而是「往 NPU 上已经分配好的参数里写数」。
- **权重与 KV cache 都占显存但完全不同**：权重来自磁盘、固定、只读；KV 是运行期算出来的、
  随并发与上下文增长、可被抢占丢弃。顺序硬性：权重 → profile → KV，
  所以「权重变大 = KV 块变少 = 并发变低」。
- 计算衔接：`layer.weight` 在 forward 里被 GEMM 直接读（`F.linear` / `torch.ops.vllm.unquantized_gemm`），
  中间没有第二次拷贝；NZ 是就地换 `weight.data`。

---

## 6. 自查与重生成

```bash
cd assets/vllm-ascend/drawio/tools
python proc/build.py                    # 重建 ../vLLM进程与线程.drawio（16 页）
python checkdrawio.py ../vLLM进程与线程.drawio
python render_check.py ../vLLM进程与线程.drawio -o /tmp/r --scale 0.7
python color_audit.py ../vLLM进程与线程.drawio
```

配色遵守同目录 `README.md` 的规范；本套图里：**橙＝前端/引擎侧进程，绿＝worker 侧线程，
黄＝数据面/daemon，红＝死亡与失败路径**。

---

## 7. 图的来历

六份并行只读走读产出，全部要求带 `文件:行号`、不确定写 `(未确认)`：

1. **进程树与进程数**：7 类进程（含 2 类 ray 形态、EPLB planner、启动期短命探测进程）、
   `world_size` 公式、internal/external/hybrid 三种 DP LB、创建/ready/shutdown 时间线；
2. **线程与 ZMQ**：frontend 的 loop/Task/线程、EngineCore 的 4 个线程、三条通道的
   socket 类型与 bind 侧、一次请求的 13 步往返、背压与超时；
3. **本仓部署实况**：4 个部署包的进程清单、参数→进程数推导、端口表、启动顺序、
   sidecar daemon 的 spawn/去重/空闲自杀；
4. **Scheduler**：四个容器、`schedule()` 五段流程、token/input 双预算、抢占（只有重算）、
   前缀缓存命中、PD 挂起/放行、DP 波次节流、`_inflight_prefills` 准入闸门；
5. **dp2×tp4 的 PD 流转**：dense 独立引擎塌缩、Proxy 轮询选 DP、TP=4 的列/行并行与回传 rank、
   P→D 的 TP 一对一 KV 映射、该配置的 8 个坑；
6. **模型加载**：`config.json` → 分片 → loader → NPU 空骨架 → `weight_loader` 切片 →
   `copy_` → 量化/NZ → profile → KV 预算；以及 forward 里权重被 GEMM 读取的调用链。

六份走读一致纠正的两点：`VLLM_ENGINE_ITERATION_TIMEOUT_S` 未被消费；
`ModelRunner` 没有任何线程（worker 主线程里同步调用）。
