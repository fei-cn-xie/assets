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

三条路径：

- **只想搞懂架构**：① → ② → ③ → ⑤
- **要排障/看现场**：⑨ → ⑦ → ⑩ → ⑧
- **要调启动参数**：② → ③ → ⑥

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

## 5. 自查与重生成

```bash
cd assets/vllm-ascend/drawio/tools
python proc/build.py                    # 重建 ../vLLM进程与线程.drawio
python checkdrawio.py ../vLLM进程与线程.drawio
python render_check.py ../vLLM进程与线程.drawio -o /tmp/r --scale 0.7
python color_audit.py ../vLLM进程与线程.drawio
```

配色遵守同目录 `README.md` 的规范；本套图里：**橙＝前端/引擎侧进程，绿＝worker 侧线程，
黄＝数据面/daemon，红＝死亡与失败路径**。

---

## 6. 图的来历

三份并行只读走读产出，全部要求带 `文件:行号`、不确定写 `(未确认)`：

1. **进程树与进程数**：7 类进程（含 2 类 ray 形态、EPLB planner、启动期短命探测进程）、
   `world_size` 公式、internal/external/hybrid 三种 DP LB、创建/ready/shutdown 时间线；
2. **线程与 ZMQ**：frontend 的 loop/Task/线程、EngineCore 的 4 个线程、三条通道的
   socket 类型与 bind 侧、一次请求的 13 步往返、背压与超时；
3. **本仓部署实况**：4 个部署包的进程清单、参数→进程数推导、端口表、启动顺序、
   sidecar daemon 的 spawn/去重/空闲自杀。

三份走读一致纠正的两点：`VLLM_ENGINE_ITERATION_TIMEOUT_S` 未被消费；
`ModelRunner` 没有任何线程（worker 主线程里同步调用）。
