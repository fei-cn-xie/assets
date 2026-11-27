# vLLM 组合配置与参数联动 · 读图指南

配套文件：`vLLM组合配置与参数联动.drawio`（10 页）。
定位：**前七套图各讲一个机制，这一套讲「同时打开时它们怎么互相影响」**——
它是整套材料的收口，也是调参/排障时最常翻的一套。

---

## 1. 十页清单

| 页 | 讲什么 | 一句话结论 |
| --- | --- | --- |
| ① | 四机制抢什么 | PD / Mooncake / 图捕获 / MTP 花的是同一份显存、算力、带宽、形状自由度 |
| ② | 参数地图 | 本仓 13 个 launcher 只动了十来个参数；六个参数零命中（含 chunked prefill） |
| ③ | 联动一：K 改写桶 | K=3 → query 长度 4 → 图桶必须是 4 的倍数（自动取整，或直接报错） |
| ④ | 联动二：显存三条链 | 权重↑ / 图池↑ / MTP↑ 都会让 KV 变少，路径不同 |
| ⑤ | 联动三：调度推导链 | `max_num_batched_tokens` 是引擎算出来的，还会反推 seqs、截断桶 |
| ⑥ | 联动四：PD 硬约束 | 块大小、并行度、拓扑必须对齐；P 侧请求被强制改写 |
| ⑦ | 联动五：两个 enforce-eager | 命令行那个管 target，JSON 里那个只管 draft |
| ⑧ | 降级链与决策树 | FULL→PIECEWISE→NONE 逐级降（会打日志）；三条硬拒绝 |
| ⑨ | 五套真实配方 | 参数差异归根到底是「一卡放不放得下整模型」 |
| ⑩ | 速查与自测 | 三张表：静默改写 / 强制覆盖 / 拒绝启动 + 五道自测 |

三条路径：

- **要调参**：② → ③④⑤ → ⑩
- **要排障**：⑧ → ⑩ → ⑦
- **要理解取舍**：① → ④ → ⑨

---

## 2. 一页纸答案

**资源只有四格**：显存 / 算力 / 带宽 / 形状自由度。四个机制都在抢这四格：
PD 分算力、Mooncake 花带宽、图捕获吃显存换形状固定、MTP 吃显存与算力换 token 数。

**三条最重要的联动**
1. **K → 桶**：`uniform_decode_query_len = 1+K` → capture size 必须按 K+1 对齐
   （自动取整；取整后超上限则报错；运行时对不上则**静默**回 eager）。
2. **显存三条链**：权重变大、开图池、开 MTP，都会减少 KV 块数与并发上限。
3. **调度推导链**：不设 `max_num_batched_tokens` 时，引擎按「显存→用途→性能模式→是否分块→多模态→截断」六步推导，
   再反推 `max_num_seqs`、截断图桶、扣掉投机占位。

**两个作用域**：`--enforce-eager`（target，全局 NONE） vs `speculative-config.enforce_eager`（只 draft）。
本仓 D 侧用「target 跑图 + draft 跑 eager」；P 与 edge 用全局 eager。

**三种结局**：静默改写（多数）、强制覆盖（你的值作废）、拒绝启动（会点名原因）。

---

## 3. 本仓五套配方对照（⑨ 页）

| 包 | TP/DP | max-model-len | max-num-seqs | gpu-mem | K | 图（D 侧） |
| --- | --- | --- | --- | --- | --- | --- |
| sidecar / in_process | 2 / 1 | 65536 | 128 | 0.90 | 3 | FULL_DECODE_ONLY |
| dp2_tp1_194_195 | 1 / 2 | 32768 | 32 | 0.95 | 3 | FULL_DECODE_ONLY |
| mooncake_pd2_tp1_195 | 1 / 2 | **512** | 32 | **0.98** | 3 | FULL_DECODE_ONLY |
| mooncake_pd_195 | 2 / 1 | 65536 | 128 | 0.95 | 3 | FULL_DECODE_ONLY + 手写 25 档桶 |
| fix_1120 | **8** / 1 | 65536 | 128 | 0.90 | **1** | FULL_DECODE_ONLY（+EP+量化+block128） |

四条「为什么」：TP=2 切一半权重才敢报长上下文；TP=1+MTP 只能给 32 并发；
`gpu-memory-utilization` 三档对应三种余量；只有 DeepSeek 那套要显式量化 + 专家并行 + 块对齐。

---

## 4. 五个必须记住的坑

1. **只抄 K 不抄桶**：K 变了，桶必须是 K+1 的倍数；抄错 → 运行时静默回 eager。
2. **只抄 max-model-len 不抄 seqs/gpu-mem**：三者是一条链，单独抄大 mml 会启动失败。
3. **以为「draft eager」不影响 target**：反方向才会——target eager ⇒ 全局 NONE ⇒ draft 也别想用图。
4. **以为调大 gpu-memory-utilization 就等于提高并发**：它只是抬高水位线，KV 仍被权重/图池/MTP 分走。
5. **以为降级会报错**：降级链只打 warning；而「形状不命中桶」连 warning 都没有。

---

## 5. 六个「零命中」的参数（别以为调过）

`--kv-cache-memory`、`--swap-space`、`--enable-chunked-prefill`、
`--long-prefill-token-threshold`、`--scheduling-policy` 在本仓 13 个 launcher 里**全部走默认**；
`--enable-expert-parallel` 只有 `fix_1120`（TP=8）显式开启。
→ 本仓的 chunked prefill 是**默认开启**且未调阈值。

---

## 6. 三处需要留意的核对结果

| 项 | 结论 |
| --- | --- |
| `sidecar` 与 `in_process` | 四个脚本**逐字节相同**（同一配置的两种目录布局），对比时应合并 |
| `rdma/`、`tcp/` 下的 `dsv4_*` 三件套 | **6 个 0 字节空壳**；目录名里的规模参数不代表任何生效配置 |
| `base/centralized_195` | 头注与 env 注释写 TP=4，实际生效是 `tp=2` + `ASCEND_RT_VISIBLE_DEVICES=0,1`；本套按**代码取值** |

---

## 7. 自查与重生成

```bash
cd assets/vllm-ascend/drawio/tools
python cfg/build.py                      # 重建 ../vLLM组合配置与参数联动.drawio
python checkdrawio.py ../vLLM组合配置与参数联动.drawio
python render_check.py ../vLLM组合配置与参数联动.drawio -o /tmp/r --scale 0.7
python color_audit.py ../vLLM组合配置与参数联动.drawio
```

配色遵守同目录 `README.md`；本套图里：**橙＝算力/图侧，绿＝draft/投机侧，
黄＝权重/数据面压力，红＝硬限制、静默失效与坑**。

---

## 8. 图的来历

两份并行只读走读（全部要求带 `文件:行号`）：

1. **部署侧**：13 个 `vllm serve` 入口脚本的完整参数抽取、`env.sh` 的变量展开关系、
   空文件与被注释项、十个「同参数不同值」的例子；
2. **代码侧**：「改 A → 影响 B」的 15 条因果（含 6 条静默改写、2 条强制覆盖、4 条硬拒绝）、
   调度参数的七步推导、K 的连锁影响、图捕获的自动调整与五级降级、PD/LWD 的必填校验。

**明确标注为「未确认」的一处**：LWD 侧是否存在对 `max_num_seqs` 的额外强制改写（本轮未找到写回点）。
