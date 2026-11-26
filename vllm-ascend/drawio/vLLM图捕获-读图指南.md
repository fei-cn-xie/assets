# vLLM 图捕获（CUDA / ACL Graph）· 读图指南

配套文件：`vLLM图捕获.drawio`（10 页）。
主题：**什么是图捕获、它为什么有用、vLLM 怎么实现、昇腾上有什么不同、怎么排障**。
锚点按本仓当前检出分支写（`vllm` / `vllm-ascend` 均为 `fix/lwd-multidp-dense-pd`）。

---

## 1. 页面清单与阅读顺序

| 页 | 讲什么 | 一句话结论 |
| --- | --- | --- |
| ① | 为什么需要图捕获 | 推理每步要逐个提交上千个算子；decode 每步只算 1 个 token，于是「提交」成了瓶颈 |
| ② | 捕获与回放的本质 | 捕获＝真跑一遍顺便录下来；回放＝一条命令重放，**Python 逻辑不再执行** |
| ③ | 一次捕获的 10 步 | 开闸门 → 侧流 → 逐桶（warmup → 同步 → 在 graph 上下文里跑一遍）→ 锁 workspace |
| ④ | 三个硬约束 | 固定形状（按桶+padding）、固定地址（持久 buffer+copy_）、不能问 CPU |
| ⑤ | 四种图模式 | PIECEWISE 把 attention 挖出去；FULL 整张图只吃「整齐的 decode」 |
| ⑥ | 形状桶与 dispatch | 向上取整到桶 → 先查 FULL 精确键 → 再查 PIECEWISE 放宽键 → 都不中静默回 eager |
| ⑦ | 启动期捕获流程 | 排在「权重 → profile → KV」之后；图池进程级共用、必须从大到小抓 |
| ⑧ | 回放期一次 step | 图只包住 forward；进图前备静态输入，出图后算 logits / 采样 / 回 CPU |
| ⑨ | 昇腾 ACL Graph | 用法同构，但**回放前要先在 host 侧刷 attention 参数**；本仓只有 D 侧开图 |
| ⑩ | 排障与坑 | 六种「以为开了图、其实没有」；四步排查法 + 六条必背日志 |

三条路径：

- **只想懂概念**：① → ② → ④（跳过所有代码锚点也能读）
- **要看实现**：③ → ⑤ → ⑥ → ⑦ → ⑧
- **要排障/调参**：⑩ → ⑥ → ⑨

---

## 2. 一页纸答案

**是什么**：把「固定形状的一段前向」在设备上真跑一遍，把算子序列与地址关系录成图；
以后每步只 `replay()` 一次，CPU 的提交次数从「上千次」变成「一次」。

**为什么有用**：只省「提交」，不省计算。所以对 decode（每请求 1 token、
但要走完整模型）收益最大；prefill 大矩阵本来就能摊薄提交开销，收益小。

**三个硬约束**（任何一条破了就出问题）：

| 约束 | 本仓怎么满足 | 破了的后果 |
| --- | --- | --- |
| 形状固定 | 有限个桶（默认 1,2,4,8…512）+ padding | 超过最大桶 → 静默回落 eager（无日志） |
| 地址固定 | 持久 buffer + 每步 `copy_` | 报 `Input addresses ... different during replay` |
| 图内不能问 CPU | 一切动态准备放图外；捕获闸门 | `.item()` 被烤进图 → 永远返回旧结果，且不报错 |

**四种模式**：`NONE`（=`--enforce-eager`）/ `PIECEWISE`（attention 在图外）/ `FULL`（整张图）/
`FULL_DECODE_ONLY`（只有 decode 用 FULL）；vLLM 默认 `FULL_AND_PIECEWISE`（`-O2`）。

**启动顺序（容易记错）**：加载权重 → profile → 切 KV 块池 → 估算图显存 → **逐桶捕获** → 锁 workspace。

---

## 3. 本仓部署的真实配置（⑨ 页）

| 角色 | 图配置 | 锚点 |
| --- | --- | --- |
| **D（decode）** | `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` | `sidecar/cloud_d.sh:31`、`mooncake_pd_195/decode.sh:34` |
| **P（prefill）** | `--enforce-eager`（不捕获） | `sidecar/cloud_p.sh:26`、`mooncake_pd_195/prefill.sh:33` |
| **edge** | `--enforce-eager`，且代码里**直接跳过** warmup/捕获 | `edge.sh:30`；`lwd_edge_worker.py:122-131`（日志 `[lwd-edge] skip model warmup/capture`） |

两个澄清：
1. **「LWD 下必须 enforce-eager」在代码里不存在**——只是部署脚本的约定。
2. **eager 与 `--compilation-config` 同时给不会报错**：eager 静默胜出，你配的 `cudagraph_mode` 被丢弃。

---

## 4. 六个坑（⑩ 页的浓缩）

1. **超桶静默掉出**：`num_tokens > max_cudagraph_capture_size` → dispatcher 直接返回 NONE，**一行日志都没有**。
   「低并发快、并发一高就慢」十有八九是这个。
2. **捕获 OOM 不会回退**：捕获段没有 try/except，直接启动失败；要自己缩桶或降 `gpu-memory-utilization`。
3. **`.item()` / `print` / 条件分支被烤进图**：不报错，只是永远用捕获那一刻的值。
4. **必须从大到小捕获**：图池是**进程级**的（≈各桶峰值取 max），先抓小桶会把池撑大。
5. **昇腾多一步**：回放前要 `graph_task_update_begin/end` 刷 attention 的
   `actual_seq_lengths*` / `block_table`，否则用的是上一轮的值。
6. **`--enforce-eager` 是最好的对照实验**：怀疑图有问题时先关掉它二分。

---

## 5. 六条必背日志

| 日志 | 含义 |
| --- | --- |
| `Enforce eager set, disabling torch.compile and CUDAGraphs...` | 被 eager 一票否决 |
| `Capturing ACL graph (FULL, BatchDescriptor(...))` | 昇腾每抓一张图（需开 debug 日志） |
| `Graph capturing finished in N secs, took X GiB` | 捕获完成：耗时 + 图显存 |
| `CUDA graph pool memory: X GiB (actual), Y GiB (estimated)` | 估算 vs 实际（决定要不要缩桶） |
| `ACL graph replay is active (logged once).` | 确认图真的在用 |
| `cudagraph_mode X is unsupported on NPU; falling back to NONE.` | 被平台降级了 |

---

## 6. 自查与重生成

```bash
cd assets/vllm-ascend/drawio/tools
python graph/build.py                    # 重建 ../vLLM图捕获.drawio（10 页）
python checkdrawio.py ../vLLM图捕获.drawio
python render_check.py ../vLLM图捕获.drawio -o /tmp/r --scale 0.7
python color_audit.py ../vLLM图捕获.drawio
```

配色遵守同目录 `README.md`；本套图里：**沙色＝eager/中立，绿＝分段图（PIECEWISE），
橙＝整张图（FULL），黄＝数据面（KV 写入），红＝失效/失败路径**。

---

## 7. 图的来历

三份并行只读走读（全部要求带 `文件:行号`、不确定写 `(未确认)`）：

1. **vLLM 实现**：配置模式与桶推导、捕获入口与启动位置、捕获十步、桶与 dispatch、回放路径、
   piecewise vs full、代价与开关、日志原文；
2. **昇腾侧**：ACL Graph 的实体与 API、`UpdatableGraph`/参数刷新、`--enforce-eager` 在本仓脚本里的分布、
   attention 的动态参数与 graph 友好路径、已知限制与错误码；
3. **原理与性能账**：仓内原文依据（「reduce host launch overhead」「at the cost of steady-state decode
   performance」）、decode 形状判据、三个硬约束的代码体现、与 `torch.compile` 的正交关系。

**注意**：本仓内没有 cudagraph vs eager 的量化对比数据（没有 TPOT/吞吐表格），
所以图里凡是「几微秒」「同量级」这类量级说法都只作定性理解，不做数据引用。

三份走读一致纠正的两点：① 捕获在 **profile 与 KV 分配之后**（不是之前）；
② 图池是 **进程级单例**（不是每桶一份）。
