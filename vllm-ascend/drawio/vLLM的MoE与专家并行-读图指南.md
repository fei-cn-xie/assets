# vLLM 的 MoE 与专家并行 · 读图指南

配套文件：`vLLM的MoE与专家并行.drawio`（10 页）。
定位：最后一块没成套的机制——**混合专家（MoE）**。它解释了本仓里那几个「只在 MoE 时生效」的
开关（`--enable-expert-parallel`、EPLB、wave 协调）。

---

## 1. 十页清单

| 页 | 讲什么 | 一句话结论 |
| --- | --- | --- |
| ① | 什么是 MoE | 很多专家 + 一个门卫 + 每 token 只走 top-k：总参数按专家数涨、计算按 top-k 涨、显存按专家数涨 |
| ② | 一个 token 的六步 | 打分 → top-k → 重排 → 分组矩阵乘 → 散射 → 加权求和 |
| ③ | 三本账 | 显存、通信、负载；且 prefill 划算、decode 要还回去 |
| ④ | 共享专家 | 所有 token 必走的 dense 旁路：兜通用能力，代价是不受稀疏保护 |
| ⑤ | 专家并行（EP） | 把专家切到多卡：省「单卡放得下」，**不省总显存** |
| ⑥ | 三种通信实现 | all-gather（默认）/ MC2（融合）/ all-to-all（按需）——**EP 不必然走 all-to-all** |
| ⑦ | 负载均衡 | 训练期改源头（辅助损失 / 路由偏置），推理期改放置（EPLB + 冗余副本） |
| ⑧ | 昇腾主链 | 融合算子串：重排顺带量化、GEMM 与激活融合、共享专家多流并行 |
| ⑨ | 本仓实测 | 十套包只有 DeepSeek 那套是 MoE 且开 EP；EP 与 TP 在 MoE 层里互斥 |
| ⑩ | wave + 术语 + 排障 | 为什么只有 MoE 需要「一起走一起停」；12 条术语；8 条排障 |

三条路径：

- **只想懂原理**：① → ② → ③ → ④
- **要看实现/配置**：⑤ → ⑥ → ⑧ → ⑨
- **要调优/排障**：③ → ⑦ → ⑩

---

## 2. 一页纸答案

**是什么**：把 Transformer 里那层 FFN 换成「N 个 FFN（专家）+ 一个路由器（门卫）」，
每个 token 只被送进 top-k 个专家，结果按门卫给的权重加权相加。

**三个量必须分开**：总参数（按 N 涨）、每 token 计算量（按 k 涨）、显存（按 N 涨）。
所以 MoE 是「用显存换参数容量」的结构，它的「参数量」与「跑多快」几乎无关。

**三笔账**（MoE 推理的全部复杂度来源）：

| 账 | 内容 | 解法 |
| --- | --- | --- |
| 显存 | 全部专家都要驻留（占权重 **85~90%**） | EP 切分（只摊开不减少）、量化 |
| 通信 | token 要跨卡（前后各一次） | 选通信实现：默认 all-gather+reduce-scatter，可换 MC2 / 真 all-to-all |
| 负载 | 热门专家把某张卡拖成瓶颈 | 训练期：辅助损失 / 路由偏置；推理期：EPLB 换位 + 冗余副本 |

**为什么 decode 更吃亏**：小 batch 下要读的专家权重零碎、两次集合通信的固定开销盖过计算、
共享专家不受稀疏保护。所以 MoE 的收益主要在 prefill（大批），decode 要还回去。

**本仓实测**：十套部署包里只有 `fix_1120`（DeepSeek-V4-Flash，MoE + w8a8 + MTP）开了
`--enable-expert-parallel`，其余九套是 dense 的 Qwen3.6-27B。所有包都**没开 EPLB**
（属「能开但权衡后没开」）。

---

## 3. 四个最容易搞错的点

1. **「开了 EP 就走 all-to-all」**——错。本仓两侧的默认都是 **all-gather（发）+ reduce-scatter（收）**；
   真 all-to-all 要显式选 DeepEP / MoRI / NIXL / FlashInfer 那一类后端。
   昇腾侧还有更多回落条件（每卡专家数、EP 规模、是否 LoRA）。
2. **「EP 能省显存」**——只省「单卡必须放下全部专家」，**总量一点没少**；
   冗余专家（EPLB）还会再加显存（本仓文档给的数量级：DeepSeek-V3 每卡多一个冗余专家约 2.4 GB）。
3. **「EP 和 TP 可以叠加」**——在 MoE 层里**互斥**：开 EP 就把这一层的 TP 规模改成 1
   （每卡拥有完整的一份专家），EP 规模 = DP × PCP × TP 的乘积。
4. **「token drop 在推理时也会发生」**——本仓推理路径**不做容量截断**：重排行数就是 token×k。
   容量因子与丢 token 是训练期概念。

---

## 4. 为什么「wave 协调」只有 MoE 需要（⑩ 页）

- **dense 的 DP**：各卡跑各自请求，互不通信 → 一卡忙一卡闲完全没问题。
- **MoE 的 DP**：每层都要跨 rank 搬 token / 归约，而集合通信**要求所有 rank 同时参与** →
  必须「同一时刻要么都在跑、要么都在停」。
- 代码里三个硬证据：一行 `enable_wave_coordination = is_moe`；
  一条断言「MoE 专用的引擎进程只应用于 MoE 模型」；
  一条纪律「同步状态里的每个字段在所有 rank 上必须相同」。
- 还有一个容易忽略的推论：**空微批也要跑成 dummy 批**，因为大家必须切得一样多——
  否则有的 rank 没有通信要发，就会卡死。

---

## 5. 六条 MoE 专属日志（照这个 grep）

| 日志关键字 | 含义 |
| --- | --- |
| `EPLB is enabled for MoE part of model …` | EPLB 已启用（模型/草稿模型各一条） |
| `EPLB step: … avg_tokens=… max_tokens=… balancedness=…` | 负载窗口统计（看 balancedness 是否接近 1） |
| `Rearranged experts … in … s.` | 真的搬过专家权重 |
| `[EPLB] Skip rearrange: imbalance … (no material gain)` | 判定「搬了也没用」而跳过 |
| `Using allgather_reducescatter all2all manager.` | 当前用的是默认通信实现（不是真 all-to-all） |
| `[Elastic EP] Starting expert resharding…` | 弹性 EP 在重排（会改并行结构，与普通 EPLB 不同） |

> 注意：本仓**没有**「专家总数 / EP 大小 / 路由类型」的启动横幅日志——
> 要看这些得开 debug，或用 `--enable-return-routed-experts` 把路由结果返回出来。

---

## 6. 与其它十套图的接口

- ①③ ← 站 2 ⑤（算术强度）+ 站 2 ⑦（显存账）：MoE 的三本账是同一套账的延伸。
- ② ← 站 1 ⑫（FFN / SwiGLU）：MoE 的每个专家就是一个 FFN。
- ③④ ← 站 0 ④（训练显存四件套）：专家占权重 85~90%，是「权重吃显存」的极端版本。
- ⑤⑥ ← 站 4 ②③（进程数 / executor）：EP 会把进程内的 MoE 通信组重排；
  站 7（MTP）：**MTP 层本身就是 MoE 层**，开 MTP 等于增加 MoE 层数。
- ⑦ ← 站 9（故障案例）：负载倾斜是「某些卡慢」的常见原因之一。
- ⑧ ← 站 0 ⑦（量化）：MoE 的 w13/w2 + per-channel scale 是量化在专家上的形态。
- ⑩ ← 站 3 ⑨（Step 循环）：wave 就是「一步」在 MoE DP 场景下的边界。

---

## 7. 自查与重生成

```bash
cd assets/vllm-ascend/drawio/tools
python moe/build.py                      # 重建 ../vLLM的MoE与专家并行.drawio
python checkdrawio.py ../vLLM的MoE与专家并行.drawio
python render_check.py ../vLLM的MoE与专家并行.drawio -o /tmp/r --scale 0.7
python color_audit.py ../vLLM的MoE与专家并行.drawio
python verify_all.py   # 一次跑完 12 套的结构 / 渲染 / 配色校验
```

配色遵守同目录 `README.md`；本套图里：**橙＝路由/决策侧，绿＝专家计算侧，
黄＝数据搬运（重排/通信/换位），红＝限制与坑**。

---

## 8. 图的来历与边界

三份并行只读走读（全部要求带 `文件:行号`、区分「本仓事实」与「通用知识」）：

1. **vLLM 侧 MoE**：本分支已把单体 `FusedMoE` 拆成「工厂 + MoERunner + RoutedExperts」；
   四段流水线原文（Router → Quantize-Dispatch → Permute-Experts-Unpermute → Combine）、
   路由策略的优先级链、EP/TP 的互斥关系、EPLB 的异步搬层、wave 协调的代码依据。
2. **昇腾侧 MoE**：`AscendMoERunner` 主链、`npu_moe_*` 算子族、重排时顺带量化、
   共享专家多流与拆算一致性校验、EP 布局坑、20+ 条 `NotImplementedError` 原文。
3. **原理与数学**：配置字段（192 专家 / top-8 / shared=1 / 每专家 1536）、参数量 vs 激活量、
   decode 为什么更吃亏（性能模型公式）、辅助损失与路由偏置的原文、all-to-all 量级公式。

**明确标为「通用知识」的三处**：① k/N 激活比这条结论（公式在本仓、结论属推导）；
② all-to-all 的字节数公式；③「MoE decode 读权重总量通常仍高于 dense」的定性结论。

**明确标为「未确认」的两处**：① DeepSeek-V4 那套包的具体专家数（脚本无硬编码，
模型目录里没有 config.json）；② 是否存在更上层的「专家数/EP 大小」启动日志（grep 未见）。
