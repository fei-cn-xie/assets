# vLLM 的 MTP 与投机解码 · 读图指南

配套文件：`vLLM的MTP与投机解码.drawio`（10 页）。
主题：**draft 是什么、为什么能加速、vLLM/昇腾怎么实现、开了之后什么时候该关**。
锚点按本仓当前检出分支写（`vllm` / `vllm-ascend` 均为 `fix/lwd-multidp-dense-pd`）。

---

## 1. 十页清单

| 页 | 讲什么 | 一句话结论 |
| --- | --- | --- |
| ① | draft（草案）/ MTP 是什么 | 先猜 N 个，再一次前向验证；猜对白拿、猜错截断 |
| ② | 为什么划算 | 一次前向的大头是搬权重（与 query 数无关）→ 1+N 个位置几乎免费 |
| ③ | 草案从哪来 | 四种来源：MTP 头 / EAGLE 小头 / 独立小模型 / n-gram 查表 |
| ④ | 一次 draft step 十步 | 排 1+N 位置 → 一次前向 → 逐位判 + 补位 → 回退 KV → 起草下一步 |
| ⑤ | 验证规则 | 贪心比 argmax、随机比 p/q；拒绝后从残差补位 → 输出分布不变 |
| ⑥ | 被拒的 KV 怎么办 | 不擦数据，只回退 `num_computed_tokens`；下一轮同槽覆盖 |
| ⑦ | 三条系统约束 | 形状变 1+K（图桶要按 K+1 对齐）、显存多留 lookahead、调度要 uniform |
| ⑧ | 昇腾侧落地 | MTP 层在 backbone 之后、共享权重不共享 KV；draft 跑 eager、target 跑图 |
| ⑨ | LWD 下的 MTP | 只在云侧跑；`(p_end, i_end, r_pos)` 划边界；接受数要改写成真实提交行数 |
| ⑩ | 指标与排障 | 先看平均接受长度：>2.5 加 K，1.5~2.5 看逐位，<1.3 关掉 |

三条路径：

- **只想懂概念**：① → ② → ⑤ → ⑩
- **要看实现**：③ → ④ → ⑤ → ⑥ → ⑧
- **要排障/调参**：⑩ → ⑦ → ⑨ → ⑥

---

## 2. 一页纸答案

**是什么**：`draft` = 便宜的候选 token；`MTP` = 模型自带的「多 token 预测」头，推理时正好拿来当草案。
一次 step 里做三件事：**起草 K 个 → 一次前向验证 → 按接受数推进**。

**为什么快**：decode 每步的开销大头是「把全部权重搬一遍」，与 query 数几乎无关；
把 1+K 个位置塞进同一次前向，权重只搬一遍 → 算术强度从 ~2 提到 ~2(1+K)。
期望产出 = `1 + α₁ + α₁α₂ + … `（含 bonus），加速比 ≈ 期望产出 ÷ (1+额外开销)。

**为什么无损**：接受与否由 **target 自己的概率**决定（贪心比 argmax；随机比 `p_target/p_draft`），
被拒位置从残差 `max(p_t − p_d, 0)` 补采 → 最终分布严格等于 target 分布。

**K 为什么不能大**：逐位接受率天然衰减 + 验证成本线性增长；本仓部署用 **K=3**（个别用 1），
昇腾还有硬上限 `K+1 ≤ 16`（TND 布局）。

---

## 3. 本仓部署的实际配置（⑧ 页汇总）

| 项 | 值 |
| --- | --- |
| `method` | 一律 `mtp`（个别写 `qwen3_5_mtp`，会被归一成 `mtp` 并打 deprecation 警告） |
| `num_speculative_tokens` | 3（`fix_1120` 那套是 1） |
| `enforce_eager` | `true` —— **只作用于 draft**，target 仍走图 |
| `model`（draft 路径） | 不给：默认取 target 权重目录，MTP 权重就是同目录里的 `mtp.*` 张量 |
| target 图模式 | P/edge `--enforce-eager`；D `FULL_DECODE_ONLY` |
| 谁跑 MTP | **只有云侧 P/D；edge 不跑**（`edge.sh` 无 `--speculative-config`） |

**实测代价**：TP=1 且开 MTP 时 KV 只剩几十个块（32–63 个 GDN 块），`max_num_seqs` 只能给 32；
关掉 MTP 后可用 KV 明显回升。→ 「开 MTP 是拿并发换单请求速度」。

---

## 4. 五个常见误解

1. **「MTP 是训练技巧，推理无关」**——这个词有两种含义：训练期是多 token 监督的辅助头，
   推理期是同一条权重当 drafter。本仓库的 `method=mtp` 是后者。
2. **「draft 用同一个模型的 KV」**——共享的是 **权重**（embedding / lm_head），
   **KV 是独立的**（MTP 层在 backbone 之后，有自己的层号）；PD 传输时只传一次是因为多层共用同一份。
3. **「被拒的 KV 要清理」**——不清理，只回退计数，下一轮覆盖同一批槽位。
4. **「K 越大越快」**——逐位接受率衰减 + 验证成本上升；K 大反而可能更慢。
5. **「投机解码不占显存」**——draft 权重 + lookahead 槽 + 与 1+K 成正比的缓冲区，
   都会从 KV 里抢走一块。

---

## 5. 六条必背日志

| 日志 | 含义 |
| --- | --- |
| `SpecDecoding metrics: Mean acceptance length: …` | 主指标一行（含逐位接受率） |
| `[spec_decode/base] Initializing spec decode proposer: method=…` | 确认 draft 真的在跑 |
| `… Sharing target model embedding weights …` | MTP 复用了 embedding（省显存） |
| `Enabling num_speculative_tokens > 1 will run multiple times of forward …` | K>1 的官方警告 |
| `decode_threshold exceeded npu_fused_infer_attention_score TND layout's limit of 16` | K 太大被硬上限拦住 |
| `[Lwd][edge-worker] UNEMBED seqno=… accepted=…` | LWD 下每次 UNEMBED 实际吐了几个 token |

---

## 6. 自查与重生成

```bash
cd assets/vllm-ascend/drawio/tools
python mtp/build.py                      # 重建 ../vLLM的MTP与投机解码.drawio
python checkdrawio.py ../vLLM的MTP与投机解码.drawio
python render_check.py ../vLLM的MTP与投机解码.drawio -o /tmp/r --scale 0.7
python color_audit.py ../vLLM的MTP与投机解码.drawio
```

配色遵守同目录 `README.md`；本套图里：**绿＝draft/验证侧，橙＝算力密集或 target 侧动作，
黄＝数据面（KV/DOWN），红＝硬限制与坑**。

---

## 7. 图的来历与边界

三份并行只读走读（全部要求带 `文件:行号`、不确定写 `(未确认)`）：

1. **vLLM 实现**：字段与流转（`num_speculative_tokens` / `uniform_decode_query_len` /
   `scheduled_spec_decode_tokens`）、proposer 家族、一次 step 十步、拒绝采样两套 kernel、KV 回退与 async 校正；
2. **vllm-ascend 落地**：`AscendEagleProposer` 外壳、MTP 层与共享权重、12 处部署脚本实测配置、
   TND ≤16 与 padded×整图互斥、LWD 的三元组与接受数改写；
3. **原理与数学**：memory-bound 与「闲着算力」的仓内原文、MTP 两义、三种 draft 来源、
   接受率/加速比公式（明确标注「通用知识」的部分）。

**三处只能标「通用知识」**：① 「N 个位置不比 1 个位置贵多少」的定量论断（仓内只有 memory-bound 定性
与 1+K 结构证据）；② 加速比与期望接受长度公式（仓内只有各分量的代码与文档线索）；
③ EAGLE/Medusa 的领域背景解释（仓内只有用法与定性表）。

**两处标「未确认」**：① P 侧是否真的执行 draft（配置带了，但 P 被压成 `max_tokens=1`，
代码里没找到按 `stage_role` 跳过 draft 的守卫）；② 是否存在「MTP 只能用 FULL_DECODE_ONLY」
的硬断言（没找到断言，实际是「draft 是否包 ACLGraphWrapper 取决于 target 是否开整图 +
draft 自己的 `enforce_eager`」）。
