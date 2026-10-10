# 边云 LWD(prefill_only)按角色加载模型参数 —— 现状分析与落地方案

> 分析对象：`vllm`(`vllm/vllm/...`) 与 `vllm-ascend`(`vllm-ascend/vllm_ascend/...`) 两侧的 LWD(layerwise disaggregated)prefill_only 实现。
>
> 触发问题："边侧只做 embedding 与 unembedding、云侧做前向计算，那么模型加载阶段是不是应该边侧只加载 embedding/unembedding 相关参数、云侧只加载 transformer 层参数？现在代码里似乎两边都把模型的所有参数都加载了。"
>
> 目标部署（按 `private_home/lauch.sh` 的真实配置）：`Qwen3.6-27B`，`--edge-npu-count 1 --cloud-npu-count 4`，`lwd_config = {enabled, role, mode: "prefill_only", edge_head_tail_layers: [0,0]}`，`--max-num-batched-tokens 128`，`enable_weight_nz_layout: true`，`cudagraph_mode: NONE`。
>
> 关注资源（按用户选择）：**HBM 常驻权重显存**。
>
> 本文所有结论均给出 `文件:行` 证据；凡属推断或需现场确认的，均显式标注「待确认」。

---

## 0. 结论速览

| 问题 | 结论 |
|---|---|
| **边侧现在是否加载了 transformer 层？** | **没有。** prefill_only + `[0,0]` 下 `LwdConfig.local_layer_indices()` 对边侧返回空集（`vllm/config/lwd.py:88-96`），`make_layers()` 走 LWD 分支（`vllm/model_executor/models/utils.py:695-698`）→ 边侧每个 layer 位置都是 `PPMissingLayer`（`utils.py:632-671`），**既不建模块也不装权重**（`utils.py:260-267`）。 |
| **云侧是否加载了 transformer 层？** | 是，而且**应该**如此：`local_layer_indices()` 对云侧返回 `range(0, N)`。 |
| **"两边都加载了所有参数"这个观察从哪来？** | 两个真实来源：① head/tail（`embed_tokens`/`norm`/`lm_head`）被 `_LWD_FULL_HEAD_TAIL` **刻意在两侧都全量建**（`vllm/distributed/parallel_state.py:354-368,559-581,1806-1809`）；② **加载器仍会把整份 checkpoint 读一遍** —— `AutoWeightsLoader` 跳过的是"写入"，不是"读盘"（`utils.py:266-267` 提前 return，但 `_groupby_prefix` 的 `itertools.groupby` 在推进外层迭代时会把整组消费掉，tensor 照样被 `f.get_tensor()` 物化）。所以从"主机内存/日志/加载耗时"看确实像"全部加载了"；只有 EP 过滤那类**放在 `f.get_tensor()` 之前**的判断才真正省 I/O（`weight_utils.py:1022-1026`）。 |
| **边侧/云侧"真正需要"的参数集是否等于设想的"边=embedding/unembedding、云=transformer"？** | **不等于，而且这个设想在现有协议下不可能实现。** 云侧**必须**持有 `embed_tokens`（decode 行的 token id 由云侧本地嵌入，`vllm/v1/worker/gpu_model_runner.py:3632-3662`、`lwd_cloud_model_runner.py:255-296`）与 `lm_head`（采样出 rank，`lwd_cloud_model_runner.py:733-746`）；边侧**必须**持有 `embed_tokens`（prefill/多模态嵌入）与 `lm_head`（rank-replay 还原 token，`lwd_edge_worker.py:474-563`）。 |
| **按"只加载执行所需"能真正省下什么（保持现协议）？** | 只有三处：① **云侧视觉塔**（多模态模型下云侧永不调用 `embed_multimodal`，可省，量级 ~1.4 GB/整份）；② 边侧 `norm`/`hc_head` 之类 tail-only 残渣（可忽略的量级，但语义上确实不该装）；③ 边侧误配 `--speculative-config` 时会把整份 MTP drafter 装进 HBM（当前 launch 没加，属防呆项）。 |
| **想省"大钱"（如边侧 `lm_head` 约 2 GB）怎么办？** | 必须动**协议**：让云侧采样后只回 token id、取消 rank-replay（本文附录 A）。代码里已有两条诊断旁路证明技术上可行（`vllm_ascend/envs.py:35-52`），缺的是正确性论证。 |
| **能不能"各机只放一半权重文件、分别启动"？** | 磁盘上可以，但**不能只做这一件事**：vLLM 默认会拒绝启动"参数存在但权重缺失"的模型（`default_loader.py:414-437`，非量化默认开启），量化模型则会静默留下未初始化权重。必须同时做构造期删模块；且"云侧纯 transformer"仍要付出每步跨机往返（附录 C）。 |
| **怎么做（推荐路径）** | 三闸门 + 复用既有原语：**构造期**决定 HBM（`no_init_weights` + `StageMissingLayer`，已有 `mm_encoder_only`/`limit_mm_per_prompt=0` 两个成熟先例），**加载期**决定磁盘 I/O 与主机内存（`DefaultModelLoader` 前缀过滤，`ep_weight_filter.py` 是现成先例），**后处理期**只处理存在的参数（NZ/量化）。分 P0~P4 落地，见 §5/§6。 |

---

## 1. 事实基线

### 1.1 部署拓扑（来自 `private_home/lauch.sh`）

| 项 | 边（edge） | 云（cloud） |
|---|---|---|
| NPU 数 | 1（`--edge-npu-count 1`） | 4（`--cloud-npu-count 4`） |
| TP 组 | `[[0]]`（单例） | `[[1,2,3,4]]` |
| PP 组 | `[0,1]` 两段 → 运行期**中性化**（两侧都算 first+last rank） | 同左 |
| EP / DCP / PCP / DP | 单例 | 单例 |
| `lwd_config.mode` | `prefill_only`，`edge_head_tail_layers=[0,0]` | 同（`role=cloud`） |
| KV cache | **0**（`LwdEdgeWorker.get_kv_cache_spec()` 返回 `{}`，`lwd_edge_worker.py:165-167`） | 正常 |
| 运行期前向 | 被劫持：只走 embed / unembed（`lwd_edge_worker.py:169-209`） | 真前向（`Qwen3_5Model.forward`）+ 采样 |

拓扑构建见 `vllm/distributed/parallel_state.py:1758-1809`（TP 按边/云分组、PP 两段、`_LWD_FULL_HEAD_TAIL = True`）。

### 1.2 关键代码地图

| 主题 | 位置 |
|---|---|
| LWD 配置与层分配 | `vllm/vllm/config/lwd.py:43-101`（`LwdConfig`、`local_layer_indices`、`validate`） |
| 层裁剪实现 | `vllm/vllm/model_executor/models/utils.py:632-671`（`make_lwd_layers`）、`674-712`（`make_layers` 的 LWD 分支） |
| 缺失层占位与判据 | `utils.py:619-629`（`PPMissingLayer`）、`521-539`（`StageMissingLayer`）、`719-745`（`is_pp_missing_parameter`） |
| "不落显存"的既有原语（构造期占位） | `utils.py:541-570`（`collect_children`）、`573-612`（`no_init_weights`） |
| 加载期跳过权重 | `utils.py:116-160,188-191,260-267`（`AutoWeightsLoader`） |
| 加载主流程 | `vllm/vllm/model_executor/model_loader/base_loader.py:42-82`（`initialize_model` → `load_weights` → `process_weights_after_loading`） |
| 权重迭代器与 EP 过滤先例 | `model_loader/default_loader.py:211-307,318-380`、`model_loader/ep_weight_filter.py:64-81`、`model_loader/weight_utils.py:893-1029` |
| head/tail 中性化 | `parallel_state.py:354-368,559-581,1806-1809` |
| 云侧 decode 行本地嵌入（证据） | `vllm/v1/worker/gpu_model_runner.py:3632-3662`；ascend 侧 `worker/lwd_cloud/lwd_cloud_model_runner.py:255-296` |
| UP 只在 prefill 下发（decode 云内闭环的证据） | `v1/lwd_control/control_cloud_scheduler/lwd_cloud_phase_scheduler.py:140-158,185`、`lwd_cloud_mixed_scheduler.py:444-455,471` |
| 嵌入自带 TP all_reduce | `vllm/model_executor/layers/vocab_parallel_embedding.py:477-496` |
| "缺权重"会启动失败 | `model_loader/default_loader.py:403-412,414-437`（开关 `enable_weights_track`，`:93-94`） |
| 云侧采样与 rank 计算 | `lwd_cloud_model_runner.py:692-746`、`lwd_cloud_worker.py:565-607` |
| 边侧 embed / unembed | `worker/lwd_edge_worker.py:240-315`、`474-563` |
| 边侧加载入口 | `lwd_edge_worker.py:69-97`（`load_model`）、`156-167`（跳 warmup、KV=0） |
| 云侧加载入口 | `worker/lwd_cloud/lwd_cloud_worker.py:128-149`（`load_model` + 层数日志） |
| 通用加载/画像 | `worker/model_runner_v1.py:3719-3832`（`load_model`）、`3685-3698`（`profile_run`）、`worker/worker.py:531-600`（`determine_available_memory`）、`677-700`（`load_model`） |
| 模型构造点（Qwen3.5 系列） | `vllm/vllm/model_executor/models/qwen3_5.py:208-248,433-489,554-618,670-675` |
| 配置默认值 | `vllm/vllm/transformers_utils/configs/qwen3_5.py:44-55`（文本）、`130-159`（视觉塔） |
| 两条协议级诊断旁路 | `vllm-ascend/vllm_ascend/envs.py:35-52` |

---

## 2. 现状逐段走查：参数是怎么被"建"和"装"的

### 2.1 三个闸门，各自决定哪份资源

```
闸门①  构造期  initialize_model() 内部 —— 模块是否被 new 出来、参数是否落在 device 上
       ↓ 决定 **HBM**（torch.empty/Parameter 在 device 上创建即占显存）
闸门②  加载期  DefaultModelLoader 迭代权重 → model.load_weights()
       ↓ 决定 **磁盘 I/O + 主机内存峰值**（safetensors 每个 tensor 都被 f.get_tensor 物化，
         即使下游 load_weights 判断"不要"）
闸门③  后处理  process_weights_after_loading()（NZ layout、量化、格式转换）
       ↓ 决定启动耗时与显存峰值
```

**关键认识**：`is_pp_missing_parameter()` / `AutoWeightsLoader._can_skip()` 只作用于闸门②的"写入"环节。要省 HBM，必须在**闸门①**让模块根本不存在（`PPMissingLayer`/`StageMissingLayer` 就是这样做的）。

### 2.2 层粒度：**已经做到了**（这是当前实现的优点，不是缺口）

边侧（`role=edge`，`mode=prefill_only`，`[0,0]`）：

1. `LwdConfig.local_layer_indices(N)`：`is_edge` 分支且 `(head_k, tail_k)==(0,0)` → 返回 `set()`（`config/lwd.py:92-95`）。
2. `make_layers()` 检测到 `lwd_config.enabled` → 调 `make_lwd_layers()`（`models/utils.py:695-698`）。
3. `make_lwd_layers()` 对 `range(N)` 逐位置：命中本地索引则建真层，否则 `PPMissingLayer()`（`utils.py:658-668`）。边侧无本地索引 → **全是 `PPMissingLayer`，零参数、零显存**。
4. 加载期双保险：`AutoWeightsLoader._load_module()` 对 `PPMissingLayer` 直接 `return`（`utils.py:266-267`）；模型自带 `load_weights`（如 `models/deepseek_v4.py:1433-1434,1496-1497,1534-1535`）还会用 `is_pp_missing_parameter()` 二次跳过。

云侧：`local_layer_indices()` 返回 `range(0, N)`（`config/lwd.py:96`）→ 全部真层、全部真权重。✅ 符合预期。

> ⚠️ **P0 必查**：以上成立的前提是"该模型的 decoder 层通过 `make_layers()` 创建"。`lauch.sh` 带了 `--trust-remote-code`，若 `/home/extra/Qwen3.6-27B/` 自带的 modeling 文件用自己的 `nn.ModuleList` 建层，**`make_lwd_layers` 就不会生效**，边侧会退化成"建全部层 + 装全部层权重"——这正好能解释你观察到的现象。判定方法与防呆见 §5.1。

### 2.3 head / tail：被**刻意**在两侧全量建

`parallel_state.py` 在 LWD 分支里把 `_LWD_FULL_HEAD_TAIL` 置 True（`:1806-1809`），于是 PP 组的 `is_first_rank` / `is_last_rank` 恒为 True（`:559-581`）——两侧都建 `embed_tokens` 与 `norm`/`lm_head`。文档字符串写明这是"prefill_only 两侧都需要 full head/tail"的有意设计。

对具体模型（Qwen3.5 系列）的落点：

| 模块 | 构造判据 | 边侧 | 云侧 |
|---|---|---|---|
| `model.embed_tokens` | **无条件建**（`qwen3_5.py:224-227`，不判 PP rank） | 有 | 有 |
| `model.layers[i]` | `make_layers`（`:236-238`） | 全 `PPMissingLayer` | 全真 |
| `model.norm` | `is_last_rank`（`:243-246`） | 有（中性化后） | 有 |
| `lm_head` | `is_last_rank`（`:473-484`） | 有 | 有 |
| `visual`（VL 类） | **无条件建**（`qwen3_5.py:577-583`；`qwen3_vl.py:1671-1677` 同款） | 有 | **有（但永不使用）** |

### 2.4 于是边侧 HBM 里到底有什么

| 边侧常驻项 | 是否用到 | 备注 |
|---|---|---|
| `embed_tokens.weight` | ✅ prefill 嵌入 / 多模态 merge | `lwd_edge_worker.py:275-281` |
| `visual.*`（VL 模型） | ✅ 图像/视频编码 | `lwd_edge_worker.py:352-368` |
| `lm_head.weight` | ✅ rank-replay 还原 token | `lwd_edge_worker.py:531-540` |
| `model.norm.weight` | ❌ **不用** | 下行 hidden 已是 post-norm（`qwen3_next.py:541` 在 `forward` 内做 norm；`compute_logits` 只做 lm_head，`:816`），边侧 `compute_logits` 不再做 norm |
| `model.layers.*` | ❌ 未构建 | `PPMissingLayer` |
| KV cache | ❌ 不需要 | `get_kv_cache_spec() → {}` |
| spec drafter（若误配） | ❌ 不用 | 见 §7.7 |

**所以边侧的可省项，只有 `norm` 这种"KB 级残渣"**（`hidden_size × 2 B`，4096 维 ≈ 8 KB）。真正占边侧 HBM 的 `embed + lm_head`，恰恰是协议要求它必须持有的。

---

## 3. "所需参数"到底是谁的：逐模块判据（本报告的结论核心）

判据：**该角色的实际执行路径是否引用到这个模块**。三个引用点必须分开看：

- `embed_input_ids(...)`（含 `embed_multimodal`）
- `compute_logits(...)`（只做 `logits_processor(lm_head, h)`)
- `forward(...)`（层 + 末尾 norm）

### 3.1 云侧确实需要 `embed_tokens`（这是最容易误判的一条）

云侧虽然从前向输入看"吃的是边侧送来的 embeds"（`lwd_cloud_model_runner.py:270-296`、`_model_forward` 走 `inputs_embeds`），但**生成阶段的 token 行是云侧本地嵌入的**：

```python
# vllm/vllm/v1/worker/gpu_model_runner.py:3632-3662（prompt-embeds 分支）
is_token_ids = self.is_token_ids.np[:num_scheduled_tokens]
token_ids_idx_np = np.nonzero(is_token_ids)[0]      # decode 行
tokens_to_embeds = self.model.embed_input_ids(input_ids=token_ids)
self.inputs_embeds.gpu[token_ids_idx] = tokens_to_embeds
```

ascend 侧同款逻辑在 `lwd_cloud_model_runner.py:255-296`（`_lwd_preprocess_prompt_embeds`，`embeds.copy_(torch.where(is_token_ids, tokens_to_embeds, embeds))`），掩码由 `_lwd_refresh_decode_embedding_mask`（`:298-314`）算出"position ≥ prompt_len 的行"。

设计动机也清楚：云侧自己采样出 token（`lm_head`）→ 下一步立即本地嵌入，**不需要 edge→cloud 回传 token**，省掉每步一个往返。

### 3.2 云侧确实需要 `lm_head`

`lwd_cloud_model_runner.py:733-769`：采样后取 `logits`，算出采样 token 在降序 logits 中的**秩**（`(lg_sel > thresh).sum(dim=1)`），随 DOWN 元数据回边；下行载荷是 `sample_hidden_states`。没有 `lm_head` 就没有 logits，采样与秩都无从谈起。

### 3.3 边侧确实需要 embed + lm_head

- `embed_input_ids` / `embed_multimodal`：`lwd_edge_worker.py:275-281,352-368`。
- `lm_head`：`lwd_edge_worker.py:510`（`logits = model.compute_logits(hidden_states)`）+ `531-540`（`select_token_batch` 按云侧给的行内秩取名）。

### 3.4 结论

| 模块 | 边侧需要？ | 云侧需要？ |
|---|---|---|
| `embed_tokens` | ✅ | ✅（decode 行本地嵌入） |
| `model.layers.*` | ❌（已不建） | ✅ |
| `model.norm` | ❌（post-norm 的 hidden 由云侧给） | ✅ |
| `lm_head` | ✅（rank-replay） | ✅（采样/秩） |
| `visual`（VL） | ✅ | ❌ **永不使用** |

**因此："边侧只加载 embedding/unembedding、云侧只加载 transformer 层"在现有协议下是一个不成立的划分。** 现有划分是"两侧都持有词表投影（embed + lm_head）+ 各自持有自己那份 transformer 层"。要让设想成立，必须改协议（附录 A），或者接受"只有云侧视觉塔是纯浪费"这个现实结论。

> 补充：若模型的 `tie_word_embeddings=True`，则 `lm_head` 就是 `embed_tokens` 同一个张量（`qwen3_5.py:474-475`），"省一份"这件事自动消失，只剩"两边各留一份共享权重"。仓库默认 `tie_word_embeddings=False`（`transformers_utils/configs/qwen3_5.py:55,178`），**待确认** Qwen3.6-27B 的真实取值。

---

## 4. 冗余量化（HBM 视角）

### 4.1 公式

- 词表投影一整份：`vocab × hidden × sizeof(dtype)`
- 边侧单卡持有一整份（TP=1）；云侧每卡持有 `1/TP` 份（`VocabParallelEmbedding`/`ParallelLMHead` 按 TP 切词表）
- 视觉塔（Qwen3-VL 系 ViT，约 0.7 G 参数）：默认 `mm_encoder_tp_mode="weights"`（`vllm/vllm/config/multimodal.py:145`）→ 视 TP 分片；若为 `"data"` 则每卡一份完整副本

### 4.2 估算（假设取自仓库默认配置，bf16，**待现场 config 校正**）

假设：`vocab=248320`、`hidden=4096`（`transformers_utils/configs/qwen3_5.py:44-45`）、`tie_word_embeddings=False`、边 TP=1、云 TP=4。

| 项 | 一整份 | 边侧单卡 | 云侧单卡（TP=4） |
|---|---|---|---|
| `embed_tokens` | 2.03 GB | 2.03 GB | 0.51 GB |
| `lm_head`（非 tie） | 2.03 GB | 2.03 GB | 0.51 GB |
| 视觉塔（VL，≈0.7 G 参数） | ≈1.42 GB | 1.42 GB | ≈0.36 GB（weights 模式）/ ≈1.42 GB（data 模式） |
| 边侧 `norm` | ≈8 KB | 8 KB（**可省**） | — |

（若真实 `hidden_size` 不是 4096，按比例缩放；`hidden=5120` 时上表 ×1.25。）

### 4.3 可省 / 不可省分类

| 分类 | 项 | 量级 | 前提 |
|---|---|---|---|
| **可省 · 零风险** | 云侧 `visual.*` | 云 4 卡合计 ≈1.42 GB（data 模式 ≈5.7 GB） | 仅 VL 模型；且要验证 `supports_mm_inputs` 副作用（§7.2） |
| **可省 · 零风险** | 边侧 `norm`（及 DeepSeek-V4 的 `hc_head_*`） | KB 级 | 需先摘掉边侧 `profile_run`（§5.3） |
| **可省 · 防呆** | 边侧 spec drafter（误配 `--speculative-config` 时） | MTP 整份（含自身 embed/lm_head/层） | 配置纪律 + fail-fast |
| **可省 · 需改协议** | 一侧的 `lm_head`（≈2.03 GB 或 0.51 GB/卡） | 见附录 A | 采样权威方变更 + 正确性论证 |
| **不是 HBM 收益，但同源** | 加载期磁盘 I/O 与主机内存（边侧几乎整份 checkpoint 白读） | 启动时间 | `DefaultModelLoader` 前缀过滤（§5.4） |

> 一句话：**在"保持现协议 + Qwen3.6 文本部署"的交叉条件下，HBM 层面的可省额度接近于零。** 这正是本报告最需要你先接受的事实判断；能省的两块（云侧视觉塔、边侧残渣）都需要额外条件（VL 模型 / 先摘 profile_run）。

---

## 5. 应该怎么做：机制与落点

### 5.0 设计原则

1. **作用域（scope）由角色派生，不靠散落的 `if`**：在 `LwdConfig` 里集中定义"本角色拥有哪些模块前缀"，模型侧只查询它。
2. **双闸门分离**：HBM 由**构造期**决定；I/O/主机内存由**加载期**决定。只想省 HBM 就只改构造期。
3. **复用既有原语**，不要新造轮子：`collect_children` + `no_init_weights` + `StageMissingLayer`（`utils.py:521-612`）已经在 `mm_encoder_only` 与"`limit_mm_per_prompt` 全 0"两条产品化路径上跑通（`interfaces.py:215-292`）。
4. **占位必须"语义兼容"或者"永不被调用"**：`StageMissingLayer.__call__` 抛错、`PPMissingLayer`(=Identity) 只回第一个参数；两者都可能踩到调用点的**解包假设**（§7.1 是最硬的坑）。

### 5.1 P0：审计 + 防呆（不改行为，先确认根因）

**必须回答的三个问题**：

1. **模型类是谁？** 读 `/home/extra/Qwen3.6-27B/config.json` 的 `architectures` 与 `auto_map`；对照 `vllm/vllm/model_executor/models/registry.py:203,569`（`Qwen3ForCausalLM` / `Qwen3_5ForConditionalGeneration` 等）。若 `architectures` 指向模型目录自带的类（trust-remote-code），就必须查它的层是否经 `make_layers` 创建。
2. **边侧到底装了多少？** 在 `LwdEdgeWorker.load_model` 之后（`lwd_edge_worker.py:69-97`）打一条参数汇总日志：`sum(p.numel()*p.element_size() for p in model.named_parameters())` + 按前缀分组（`embed_tokens`/`layers.`/`norm`/`lm_head`/`visual`）。
   判据：**`layers.` 分组必须为 0**；否则就是"层裁剪没生效"（trust-remote-code 或 LWD 分支未进入）。
3. **边侧是否被误配了投机解码？** 查 launch 参数与 `vllm_config.speculative_config`。若存在，`model_runner_v1.py:3745-3750` 会加载整份 drafter，而边侧 `execute_model` 被劫持、永不使用它。

**防呆（一次性成本，长期收益）**：在 `LwdEdgeWorker.load_model` 末尾加断言——

```
assert edge_layer_params == 0          # 层裁剪必须生效
assert speculative_config is None      # 边侧不该有 drafter
assert 实际权重字节 < BUDGET           # 例如 embed+lm_head+visual 的公式预算 × 1.1
```

### 5.2 P1：云侧视觉塔裁剪（唯一一块"纯浪费"的大权重）

两条路，A 可立刻验证，B 才是长期形态。

**路 A（零改码，先量收益）**：给**云侧**启动参数加 `--limit-mm-per-prompt '{"image":0,"video":0}'`。
链路：`Qwen3_5ForConditionalGeneration.__init__` 把 `self.visual` 的构造包在 `self._mark_tower_model(vllm_config, {"image","video"})`（`qwen3_5.py:577-583`；`qwen3_vl.py:1671-1677` 同款）→ 该上下文若判定"所有模态上限为 0"，则用 `no_init_weights(..., targets=...)` 让子模块建在 **meta** 上并替换为 `StageMissingLayer`（`interfaces.py:280-290`、`utils.py:573-612`）→ **不占显存、不装载权重**（`utils.py:266-267`）；但**磁盘仍会被读一遍**（原因见 §5.4 第 3 条）。
⚠️ **副作用必须先验证**：同一条设置会让 `supports_multimodal_inputs()` 返回 False（`vllm/vllm/multimodal/registry.py:103-127`），从而改变云侧 `_model_forward` 的 SP 切分分支（`lwd_cloud_model_runner.py:327-336` 的 `not self.supports_mm_inputs`）与 `_preprocess` 分支（`:257-263`）。数值必须逐层对拍（§6.2），不能只看"起来了"。

**路 B（LWD 角色感知，推荐长期形态）**：把"该不该建"的条件从"模态上限"扩展为"角色 + 模态上限"。即在 `_mark_tower_model` 的同款机制上，增加 LWD 云角色判据（在 vllm 侧改 `qwen3_5.py`/`qwen3_vl.py` 的 `visual` 构造点，或在 vllm-ascend `patch/platform/` 下以 patch 形式包一层）。这样云侧**保留** `supports_mm_inputs` 语义（它是从 `model_config`/registry 推导，`gpu_model_runner.py:503`，与模块树无关），只去掉权重——避免路 A 的副作用。

### 5.3 P2：边侧 tail-only 裁剪（先摘 `profile_run`，再谈裁剪）

**前提认知**：边侧目前仍会执行一次"画像前向"——`worker/worker.py:531-600`（`determine_available_memory`）→ `model_runner.profile_run()` → `model_runner_v1.py:3685-3698` → `_dummy_run`。而边侧 KV cache 恒为 0（`lwd_edge_worker.py:165-167`），这次画像**没有任何用途**，却：
- 占一份按 `max_num_batched_tokens` 计的激活峰值；
- 逼着边侧把 `embed → norm` 这条"0 层前向"保持可跑——这正是裁剪 `norm` 的拦路虎（§7.1）。

**动作**：
1. 在 `LwdEdgeWorker` 覆写 `determine_available_memory()`（返回 0 或固定小值，跳过 `profile_run`）——与已有的 `compile_or_warm_up_model`（跳 warmup）和 `get_kv_cache_spec`（KV=0）同风格。
2. 再把边侧的 `norm`（以及 DeepSeek-V4 的 `hc_head_fn/base/scale`、`_mtp_hidden_buffer`、`topk_indices_buffer`）改为不建/不装。
   - 注意**不能用** `PPMissingLayer`/`StageMissingLayer` 直接顶替 `norm`：`Qwen3NextModel.forward` 里是 `hidden_states, _ = self.norm(hidden_states, residual)`（`qwen3_next.py:541`），Identity 返回单值会**解包失败**；若仍要走 dummy 前向，必须用一个"保持返回元数"的兼容占位（或干脆按第 1 步摘掉 profile 路径，使该模块永不被调用）。
   - 这也解释了 `lwd_edge_worker.py:156-163` 那条"0 层拓扑下 full-forward 会在 final norm 解包失败"的注释：同一个坑。

**收益预期**：HBM 上只有 KB 级；真实价值是"语义变干净 + 少一次画像 + 少一次 warmup 开销"。

### 5.4 P3：加载期前缀过滤（省磁盘 I/O 与主机内存；边侧收益最大）

现状：`DefaultModelLoader._get_weights_iterator()`（`default_loader.py:211-286`）会把每份 shard 的每个 tensor `f.get_tensor()` 物化一次，再交给 `load_weights`。跳过发生在**之后**（`utils.py:266-267`、`is_pp_missing_parameter`），所以"边侧把整份 checkpoint 读了一遍"是真实成本（HBM 不涨，但启动时间与主机内存峰值涨）。

做法（有现成范式）：

1. **按前缀过滤**：给 `DefaultModelLoader` 增加 `skip_prefixes`（与 `local_expert_ids` 同型），在 `get_all_weights` / `_get_weights_iterator` 出口处丢弃（`default_loader.py:288-307`）；判定函数照抄 `ep_weight_filter.should_skip_weight`（`ep_weight_filter.py:64-81`）的形状，由 `LwdConfig` 提供前缀集合。
2. **整 shard 跳过**：`_prepare_weights`（`default_loader.py:97-209`）已解析 `model.safetensors.index.json`；边侧[0,0]下绝大多数 shard 只含 decoder 层权重，可按 index 直接过滤掉整个文件——这是启动时间数量级的改善。
3. **不要指望"占位模块"能顺带省 I/O（重要纠正）**：`_load_module` 对 `StageMissingLayer`/`PPMissingLayer` 是提前 `return`（`utils.py:266-267`），但组织权重的 `_groupby_prefix` 用的是 `itertools.groupby`（`utils.py:160-178`）——**外层迭代推进时会消费掉上一组未读完的成员**，所以被"跳过"的子树里的 tensor 依然会被 `f.get_tensor()` 从磁盘读出再丢弃。EP 过滤之所以能省 I/O，正是因为它把判断放在了 `f.get_tensor()` **之前**（`weight_utils.py:1022-1026`，文档字符串也点明了这个动机：`ep_weight_filter.py:1-9`）。
   → 结论：**构造期裁剪只省 HBM；要省 I/O/主机内存必须做本节的迭代器层过滤**。当前状态下，边侧（实际只需要 embed/lm_head/visual 三类张量）仍会完整读一遍 checkpoint，这是它最"冤"的一笔开销。

### 5.5 P4：通用化（把一次性改动收敛成机制）

1. `vllm/vllm/config/lwd.py`：给 `LwdConfig` 增加派生属性，例如

```python
@property
def owned_module_prefixes(self) -> set[str]:
    """本角色声明拥有的模块前缀；未列出的前缀 = 不建、不装。"""
# edge(prefill_only): {"embed_tokens", "lm_head", "visual"}   # visual 仅 VL
# cloud(prefill_only): {"layers", "norm", "lm_head", "embed_tokens"}   # 不含 visual
```

   并把它接进 `validate()`（拒绝"角色需要的模块被用户配掉"这类致命组合）。
2. `parallel_state.py:559-581`：把 `_LWD_FULL_HEAD_TAIL` 从"整组恒 True"细化为"按模块语义的角色矩阵"（head/tail 在中性化语义上仍需成立，但"建不建"由 `owned_module_prefixes` 决定）。这是把"云侧不该有 visual / 边侧不该有 norm"落到统一机制的唯一干净位置。
3. 长期（不在本次范围）：给边侧一个**专用轻量模型适配器**（只暴露 `embed_input_ids`/`embed_multimodal`/`compute_logits` 的 shell），彻底摆脱"复用完整模型类"的包袱。

### 5.6 落点总表

| 落点 | 文件:行 | 做什么 | 收益 | 风险 |
|---|---|---|---|---|
| 配置 | `vllm/config/lwd.py:43-101` | 增 `owned_module_prefixes` + 校验 | 机制化 | 低 |
| 构造（云·VL） | `qwen3_5.py:577-583`、`qwen3_vl.py:1671-1677`（或 `patch/platform/`） | 云角色不建 `visual` | HBM ≈0.36~1.42 GB/卡 | 中（须验 SP 分支） |
| 构造（边） | `lwd_edge_worker.py:69-97` + 通用 `determine_available_memory` | 摘 profile、裁 `norm` 等 tail-only | KB 级 + 启动 | 中（解包兼容，§7.1） |
| 中性化语义 | `parallel_state.py:354-368,559-581,1806-1809` | head/tail 按角色-模块矩阵 | 机制化 | 中 |
| 加载 | `default_loader.py:211-307`、`weight_utils.py:893-1029` | 前缀/整 shard 过滤 | I/O + 主机内存 | 低 |
| 后处理 | `base_loader.py:80`、`process_weights_after_loading` | 只处理存在参数（现状天然满足，需回归） | — | 低 |
| 防呆 | `lwd_edge_worker.py:69-97`、云 `lwd_cloud_worker.py:128-149` | 参数预算/层数断言 | 防回归 | 低 |

---

## 6. 分阶段实施与验收

### 6.1 阶段

| 阶段 | 内容 | 改动面 | 预计 HBM 收益 | 风险 | 前置 |
|---|---|---|---|---|---|
| **P0** | 参数清单审计 + 三层防呆断言；确认模型类是否走 `make_layers` | 仅日志/断言（ascend worker） | 0（但决定后续一切） | 极低 | — |
| **P1** | 云侧视觉塔裁剪（先路 A 验证，再落路 B） | 配置 / 1 个构造点 | ≈1.42 GB（云合计） | 中（SP 分支） | VL 模型确认 |
| **P2** | 边侧摘 `profile_run` + 裁 tail-only | `LwdEdgeWorker` 覆写 | KB 级 + 启动 | 中（解包） | P0 |
| **P3** | 加载期前缀/整 shard 过滤 | `DefaultModelLoader` | 0（省 I/O/主机内存） | 低 | P0 的角色前缀集合 |
| **P4** | `LwdConfig.owned_module_prefixes` 机制化 + 中性化语义细化 | 配置 + parallel_state | 巩固 P1/P2 | 中 | P1/P2 验证通过 |
| **附录 A** | 协议级方案（一侧省 `lm_head`） | 控制面/数据面 + 采样契约 | 边 2.03 GB 或 云 0.51 GB×4 | 高（正确性） | 需专项决策 |

### 6.2 验收方法

| 验收项 | 方法 | 判据 |
|---|---|---|
| 参数清单对拍 | 每角色 dump `{前缀: (参数量, 字节)}`，与基线 diff | 只减少"声明要裁的前缀"，其余逐项一致 |
| HBM 读数 | `torch.npu.memory_allocated()/max_memory_allocated()/mem_get_info()`（已有先例：`lwd_cloud_worker.py:237-249`） | 峰值下降量 ≈ 预期字节；云侧 4 卡一致 |
| 启动耗时 | `model_runner_v1.py:3828-3832` 已有的 `load_model total time` | P3 生效后边侧应显著下降 |
| 数值对拍 | `VLLM_ASCEND_LWD_LAYER_TRACE=1`（`lwd_layer_trace.py`）+ 边侧 `[layer-trace] ... top20`（`lwd_edge_worker.py:514-529`） | 逐层校验和与基线逐位一致（尤其 P1 之后） |
| 功能回归 | 纯文本 + 多模态 + chunked prefill + abort（参照 `reports/abort_handling_analysis_v23.0_prefill_only_chunked_prefill_mm.md` 的用例面） | 无 OOM、无 rank 漂移、abort 不悬挂 |
| 回滚 | 所有裁剪挂在配置开关下（默认关，逐步打开） | 关掉开关行为与基线一致 |

---

## 7. 风险与坑（逐条带证据）

1. **占位模块的"元数"陷阱（最硬的坑）**：`Qwen3NextModel.forward` 是 `hidden_states, _ = self.norm(hidden_states, residual)`（`qwen3_next.py:541`）。用 `PPMissingLayer`(=Identity) 顶替 `norm` 会解包失败；`StageMissingLayer.__call__` 直接 `raise RuntimeError`（`utils.py:534-535`）。所有裁剪都必须回答："这个模块在本角色的任何执行路径（含画像/暖机）里还会不会被调用？"
2. **`supports_mm_inputs` 不可被间接改变**：它由 `mm_registry.supports_multimodal_inputs(model_config)` 推导（`gpu_model_runner.py:503`、`multimodal/registry.py:103-127`），而云侧 `_model_forward` 的 SP 切分与 `_preprocess` 分支**依赖它**（`lwd_cloud_model_runner.py:257-268,327-336`）。因此"用 `--limit-mm-per-prompt {"image":0}` 省视觉塔"这条路必须做数值对拍，不能只看能起。
3. **`tie_word_embeddings`**：为真时 `lm_head is embed_tokens`（`qwen3_5.py:474-475`）。任何"按名字裁 lm_head"的逻辑都必须先判 tie，否则会把还要用的词表张量一起删掉。仓库默认 False，**Qwen3.6-27B 待确认**。
4. **meta 参数残留**：`no_init_weights` 会把模块留成 meta（`utils.py:596-605`）。若某模块最终既在 `named_parameters()` 里、又在 meta 上，后续 `process_weights_after_loading`、NZ layout 转换（launch 开了 `enable_weight_nz_layout`）、EPLB/EP filter 都可能踩到。裁剪后必须确认"要么整棵子树换成 `StageMissingLayer`（不进 `named_parameters`），要么是真张量"。
5. **trust-remote-code 的模型类可能绕过 `make_layers`**：一旦绕过，边侧会建并装全部层权重（`is_pp_missing_parameter` 也无从命中）。这就是"两边都加载了所有参数"最可能的根因，必须由 P0 排除。
6. **边侧误配 `--speculative-config`**：`model_runner_v1.py:3745-3750` 会 `self.drafter.load_model(self.model)`，把整份 drafter（自带 embed/lm_head/层）装进边侧 HBM，而边侧运行时永不使用（`execute_model` 被劫持）。当前 `lauch.sh` 未配，建议加 fail-fast。
7. **EP/EPLB 与裁剪的交互**：`_init_ep_weight_filter` 会因 EPLB 主动跳过过滤（`default_loader.py:318-380`）。P3 的前缀过滤实现要与它并列而非覆盖，避免"该留的专家权重被过滤掉"。
8. **混合线性注意力模型（Qwen3.5 系）**：`Qwen3_5TextConfig.layer_types` 是 `linear_attention`/`full_attention` 交替（`transformers_utils/configs/qwen3_5.py:88-96`）。云侧需要 mamba/GDN 状态与 KV；边侧已通过 `get_kv_cache_spec() → {}` 撇清（`lwd_edge_worker.py:165-167`）。裁剪时**不要**把 linear-attention 相关参数当成"只在 forward 里用"的层参数误判。
9. **数据面契约不受影响**：UP/DOWN 的载荷语义与 seqno/numel 校验与权重加载无关（`distributed/lwd_comm/types.py:15-22`、`channel.py`），但注意 UP 是**全 world 广播**（每个云 TP rank 都要拿到 embeds）、DOWN 是点对点。裁剪不得改变 `hidden_size`/`hash_payload_numel` 的计算口径。
10. **边侧 TiD2EiD 表的额外读盘（DeepSeek-V4 专用）**：`load_model` 里用 `DefaultModelLoader._prepare_weights` 重新解析权重清单并逐 shard 打开 safetensors 取 hash 路由表（`lwd_edge_worker.py:83-97` → `lwd_hash_tables.py:14-73`）。所幸它只对匹配名字调 `get_tensor`（`lwd_hash_tables.py:36-42`），不物化专家权重，属于"打开全部 shard 文件"而不是"整份读取"。对 Qwen3.6 不适用，但说明"边侧加载路径里存在与主加载并行的第二次读盘"这一模式。

---

## 8. 附录 A：协议级方案（要让"边=embed/unembed、云=前向"真正成立）

现状的 `rank-replay` 是**有意的生产路径**，两条诊断旁路明确标注 "Never enable in production"（`vllm_ascend/envs.py:35-52`）：
- `VLLM_ASCEND_LWD_EDGE_SKIP_SAMPLE=1`：云侧捎带 token id，边侧跳过 lm_head + rank-replay（DOWN hidden 仍收）；
- `VLLM_ASCEND_LWD_DISABLE_DOWN=1`：云侧不做 DOWN 采集与发送，只回 token id。

### A1. 云侧采样为准（DOWN 只回 token id）

- 边侧省下 `lm_head`（≈2.03 GB 或 1/2 份若 tie）+ DOWN 隐藏态带宽（`[rows, hidden]` bf16）归零。
- 云侧仍需 `embed_tokens`（下一步 decode 输入）。
- 代价：采样权威方从"边侧 rank-replay"改为"云侧 token id"；需论证与集中式基线的**逐位一致性**（云侧 `lm_head` 在 TP=4 下是词表分片归约，边侧是单卡全词表 GEMM，低位可能有差异——rank-replay 本质上是在"容忍这种漂移"的前提下对齐两端）。

### A2. 边侧采样为准（云侧不做 lm_head）

- 云侧每卡省 `lm_head`（≈0.51 GB × 4）。
- 代价：每步 decode 需要 edge→cloud 回传 token id 才能继续（云端无法本地嵌入下一步输入）→ 引入 **ITL 串行往返**，对时延敏感场景不划算。

### A3. 折中：保留 rank-replay，但 DOWN 只带 token id（兜底）

省带宽不省权重，可作为 A1 的灰度过渡。

### 决策点（需你/团队拍板）

1. 是否允许变更采样权威方？正确性验收口径是什么（与集中式单机基线逐位对拍？还是允许 tie 情形差异）？
2. 若目标是"边侧极简"，A1 是唯一同时省权重与带宽的方案，但要接受协议变更与回归面扩大。
3. 在协议变更落地前，本报告 P0~P3 是与之正交、可独立收益的工作，建议先行。

---

## 9. 附录 B：自查清单（现场用）

**B1. 确认模型类与是否走 `make_layers`（P0 第一步）**

```
# 1) 看 config.json
python - <<'PY'
import json; c=json.load(open("/home/extra/Qwen3.6-27B/config.json"))
print("architectures:", c.get("architectures"))
print("auto_map:", c.get("auto_map"))
tc = c.get("text_config", c)
print("vocab/hidden/layers:", tc.get("vocab_size"), tc.get("hidden_size"), tc.get("num_hidden_layers"))
print("tie_word_embeddings:", c.get("tie_word_embeddings", tc.get("tie_word_embeddings")))
print("vision_config:", c.get("vision_config"))
PY
```

- `architectures` 命中 `vllm/model_executor/models/registry.py` 里的类（如 `Qwen3_5ForConditionalGeneration`，`:569`）→ 走 `make_layers`，层裁剪生效（`qwen3_5.py:236`）。
- 命中模型目录自带类（trust-remote-code）→ **必须人工检查它的 layers 是否用 `make_layers`**；不是则边侧会全量加载。

**B2. 参数清单对拍（两个角色都跑）**

```
# 在 worker.load_model() 之后调用；按前缀聚合参数量与字节
def dump_param_inventory(model, tag):
    buckets: dict[str, list[int]] = {}
    for n, p in model.named_parameters():
        parts = n.split(".")
        key = ".".join(parts[:2]) if parts[0] == "model" else parts[0]
        buckets.setdefault(key, [0, 0])
        buckets[key][0] += p.numel()
        buckets[key][1] += p.numel() * p.element_size()
    for k, (numel, nbytes) in sorted(buckets.items()):
        print(f"[inv][{tag}] {k:28s} params={numel/1e6:9.2f}M bytes={nbytes/2**30:7.3f}GiB")
    print(f"[inv][{tag}] TOTAL bytes={sum(v[1] for v in buckets.values())/2**30:.3f}GiB")
```

判据：`model.layers` 在**边侧**必须为 0；`visual` 在**云侧**应为 0（P1 之后）。

**B3. 需要现场补充/确认的信息（本文估算依赖它们）**

| 项 | 为什么需要 |
|---|---|
| `config.json` 的 `vocab_size` / `hidden_size` / `num_hidden_layers` | 把 §4.2 的估算换成真数 |
| `tie_word_embeddings` | 决定 `lm_head` 是不是独立的一份（§3.4、§7.3） |
| 是否上多模态请求 | 决定云侧视觉塔（P1）是否有收益 |
| `mm_encoder_tp_mode`（默认 `weights`） | 决定视觉塔是分片还是每卡复制（0.36 vs 1.42 GB/卡） |
| 云侧启动命令 | 与边侧命令不同（本次只看到边侧 `private_home/lauch.sh`），需确认云侧是否带 `--speculative-config`（MTP 在云侧是必需的） |

---

## 10. 附录 C：物理拆分（各机只放一半 checkpoint）的可行性与难点

> 问题："边侧服务器上只放 embedding/unembedding 权重、云侧只放 transformer 权重，两边分别启动，可行吗？"

### C.1 结论

| 口径 | 结论 |
|---|---|
| **显存/加载**："云侧只加载 transformer 参数" | **不可行**（保持现协议）。云侧必须自给 `embed_tokens`（decode 本地嵌入）与 `lm_head`（采样），见 §3.1/§3.2。 |
| **磁盘**：各机只放自己那半 checkpoint、各自启动 | **技术上可行，但不能只做这一件事**：必须同时做**构造期模块裁剪**，否则非量化模型直接启动失败、量化模型静默算错（C.3-1）。 |
| **推荐终态** | 变体 **A1**（附录 A1）：云侧保留 `embed`+`lm_head` 自给自足，**边侧删掉 `lm_head`**、只留 `embed`(+视觉塔)。这样"边=纯 embedding、云=transformer(+embed/lm_head)"两边各自启动成立，且**不新增任何延迟**。 |

### C.2 为什么"云侧只有 transformer"必须付出每步跨机往返

现状的 **decode 是云内自闭环**，边侧不在云下一步的关键路径上：

- 上行（UP，边→云）**只在 prefill 发生**：`LwdBatchType.LWD_EMBED` 仅在消费掉一条 prefill 预告（`prefill_notify_queue`）时挂到批上——`lwd_cloud_phase_scheduler.py:140-151`（`:185` 注释"lwd_batch=EMBED 即纯 prefill 步"）、`lwd_cloud_mixed_scheduler.py:444-455`（`:471` 注释"EMBED 且批内还有 decode 请求即 MIXED"）。纯 decode 步 `_schedule_pure_decode`（`lwd_cloud_phase_scheduler.py:153-158`）不挂 `lwd_batch` → 云侧不挂 UP recv。
- 云侧的 decode 输入靠**自己嵌入**：混合批里 prefill 行用 UP 来的 embeds、decode 行本地嵌入，两者 `torch.where` 合并（`lwd_cloud_model_runner.py:270-296`）。这次嵌入还带**一次 TP all_reduce**（`vllm/model_executor/layers/vocab_parallel_embedding.py:496`）。
- 云侧采样用**自己的 `lm_head`**（`lwd_cloud_model_runner.py:733-769`），采样完立刻本地嵌入，不等任何跨机消息。

因此，**一旦云侧丢掉 `embed_tokens`（或 `lm_head`），下一步的输入就只能由边侧产生**，decode 变成：

```
云前向(t) → DOWN → 边 unembed/采样(t) → UP(embedding 或 token id) → 云前向(t+1)
```

严格串行、云被边门控。带宽不是问题（1×hidden bf16 ≈ 8 KB/请求/步；token id 仅 4 B），**延迟与流水线才是代价**——现有设计把 `embed`+`lm_head` 放在云上，正是为了砍掉这个环。

### C.3 磁盘拆分的具体难点

1. **vLLM 默认拒绝"缺权重"（最容易踩的一条）**：`DefaultModelLoader.track_weights_loading`（`model_loader/default_loader.py:414-437`）会取"模型里存在的全部参数名 − 已加载集合"，非空即 `raise ValueError("Following weights were not initialized from checkpoint: ...")`；默认对**非量化**模型开启（`default_enable_weights_track = quantization is None and loaded_weights is not None`，`:403-405`），开关是 `--model-loader-extra-config '{"enable_weights_track": false}'`。
   - 好消息：这是 fail-fast，不会让你在垃圾权重上跑出结果。
   - **坏消息**：**量化模型默认关闭该追踪** → 少权重不报错，参数停留在 `torch.empty` 的未初始化值 → **静默算错**。所以"拆磁盘"绝不能靠关追踪来绕过，必须与"构造期删模块"绑定。
2. **模块骨架与 checkpoint 无关地照样建**：`base_loader.load_model` 先 `initialize_model`（`with target_device:`，`base_loader.py:53-59`），**再** `load_weights`。除 `PPMissingLayer`/`StageMissingLayer` 之外的模块一律先占显存。
3. **index.json / weight_map / 缓存链路**：`_prepare_weights`（`default_loader.py:97-209`）依赖 shard 列表与 `filter_files_not_needed_for_inference` / `filter_duplicate_safetensors_files`（`:198-202`）；裁文件必须同步维护 `model.safetensors.index.json`，并注意 HF 缓存 hash、`--revision` 校验。
4. **`tie_word_embeddings`**：为真时 `lm_head is embed_tokens`（`qwen3_5.py:474-475`），且 checkpoint 通常不含 `lm_head.weight`。两侧"缺什么"的判定必须一致，否则一边被追踪拦下、一边静默通过。
5. **量化参数成组**：`.weight_scale` / `.input_scale` / kv-scale 与权重同域（`maybe_remap_kv_scale_name`），前缀过滤要成组；NZ layout 与 `process_weights_after_loading` 只能处理真实存在的参数。
6. **多模态与投机**：边侧还需要视觉塔（"边=embedding"应写成"embedding + vision tower"）；云侧开 MTP 时还有 drafter（自带 embed/lm_head/层，`lwd_cloud/lwd_mtp_proposer.py`），"云只有 transformer"在 MTP 下更不成立。
7. **两侧必须同时起来**：world/TP/PP 组是集合创建（`parallel_state.py:1758-1809`），且 `lwd_wire.init_lwd_duplex_channels` 需要两侧按固定顺序完成双向 P2P 预热（`lwd_wire.py:124-167`；`lwd_edge_worker.py:125-136` 注释）。所谓"分别启动"= 加入同一 world 的两个进程，单边无法独立就绪；云侧还强依赖 `--master-addr`（`v1/lwd_control/__init__.py:73-95`）。
8. **可回退性**：拆过的 checkpoint 无法再单机跑。建议把"拆分产物"当作**分发选项**，而加载期用前缀过滤（§5.4）保持源 checkpoint 完整。

### C.4 三条路线的性价比

| 路线 | 边侧 HBM | 云侧 HBM | 每步跨机往返 | 磁盘 | 工程量 |
|---|---|---|---|---|---|
| 现状 | `embed`+`lm_head`+视觉塔 | 全套 | **无**（decode 云内闭环） | 两侧都需全量文件 | — |
| **A1（推荐）**：云采样，token id 随 DOWN 捎带 | **`embed`+视觉塔**（省 `lm_head` ≈2 GB） | 全套 | **无**（同方向捎带，不新增往返） | 边侧可只放 embed(+visual) | 小 |
| **X（你设想的终态）**：云无 embed/lm_head、边无 lm_head | `embed`+视觉塔 | transformer+norm | **每步 1~2 跳** | 真正各一半 | 大（协议+调度+正确性） |
| P3 前缀过滤（只省 I/O） | 不变 | 不变 | 无 | 源文件不动 | 小 |

> 路线 X 相对 A1 只多省云侧 `embed`（TP=4 下 ≈0.5 GB/卡），却把 decode 变成串行环，通常不划算。**要"云侧纯 transformer"，先算清这一步往返值不值。**

---

## 11. 一句话总结

> 边侧**已经**不加载 transformer 层了（`make_lwd_layers` + `PPMissingLayer`），你看到的"两边都加载了全部参数"主要来自 ① head/tail 被 `_LWD_FULL_HEAD_TAIL` 刻意两侧全建、② 加载器仍读完整份 checkpoint。而"边=embed/unembed、云=transformer"这个划分在**现有协议下不成立**——云侧必须持有 `embed_tokens`（decode 行本地嵌入）与 `lm_head`（采样/秩），边侧必须持有两者（rank-replay）。保持协议的前提下，唯一真正的 HBM 浪费是**云侧的视觉塔**（VL 模型）与**边侧的 tail-only 残渣**；再想省权重就必须改协议（附录 A）。落地建议：先做 P0 审计与防呆，再按 P1→P4 推进，全部挂在配置开关下、以逐层数值对拍兜底。
