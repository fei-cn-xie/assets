# commit `24f7d84f5` 逐文件改动说明

> 提交：`24f7d84f570875bddd127c0470143afb2b7f0dc7`
> `feat(lwd): load only the parameters each edge/cloud role runs`
> 范围：本文按 IDE 里列出的顺序（config → model_loader → models → multimodal）逐个文件说明**为什么要改、作用是什么**。
> 两个 UT 文件（`tests/config/test_lwd_module_scope.py`、`tests/model_executor/model_loader/test_lwd_weight_filter.py`）不在本文范围。

---

## 改动前的三个问题（所有文件都在解决这三件事）

1. **两侧多建了永不执行的模块**：`prefill_only` 下 `_LWD_FULL_HEAD_TAIL`（`vllm/distributed/parallel_state.py:1806-1809`）把 PP 首/尾判定中性化，让边、云都当成首/尾段 —— 于是**边侧多建语言模型尾部的 `norm`**，**云侧多建多模态视觉塔 `visual`**。两者都不会被执行（边侧只 embed/unembed，收到的 hidden states 已是 post-norm；云侧吃的是边侧算好的 embeds，`embed_multimodal` 永不调用），却白占 HBM。
2. **导致"按角色裁剪 checkpoint"起不来**：`DefaultModelLoader.track_weights_loading`（`default_loader.py:466`）严格要求 `named_parameters()` 里的参数都能在 checkpoint 里找到；边侧目录若不含 `model.norm.weight`、云侧目录若不含 `model.visual.*`，就会直接报 `Following weights were not initialized from checkpoint`。
3. **云侧还多建了 tokenizer 与多模态 processor**：云侧不对外服务、不分词，但启动期仍会因 `supports_multimodal_inputs()` 去加载 tokenizer 与 `preprocessor_config.json`，导致云侧目录必须保留一堆用不到的文件。

---

## 1. `vllm/config/lwd.py` —— 角色裁剪策略的唯一来源

**改动**

- `:37-38` 新增两张后缀表：
  ```python
  _EDGE_UNUSED_MODULE_SUFFIXES  = ("model.norm",)   # 边侧:语言模型尾部 norm
  _CLOUD_UNUSED_MODULE_SUFFIXES = ("visual",)       # 云侧:多模态视觉塔
  ```
- `:109-135` 新增 `LwdConfig.skips_module(path)`：
  ```python
  if not self.enabled or self.mode != "prefill_only":
      return False                      # head_tail/embedding_only 下 tail 在边侧,不能裁
  suffixes = _EDGE_UNUSED_MODULE_SUFFIXES if self.is_edge else _CLOUD_UNUSED_MODULE_SUFFIXES
  return any(path == suffix or path.endswith("." + suffix) for suffix in suffixes)
  ```

**为什么这么改**

- 策略必须**集中在一处**：否则"哪些模块该裁"会散落到每个模型文件里，无法单测、无法统一。
- 必须是**按路径后缀**匹配，而不是按属性名：Qwen3-VL 里存在 `visual.merger.norm` 这种同名子模块，若按裸 `norm` 匹配，边侧规则会把视觉塔里的 norm 一起裁掉（错删）。
- 必须**只在 `prefill_only`** 生效：`head_tail` / `embedding_only` 模式下尾段在边侧、边侧要自己算 norm，裁掉就错了。
- 必须是"角色"驱动而不是"模型"驱动：加新模型时只补构造点，不重写策略。

**作用**：所有模型文件只需向这一个函数提问；非 LWD 或非 `prefill_only` 时恒返回 `False`，行为与上游完全一致。

---

## 2. `vllm/config/multimodal.py` —— 新增 `mm_encoder_remote` 字段

**改动**（`:190`）

```python
mm_encoder_remote: bool = False
"""多模态输入由**远端**（LWD 边侧）编码：本进程只消费 prompt embeddings，
因此永不运行多模态 processor 与编码器。"""
```

**为什么这么改**

- 云侧需要在**注册表层**被短路掉，但又不能复用 `limit_mm_per_prompt=0` / `language_model_only`：上游 `supports_multimodal_inputs()` 是**先构建 processor（连带加载 tokenizer）、再判断模态上限**，所以把上限配成 0 也绕不过 tokenizer 加载；而且改 limits 会连带改变调度语义。
- 新增一个语义明确的开关，只表达"编码在远端"，不触碰 limits/调度。

**作用**：给第 10 个文件（`multimodal/registry.py`）提供判据，从而让云侧不建 HF 多模态 processor（连带不加载它自己的 tokenizer）。

---

## 3. `vllm/config/reasoning.py` —— `None` tokenizer 保护

**改动**（`:72-77`）

```python
tokenizer = cached_tokenizer_from_config(model_config=model_config)
if tokenizer is None:
    # skip_tokenizer_init(如 LWD 云侧,请求已由边侧分词):没有 tokenizer
    # 可以推导 reasoning token id,保持禁用而不是在 None 上崩。
    return
```

**为什么这么改**

- 云侧被强制 `skip_tokenizer_init=True` 后，`cached_tokenizer_from_config()` 会直接返回 `None`（`vllm/tokenizers/registry.py:268-270`）。
- 而 `VllmConfig.__post_init__`（`config/vllm.py:1498-1499`）在**任何**配置下都会调 `ReasoningConfig.initialize_token_ids()`，原代码随后会执行 `tokenizer.encode(...)` → 在 `None` 上 `AttributeError`，云侧启动即崩。

**作用**：云侧不会因为 reasoning 配置而启不来；该场景下 reasoning 保持禁用。这是通用修复——任何使用 `skip_tokenizer_init` 的部署都受益。

---

## 4. `vllm/config/vllm.py` —— 云侧强制三个开关

**改动**（`:900-917`，仅 `prefill_only` 且非边侧）

```python
if self.model_config is not None:
    self.model_config.skip_tokenizer_init = True      # :907
    multimodal_config = getattr(self.model_config, "multimodal_config", None)
    if multimodal_config is not None:
        multimodal_config.skip_mm_profiling = True    # 多模态显存 profile 无意义
        multimodal_config.mm_encoder_remote = True    # :913
logger.info_once("[LWD] cloud: tokenizer and multimodal encoder are not initialized (the edge owns both)")  # :915
```

**为什么这么改**

- 云侧**不 tokenize**（边侧分词后把 token ids/embeds 送过来）、**不编码媒体**，所以 tokenizer 与 MM processor 都不该初始化。
- 放在 `VllmConfig.__post_init__` 里：这一步在 worker/子进程启动**之前**执行，配置会随对象传给各进程，因此能覆盖所有进程与所有入口。
- 保留 `skip_mm_profiling`：作为兜底——即使还有别的路径按"多模态"走，也不会去 `profile_run` 里调 `embed_multimodal` 打到已被裁掉的视觉塔上。
- 加一条 INFO 日志：让"云侧确实没建 tokenizer/编码器"这件事在启动日志里可自证。

**作用**：云侧最小模型目录可以只剩 `config.json` + 自己的权重；启动更快、依赖更少。

---

## 5. `vllm/model_executor/model_loader/default_loader.py` —— 把过滤器接进权重迭代

**改动**

- `:22` 导入 `lwd_weight_filter` 的两个函数。
- `:292-315` `get_all_weights`（定义在 `:292`）把 **primary 与 secondary 两个**权重迭代器都包进过滤器：
  ```python
  yield from self._skip_unbuilt_weights(model, self._get_weights_iterator(primary_weights))
  for source in secondary_weights:
      yield from self._skip_unbuilt_weights(model, self._get_weights_iterator(source))
  ```
- `:317-346` 新增 `_skip_unbuilt_weights(...)` 生成器：命中占位模块名字的张量**不往下传**（也就是不产生 `get_tensor` 调用）。
- `:348-359` 新增 `_placeholder_module_names(model)`：只有 `lwd_config.enabled` 时才返回 `get_pp_missing_layer_names(model)`，否则返回 `[]`。

**为什么这么改**

- 过滤点必须在 `safetensors.get_tensor` **之前**：只有"不产生"才能真正省掉读盘与反序列化内存；放在装载之后只是"读完再扔"。
- 必须包住两个迭代器：否则 secondary 权重（如 LoRA/额外权重源）会绕过过滤。
- 必须以 `lwd_config.enabled` 为门槛：这是"非 LWD 部署零开销、零行为变化"的保证。
- 占位名单必须**从模型自身取**（`get_pp_missing_layer_names`，同时匹配 `PPMissingLayer` 与 `StageMissingLayer`），而不是按模型名硬编码——这样对任何模型都成立。

**作用**：把"完整 checkpoint + 已裁剪的角色"这种过渡期组合变成可启动；而当目录本来就已经裁剪过（本来就没有那些张量）时，过滤器无事可做（`skipped=0`），行为等价于原逻辑。

**注意**：这是两个调用者的共用路径 —— `DefaultModelLoader.load_weights`（`default_loader.py:446`）与权重热更 `GPUModelRunner.reload_weights`（`v1/worker/gpu_model_runner.py:5554`）；NPU 的 `NPUWorker.update_weights`（`vllm-ascend`）走 `weight_transfer_engine`，**不经过**这里。

---

## 6. `vllm/model_executor/model_loader/lwd_weight_filter.py` —— 过滤算法（新增文件，74 行）

**改动**

- `:38-49` `parse_skip_keys()`：把占位模块名（`"model.layers.3."`）转成组件序列（`["model","layers","3"]`，尾点去掉）。
- `:52-74` `name_is_skipped()`：把 checkpoint 张量名按 `.` 拆开，对每个 key 做**连续子序列**匹配。

**为什么这么改**（为什么不用 `startswith`）

| 场景 | `startswith` 的问题 | 组件级匹配 |
|---|---|---|
| `layers.3` vs `layers.30` | 字符串前缀会把 `layers.30.*` 一起吞掉（错删） | 组件不同 → 不匹配 |
| VL checkpoint 与模块名布局不一致：checkpoint 存 `model.visual.*`，vLLM 模块名是 `visual` | 前缀根本对不上 → **漏删** | 作为连续子序列能匹配 |
| 逐层判断（`num_hash_layers` 等） | 需要为每个模型写死前缀 | 组件序列天然通用 |

**作用**：跨布局安全地"只丢该丢的"；对 `StageMissingLayer`（上游因 `--limit-mm-per-prompt 0` 自动裁塔产生的占位）同样生效，因为名单来自 `models/utils.py:741-756`。风险面是"可能漏"（结果是多读一张量）、不是"错删"。

---

## 7. `vllm/model_executor/models/qwen3_5.py` —— 语言模型 norm + 两处视觉塔

**改动**

| 行 | 内容 |
|---|---|
| `:102` | 导入 `lwd_skips_module` |
| `:244-251` | 尾部 norm：`if get_pp_group().is_last_rank and not lwd_skips_module(vllm_config, maybe_prefix(prefix, "norm")):` 建真实 norm，否则 `PPMissingLayer()`（并补注释说明"边侧只反嵌入云侧已 norm 的隐状态"） |
| `:583-590` | `Qwen3_5ForConditionalGeneration.visual`：命中则 `PPMissingLayer()` |
| `:804-811` | `Qwen3_5MoeForConditionalGeneration.visual`：同上（MoE 变体） |

**为什么要改 norm**

- `_LWD_FULL_HEAD_TAIL` 让**边侧 `is_last_rank` 也是 `True`**（否则边侧连 `lm_head` 都不建，而边侧恰恰需要 `lm_head` 做 rank-replay）。所以单靠 `is_last_rank` 会把 `norm` 一起建出来。
- 但边侧的 unembed 路径只用 `lm_head`：`vllm-ascend/.../lwd_edge_worker.py:497` 收到云侧 DOWN 通道的 hidden states（**已 post-norm**），`:510` 直接 `model.compute_logits(hidden_states)`，而 `qwen3_5.py:520-524` 的 `compute_logits` 只调 `logits_processor(self.lm_head, ...)`——**全程没有 norm**。
- 不裁的后果：白占 HBM，并且边侧裁剪目录必须保留 `model.norm.weight`，否则 `track_weights_loading` 报缺权重。

**为什么要改 visual（两处）**

- 云侧永远不调 `embed_multimodal`（图像在边侧编码成 embeds 后经控制面传来），整座塔都是死权重。
- 用 `PPMissingLayer()` 而不是 `None`：占位层零参数、不出现在 `named_parameters()`（满足权重校验），但**名字仍留在模块树里**，于是完整 checkpoint 里多出来的塔张量会被 `AutoWeightsLoader` 当成占位模块跳过，而不是报 `no module or parameter named ...`。

**作用**：边侧省一个 `norm`、云侧省整座视觉塔；两侧目录都能只放自己那份权重（Qwen3.6-27B 的视觉塔是 333 个张量，云侧裁剪后可去掉这些张量，具体体积用拆分脚本 dry-run 的 `[size]` 行核对）。

**注意**：`norm` 的实际路径是 `model.language_model.model.norm`，以 `model.norm` 结尾 → 与 `lwd.py:37` 的后缀表匹配，**不需要改策略**。另外 `qwen3_next.py` 那种"双返回值 norm"（`:541`）本次**没有**改：它用 `is_last_rank` 提前 return（`:537`），在 LWD 下会被中性化，若将来给该模型加边侧 norm 裁剪需注意 arity。

---

## 8. `vllm/model_executor/models/qwen3_vl.py` —— 视觉塔 + deepstack 守卫

**改动**

| 行 | 内容 |
|---|---|
| `:137` | 导入 `lwd_skips_module` |
| `:1673-1685` | `Qwen3VLForConditionalGeneration.visual`：命中则 `PPMissingLayer()`，并注释"塔永不运行（embeddings 由边侧算），占位只为让名字留在模块树里供权重装载" |
| `:1687` | **`if self.use_deepstack and not isinstance(self.visual, PPMissingLayer):`** |

**为什么这么改**

- 与 `qwen3_5.py` 的 visual 同因（云侧不跑塔）。
- **deepstack 守卫是关键补充**：`use_deepstack` 时原代码会基于塔的输出维度创建 buffer（`deepstack_input_embeds` 等），塔被换成占位层后这些 buffer 没有意义、甚至会在构造/前向时引用不存在的属性 —— 因此必须在裁掉塔时一并跳过。

**作用**：Qwen3-VL 全系列（含 MoE 变体经 `qwen3_5.py`）云侧都不再构建视觉塔；避免"裁了塔却还在建 deepstack buffer"的半裁状态。

---

## 9. `vllm/model_executor/models/utils.py` —— `lwd_skips_module()` 薄封装

**改动**（`:674-695`）

```python
def lwd_skips_module(vllm_config: VllmConfig, path: str) -> bool:
    lwd_config = getattr(vllm_config, "lwd_config", None)
    return lwd_config is not None and lwd_config.skips_module(path)
```

**为什么这么改**

- 模型文件不应该自己去翻 `vllm_config.lwd_config`（有的调用点拿不到 config、有的 config 可能是 `None`）；用 `getattr` 容错保证非 LWD 场景不报错。
- 统一入口便于写单测（`tests/config/test_lwd_module_scope.py`）与后续替换实现。

**作用**：4 个模型文件、6 处调用点用同一种写法；语义是"本角色是否**不**构建该路径的模块"，与 `PPMissingLayer` 的替换动作一一对应。

---

## 10. `vllm/multimodal/registry.py` —— `supports_multimodal_inputs()` 提前返回

**改动**（`:114-118`）

```python
mm_config = model_config.get_multimodal_config()
if mm_config is not None and mm_config.mm_encoder_remote:
    # 多模态输入由别的进程(边侧)编码:processor 及其 tokenizer 永不被用到,
    # 没必要为了"探测支持哪些模态"把它们建出来。
    return False
try:
    info = self._create_processing_info(model_config, tokenizer=None)
```

**为什么这么改**

- 这是"**初始化期建多模态 processor**"的四处入口的**唯一闸门**，全部以它为判据：
  | 入口 | 位置 |
  |---|---|
  | 模型 runner 初始化 | `v1/worker/gpu_model_runner.py:503`（`:830-834` 决定 `mm_budget`） |
  | API 侧 InputProcessor | `v1/engine/input_processor.py:58-62`（建 `MultiModalBudget`） |
  | Renderer 构建 | `vllm/renderers/base.py:110-118` |
  | Scheduler 初始化 | `v1/core/sched/scheduler.py:186-190` |
- 上游是"**先** `_create_processing_info(tokenizer=None)`（内部会真去加载 tokenizer、读 `preprocessor_config.json`）、**再**判断模态上限"，所以必须在建之前返回 False —— 这也是当初云侧会崩在 `OSError: Can't load image processor ... preprocessor_config.json` 的直接原因。
- 改一处即可覆盖全部入口，避免到处打补丁。

**作用**：云侧 `supports_mm_inputs=False` → 不建 processor、不加载 tokenizer、`mm_budget=None`；配合第 4 个文件的三个开关，云侧目录只需 `config.json`。

**注意**：`supports_mm_inputs` 由 `True→False` 会让云侧 `_model_forward` 走进 `not self.supports_mm_inputs` 的序列并行分支（`vllm-ascend/.../lwd_cloud_model_runner.py:327-336`）。没开 FlashComm/SP 时该分支不执行；开了则需要逐层对拍。

---

## 串起来看：一次改动的整体效果

| 维度 | 改动前 | 改动后 |
|---|---|---|
| 边侧建的模块 | 层占位 + embed + lm_head + **norm** + visual | 层占位 + embed + lm_head + visual（**norm 不建**） |
| 云侧建的模块 | 全层 + embed + lm_head + norm + **visual** + MTP | 同左但**不含 visual** |
| 云侧启动期 tokenizer / MM processor | **建** | **不建** |
| 云侧最小目录 | `config.json` + tokenizer/processor + 权重 | **`config.json` + 权重** |
| 用完整 checkpoint 启动裁剪角色 | 失败（`Following weights were not initialized`） | 成功（占位模块的张量在读盘前被丢弃） |
| 非 LWD / 非 `prefill_only` 部署 | — | **完全不变**（策略函数恒 False、过滤器门槛不满足） |

## 回退方式（按文件）

| 想回退什么 | 怎么退 |
|---|---|
| 模块裁剪（文件 1/7/8/9） | 删掉模型文件里的 `lwd_skips_module(...)` 判断，或让 `skips_module` 直接返回 `False` |
| 云侧去 tokenizer/processor（文件 2/3/4/10） | 去掉 `config/vllm.py:900-916` 里的三个赋值 |
| 加载期权重过滤（文件 5/6） | `_skip_unbuilt_weights` 直接 `yield from weights`，或整个 revert 本提交 |

## 配套脚本：`vllm/scripts/lwd/split_lwd_checkpoint.py` 用法

> 本次提交解决的是"**运行期不建、不读**"（不构建永不执行的模块、不建 tokenizer/processor、把占位模块的权重在读盘前丢掉）；
> 本脚本解决的是"**磁盘上不放**"（把完整 checkpoint 按角色拆成子集）。两者是配套关系：**只做脚本不做提交 → 启动报缺权重；只做提交不做脚本 → 能跑但磁盘不省**。
>
> （历史上另有一份 Java 实现，已弃用；现在只保留这一份 Python 脚本。）

### 1. 运行环境

- Python 3.8+（建议 3.10+，脚本用 `from __future__ import annotations`）。
- 依赖 `safetensors`：**写分片**（`safetensors.torch.save_file`）与**无 `index.json` 时枚举分片键**需要它；
  有 `index.json` 的 **dry-run 只需标准库**（体积统计是纯 stdlib 读头部算出来的）。

### 2. 快速开始（先 dry-run，再真拆）

```bash
# 只看规则会保留什么(不写文件)
python vllm/scripts/lwd/split_lwd_checkpoint.py --src /data/Qwen3.6-27B --role edge  --dry-run
python vllm/scripts/lwd/split_lwd_checkpoint.py --src /data/Qwen3.6-27B --role cloud --dry-run

# 真正拆分
python vllm/scripts/lwd/split_lwd_checkpoint.py \
    --src /data/Qwen3.6-27B --dst /data/Qwen3.6-27B-edge  --role edge  --max-shard-gb 8
python vllm/scripts/lwd/split_lwd_checkpoint.py \
    --src /data/Qwen3.6-27B --dst /data/Qwen3.6-27B-cloud --role cloud --max-shard-gb 8
```

### 3. 参数

| 参数 | 必填 | 说明 |
|---|---|---|
| `--src <目录>` | 是 | 完整模型目录（含 `*.safetensors`）。支持相对路径与 `~`（脚本内部 `expanduser` + `realpath`） |
| `--dst <目录>` | 非 dry-run 必填 | 输出目录；**不能等于或包含 `--src`**（安全校验），不存在会自动创建 |
| `--role {edge,cloud}` | 是 | 拆给哪一侧，决定保留规则 |
| `--max-shard-gb <GB>` | 否 | 输出分片大小上限，默认 `5`。写盘的判定是"当前缓冲 ≥ 该值就落一个分片"，因此单分片可能略超此值；盘够大时给 `8` 可减少分片数 |
| `--keep-prefix <前缀>` | 否 | 追加保留前缀（可重复），排在角色规则之前 |
| `--keep-name-regex <正则>` | 否 | **按整个张量名匹配**的追加规则（可重复）。用于前缀表达不了的场景，例如"只保留某些层里的某个参数"。非法正则会报错并退出 `2` |
| `--metadata {auto,all}` | 否 | 非权重文件策略，默认 `auto`（见第 5 节） |
| `--no-progress` | 否 | 关闭写盘进度显示（默认：终端原地刷新一行；重定向到日志时每 10 秒一行） |
| `--dry-run` | 否 | 只打印将保留/丢弃的清单与体积，不写任何文件 |

### 4. 角色保留规则

**前缀规则**（`startswith` 匹配；默认覆盖三种命名布局：VL 包装 `model.language_model.*`、`model.*`（Qwen 系）、以及 **DeepSeek 原生布局**（无 `model.` 前缀））：

| 角色 | 保留前缀 |
|---|---|
| `edge` | 嵌入：`model.language_model.embed_tokens.`、`model.embed_tokens.`、`model.embed.`、`embed_tokens.`、`embed.`　输出头：`lm_head.`、`model.head.`、`head.`　顶层参数：`hc_head`、`model.hc_head`　视觉塔：`model.visual.` |
| `cloud` | 层与 norm：`model.language_model.`、`model.layers.`、`layers.`、`model.norm.`、`norm.`　嵌入：`model.embed_tokens.`、`model.embed.`、`embed_tokens.`、`embed.`　头与顶层参数：`lm_head.`、`model.head.`、`head.`、`hc_head`、`model.hc_head`　MTP：`mtp.`、`model.mtp.` |

> `hc_head*` 是 DeepSeek-V4 顶层**无条件创建**的真参数（`vllm_ascend/models/deepseek_v4.py:1082-1084`），两侧都要；`head.`/`embed.` 是 DeepSeek 原生布局里的 lm_head/embed 别名（`head.weight`→`lm_head.weight`、`embed.weight`→`embed_tokens.weight`）。

**按名匹配规则**（前缀表达不了的，`ROLE_NAME_PATTERNS`）：

| 角色 | 模式 | 为什么需要 |
|---|---|---|
| `edge` | `^(?:model\.)?layers\.\d+\.(?:ffn\|mlp)\.gate\.tid2eid(?:\.weight)?$` | DeepSeek-V4 的 Hash MoE 路由表：边侧不建层，但会**直接从 checkpoint 分片读**这些表（`lwd_edge_worker.load_model` → `lwd_hash_tables.load_tid2eid_tables`），漏了会报 `Missing tid2eid tables for Hash MoE layers` |
| `cloud` | （空） | 云侧建全部层，这些表就是普通参数，已被 `layers.`/`model.layers.` 覆盖 |

配套的子命令（查某个前缀/模式命中多少）：

```bash
python vllm/scripts/lwd/split_lwd_checkpoint.py --src <目录> --role cloud --dry-run | grep -A20 "matched tensors per rule"
```

### 5. 非权重文件策略（`--metadata`）

| 取值 | 边侧 | 云侧 |
|---|---|---|
| `auto`（默认） | 全拷**文件**（`config.json`、`tokenizer*`、`vocab.json`、`merges.txt`、`chat_template.jinja`、`preprocessor_config.json`、`video_preprocessor_config.json`、`generation_config.json` 等） | `config.json`、`configuration.json`、`*quant*description*.json`（ModelSlim 量化描述，**必须带**）、`chat_template.jinja` |
| `all` | 全拷（文件 + 目录） | 全拷（文件 + 目录） |

- **目录默认不拷**：只有角色规则里**显式点名**某个目录（例如 `"yaml"`）或 `--metadata all` 时才整目录递归拷贝。这样 `auto` 不会顺手把导出附带的目录塞进模型目录（保持"每侧只放自己需要的文件"）。
- 云侧保留 `chat_template.jinja` 只为兼容启动命令里的 `--chat-template <模型目录>/chat_template.jinja`（云侧不渲染模板）；若你的启动命令没传该参数，可以从规则里去掉它。
- 两侧都跳过：`.git*`、`README.md`、`LICENSE`、`.gitattributes`、`model.safetensors.index.json`（索引由脚本按需重建）。

> ⚠️ **云侧 `auto` 的前提是云侧 vllm 已包含本提交**（`skip_tokenizer_init` + `mm_encoder_remote`）。
> 如果云侧仍跑未打补丁的 vllm，必须用 `--metadata all`，否则云侧会报
> `OSError: Can't load image processor for ... containing a preprocessor_config.json`。
>
> 🔍 **改动安全性**：本次对脚本的连续改动已用"旧版 vs 新版逐张量对拍"验证过，Qwen 不受影响——做法是把 `cfa654575` 的脚本与新版都当模块加载，对同一条权重表分别调用 `select_keys()`，比较最终保留集合（结果：Qwen3.6 的 `model.*` 布局与 Qwen3-VL 的 `model.language_model.*` 布局在 edge/cloud 上**完全一致**；只有云侧 metadata 计划按上面的表格**新增**了 2 个文件，没有任何移除）。

### 6. 输出物与写盘行为

| 行为 | 细节 |
|---|---|
| 分片命名 | 先写成临时名 `model-XXXXX-of-XXXXX.safetensors`，全部写完后再按最终数量 `os.replace` 成 `model-00001-of-000NN.safetensors` |
| 索引 | 分片数 `> 1` → 写新的 `model.safetensors.index.json`（`weight_map` 指向新分片，并校验"计划里的每个 key 都在新 map 里"，否则 `RuntimeError`）；分片数 `== 1` → **删除**同目录旧索引（避免旧索引把自建分片过滤掉） |
| 内存 | 按源分片顺序读，攒够 `--max-shard-gb` 就落盘（峰值 ≈ 一个分片） |
| 进度显示 | 写盘期间在 **stderr** 输出 `百分比 / 已写 GiB / 速率 / ETA`，并带 `shards=`、`tensors=`。终端（tty）用回车原地刷新一行；输出被重定向（非 tty）时每 10 秒一行，避免刷满日志；体积统计失败时自动关闭。示例：`[write]  42.3% (118.3/279.4 GiB, 1.83 GiB/s, ETA 1:28) shards=15 tensors=22114` |
| 空间预检 | 写盘前检查目标盘（取 `--dst` 最近的已存在祖先）可用空间，需 ≥ 将写出体积 + 256 MiB，否则打印 `[space] ... 空间不足...` 并以退出码 **5** 退出 |
| 失败清理 | 写分片失败会删除本次已写出的分片（临时名分片也会删），只保留已拷贝的非权重文件，打印 `[clean] 写出失败...`，可直接重跑 |
| 复核 | 写完后回读所有新分片的键集合，必须与计划集合**完全相等**，否则退出码 **4** |

### 7. 退出码

| 退出码 | 含义 |
|---|---|
| `0` | 成功（`--dry-run` 走完也算成功） |
| `2` | 参数/路径非法：缺 `--src`、`--role` 取值错、非 dry-run 未给 `--dst`、`--dst` 等于/包含 `--src`、源目录不存在、`--keep-name-regex` 非法 |
| `3` | 没有任何张量命中保留规则（通常 `--role` 选错，或模型布局与规则不匹配） |
| `4` | 写完复核失败（输出键集合与计划不一致） |
| `5` | 目标盘空间不足 |
| 其它（Python 异常 → 1） | IO/JSON/`safetensors` 报错、`write_index` 发现缺张量等 |

### 8. 日志样例

```text
[in ] 1199 tensor(s), 51.75 GiB on disk
[rule] keep prefixes: ['lm_head.', 'model.language_model.', ...]
[rule] keep name patterns: ['re:^(?:model\\.)?layers\\.\\d+\\.(?:ffn|mlp)\\.gate\\.tid2eid(?:\\.weight)?$']

  matched tensors per rule:
    lm_head.                                 1 tensor(s)
    model.language_model.                    863 tensor(s)
    model.mtp.                               0 tensor(s)  <-- 没命中,检查命名布局/是否该保留
    re:^(?:model\.)?layers\.\d+\.(?:ffn|mlp)\.gate\.tid2eid(?:\.weight)?$   2 tensor(s)

  => keep 866 tensor(s) of 1199; drop 333

  dropped tensors grouped by leading components (top 15):
    model.visual.patch_embed                 3
    model.visual.blocks                      330

  [size] 将写出 50.31 GiB / 源 51.75 GiB (97.2%)

[meta] 将拷贝 3 个非权重文件: ['chat_template.jinja', 'config.json', 'configuration.json']
[meta] 主动跳过 8 个: ['generation_config.json', 'gitattributes', 'merges.txt', ...]
[meta] 云侧最小化依赖 vllm 的 LWD 改动(skip_tokenizer_init + mm_encoder_remote);若云侧仍跑未打补丁的 vllm,请加 --metadata all

[space] 目标 /data 可用 851.28 GiB,需要 50.56 GiB

[copy] 3 metadata file(s): ['chat_template.jinja', 'config.json', 'configuration.json']

# 以下是 stderr 上的写盘进度(终端会原地刷新同一行;重定向到日志时每 10 秒一行)
[write]   6.7% ( 3.4/50.3 GiB, 1.72 GiB/s, ETA 0:27) shards=0 tensors=812
[write]  42.3% (21.3/50.3 GiB, 1.83 GiB/s, ETA 0:16) shards=4 tensors=6120
[write] 100.0% (50.3/50.3 GiB, 1.79 GiB/s, ETA 0:00) shards=13 tensors=866

[write] 13 shard(s): ['model-00001-of-00013.safetensors', ...]
[verify] OK: 866 tensor(s), 50.31 GiB in /data/Qwen3.6-27B-cloud
[next ] 用 split 出来的目录启动该侧;若仍报 'Following weights were not initialized',按报错名单补张量
```

关键行：`[size]`（精确体积，用来判断盘够不够）、`[rule] ... name patterns`（按名规则是否命中）、`[verify] OK`（写完的文件确实包含计划的全部张量）。

### 9. 常见问题

| 现象 | 原因与处理 |
|---|---|
| `No space left on device (os error 28)` | 目标盘不够。**换 `--dst` 到另一块盘**（调小 `--max-shard-gb` 只改分片粒度、不减总量）；或只拆一侧（云侧直接用完整目录 + 本提交的读盘前过滤） |
| 云侧报 `Can't load image processor ... preprocessor_config.json` | 云侧 vllm 未含本提交 → 临时用 `--metadata all`，正式做法是同步补丁后再用 `auto` |
| 启动报 `Following weights were not initialized: [...]` | 按名单补张量；常见原因：`--role` 选错、手工删过文件、开了 MTP 但目录里没有 `mtp.*` |
| 边侧报 `Missing tid2eid tables for Hash MoE layers` | 用了 DeepSeek-V4 但边侧目录没保留 `tid2eid`（旧版脚本或手工删过）；确认走的是带 `ROLE_NAME_PATTERNS` 的版本，或临时 `--keep-prefix model.layers.0.mlp.gate.tid2eid`（每个 hash 层一条） |
| 日志里没有 `LWD weight filter: skipped N tensor(s)` | 这条来自 vllm 侧（`debug` 级）：① 需 `VLLM_LOGGING_LEVEL=DEBUG`；② 用**完整 checkpoint** 才会命中（用拆分目录时本来就没有可丢的，`skipped=0` 是正常）；③ vllm 代码需含本提交 |

### 10. 校验拆分结果

```bash
python - <<'PY'
import json

for role, path in (("edge", "./Qwen3.6-27B-edge"), ("cloud", "./Qwen3.6-27B-cloud")):
    wm = json.load(open(f"{path}/model.safetensors.index.json"))["weight_map"]
    layers = sum(k.startswith(("model.language_model.layers.", "model.layers.")) for k in wm)
    norm = sum(k.endswith("norm.weight") for k in wm)
    visual = sum(".visual." in k or k.startswith("visual.") for k in wm)
    print(f"{role:5s} tensors={len(wm):5d} layers={layers:4d} norm={norm:2d} visual={visual:4d}")
# 期望:edge 的 layers=0;cloud 的 visual=0
PY
```

单分片输出（`--max-shard-gb` 足够大）不会生成索引文件，此时可改为直接读分片头部的键集合。

### 11. 与本次提交的分工

| 关注点 | 本次提交（vllm 代码） | 本脚本 |
|---|---|---|
| 边侧不建 `norm` / 云侧不建 `visual` | ✅ `LwdConfig.skips_module` + 模型构造点 | — |
| 云侧不建 tokenizer/MM processor | ✅ `skip_tokenizer_init` + `mm_encoder_remote` | — |
| 完整 checkpoint 喂给裁剪角色也能启动 | ✅ `lwd_weight_filter`（读盘前丢弃占位模块张量） | — |
| 磁盘上只放本角色需要的张量 | — | ✅ 按角色重打包分片（本脚本） |
| 每侧需要哪些非权重文件 | ✅ 云侧只需 `config.json` | ✅ `--metadata auto` 按角色拷贝 |

---

## 换新模型时需要做的适配

> 前提：本提交的"通用部分"（加载期权重过滤、云侧去 tokenizer/processor）**与具体模型无关**，换模型不用动；
> 需要逐个模型适配的只有"**哪些模块本角色不建**"这件事——也就是"在模型文件的构造点加一个判断"。

### A. 先回答三个问题，就知道要改哪里

| 问题 | 怎么查 | 结论 |
|---|---|---|
| 1) 它的主干层是走 `make_layers` 建的吗？ | `grep -n "make_layers(" <model>.py` | 有 → **层切分自动生效**（`models/utils.py:717-720` 内部分派到 `make_lwd_layers`），不用改；没有（自建 ModuleList）→ 需要自己接 LWD 层切分 |
| 2) 它有"尾部 norm / final norm"吗？ | `grep -n "is_last_rank" <model>.py` | 有 → **要加 norm 判断**（否则边侧白建、且边侧目录必须带该权重） |
| 3) 它是多模态吗？塔的属性叫什么？ | `grep -n "_mark_tower_model" <model>.py` | 属性名是 `visual` → 直接加判断；是别的名字 → 还要**扩后缀表**（`config/lwd.py:37-38`） |

### B. 必改清单（按模型类型）

| 模型类型 | 必改点 | 改哪个文件 |
|---|---|---|
| **纯文本 LLM**（llama / qwen3 / qwen3_moe / deepseek_v2 / gemma…） | ① 尾部 norm 构造点 | 该模型文件（如 `llama.py` 的 norm 分支） |
| **多模态 VL，塔属性名 `visual`**（Qwen 系 / GLM-4.1V / Hunyuan / Intern-S1 / OpenPangu / MiMo / EXAONE4.5…） | ① norm ② `visual` 构造点 ③ 若有 deepstack/多尺度 buffer → 加"占位时不建 buffer"的守卫 | 该模型文件（**dense 与 MoE 变体常常是两份类，两处都要改**，例：`qwen3_5.py:583` 与 `:804`） |
| **多模态 VL，塔属性名 `vision_tower`**（SigLIP/LLaVA 系 / Gemma3-MM / Kimi-VL…） | ① norm ② `vision_tower` 构造点 ③ **扩展 `_CLOUD_UNUSED_MODULE_SUFFIXES` 加 `"vision_tower"`** | 该模型文件 + `vllm/config/lwd.py:38` |
| **多模态 VL，塔属性名 `vision_model`**（InternVL / CLIP / BLIP2 / Step3-VL / Nemotron-VL…） | 同上，后缀换成 `"vision_model"` | 该模型文件 + `config/lwd.py:38` |
| **另有独立投影层**（如 Kimi-K2.5 的 `mm_projector`） | 视需要再加一条后缀/构造点判断（塔不建时投影层通常也一起不建） | 该模型文件 + `config/lwd.py:38` |
| **带 MTP/drafter**（Qwen3.5 MTP / DeepSeek MTP…） | 代码侧**通常不用改**（边侧根本不建 drafter）；但要保证**云侧拆分目录保留 MTP 权重**（现有 `mtp.` / `model.mtp.` 规则已覆盖） | 只用检查拆分规则 |
| **"藏在 `layers.*` 下但边侧必须保留"**（典型：DeepSeek-V4 的 Hash MoE 路由表 `tid2eid`） | 改**拆分脚本**的 edge 规则（代码侧不用改） | `vllm/scripts/lwd/split_lwd_checkpoint.py` 的 `ROLE_NAME_PATTERNS` |

### C. 不用改的文件（以及为什么）

| 文件 | 为什么不用为"新模型"改 |
|---|---|
| `vllm/config/multimodal.py`、`vllm/multimodal/registry.py` | 在 registry 层统一短路 processor，与模型类无关 |
| `vllm/config/reasoning.py` | 与模型无关的 `None` tokenizer 保护 |
| `vllm/config/vllm.py` | 云侧三个开关按角色设置，与模型无关 |
| `vllm/model_executor/model_loader/default_loader.py`、`lwd_weight_filter.py` | 用模型自身的占位模块名单（`get_pp_missing_layer_names`，`:741-756`），不依赖模型名 |
| `vllm/model_executor/models/utils.py` | `lwd_skips_module()` 是通用薄封装 |
| `vllm/config/lwd.py` | **多数情况不用改**；只有"模块路径后缀与现有表不匹配"时才加一行 |

### D. 两种改法模板（直接照抄）

**D1. 尾部 norm（把"只在尾段建"改成"只在尾段且本角色需要时建"）**

```python
# 改前
if get_pp_group().is_last_rank:
    self.norm = SomeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
else:
    self.norm = PPMissingLayer()

# 改后
if get_pp_group().is_last_rank and not lwd_skips_module(
    vllm_config, maybe_prefix(prefix, "norm")
):
    self.norm = SomeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
else:
    # 非 PP 尾段，或只反嵌入云侧已 norm 隐状态的 LWD 边侧
    self.norm = PPMissingLayer()
```

**D2. 多模态塔（含可选的 buffer 守卫）**

```python
# 改前
with self._mark_tower_model(vllm_config, {"image", "video"}):
    self.visual = SomeVisionTransformer(
        config.vision_config, quant_config=quant_config,
        prefix=maybe_prefix(prefix, "visual"),
    )

# 改后
with self._mark_tower_model(vllm_config, {"image", "video"}):
    if lwd_skips_module(vllm_config, maybe_prefix(prefix, "visual")):
        # LWD 云侧：embeddings 由边侧算好经 UP 通道送来，塔永不运行。
        # 占位层零参数，但名字留在模块树里，完整 checkpoint 的多余张量会被跳过。
        self.visual = PPMissingLayer()
    else:
        self.visual = SomeVisionTransformer(
            config.vision_config, quant_config=quant_config,
            prefix=maybe_prefix(prefix, "visual"),
        )
    # 若塔被替换成占位，基于塔输出维度创建的 buffer 也要跳过
    if self.use_deepstack and not isinstance(self.visual, PPMissingLayer):
        ...   # 原有 buffer 创建逻辑
```

**D3. 后缀表（只有当模块路径后缀与现有表不匹配时才需要）**

```python
# vllm/config/lwd.py:37-38
_EDGE_UNUSED_MODULE_SUFFIXES  = ("model.norm",)                      # 边侧：尾部 norm
_CLOUD_UNUSED_MODULE_SUFFIXES = ("visual", "vision_tower")           # 云侧：视觉塔（按模型命名追加）
```

匹配规则（`:126-135`）：`path == suffix or path.endswith("." + suffix)` —— 所以写 `"vision_tower"` 能同时命中 `vision_tower` 与 `language_model.vision_tower`。

**另外**：别忘在模型文件的 import 列表里加 `lwd_skips_module`（例：`qwen3_5.py:102`、`qwen3_vl.py:137`、`deepseek_v4.py:60-66`）。

### E. 查"该改哪一行"的四个命令

```bash
# 1) 层：是否自动获得 LWD 层切分
grep -n "make_layers(" vllm/model_executor/models/<model>.py
# 2) 尾段：norm / lm_head / embed 的构造点
grep -n "is_last_rank" vllm/model_executor/models/<model>.py
# 3) 塔：构造点与属性名
grep -n "_mark_tower_model" vllm/model_executor/models/<model>.py
# 4) 精确点分路径（最权威）：打开模型结构日志
VLLM_LOG_MODEL_INSPECTION=1 <启动命令> 2>&1 | less     # 见 base_loader.py:61 log_model_inspection
#    或自己打印 [n for n,_ in model.named_modules()]，确认 norm / 塔的真实路径
```

### F. 已经做过的两个范例（照这个抄）

| 模型 | 改动 |
|---|---|
| **Qwen3.5 / Qwen3.6 + Qwen3-VL**（本次提交） | `qwen3_5.py:102`（import）、`:244-251`（norm）、`:583-590`（dense 的 visual）、`:804-811`（MoE 的 visual）；`qwen3_vl.py:137`、`:1673-1685`（visual）、`:1687`（deepstack 守卫） |
| **DeepSeek-V4（NPU）**（本次提交之后的未提交改动） | `vllm-ascend/vllm_ascend/models/deepseek_v4.py`：`:60-66` 加 import、`:1051-1058` 加 norm 判断；拆分脚本 `split_lwd_checkpoint.py` 的 `ROLE_NAME_PATTERNS` 为 edge 增加 `tid2eid` 规则（V4 边侧不建层，但要直接从 checkpoint 读 Hash 路由表） |

### G. 还没改、将来可能要补的（附确切位置）

| 模型/文件 | 要补什么 | 注意 |
|---|---|---|
| `vllm/model_executor/models/qwen3_next.py:497-500` | 尾部 norm 判断 | **arity 陷阱**：`:541 hidden_states, _ = self.norm(hidden_states, residual)` 是双返回值，`PPMissingLayer` 只返回一个值；该文件 `:537` 用 `is_last_rank` 提前 return，而 LWD 会把它中性化 → 边侧若走到这行会解包失败（边侧不 forward，所以只是 fail-fast，但要清楚） |
| `gemma3_mm.py:516`、`aya_vision.py:344`、`cohere2_vision.py:344`、`granite4_vision.py:488`、`paligemma.py:296`、`kimi_vl.py:320` | 塔判断 + 后缀 `vision_tower` | 这些是 `vision_tower` 命名 |
| `internvl.py:589`、`nemotron_vl.py:121`、`skyworkr1v.py:185`、`eagle2_5_vl.py:173`、`blip2.py:554` | 塔判断 + 后缀 `vision_model` | 这些是 `vision_model` 命名 |
| `kimi_k25.py:336/349` | 塔 + `mm_projector` | 塔与投影层都要处理 |
| `vllm/models/deepseek_v4/{nvidia,amd,xpu}/model.py:487/934/1003` | 与 ascend 版同样的 norm 判断 | 你当前 NPU 部署用不到（ascend 注册覆盖），**可选** |

### H. 适配后的验证清单

1. **拆分脚本 dry-run**：目标规则命中数 > 0（例如云侧 `visual` 行、边侧 `norm`/`tid2eid` 行），`=> keep N` 与预期一致；
2. **启动**：边上不出现 `layers.*`/`norm` 参数；云侧不出现塔参数；云侧无 `Can't load image processor ... preprocessor_config.json`；
3. **权重校验**：两边都不出现 `Following weights were not initialized from checkpoint`；
4. **数值**：`VLLM_ASCEND_LWD_LAYER_TRACE=1` 对拍（重点看云侧 `supports_mm_inputs` True→False 的 SP 分支，`lwd_cloud_model_runner.py:327-336`）；
5. **可选**：给 `tests/config/test_lwd_module_scope.py` 补一条新模型路径的用例（含"不该命中的同名子模块"，如 `vision_tower.merger.norm`）。

### I. 适配时的风险清单

| 风险 | 说明 |
|---|---|
| **arity** | 双返回值/多返回值的 norm（`hidden_states, _ = self.norm(...)`）被换成 `PPMissingLayer` 后，若该路径真被执行会解包失败；只在"该角色永不 forward"的前提下可接受 |
| **塔的衍生模块** | 塔被裁后，基于塔输出创建的 buffer/子模块（deepstack、merger、projector、pos-embed 等）要一并跳过，否则会出现"半裁"状态 |
| **checkpoint 与模块名不一致** | 有 `WeightsMapper`/重命名的模型（如 VL 的 `model.visual.*` ↔ 模块 `visual`），过滤靠组件级匹配覆盖；新增后缀不要选过短的词，避免误删 |
| **`StageMissingLayer` 也会被过滤** | 上游因 `--limit-mm-per-prompt 0` 自动裁塔产生的占位同样在名单里（`models/utils.py:749`），不要重复处理 |
| **只在 `prefill_only` 生效** | `head_tail` / `embedding_only` 下尾段在边侧，**不能**照抄 norm 裁剪；`skips_module` 已用模式判断挡住，别绕过它直接写死 |

---

## 附：DeepSeek-V4 的边云拆分（V4 适配落地）

> 解决"把 `DeepSeek-V4-Flash-w8a8-mtp` 按角色拆成边/云两份目录，并让边侧正常起来"。

### 1. 新增的两个提交

| 仓 | 提交 | 文件 | 作用 |
|---|---|---|---|
| vllm | `9a6ef7283` | `scripts/lwd/split_lwd_checkpoint.py` | ① 默认规则同时覆盖三种命名布局（含 **DeepSeek 原生布局**）② 边侧补 `head.`/`embed.`/`hc_head` ③ `ROLE_NAME_PATTERNS`：边侧自动保留 `tid2eid` ④ `--keep-name-regex` ⑤ 云侧 metadata 带上量化描述 + `*` 通配 + 目录整拷 |
| vllm | `5b20bf947` | `scripts/lwd/split_lwd_checkpoint.py` | 写盘进度：`百分比/已写 GiB/速率/ETA`（tty 原地刷新、日志每 10 秒一行、`--no-progress` 关闭、只写 stderr） |
| vllm | `78edc0aab` | `scripts/lwd/split_lwd_checkpoint.py` | 目录默认不拷（只有规则显式点名或 `--metadata all` 才整目录拷贝），避免 `auto` 顺手带走 `yaml/` 之类 |
| vllm-ascend | `b189aed65e` | `vllm_ascend/models/deepseek_v4.py` | 尾部 `norm` 按 `lwd_skips_module()` 裁剪 |

**norm 守卫为什么必须**：边侧只反嵌入云侧**已经 norm 过**的隐状态；不裁的话边侧会白建 `norm`，而且**边侧目录必须保留 `model.norm.weight`**（splitter 的 edge 规则并不保留它）→ 拆盘后边侧直接报缺权重。

**为什么不用改 vllm 侧的上游 V4 实现**：上游 `nvidia`/`xpu` 实现里 `get_expert_mapping()` 用了 `next(iter(islice(self.layers, start, end)))`，在"本进程零层"（`edge_head_tail_layers=[0, 0]` → `make_lwd_layers()` 返回 `start = end = 0`，`vllm/model_executor/models/utils.py:669-670`）时会抛 `StopIteration`，并被 PEP 479 转成 `RuntimeError: generator raised StopIteration`。但 **Ascend 上跑的不是它**：`vllm-ascend/setup.py:545-550` 把 `ascend_model = vllm_ascend:register_model` 注册进 `vllm.general_plugins`，该注册用 `AscendDeepseekV4ForCausalLM` 覆盖 `DeepseekV4ForCausalLM`（`vllm_ascend/models/__init__.py:5`）；而 ascend 实现的 `get_expert_mapping()`（`vllm_ascend/models/deepseek_v4.py:1287`）不依赖本地层，天然安全。

> ⚠️ 前提：**不要用 `VLLM_PLUGINS` 排除 `ascend_model`**（general 插件受它过滤：`vllm/plugins/__init__.py:32,56`）。否则会退回上游 `nvidia` 实现，`RuntimeError: generator raised StopIteration` 立刻复现——2026-10-14 那两次崩溃栈正是上游 `nvidia/model.py:1149`。

### 2. 这个 checkpoint 与 Qwen 的差异（决定了脚本怎么用）

| 观察 | 影响 |
|---|---|
| **张量名没有 `model.` 前缀**（DeepSeek 原生布局：`layers.`/`norm.`/`embed.`/`head.`/`hc_head`/`mtp.`） | **首次拆错的直接原因**：旧规则只认 `model.*`，于是只有 `mtp.` 命中（103176 → 2347 个、8.24 GiB）。现已把该布局写进默认规则 |
| 分片名 `quant_model_weights-XXXXX-of-00070.safetensors`，**没有** `model.safetensors.index.json` | 脚本自动走"无索引回退"（逐分片读头部建 `weight_map`），无需改代码；环境里要有 `safetensors` |
| `quant_model_description.json`（ModelSlim W8A8 量化描述，6.2M） | **每一侧都要有**：`vllm_ascend/quantization/utils.py:114-117` 靠它判定量化，缺了会被当成 float 加载 → 已加入云侧规则 `*quant*description*.json` |
| `gitattributes`（无点）、`yaml/` 子目录 | `gitattributes` 无需处理；`yaml/` 运行时不读，默认不拷（规则里点名目录时才整目录拷贝） |

checkpoint 名 → 模型内名（取自 loader 的权重映射表 `vllm/models/deepseek_v4/nvidia/model.py:1184-1201`，这是"checkpoint 到底叫什么"的权威来源，比猜名字可靠）：

| checkpoint | 模型内 |
|---|---|
| `layers.` | `model.layers.` |
| `embed.` | `model.embed.`（`embed.weight` → `embed_tokens.weight`） |
| `norm.` | `model.norm.` |
| `head.weight` | `lm_head.weight` |
| `hc_head_fn` / `hc_head_base` / `hc_head_scale` | `model.hc_head_*` |
| `mtp.` | `model.mtp.` |

### 3. 命令（默认规则已覆盖，不需要额外参数）

```bash
SRC=/weight/DeepSeek-V4-Flash-w8a8-mtp

# ① 干跑：核对"每条规则的命中数"与 [meta] 计划
python vllm/scripts/lwd/split_lwd_checkpoint.py --src $SRC --role edge  --dry-run
python vllm/scripts/lwd/split_lwd_checkpoint.py --src $SRC --role cloud --dry-run

# ② 真拆
python vllm/scripts/lwd/split_lwd_checkpoint.py --src $SRC --dst ${SRC}-edge  --role edge  --max-shard-gb 8
python vllm/scripts/lwd/split_lwd_checkpoint.py --src $SRC --dst ${SRC}-cloud --role cloud --max-shard-gb 8
```

**干跑的期望值（一眼就能看出规则对不对）**

| | 期望 | 危险信号 |
|---|---|---|
| 云侧 | `layers.` ≈ 10 万（≈99%），`=> keep` 接近总量，`[size]` ≈ 280 GiB | `=> keep 2347`、`[size] 8.24 GiB` → 命名/规则不对（**首次拆错就是这个样子**） |
| 边侧 | `embed.` 1、`head.` 1、`hc_head` 2~3、tid2eid 正则 = hash 层数；`[size]` 很小 | 边侧 keep 里出现 `layers.`/`mtp.` |

> ⚠️ `[verify] OK` 只证明"**写出的 == 计划的**"，**不证明"计划是对的"**——那次错误的云侧目录同样打印了 `[verify] OK`。所以**必须先看 dry-run 的命中数**。

### 4. 两侧目录里"应该有什么"（DeepSeek 原生命名）

| | 边侧 `${SRC}-edge` | 云侧 `${SRC}-cloud` |
|---|---|---|
| 权重 | `embed.*`、`head.*`、`hc_head*`、`layers.<i>.{ffn,mlp}.gate.tid2eid` | **全部**：`layers.*` + `norm.*` + `embed.*` + `head.*` + `hc_head*` + `mtp.*` |
| 非权重 | 全部（config / tokenizer / chat_template / 量化描述 / yaml…） | `config.json`、`configuration.json`、`*quant*description*.json`、`chat_template.jinja` |
| 体积 | 很小（embed/head/hc_head + 路由表） | ≈ 全量（280G 级） |

> `hc_head_fn` / `hc_head_base` / `hc_head_scale` 是**无条件创建的真参数**（`vllm_ascend/models/deepseek_v4.py:1082-1084`；上游 `vllm/models/deepseek_v4/nvidia/model.py:939-957` 同样无条件），**两侧都必须保留** —— 这是最容易漏的一项，已写进默认规则。

> V4 是纯文本模型，云侧要用到全部层与 embed/lm_head，所以**云侧没有可裁的"脂肪"**——云侧拆分只去掉 tokenizer/生成配置等少量文件。磁盘紧张时可以**只拆边侧**，云侧继续用完整目录（运行期 `lwd_weight_filter` 会在读盘前丢掉用不到的张量）。
> 云侧保留 `chat_template.jinja` 只为兼容启动命令里的 `--chat-template <模型目录>/chat_template.jinja`；云侧不渲染模板。

### 5. 启动与验证

```bash
# 边侧应出现
grep -E "lwd-edge|duplex channels" edge.log
# 云侧应出现
grep -E "\[LWD\] cloud: tokenizer and multimodal encoder are not initialized|LwdCloudEngineCore" cloud.log
# 两侧都不应出现（缺权重 / 量化描述没带上）
grep -c "Following weights were not initialized" edge.log cloud.log   # 期望 0
grep -c "No quantization signature detected"     edge.log cloud.log   # 期望 0
```

### 6. 回退与排错

| 症状 | 处理 |
|---|---|
| 边侧报缺 `model.norm.weight` | norm 守卫没生效：Ascend 上对应 `b189aed65e`（`vllm_ascend/models/deepseek_v4.py`）；若你实际走的是上游实现，对应文件是 `vllm/models/deepseek_v4/{nvidia,xpu}/model.py` |
| 边侧报 `Missing tid2eid tables ...` | 边侧目录是旧脚本拆的；用新脚本重拆，或临时 `--keep-prefix model.layers.<i>.mlp.gate.tid2eid`（每个 hash 层一条） |
| 云侧报 `Can't load image processor` / 缺 tokenizer | 用 `--metadata all` 重新生成云侧目录（或让云侧直接跑完整目录） |
| 云侧按 float 加载了 int8 权重 | 云侧目录缺 `quant_model_description.json` |
| 边侧崩在 `nvidia/model.py` 的 `get_expert_mapping` → `StopIteration` | 说明这次走的是**上游** V4 实现（`ascend_model` 注册没生效）：检查 `VLLM_PLUGINS` 是否排除了它、vllm-ascend 是否为含 `setup.py:545-550` 的版本。Ascend 上正确的实现是 `vllm_ascend/models/deepseek_v4.py` |

---

## 相关文档

- 全链路逐行走读：[边云分侧模型加载逐行走读-NPU路径.md](边云分侧模型加载逐行走读-NPU路径.md)
- 改动逻辑（含图示）：[LWD按角色加载-commit24f7d84f5详解.md](LWD按角色加载-commit24f7d84f5详解.md)
- 加载流程逐行位置：[模型加载全链路-边云分侧.md](模型加载全链路-边云分侧.md)
