# vllm/model_executor 架构地图 (HEAD 70c00163ffa80)

`vllm/model_executor/` 共 665 个 `.py`。一级：`layers/`(284)、`models/`(298)、`model_loader/`(20)、`kernels/`(51)、`offloader/`(5)、`warmup/`(3)，顶层 `custom_op.py`(353 行)、`parameter.py`(617)、`utils.py`(142)、`__init__.py`(9)。

## 目录总览

| 名称 | 作用 | 规模 | 代表文件（file:line） |
|---|---|---|---|
| `__init__.py` | 仅再导出 `BasevLLMParameter`/`PackedvLLMParameter`(:4-9) | 9 行 | — |
| `parameter.py` | vLLM 参数子类，把 TP 切片/打包放进参数本身 | 617 行 | `BasevLLMParameter`:31, `RowvLLMParameter`:204, `ModelWeightParameter`:233, `PackedvLLMParameter`:353, `SharedWeightParameter`:406 |
| `utils.py` | 权重属性/参数替换/映射 | 142 行 | `set_weight_attrs`:13, `replace_parameter`:47, `get_moe_expert_mapping`:124 |
| `custom_op.py` | `CustomOp` 平台分派 + `PluggableLayer` OOT 替换 | 353 行 | `PluggableLayer`:32, `CustomOp`:103, `dispatch_forward`:174 |
| `kernels/` | 平台无关 kernel 抽象 | 51 | `linear/`(45), `mhc/`(5) |
| `layers/` | 可复用层与算子 | 284 | `linear.py`, `attention/`, `fused_moe/`, `quantization/` |
| `models/` | 架构注册表 + 模型实现 + 接口协议 + 加载适配 | 298 | `registry.py`, `interfaces.py`, `interfaces_base.py`, `utils.py`, `adapters.py` |
| `model_loader/` | 权重下载/迭代/装载，`--load-format` 分派 | 20 | `__init__.py`, `base_loader.py`, `default_loader.py`, `weight_utils.py`, `reload/` |
| `offloader/` | 权重卸载与预取 | 5 | `base.py`, `prefetch.py`, `uva.py` |
| `warmup/` | JIT kernel 预热 / autotune | 3 | `kernel_warmup.py`, `deep_gemm_warmup.py` |

## 模型注册与构建

1. 架构清单 `_TEXT_GENERATION_MODELS`(`models/registry.py:70`)、`_EMBEDDING_MODELS`:223、`_MULTIMODAL_MODELS`:343 等合并为 `_VLLM_MODELS`:677-688，值 = `(模块相对名, 类名)`。
2. `ModelRegistry = _ModelRegistry({...})`:1325-1333，每项为 `_LazyRegisteredModel`(`module_name="vllm.model_executor.models.<x>"`)。`register_model`:945 接受模型类(→`_RegisteredModel.from_model_cls`:788)或 `"mod:Cls"`。
3. 解析：`inspect_model_cls`:1130 / `resolve_model_cls`:1182 → transformers 后端 `_try_resolve_transformers`:1036（读 `auto_map`）→ `_normalize_arch`:1104（`try_match_architecture_defaults`，处理 `convert_type`）→ `_try_*_model_cls`:1024/:1030 → 失败 `_raise_for_unsupported`:991（:696 已下线表、:707 插件表专门报错）。
4. 惰性探测 `_LazyRegisteredModel.inspect_model_cls`:868：先按源文件 hash 查 `VLLM_CACHE_ROOT/modelinfos/*.json`(:811-865)，未命中则 `_run_in_subprocess`:892→1338（`_SUBPROCESS_COMMAND`:694）在子进程 import 模型执行 `_ModelInfo.from_model_cls`，避免主进程初始化 CUDA。
5. `_ModelInfo`:715-765 冻结 dataclass（21 字段），`from_model_cls`:737 逐项调接口判定函数。
6. `supports_*` 判定全在 `interfaces*.py`：`is_text_generation_model`(:135, Protocol `VllmModelForTextGeneration`:114)、`is_pooling_model`(:223, `VllmModelForPooling`:148)、`supports_multimodal`(:457)、`supports_pp`(:682)、`has_inner_state`(:753)、`is_hybrid`(:837)、`supports_transcription`(:1247)、`MixtureOfExperts`(:844)、`SupportsLoRA`(:536)、`SupportsEagle3`(:1333)、`SupportsQuant`(:995，`__new__` 自动抓 `quant_config` 并套 mapper)。
7. 消费者：`config/model.py:566`（决定 runner_type/convert_type；`registry` 属性 :789）、`model_loader/utils.py:184`（经 `_get_model_architecture`:179，随后 `adapters.as_embedding_model`:230 / `as_seq_cls_model`:264 包装）、`platforms/interface.py:598`、`config/speculative.py:924`(draft)。`is_text_generation_model`:1236 / `is_pooling_model`:1244 / `is_multimodal_model`:1252 / `is_pp_supported_model`:1268 均为 `_ModelInfo` 薄封装。
8. 实例化 `model_loader/utils.py:40 initialize_model` → `model_class(vllm_config=..., prefix=...)`:56-63，old-style 走 :65-95。

模型文件组织（文件名=HF 架构名，常含 `XxxModel`+`XxxForCausalLM`+`load_weights`）：
- **dense LLM** `models/llama.py`：`LlamaAttention`:124、`LlamaDecoderLayer`:253、`LlamaModel`:350、`LlamaForCausalLM`:501、`load_weights`:436/:584；pooling 变体 :592/:598 由 adapters 生成。
- **MoE LLM** `models/deepseek_v2.py`：`DeepseekV2MoE`:245、`DeepseekV2MLAAttention`:865、`DeepseekV2ForCausalLM`:1590、`DeepseekV3ForCausalLM`:1725；`models/qwen3_moe.py`：`Qwen3MoeSparseMoeBlock`:137、`Qwen3MoeForCausalLM`:675。
- **多模态** `models/llava.py`：`LlavaMultiModalProjector`:128、`LlavaForConditionalGeneration`:504；`models/qwen2_vl.py`：`Qwen2VisionTransformer`:526、`Qwen2VLForConditionalGeneration`:1177。复合模型用 `SupportsMultiModal`:94 的 `_mark_language_model`(:214)/`_mark_tower_model`:249/`_mark_composite_model`:293。
- **pooling/embedding** `models/bert.py`：`BertPooler`:96（继承 `pooler/seqwise/poolers.py:44 SequencePooler`）、`BertModel`:367、`BertEmbeddingModel`:470、`BertForSequenceClassification`:792。
- **encoder-decoder/Whisper** `models/whisper.py`：`WhisperEncoderLayer`:354、`WhisperDecoderLayer`:400、`WhisperModel`:588、`WhisperForConditionalGeneration`:798；其他 enc-dec 已移除（`registry.py:696-705`）。
- **PP/共享工具** `models/utils.py`：`WeightsMapper`:44、`AutoWeightsLoader`:117（`load_weights`:342，`_groupby_prefix`:161、`_load_module`:261、`_load_param`:199、`_can_skip`:189）、`init_vllm_registered_model`:359(:379 调 `initialize_model`)、`PPMissingLayer`:607、`make_layers`:620（`get_pp_indices` 求 start/end，:644-650 拼 `ModuleList`，本 rank 外的层用 `PPMissingLayer`，真实层经 `get_offloader().wrap_modules`）、`get_pp_missing_layer_names`:659 / `is_pp_missing_parameter`:677、`StageMissingLayer`:509 + `no_init_weights`:562（`--mm-encoder-only` 省显存）。

## 层与算子

`layers/` 一级（`__init__.py` 为空）：`attention/`(9)、`fla/`(18)、`fused_moe/`(80 py/397 文件)、`mamba/`(24)、`pooler/`(13)、`quantization/`(102 py/321 文件)、`rotary_embedding/`(20)，及文件 `activation.py`、`attention_layer_base.py`(`AttentionLayerBase`:12)、`batch_invariant.py`、`conv.py`(`Conv2dLayer`:110)、`deepseek_compressor.py`(`DeepseekCompressor`:177)、`deepseek_v4_attention.py`(`DeepseekV4MLAAttention`:616)、`kda.py`(`KimiDeltaAttention`:86)、`layernorm.py`、`lightning_attn.py`、`linear.py`、`logits_processor.py`、`mhc.py`(`MHCPreOp`:13)、`mla.py`(`MultiHeadLatentAttentionWrapper`:34)、`resampler.py`(`Resampler2`:210)、`sparse_attn_indexer.py`(`SparseAttnIndexer`:405)、`utils.py`(`default_unquantized_gemm`:92)、`vocab_parallel_embedding.py`。**`layers/sampler.py` 不存在**（采样在 `vllm/v1/sample/`）。

- **`linear.py`**：`LinearMethodBase`:142、`UnquantizedLinearMethod`:183、`LinearBase`:235、`ReplicatedLinear`:296、`ColumnParallelLinear`:414、`MergedColumnParallelLinear`:611、`QKVParallelLinear`:979、`RowParallelLinear`:1396。**无 `QKVCrossParallelLinear`**。TP 不靠 `partition_dim/stride`（0 处），而是 `output_partition_sizes`(:459/:461-463/:1045) 传给 `quant_method.create_weights(..., output_partition_sizes=...)`(:481-484/:1468-1471)，loader 用前缀和 `shard_offset = sum(self.output_sizes[:id])`(:812/:957-960)。loader：`:369/:537/:696/:1189/:1500`，v2 变体 `:574/:912/:1143/:1535`，合并 ckpt `_load_fused_module_from_checkpoint`:872/:1094。量化选择 :275-279；执行 `self.quant_method.apply(...)`:398/:589/:1560。
- **`attention/`**：9 文件，`Attention`:177（`AttentionLayerBase` 子类）。**本目录无 `selector.py`/`backend.py`/`mla/`**——后端选择在 `vllm/v1/attention/selector.py`(`AttentionSelectorConfig`:21 含 `kv_cache_dtype`:24/`use_mla`:26/`use_per_head_quant_scales`:30；`get_attn_backend`:52)，由 `attention/attention.py:299` 调用。MLA 在 `mla_attention.py`(`MLAAttention`:321, 后端 :385, `MLACommonImpl`:1933)；KV-cache 量化是 selector 输入而非独立文件。实现位于 `v1/attention/backends/*`(`v1/attention/backend.py` 定义 `AttentionBackend`:55、`MLAAttentionImpl`:843)。其他：`ChunkedLocalAttention`:81、`CrossAttention`:189、`EncoderOnlyAttention`:51、`StaticSinkAttention`:116、`MMEncoderAttention`:216。融合 pass 在 `compilation/passes/fusion/`。
- **`fused_moe/`**：`FusedMoE(PluggableLayer)`(`layer.py:73`；该文件管模块/权重/sharding)，kernel 与启发式在 `fused_moe.py`(`fused_experts`:1587)，modular 抽象在 `modular_kernel.py`(`FusedMoEExperts`:465、`FusedMoEExpertsModular`:763、`FusedMoEPrepareAndFinalize`:181、`FusedMoEKernel`:1500)。子目录 `experts/`(26 py：flashinfer_cutlass/triton/cutlass/deep_gemm/marlin/trtllm*/cpu/xpu/fallback)、`router/`、`runner/`(`shared_experts.py`)、`oracle/`、`prepare_finalize/`、`configs/`(JSON 调优)。TP/EP：`tp_size`/`ep_size`(`layer.py:120-121`)、`moe_parallel_config`:183、`intermediate_size_per_partition`:273-274、`ExpertMapManager`:256、`update_expert_map_info`:522-530。量化 MoE 方法继承 `FusedMoEMethodBase`(`fused_moe_method_base.py:30`)。
- **`quantization/`**：`QuantizationMethods`(`__init__.py:12-45`：awq/fp8/gptq_marlin/awq_marlin/gptq/compressed-tensors/bitsandbytes/modelopt*/quark/mxfp4/torchao/inc/humming/fp_quant/online…)、`QUANTIZATION_METHODS`:46、`register_quantization_config`:58、`get_quantization_config`:108（方法→Config 表 :144-171）。抽象 `base_config.py`(`QuantizeMethodBase`:19、`QuantizationConfig`:70)。代表：`fp8.py`(:100/:261/:565)、`awq_marlin.py`(:153/:364/:495)、`gptq_marlin.py`(:97/:334/:495)、`compressed_tensors/`(34 文件, `compressed_tensors.py:79`)、`modelopt.py`(:131)、`mxfp4.py`(:41/:136/:470)、`quark/`(`quark.py:49`, `QuarkLinearMethod`:671)。选择链路 `config/model.py:950` → `config/vllm.py:537`（校验 `get_min_capability`/`get_supported_act_dtypes`）→ `model_loader/weight_utils.py:263` → `quant_cls.from_config`:318。**无 `_VLLM_*` 量化映射表**。
- **`rotary_embedding/`**：20 文件，工厂 `get_rope`(`__init__.py:33`, 缓存 `_ROPE_DICT`:30) 分派 `base.py`(`RotaryEmbeddingBase`:15/`RotaryEmbedding`:118)、`llama3_rope.py`、`yarn_scaling_rope.py`、`deepseek_scaling_rope.py`、`mrope.py`、`xdrope.py`、`fope.py` 等约 16 种缩放类型。
- **归一化/激活**：`layernorm.py`(`RMSNorm`:38、`GemmaRMSNorm`:133、`RMSNormGated`:183、`LayerNorm`:304)；`activation.py`(`SiluAndMul`:118、`GeluAndMul`:322、`GELU`:294、注册表 `get_act_fn`:736/`get_act_and_mul_fn`:764)，均基于 `CustomOp` 做分派。
- **`pooler/`**：`abstract.py:16 Pooler`；seqwise(`SequencePooler`:44、`SequencePoolerHead`:19、`EmbeddingPoolerHead`:33、`ClassifierPoolerHead`:102、`CLSPool`:36/`LastPool`:50/`MeanPool`:60)；tokwise(`TokenPooler`:48、`TokenPoolerHead`:20、`AllPool`:35/`StepPool`:83)；`special.py`(`DispatchPooler`:25、`BgeM3Pooler`:198)；`activations.py`(`PoolerClassify`:112)。
- **embedding/logits**：`vocab_parallel_embedding.py`(`VocabParallelEmbedding`:192、`ParallelLMHead`:503、`UnquantizedEmbeddingMethod`:35)；`logits_processor.py`(`LogitsProcessor`:19，LM head all-gather + 采样元数据)。

## 权重加载

`--load-format`（`model_loader/__init__.py:31-47` `LoadFormats`；映射 `_LOAD_FORMAT_TO_MODEL_LOADER`:48-64）：

| load_format | Loader (file:line) | 说明 |
|---|---|---|
| `auto`/`hf`/`safetensors`/`pt`/`npcache`/`fastsafetensors`/`instanttensor`/`mistral` | `DefaultModelLoader`(`default_loader.py:43`) | HF 下载 + safetensors/pt 迭代 |
| `bitsandbytes` | `BitsAndBytesModelLoader`(:56) | 加载时在线 4/8bit 量化 |
| `gguf` | `GGUFModelLoader`(`gguf_loader.py:38`) | GGUF 单文件，覆写 `load_model`:413 |
| `tensorizer` | `TensorizerLoader`(:43) | 序列化张量流式加载，`save_model`:141 |
| `sharded_state` / `runai_streamer_sharded` | `ShardedStateLoader`(:29) | 每 rank 分片 + pattern |
| `runai_streamer` | `RunaiModelStreamerLoader`(:21) | S3/GCS/Azure 流式 |
| `dummy` | `DummyModelLoader`(:22) | 随机权重（性能测试） |

`register_model_loader`:67 可扩展；`get_model_loader`:120 查表（未知格式 :123-124 报错）；`get_model`:128。

默认流程：`BaseModelLoader.load_model`(`base_loader.py:42-82`) → 解析设备(:49-52) → `set_default_torch_dtype`(:53) → **`initialize_model`**(`utils.py:40`；:51 `get_model_architecture`、:53-54 `configure_quant_config`、:56-63 构造) → `self.load_weights`:64（`default_loader.py:382-412`，:394 `model.load_weights(self.get_all_weights(...))`）→ 在线量化 `finalize_layerwise_processing`(:77-78) → **`process_weights_after_loading`**(`utils.py:99-127`：:102-111 遍历 `QuantizeMethodBase`；:115-122 处理 `Attention`/`MLAAttention`/`MMEncoderAttention`) → `model.eval()`(:82)。**`device_loading_context` 在 `model_loader/utils.py:131-172`**：临时把 CPU 参数搬到目标设备执行后还原，并重新 pin/UVA 回卸载。

`weight_utils.py`(1629 行)：`atomic_writer`:125、`get_quant_config`:263、`download_weights_from_hf`:504、`filter_duplicate_safetensors_files`:655、`np_cache_weights_iterator`:708、`safetensors_weights_iterator`:893、`multi_thread_safetensors_weights_iterator`:1030、`fastsafetensors_weights_iterator`:1109、`pt_weights_iterator`:1206、`default_weight_loader`:1383、`row_parallel_weight_loader`:1404、`sharded_weight_loader`:1422、`composed_weight_loader`:1437、`initialize_dummy_weights`:1450、`maybe_remap_kv_scale_name`:1526。**无 `WeightsIterator` 与 `hf_model_weights_iterator`**（后者已拆分）。可选依赖 `PlaceholderModule` 兜底：`runai_model_streamer`:51-55、`gguf`:57-60、`fastsafetensors`:62-67，tensorizer `tensorizer.py:35-54`，bitsandbytes 版本门 `>=0.46.1`(:201-214)。

`tensorizer` 把权重序列化成可 mmap 的 tensor 文件（`tensorizer_loader.py:68-87`、覆写 `load_model`:115-139）；`sharded_state` 让每个 TP/PP rank 只读自己的分片（`sharded_state_loader.py:29`，`DEFAULT_PATTERN`:38，`_filter_subtensors`:56，S3 感知 `_prepare_weights`:94），常与 `runai_streamer_sharded` 搭配。`ep_weight_filter.py` 跳过非本地专家 I/O（`default_loader.py:318-379`）。`reload/`(7 文件) 做 layerwise 重载，明确不与 CPU offloading 组合(`reload/__init__.py:10`)。

## 平台分派与预热

- `CustomOp`(`custom_op.py:103`)：`__new__`:109 查 `op_registry_oot`:22 做 OOT 替换；`__init__`:130 调 `dispatch_forward`:174 缓存 `_forward_method`。分支 `is_rocm()`→`forward_hip`(:196/:149，**无 `forward_rocm`**)、`is_cpu()`→`forward_cpu`(:198/:158)、`is_tpu()`→`forward_tpu`(:200/:163)、`is_xpu()`→`forward_xpu`(:202/:153)、`is_out_of_tree()`→`forward_oot`(:204/:169)、否则 `forward_cuda`(:207/:146)；`forward_native`:138 为参考实现兼默认回退。
- 开关由 `CompilationConfig.custom_ops` 决定：`enabled`:272（`+name`/`-name`）、`default_on`:292（`all`/`none`），并回写 :187-189；未启用则 `maybe_compile(forward_native)`:194。`PluggableLayer`:32 只做整类替换。
- **`dispatch_key` 不在 `custom_op.py`**：它是平台属性（`platforms/interface.py:113`、`cuda.py:162`、`rocm.py:395`、`xpu.py:36`），在 `vllm/utils/torch_utils.py:937/:958-967` 的 `direct_register_custom_op` 中使用；调用点 `layers/attention/mla_attention.py:1109`、`layers/sparse_attn_indexer.py:400`、`layers/quantization/fp_quant.py:281` 等、`quantization/bitsandbytes.py:427`。
- 预热 `warmup/kernel_warmup.py:27 kernel_warmup(worker)`：DeepGEMM 门控(:29-33)→`deep_gemm_warmup`(:37，实现 `deep_gemm_warmup.py:363-375`)；FlashInfer autotune(:39-46→:81-109)；FlashInfer attention 预热(:51-78，仅非 pooling 且所有 attn group 后端均为 `FLASHINFER`，跑 `_dummy_run(num_tokens=16, create_mixed_batch=True)`)。唯一调用点 `v1/worker/gpu_worker.py:608`（dummy-run 之后、CUDA graph 捕获之前）。`warmup/__init__.py` 为空。
- offloader：`BaseOffloader`(`base.py:47`,`wrap_modules`:55)、`NoopOffloader`:96、`get_offloader`:111/`set_offloader`:116、`create_offloader`:126-162（按 `offload_group_size`/`cpu_offload_gb` 选 prefetch/UVA/Noop）。`PrefetchOffloader`(`prefetch.py:127`)：按 group 选层、`StaticBufferPool`:60 双缓冲、专用 copy stream、`post_init`:310-365 在 `process_weights_after_loading` 后重建 CPU 存储；CUDA-graph 安全 op 在 `prefetch_ops.py`。`UVAOffloader`(`uva.py:21`) 按字节预算把参数留在 pinned CPU。调用方 `v1/worker/gpu_model_runner.py:868`/:5092、`models/utils.py:646`、`v1/worker/gpu_ubatch_wrapper.py:275/:291`、`compilation/cuda_graph.py:310/:324`。

## 与其他子系统的接口

**调用 model_executor 的方**：
- `v1/`：`model_loader` 10 处 import；入口 `v1/worker/gpu_model_runner.py:4940-4941`、`:5151`(reload)、`v1/worker/gpu/model_runner.py:271-274`、`v1/worker/cpu_model_runner.py:105`、`v1/worker/gpu_worker.py:955`(`ShardedStateLoader`)/:608(`kernel_warmup`)；draft/EAGLE `v1/spec_decode/{draft_model.py:74, medusa.py:61, extract_hidden_states.py:354}`。`models/interfaces*` 11 处；`layers/attention/mla_attention` 15 处；`attention_layer_base` 9 处。
- `lora/`：`layers/linear`(5)、`custom_op`(4)、`layers/fused_moe`(3)、`vocab_parallel_embedding`(3)、`models/utils`(3)——LoRA 通过子类化这些层实现。
- `compilation/`：`layers/quantization/utils/quant_utils`(8)、`layers/attention`(3)、`rotary_embedding`(3)、`layers/mamba/*`；fusion pass 直接改写 attention/rope 层。
- 其他：`entrypoints/`(`model_loader` 4 处)、`distributed/`(`models/extract_hidden_states`、`layers/mamba/mamba_utils`、EPLB 用 `MixtureOfExperts`)、`config/`(`model.py:566`、`speculative.py:924`、`model_loader/weight_utils`)、`platforms/interface.py:598`。

**model_executor 依赖的方**（包内 import 统计）：`vllm.config`(332)、`vllm.distributed`(203)、`vllm.platforms`(153)、`vllm.compilation.decorators`(135)、`vllm.config.multimodal`(81)、`vllm.v1.attention.backends`(59)、`vllm.v1.attention.backend`(45)、`vllm.distributed.parallel_state`(26)、`vllm.v1.kv_cache_interface`(16)、`vllm.transformers_utils.*`(32)、`vllm.v1.attention.ops`(12)、`vllm.v1.pool.metadata`(11)、`vllm.distributed.eplb.eplb_state`(10)。即配置/分布式/平台是基础设施，v1 的 attention/pool 元数据是契约。

**`_custom_ops` 边界**：model_executor 内 18 个文件 `from vllm._custom_ops import ...`；反向仅 `vllm/_custom_ops.py:663/:696/:915` 三处在函数体内惰性 import `layers.quantization.awq_triton` 与 `compressed_tensors.triton_scaled_mm`（避免循环导入）。`parameter.py`/`utils.py` 为纯 Python，不触 csrc；`layers/*` 经 `current_platform` 与 `direct_register_custom_op` 触达算子（csrc 细节不在本次范围）。

**任务前提更正（已核实不存在或命名不同）**：`model_loader/loader.py`、`layers/sampler.py`、`layers/attention/selector.py`、`layers/attention/backend.py`、`layers/attention/mla/`、`QKVCrossParallelLinear`、`CustomOp.forward_rocm`、`custom_op.dispatch_key`、`WeightsIterator`、`hf_model_weights_iterator`、量化 `_VLLM_*` 映射表、`layers/{moe, mlp.py, embedding/, minimax_rms_norm.py}` 均不存在；`device_loading_context` 位于 `model_loader/utils.py:131`。
