# GLM-5.2(w4a8c8) 边云拆分与运行手册

分支:`vllm` / `vllm-ascend` 均为 **`model_load_update2`**(基线 `prefill_only_glm_dp2`)。
原则:**边侧只放边侧需要的文件,云侧只放云侧需要的文件;拆分后的目录与完整目录都必须能正常起服务。**

---

## 1. 这个 checkpoint 的关键事实(都已从代码查到出处)

| 项 | 事实 | 出处 |
|---|---|---|
| 架构 | GLM-5.2 是 `glm_moe_dsa`,`GlmMoeDsaForCausalLM` **就是** `DeepseekV2ForCausalLM` 的空子类 → 层与尾部 norm 都在 `DeepseekV2Model` | `vllm/.../models/deepseek_v2.py:1750-1751`、`:1253 make_layers`、`:1263-1266 norm` |
| MTP 层 | 表现为"层号续接"的额外层(`model.layers.<num_hidden_layers+i>.*`),云侧 `model.layers.` 已覆盖;边侧丢弃 | `deepseek_v2.py:1756-1769`、`patch_deepseek_mtp.py:67`(`model.layers.<spec_layer>.rot.weight`) |
| `rot.safetensors` / `optional/quarot.safetensors` | QuaRot 旋转矩阵,**只在带 EAGLE3 drafter 时**被读取(drafter 在云侧):路径来自 `quant_model_description.json` 的 `optional.quarot.rotation_map.global_rotation` | `vllm-ascend/.../patch/worker/patch_draft_quarot.py:42-53`、调用点 `worker/model_runner_v1.py:3766-3769` |
| `quant_model_description.json` | W4A8C8 量化判据,**每侧都必须有**,缺了会被当 float 加载 | `vllm-ascend/quantization/utils.py:114-117`、`modelslim_config.py:67/793` |
| 索引名 | `quant_model_weights.safetensors.index.json`(非标准名)→ 脚本自动识别;输出写 vLLM 认的 `model.safetensors.index.json` | `vllm/.../model_loader/default_loader.py:120/181-204`、`weight_utils.py:655-673` |
| 伴随 safetensors 的坑 | vLLM 对本地目录是"glob 所有 `*.safetensors` 再用索引过滤";**没有索引时 `rot.safetensors` 会被当权重文件读** → 拆分目录只要含伴随文件就**必须**写索引(脚本已处理) | 同上 |

## 2. 代码改动(本分支新增的两个提交)

| 提交 | 内容 | 影响面 |
|---|---|---|
| `7f1555828` | 拆分脚本重写:**零拷贝字节直写**(不再 materialize torch 张量)、`--jobs N` 多进程、索引名自动识别、伴随 safetensors 分角色拷贝、**源目录只读校验(默认开)**、确定性输出、进度合并 | 仅 `scripts/lwd/split_lwd_checkpoint.py`;不涉及模型加载路径 |
| `f2d72ce89` | `deepseek_v2.py` 尾部 norm 加 `lwd_skips_module` 守卫(边侧不建 norm) | `DeepseekV2Model` 同时服务 DeepSeek-V2/V3 与 GLM-5.x DSA;`skips_module` 只在 `enabled and mode=="prefill_only"` 时为真 → 非 LWD 场景惰性 |

> 边侧不建 norm ⇒ **边侧目录不需要 `model.norm.weight`**;不拆(用完整目录)时由 `lwd_weight_filter` 在读盘前丢弃它,所以两种用法都能起。

## 3. 拆分命令

```bash
SRC=/home/extra/GLM-5.2-w4a8c8

# ① 干跑:只看命中数与计划(不写任何文件)
python vllm/scripts/lwd/split_lwd_checkpoint.py --src $SRC --role edge  --dry-run
python vllm/scripts/lwd/split_lwd_checkpoint.py --src $SRC --role cloud --dry-run

# ② 真拆(两端都拆;--dst 建议放另一块盘,读写不抢带宽)
python vllm/scripts/lwd/split_lwd_checkpoint.py --src $SRC --dst ${SRC}-edge  --role edge  --max-shard-gb 8 --jobs 8
python vllm/scripts/lwd/split_lwd_checkpoint.py --src $SRC --dst ${SRC}-cloud --role cloud --max-shard-gb 8 --jobs 8
```

`--jobs 0`(默认)取 `min(4, CPU 数)`;多进程不可用的环境(容器/seccomp 禁匿名管道)会自动回退单进程并打印 `[warn]`,结果一致。

**干跑期望值(判断规则对不对)**

| | 期望 | 危险信号 |
|---|---|---|
| 云侧 | `model.layers.` ≈ 全量(≈99%),`model.norm.` 1、`model.embed_tokens.` 1、`lm_head.` 1、MTP 层含在 `model.layers.` 里;`[size]` ≈ 378G | `=> keep` 只有几千个 / `[size]` 只有几 G → 命名布局不匹配(先看每条规则的命中数) |
| 边侧 | `model.embed_tokens.` 1(+量化 scale)、`lm_head.` 1;`[size]` 很小 | 边侧 keep 里出现 `model.layers.`/`model.norm.`/`mtp.` |

> `[verify] OK` 只证明"写出的 == 计划的",**不证明计划是对的**;判断标准是干跑的命中数。

## 4. 两侧目录里"应该有什么"

| | 边侧 `${SRC}-edge` | 云侧 `${SRC}-cloud` |
|---|---|---|
| 权重 | `model.embed_tokens.*`、`lm_head.*`(+ 量化 scale/offset,同前缀自动跟随) | **全部**:`model.layers.*`(含 MTP 层与 Indexer 权重)、`model.norm.*`、`model.embed_tokens.*`、`lm_head.*` |
| 非权重 | 全部文件 + `optional/`(tokenizer/chat_template/量化描述/最佳实践 yaml…) | `config.json`、`configuration.json`、`quant_model_description.json`、`chat_template.jinja`、`rot.safetensors`、`optional/` |
| 不拷 | — | `tokenizer*`、`generation_config.json`、`README.md`、`gitattributes`、`__pycache__`、`modelscope.py`、`GLM-5.2_best_practice.yaml` |
| 体积 | 很小 | ≈ 全量(378G 级) |

## 5. 启动与验证

```bash
# 边侧应出现(沿用你们现有脚本;注意 --chat-template 指向的目录要有 chat_template.jinja)
grep -E "lwd-edge|duplex channels" edge.log
# 云侧应出现
grep -E "\[LWD\] cloud: tokenizer and multimodal encoder are not initialized|LwdCloudEngineCore" cloud.log
# 两侧都不应出现
grep -c "Following weights were not initialized" edge.log cloud.log   # 期望 0
grep -c "No quantization signature detected"     edge.log cloud.log   # 期望 0(出现=量化描述没带上)
```

拆分脚本自身还会打印 `[guard] 源目录只读校验通过(N 个文件快照一致)`;若源目录被任何进程改动,脚本会以退出码 **6** 失败并列出变化项。

## 6. 排错

| 症状 | 处理 |
|---|---|
| 边侧报缺 `model.norm.weight` | norm 守卫没生效(确认 `deepseek_v2.py` 里 `lwd_skips_module` 在):或边侧目录是用旧脚本拆的 |
| 云侧报 `Can't load image processor` / 缺 tokenizer | 云侧 vllm 未含 LWD 补丁 → 用 `--metadata all` 重新生成云侧目录 |
| 云侧被按 float 加载(int4/int8 权重) | 云侧目录缺 `quant_model_description.json` |
| 用 EAGLE3 时旋转矩阵加载失败 | 云侧目录要有 `rot.safetensors` 与 `optional/`(规则里已点名;或用 `--metadata all`) |
| 拆分很慢 | `--dst` 换到另一块盘;调大 `--max-shard-gb`;`--jobs` 提到 CPU 数 |
| 多进程起不来 | 自动回退单进程(打印 `[warn]`);要强制并行可 `--jobs N`,`--jobs 1` 便于排查 |

## 7. 验证状态

| 项 | 状态 |
|---|---|
| 拆分脚本端到端(合成 GLM 风格 fixture:非标准索引名 + 伴随 safetensors + 2 分片) | ✅ 输出张量**逐字节**等于源;多分片索引正确;单分片+伴随仍写索引;`--dst==--src` 拒绝;源目录快照不变;无索引回退可用;`jobs=1`/`jobs=3` 输出一致 |
| V4/GLM 双布局 dry-run 套件、进度组件套件 | ✅ OK |
| 与旧版对拍(Qwen3.6 `model.*` / Qwen3-VL `model.language_model.*`) | ✅ 保留集合完全相同;metadata 仅边侧少一个 `gitattributes`(git 产物,刻意跳过)、云侧多 `configuration.json`+`chat_template.jinja`(刻意补充) |
| **多进程真并行** | ⚠️ 开发沙箱禁止匿名管道,本机走的是"回退单进程"路径;请在 Linux 上跑 `--jobs 8`,并用 `md5sum` 与 `--jobs 1` 的输出对比一次 |
| NPU 上真实起服务 | ⏳ 待你们上机验证(拆分目录 + 完整目录各起一次) |
