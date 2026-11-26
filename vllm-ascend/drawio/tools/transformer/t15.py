"""⑮ config.json 逐字段：每个数字都在后面某一页用过。"""
from dg import table
from mth import (ACT, CALC, ERR, NOTE, OUT, W, legend, panel, title)

TITLE = "⑮ config.json：十几个数字，决定整个网络的形状"
SUB = ("这份不到 1 KB 的 JSON 是所有形状的来源。本页把每个字段翻译成"
       "「它决定了哪一页里的哪个矩阵有多大」。")


def build(doc) -> None:
    p = doc.page("⑮ config.json", pw=1700, ph=1460)
    title(p, TITLE, SUB, w=1620)

    # --- 两个真实配置对照 -------------------------------------------------
    y = 150
    p.text(36, y - 60, 1200, 22,
           "<b>先看两个真实模型的 config（同名字段，值不同 → 网络完全不同）</b>",
           "text_b")
    rows = [
        ["architectures", "LlamaForCausalLM", "Qwen2ForCausalLM", "用哪个模型类"],
        ["hidden_size（d_model）", "4096", "3584", "每层宽度（③⑤ 的列数）"],
        ["num_hidden_layers", "32", "28", "层数（⑬ 堆几层）"],
        ["num_attention_heads", "32", "28", "Q 的头数（⑩）"],
        ["num_key_value_heads", "32", "4", "K/V 的头数（GQA，⑩）"],
        ["head_dim（隐式）", "4096/32 = 128", "3584/28 = 128", "每头维度（⑦ 的 d）"],
        ["intermediate_size", "11008", "18944", "FFN 中间宽度（⑫）"],
        ["vocab_size", "32000", "152064", "词表大小（②③⑬）"],
        ["max_position_embeddings", "4096", "32768", "最长上下文（④）"],
        ["rope_theta", "10000", "1000000", "RoPE 频率底数（④）"],
        ["rms_norm_eps", "1e-6", "1e-6", "归一化防零（⑪）"],
        ["tie_word_embeddings", "false", "false", "嵌入与输出头是否共享（③）"],
        ["torch_dtype", "float16", "bfloat16", "权重精度（⑯⑳）"],
    ]
    table(p, 36, y, [("字段", 300), ("Llama-2-7B", 200), ("Qwen2.5-7B", 200),
                     ("它决定什么", 400)], rows, row_h=32)

    # --- 三个"隐式"字段 ---------------------------------------------------
    y2 = y + 32 * 14 + 46
    p.text(36, y2 - 60, 1200, 22,
           "<b>三个「不写出来但一定存在」的量（必须自己算）</b>", "text_b")
    panel(p, 36, y2, 520, "① head_dim = hidden_size / num_heads",
          "Llama-2-7B：4096/32 = 128。<br>"
          "注意有些模型（如 Qwen2 某些尺寸）会显式写 head_dim，"
          "此时它<b>不一定</b>等于 hidden/heads——这时要以字段为准。", CALC)
    panel(p, 576, y2, 520, "② n_kv_groups = num_heads / num_kv_heads",
          "Llama-2：32/32 = 1（普通 MHA）。Qwen2.5-7B：28/4 = 7（每 7 个 Q 头共用 1 份 K/V，GQA）。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "KV cache 的显存 ∝ num_kv_heads，所以这个比值直接决定能放多少并发。</font>", CALC)
    panel(p, 1116, y2, 548, "③ ffn 倍数 = intermediate_size / hidden_size",
          "Llama-2：11008/4096 ≈ 2.69（不是整 4 倍！）。Qwen2.5：18944/3584 ≈ 5.29。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "老教程常说「FFN 是 4 倍」，那是原始 Transformer 的设置；"
          "现代模型为了对齐总参数量会取非整数倍。</font>", NOTE)

    # --- 用 config 反推参数量 ---------------------------------------------
    y3 = y2 + 200
    p.text(36, y3 - 60, 1200, 22,
           "<b>拿这几个数字反推参数量（手算一遍，你就懂「7B」是怎么来的）</b>",
           "text_b")
    rows2 = [
        ["每层注意力", "q:4096×4096 + k:4096×4096 + v:4096×4096 + o:4096×4096",
         "≈ 6700 万"],
        ["每层 FFN（SwiGLU）", "gate:4096×11008 + up:4096×11008 + down:11008×4096",
         "≈ 1.35 亿"],
        ["每层小计", "注意力 + FFN + 两个 RMSNorm（各 4096 个参数）", "≈ 2.02 亿"],
        ["32 层", "2.02 亿 × 32", "≈ 64.6 亿"],
        ["词嵌入（32000×4096）", "与输出头若共享则只算一次", "≈ 1.3 亿"],
        ["<b>合计</b>", "≈ 65.9 亿 → 四舍五入叫 <b>「7B」</b>", "≈ 6.6 B"],
    ]
    t2 = table(p, 36, y3, [("项", 260), ("算式", 700), ("结果", 180)], rows2,
               row_h=34)

    p.box(36, t2 + 40, 1628, 118,
          "<b>为什么这个手算值得做</b><br>"
          "① 你能预测权重文件的<b>总大小</b>：6.6 B × 2 字节（fp16）≈ 13.2 GB，"
          "正好对上下载到的 13~14 GB。<br>"
          "② 你能预测<b>显存占用</b>：权重 13 GB + KV cache（随并发/上下文增长）+ 激活值。<br>"
          "③ 你能判断一个模型「是不是真的那么大」——市面上虚标的模型，"
          "用 config 一算就露馅。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "注意：算出来的是<b>参数个数</b>，不是字节数；字节数还要乘精度（⑯⑳）。</font>",
          "box", NOTE)

    # --- 代码锚点 --------------------------------------------------------
    y4 = t2 + 182
    p.box(36, y4, 1628, 96,
          "<b>代码锚点</b><br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "config 的解析入口：vllm/transformers_utils/config.py（get_config）<br>"
          "vLLM 自己读这些字段的地方：vllm/model_executor/models/llama.py:347"
          "（LlamaModel.__init__ 里 config.hidden_size / num_hidden_layers / "
          "num_attention_heads 等）<br>"
          "线性层的维度就是这么来的：:165（qkv_proj 用 head 数与 head_dim 推出总输出维）"
          "、:95（gate_up_proj 用 intermediate_size）"
          "</font>", "box", NOTE)

    legend(p, y4 + 132)
