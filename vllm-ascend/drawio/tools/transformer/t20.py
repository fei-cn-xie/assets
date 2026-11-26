"""⑳ GGUF 与量化：为什么 Qwen-7B 能塞进 4 GB 的显存。"""
from dg import table
from mth import (ACT, CALC, ERR, NOTE, OUT, W, arrow, legend, panel, title)

TITLE = "⑳ GGUF 与量化：把 fp16 压成 4 bit，省下 3/4 的空间"
SUB = ("前七页讲的都是 fp16 的 HF 权重。但你在本地跑模型时常见的是 4 GB 的 .gguf 文件——"
       "这一页讲清量化在做什么、GGUF 是什么、代价是什么。")


def build(doc) -> None:
    p = doc.page("⑳ GGUF 与量化", pw=1700, ph=1440)
    title(p, TITLE, SUB, w=1620)

    # --- 量化是什么 -------------------------------------------------------
    y = 150
    p.text(36, y - 60, 1200, 22,
           "<b>第一步：量化 = 用更少的比特表示同一批数（不是删参数）</b>",
           "text_b")
    panel(p, 36, y, 520, "fp16：每个数 16 bit",
          "表示范围大、精度高，但占地方。7B 参数 × 2 字节 ≈ <b>14 GB</b>。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "这是 HuggingFace 上的默认格式（⑯ 的 safetensors）。</font>", NOTE)
    panel(p, 576, y, 520, "int8：每个数 8 bit",
          "把连续的浮点值「装进 256 个桶」，再记一个缩放系数。"
          "7B ≈ <b>7 GB</b>，精度损失很小。", CALC)
    panel(p, 1116, y, 548, "4 bit：每个数 4 bit",
          "只有 16 个桶——单个数根本不够用，于是<b>一组数共享一个缩放系数</b>"
          "（这叫 block-wise 量化）。7B ≈ <b>4 GB</b>。", CALC)

    # --- 分块量化的数字 ---------------------------------------------------
    y2 = y + 170
    p.text(36, y2 - 60, 1200, 22,
           "<b>第二步：一个 8 个数的分块，怎么变成 8 个 4bit 整数 + 1 个缩放系数</b>",
           "text_b")
    vals = ["0.42", "-0.11", "0.87", "0.03", "-0.65", "0.21", "-0.02", "0.55"]
    q = ["+6", "-2", "+12", "0", "-9", "+3", "0", "+8"]
    p.box(36, y2, 120, 42, "<b>原始（fp16）</b>", "box_center", NOTE)
    for i, v in enumerate(vals):
        p.box(170 + i * 96, y2, 92, 42, f"<font style=\"font-size:10px\">{v}</font>",
              "box_center", ACT)
    p.box(36, y2 + 56, 120, 42, "<b>4bit 整数</b>", "box_center", NOTE)
    for i, v in enumerate(q):
        p.box(170 + i * 96, y2 + 56, 92, 42,
              f"<font style=\"font-size:10px\">{v}</font>", "box_center", OUT)
    p.box(36, y2 + 112, 120, 42, "<b>缩放系数</b>", "box_center", NOTE)
    p.box(170, y2 + 112, 764, 42,
          "<font style=\"font-size:10px\">"
          "scale = 0.07（这一组里绝对值最大的 0.87 ÷ 12 ≈ 0.07）</font>",
          "box", W)
    p.box(950, y2 + 112, 714, 42,
          "<font style=\"font-size:10px\">还原：真实值 ≈ 整数 × scale</font>",
          "box", CALC)
    p.text(36, y2 + 168, 1628, 40,
           "<font style=\"font-size:9px\" color=\"#666666\">"
           "8 个数 × 4 bit = 32 bit + 1 个 scale（16 bit）= 48 bit；"
           "而 fp16 需要 128 bit。压缩比约 <b>2.7 倍</b>（分块越大越省，"
           "但精度越差——这就是 Q4_K_S / Q4_K_M 之类名字的区别）。</font>",
           "subtitle")

    # --- GGUF -------------------------------------------------------------
    y3 = y2 + 230
    p.text(36, y3 - 60, 1200, 22,
           "<b>第三步：GGUF 是「一个文件装下所有东西」的格式</b>", "text_b")
    rows = [
        ["文件个数", "多个 safetensors + index + config + tokenizer",
         "通常 <b>一个 .gguf</b>"],
        ["元数据", "散在 config.json / tokenizer_config.json 等",
         "全部塞进文件头的 KV 段（层数、头数、词表、模板…）"],
        ["量化", "单独做（如 GPTQ/AWQ 的 4bit 权重 + scales 张量）",
         "格式原生支持几十种量化类型，文件名就写清（Q4_K_M）"],
        ["谁在用", "HuggingFace / vLLM / transformers 生态",
         "llama.cpp / Ollama 生态；vLLM 也能读（部分支持）"],
        ["优点", "生态标准、加载快、可 mmap 切片",
         "单文件好分发；CPU/混合推理友好；体积小"],
        ["代价", "体积大（fp16）", "精度有损；量化内核与硬件绑定，GPU 上未必更快"],
    ]
    table(p, 36, y3, [("维度", 200), ("HF safetensors（⑯⑰）", 700),
                      ("GGUF", 700)], rows, row_h=40)

    # --- 代价 -------------------------------------------------------------
    y4 = y3 + 7 * 40 + 36
    panel(p, 36, y4, 810, "量化到底损失了什么",
          "① <b>数值精度</b>：0.87 变成 12×0.07=0.84，误差 3%；一组里的小数值误差可能到 10%。<br>"
          "② <b>长尾词</b>：罕见词、代码符号、数字推理最容易先崩——"
          "因为它们的权重本来就在分布的边角。<br>"
          "③ <b>不是线性的</b>：4bit 通常只掉 1~3 分（常识问答几乎无损），"
          "但 2bit 会断崖式下降。",
          ERR)
    panel(p, 866, y4, 798, "为什么 GPU 上 4bit 不一定更快",
          "量化的收益是<b>显存</b>（能放更大模型/更长上下文）与<b>带宽</b>"
          "（搬的字节少），不是算力。<br>"
          "解码前要先反量化回 fp16 才能做矩阵乘，这步有额外开销；"
          "所以小 batch 时收益明显，大 batch 时要专门的量化 kernel 才划算。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "vLLM 里有专门的量化层实现（layers/quantization/ 目录下按方案分子目录）。</font>",
          NOTE)

    # --- 代码锚点 --------------------------------------------------------
    y5 = y4 + 200
    p.box(36, y5, 1628, 100,
          "<b>代码锚点</b><br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "GGUF 的解析（把单文件里的元数据/tensor 信息读出来）："
          "vllm/transformers_utils/gguf_utils.py<br>"
          "量化方案的实现（不同 scheme 各自的 Linear 与反量化 kernel）："
          "vllm/model_executor/layers/quantization/ 目录（按方案名分子目录）<br>"
          "加载时怎么选：vllm/model_executor/model_loader/default_loader.py 会根据 "
          "config/权重的量化信息选择对应的 QuantizeMethodBase"
          "</font>", "box", NOTE)

    legend(p, y5 + 132)
