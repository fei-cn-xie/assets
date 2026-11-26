"""⑭ 模型文件结构（总览）：一个模型目录里到底有什么。"""
from dg import table
from mth import (ACT, CALC, ERR, NOTE, OUT, W, arrow, legend, panel, title)

TITLE = "⑭ 一个模型目录里到底有什么（先看全景，再逐个拆）"
SUB = ("你从 HuggingFace 下载到的「一个模型」，其实是一个装着 10 来个文件的文件夹。"
       "这一页把所有文件摆出来，说明各自干嘛、谁在读它。")


def build(doc) -> None:
    p = doc.page("⑭ 目录全景", pw=1700, ph=1400)
    title(p, TITLE, SUB, w=1620)

    # --- 文件表格 ---------------------------------------------------------
    y = 150
    p.text(36, y - 60, 1200, 22,
           "<b>一个 7B 模型（fp16，约 14 GB）的典型目录</b>", "text_b")
    rows = [
        ["config.json", "structure",
         "结构说明书：几层、多宽、几个头、词表多大", "~1 KB",
         "加载器读它才知道怎么搭网络（⑮）"],
        ["model.safetensors", "weight",
         "权重本体（这里只有一个文件；大模型会切成 4~10 个分片）", "3~14 GB",
         "真正的几十亿个浮点数（⑯⑰）"],
        ["model.safetensors.index.json", "weight",
         "分片索引：哪个张量在哪个文件里", "~100 KB", "分片时才有（⑰）"],
        ["tokenizer.json", "text", "分词器的完整规则（BPE 合并表）", "~1~10 MB",
         "文本 → id（②⑲）"],
        ["tokenizer_config.json", "text",
         "分词器配置：特殊 token、大小写、模板名", "~2 KB", "⑲"],
        ["special_tokens_map.json", "text", "BOS/EOS/PAD 与 id 的对应表", "~1 KB", "⑲"],
        ["generation_config.json", "text", "默认生成参数：温度、top-p、最大长度",
         "~1 KB", "⑬ 里那些默认值的来源"],
        ["chat_template.jinja（可内嵌）", "text",
         "对话模板：system/user/assistant 怎么拼成一段文本", "~2 KB", "⑲"],
        ["vocab.json + merges.txt", "text", "旧式分词器（GPT-2 系）的两份表",
         "~1 MB", "⑲"],
        ["LICENSE / README / *.md", "text", "许可与说明", "~10 KB",
         "不看也能跑，但要遵守许可"],
    ]
    cy = y
    for name, kind, what, size, who in rows:
        role = W if kind == "weight" else (ACT if kind == "text" else NOTE)
        p.box(36, cy, 330, 42, f"<b>{name}</b>", "box", role)
        p.box(378, cy, 520, 42, f"<font style=\"font-size:10px\">{what}</font>",
              "box", NOTE)
        p.box(910, cy, 110, 42, f"<font style=\"font-size:10px\">{size}</font>",
              "box_center", NOTE)
        p.box(1032, cy, 632, 42, f"<font style=\"font-size:10px\">{who}</font>",
              "box", NOTE)
        cy += 50

    # --- 三类文件 ---------------------------------------------------------
    y2 = cy + 30
    p.text(36, y2 - 60, 1200, 22,
           "<b>记住这个三分法（后面三页就按这个顺序讲）</b>", "text_b")
    panel(p, 36, y2, 520, "① 结构（config）",
          "决定「网络长什么样」：层数、宽度、头数、词表大小。<br>"
          "它很小，但<b>没有它就无法解释权重</b>——同一个 safetensors 文件，"
          "按不同 config 解释会得到完全不同的模型。<br>"
          "<font style=\"font-size:9px;color:#c0392b\">"
          "常见事故：config 被改错，模型能加载但输出乱码。</font>", CALC)
    panel(p, 576, y2, 520, "② 权重（safetensors）",
          "决定「参数是多少」：几十亿个 fp16 浮点，按名字分组存放。<br>"
          "它占了 99.9% 的体积，也是量化、分片、并发加载的战场（⑯⑰⑱）。", W)
    panel(p, 1116, y2, 548, "③ 文本处理（tokenizer 等）",
          "决定「文本怎么变成 id、id 怎么变回文本」，以及「对话怎么拼接」。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "这部分和神经网络无关，是纯字符串工程——但配错 tokenizer "
          "比配错权重更常见（⑲）。</font>", ACT)

    # --- 代码锚点 --------------------------------------------------------
    y3 = y2 + 190
    p.box(36, y3, 1628, 96,
          "<b>代码锚点</b><br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "权重迭代器（决定读哪些文件、按什么顺序读）："
          "vllm/model_executor/model_loader/weight_utils.py:866"
          "（safetensors_weights_iterator）<br>"
          "默认加载流程：vllm/model_executor/model_loader/default_loader.py"
          "（起权重迭代器 → 逐层 load_weights）<br>"
          "config 的解析与校验：vllm/transformers_utils/config.py"
          "（get_config / 把 config.json 变成 Python 对象）"
          "</font>", "box", NOTE)

    legend(p, y3 + 132)
