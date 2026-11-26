"""③ 词嵌入：整数 id → 一行浮点数（其实是「查表」）。"""
from mth import (ACT, CALC, ERR, NOTE, OUT, W, arrow, legend, matrix, strip,
                 title)

TITLE = "③ 词嵌入（Embedding）：查表，不是乘法"
SUB = ("很多人以为「把词变成向量」要做矩阵乘。其实 nn.Embedding 就是一张表："
       "id 是行号，直接把这行取出来。这一页把表摊开给你看。")


def build(doc) -> None:
    p = doc.page("③ 词嵌入", pw=1700, ph=1220)
    title(p, TITLE, SUB, w=1620)

    # --- 左：那张表 -------------------------------------------------------
    y = 150
    p.text(36, y - 34, 800, 22,
           "<b>第一步：想象一张 V 行 × d 列的浮点表</b>", "text_b")
    table = [
        ["0.11", "-0.42", "0.73", "0.02"],
        ["-0.35", "0.28", "0.19", "-0.61"],
        ["0.44", "0.09", "-0.52", "0.31"],
        ["0.07", "0.66", "-0.18", "0.25"],
        ["-0.23", "0.51", "0.12", "-0.47"],
    ]
    mr, mb, _ = matrix(p, 120, y, table, W, name="嵌入矩阵 E",
                       dim="5×4（真实：32000×4096）",
                       row_hdr=["id 12841", "id 3027", "id 882", "id 6132", "…"])
    p.box(36, mb + 20, 620, 96,
          "<b>这张表是「训练出来的参数」</b><br>"
          "一开始全是随机小数；训练就是不断微调它们，"
          "直到「语义相近的词，向量也相近」。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "表里的数字是小数（真实是高精度浮点），不像 ② 的 id 是整数</font>",
          "box", NOTE)

    # --- 右：取行 ---------------------------------------------------------
    p.text(720, y - 34, 900, 22,
           "<b>第二步：id 12841 就是「取第 12841 行」</b>", "text_b")
    p.box(720, y, 944, 104,
          "<b>nn.Embedding(32000, 4096) 的 forward 只有一句话：</b><br>"
          "<font style=\"font-size:10px\">"
          "output = weight[input_ids]　　# 按行号取行，没有任何乘法</font><br>"
          "<font style=\"font-size:9px;color:#c0392b\">"
          "所以它没有矩阵乘的计算量，却占 32000×4096×2 字节 ≈ 250 MB 显存（fp16）。"
          "</font>",
          "box", CALC)
    p.box(720, y + 116, 944, 120,
          "<b>为什么用「查表」而不是「乘矩阵」</b><br>"
          "如果一定要写成乘法：<font style=\"font-size:9px\">"
          "one_hot(id) · E</font> —— 也就是「一个只有一个 1 的向量」去乘 E。"
          "而 <font style=\"font-size:9px\">one_hot(12841)</font> 乘 E 的结果，"
          "正是 E 的第 12841 行。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "查表 = 这个乘法的特例；工程上直接按行索引，省掉 32000 次无用乘法。</font>",
          "box", NOTE)
    p.box(720, y + 256, 944, 96,
          "<b>5 个 token 的结果</b>：把取出来的 5 行摞起来，得到 <b>5×4</b> 的矩阵——"
          "这就是整个 Transformer 的输入张量（真实 5×4096）。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "行 = token 位置，列 = 特征维（d_model）</font>",
          "box", ACT)

    # --- 底部：权重共享 ---------------------------------------------------
    y2 = y + 400
    p.text(36, y2 - 34, 1100, 22,
           "<b>一个省显存的小技巧：嵌入矩阵和输出头共用同一张表</b>", "text_b")
    a = p.box(36, y2, 780, 120,
              "<b>输入侧：Embedding</b><br>"
              "32000×4096 → 取一行得到 4096 维<br>"
              "<font style=\"font-size:9px\" color=\"#666666\">id → 向量</font>",
              "box", W)
    b = p.box(876, y2, 788, 120,
              "<b>输出侧：LM Head</b><br>"
              "4096×32000 → 得到每个词的分数<br>"
              "<font style=\"font-size:9px\" color=\"#666666\">向量 → 32000 个分数</font>",
              "box", W)
    c = p.box(36, y2 + 152, 1628, 96,
              "<b>它们互为转置，可以共享同一份权重（tied embeddings）</b><br>"
              "省下 250 MB 显存；很多模型（Llama 系）就是这么干的。"
              "是否共享写在 config.json 的 "
              "<font style=\"font-size:9px\">tie_word_embeddings</font> 里（⑮）。",
              "box", OUT)
    p.edge(src_pt=(426, y2 + 120), dst_pt=(426, y2 + 152), st="arrow",
           role="http")
    p.edge(src_pt=(1270, y2 + 120), dst_pt=(1270, y2 + 152), st="arrow",
           role="http")

    # --- 代码锚点 --------------------------------------------------------
    y3 = y2 + 284
    p.box(36, y3, 1628, 96,
          "<b>代码锚点</b><br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "词嵌入层：vllm/model_executor/layers/vocab_parallel_embedding.py:205"
          "（class VocabParallelEmbedding；forward 就是按 id 取行，"
          "再按 TP 切分到各卡）<br>"
          "输出头：同文件 :569（class ParallelLMHead）——它继承自词嵌入层，"
          "因为「用 E 打分」和「用 E 查表」数学上是同一个矩阵的两种用法<br>"
          "装配处：vllm/model_executor/models/llama.py:382（embed_tokens）、:503（lm_head）、"
          ":510（tie_weights 共享权重）"
          "</font>", "box", NOTE)

    y4 = y3 + 130
    p.box(36, y4, 1628, 70,
          "<b>这一页要记住的</b>：id 是整数、向量是小数；"
          "「变成向量」= 查表 = 按行号取值，<b>没有矩阵乘法</b>；"
          "得到的 5×d 矩阵才是后面所有矩阵运算的起点。",
          "box", NOTE)

    legend(p, y4 + 104)
