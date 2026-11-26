"""① 总览：一句话从文本变成下一个 token（把 21 页的路线图一次画出来）。

这一页是"地图"，后面每页只展开其中一个方框。
"""
from mth import (ACT, CALC, ERR, NOTE, OUT, W, arrow, callout, legend, strip,
                 title)

TITLE = "① 总览：一句话是怎么变成下一个字的（21 页的路线图）"
SUB = ("你只要会线性代数就能看懂：整个 Transformer 就是「查表 → 矩阵乘 → 加权求和 → 再矩阵乘」"
       "反复 32 次，最后在词表上打一个分。本页先把全程画成一条线，后面逐段展开。")


def build(doc) -> None:
    p = doc.page("① 总览", pw=1700, ph=1180)
    title(p, TITLE, SUB, w=1620)

    # --- 第 1 行：从文本到张量 -------------------------------------------
    y = 150
    p.text(36, y - 34, 900, 22, "<b>前半程：文本 → 数字矩阵</b>", "text_b")
    x1 = p.box(36, y, 250, 78,
               "<b>文本</b><br>"
               "「我爱吃苹果」<br>"
               "<font style=\"font-size:9px\" color=\"#666666\">人看的东西</font>",
               "box", NOTE)
    x2 = p.box(316, y, 270, 78,
               "<b>② 分词 tokenizer</b><br>"
               "切成 5 个 token，每个换成整数 id<br>"
               "<font style=\"font-size:9px\" color=\"#666666\">词表里第几号</font>",
               "box", CALC)
    x3 = p.box(616, y, 300, 78,
               "<b>③ 词嵌入 + ④ 位置编码</b><br>"
               "每个 id 变成一行浮点数，再加位置信息<br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "得到 5×4 的矩阵（真实 5×4096）</font>",
               "box", ACT)
    arrow(p, 286, y + 39, 316, y + 39)
    arrow(p, 586, y + 39, 616, y + 39)

    x4 = p.box(946, y, 300, 78,
               "<b>⑤~⑫ 一层 Transformer</b><br>"
               "注意力（⑥~⑩）+ 残差/RMSNorm（⑪）+ FFN（⑫）<br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "形状不变：进去 5×4，出来还是 5×4</font>",
               "box", CALC)
    arrow(p, 916, y + 39, 946, y + 39)

    x5 = p.box(1276, y, 388, 78,
               "<b>⑬ 堆 32 层 → LM Head</b><br>"
               "最后一层输出 × 词表矩阵 → 每个 token 一个 32000 维分数<br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "取最后一个 token 的分数做 softmax → 采样</font>",
               "box", OUT)
    arrow(p, 1246, y + 39, 1276, y + 39)

    # --- 第 2 行：形状不变的秘密 -----------------------------------------
    y2 = y + 130
    p.text(36, y2 - 34, 1100, 22,
           "<b>贯穿全程的一条铁律：Transformer 层不改变张量形状</b>", "text_b")
    sx, _, _ = strip(p, 36, y2 - 4,
                     ["5×4", "5×4", "5×4", "5×4", "5×4"], NOTE,
                     name="形状（token 数 × 特征维）")
    p.text(36 + 5 * 50 + 12, y2 - 4, 900, 30,
           "<font style=\"font-size:10px\">"
           "→ 于是可以一层层往上堆：<b>第 2 层的输入，就是第 1 层的输出</b>。"
           "这就是「深度」的来源。</font>", "subtitle")
    p.box(36 + 5 * 50 + 12, y2 + 40, 1560, 62,
          "<b>为什么必须保持形状</b>：只有输入输出形状一致，才能做「残差连接」——"
          "把输入直接加到输出上（x + f(x)）。这一点是 Transformer 能堆深的根本原因（⑪ 详讲）。",
          "box", NOTE)

    # --- 第 3 行：后半程 -------------------------------------------------
    y3 = y2 + 140
    p.text(36, y3 - 34, 900, 22, "<b>后半程：分数 → 下一个字</b>", "text_b")
    b1 = p.box(36, y3, 300, 96,
               "<b>logits：每个候选字的分数</b><br>"
               "形状 5×32000（行=位置，列=词表）<br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "只取最后一行：1×32000</font>",
               "box", OUT)
    b2 = p.box(366, y3, 300, 96,
               "<b>softmax → 概率</b><br>"
               "把分数变成「加起来等于 1」的概率<br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "温度 T 控制锐利程度（⑬）</font>",
               "box", CALC)
    b3 = p.box(696, y3, 300, 96,
               "<b>采样：掷一次骰子</b><br>"
               "按概率抽一个 token（top-p 先砍掉长尾）<br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "贪心 = 永远选最大</font>",
               "box", CALC)
    b4 = p.box(1026, y3, 300, 96,
               "<b>把新 token 接回输入</b><br>"
               "序列变成 6 个 token，再跑一遍<br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "这就是「自回归」：一个字一个字生成</font>",
               "box", ACT)
    b5 = p.box(1356, y3, 308, 96,
               "<b>为什么叫「自」回归</b><br>"
               "模型的输出，变成它下一步的输入；"
               "整段对话就是这一个循环转几百次。",
               "box", NOTE)
    for a, b in ((b1, b2), (b2, b3), (b3, b4), (b4, b5)):
        p.edge(a, b, "", st="arrow", role="http")

    # --- 第 4 行：路线图 -------------------------------------------------
    y4 = y3 + 160
    p.text(36, y4 - 34, 1100, 22,
           "<b>全套图的路线（每页只展开一个方框）</b>", "text_b")
    road = [
        ("A · Transformer 怎么算（⑵~⑬）", "① 总览 → ② 分词 → ③ 词嵌入 → ④ 位置编码 → "
                                        "⑤ 矩阵乘 → ⑥ QKV → ⑦ 打分 → ⑧ softmax → "
                                        "⑨ 加权求和 → ⑩ 多头 → ⑪ 残差+归一化 → ⑫ FFN → ⑬ 输出与采样", NOTE),
        ("B · 模型文件里有什么（⑭~㉑）", "⑭ 目录全景 → ⑮ config.json → ⑯ safetensors 字节布局 → "
                                    "⑰ 分片与 index → ⑱ 权重名→形状 → ⑲ tokenizer 三件套 → "
                                    "⑳ GGUF 与量化 → ㉑ 从文件到推理", NOTE),
    ]
    cy = y4
    for t, body, role in road:
        p.box(36, cy, 1628, 74,
              f"<b>{t}</b><br><font style=\"font-size:10px\">{body}</font>",
              "box", role)
        cy += 86

    legend(p, cy + 10)
