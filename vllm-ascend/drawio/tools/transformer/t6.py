"""⑥ Q / K / V 投影：同一份输入，乘三个不同的权重矩阵。"""
import ex
from mth import (ACT, CALC, ERR, NOTE, OUT, W, arrow, legend, matrix, op,
                 panel, title)

TITLE = "⑥ Q / K / V 投影：一份输入，三次矩阵乘"
SUB = ("注意力需要的三样东西（query / key / value）不是玄学，就是同一份 X 乘三个不同的权重矩阵。"
       "这一页只做三次 ⑤ 那样的乘法，并把结果摆出来给后面两页用。")


def build(doc) -> None:
    p = doc.page("⑥ QKV 投影", pw=1700, ph=1340)
    title(p, TITLE, SUB, w=1620)

    # --- 为什么要三份 -----------------------------------------------------
    y = 150
    p.text(36, y - 60, 1100, 22,
           "<b>第一步：先用一句话说清三者的分工（后面全靠这个比喻）</b>", "text_b")
    p.box(36, y, 530, 104,
          "<b>Q（query，提问）</b><br>"
          "「我这个 token 想找什么信息？」<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "每个 token 用自己的 Q 去和别人比对</font>",
          "box", ACT)
    p.box(586, y, 530, 104,
          "<b>K（key，标签）</b><br>"
          "「我这里有什么信息？」<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "Q 和 K 做点积 = 相关度打分</font>",
          "box", ACT)
    p.box(1136, y, 528, 104,
          "<b>V（value，内容）</b><br>"
          "「如果被关注，我提供什么内容？」<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "按分数加权求和的就是 V</font>",
          "box", ACT)
    p.box(36, y + 120, 1628, 62,
          "<b>图书馆类比</b>：Q 是你在检索框里输入的关键词，K 是每本书的标签，V 是书的内容。"
          "检索 = 关键词和标签算相似度（⑦），取书 = 按相似度把内容加权搬回来（⑨）。",
          "box", NOTE)

    # --- 三次乘法 ---------------------------------------------------------
    y2 = y + 250
    p.text(36, y2 - 60, 1100, 22,
           "<b>第二步：三次独立的矩阵乘（X 是同一份，W 不同）</b>", "text_b")
    # X
    mx, my, _ = matrix(p, 60, y2 + 20, ex.X, ACT, name="X（上一层输出）",
                       dim="5×4")
    # 三个 W 竖排
    wx = 300
    wy = y2 + 10
    for i, (nm, mat, out, outname) in enumerate([
            ("W_Q", ex.WQ, ex.Q, "Q"),
            ("W_K", ex.WK, ex.K, "K"),
            ("W_V", ex.WV, ex.V, "V")]):
        matrix(p, wx + i * 470, wy, mat, W, name=f"{nm}（权重）", dim="4×3")
        op(p, wx + i * 470 + 134, wy + 30, "→", 40, 40)
        matrix(p, wx + i * 470 + 186, wy, out, OUT, name=f"{outname} = X·{nm}",
               dim="5×3")
    p.text(60, my + 16, 560, 40,
           "<font style=\"font-size:10px\">"
           "上面三行是<b>并行</b>的：都吃同一个 X，各用各的 W。<br>"
           "所以工程上会把三个 W <b>拼成一个大矩阵</b>一次算完"
           "（这就是 qkv_proj 的由来）。</font>", "subtitle")
    p.text(60, my + 82, 620, 60,
           "<font style=\"font-size:9px\" color=\"#666666\">"
           "本页数字全是整数（W 用 0/1，X 用 0~2）只为手算方便。"
           "真实模型里：X 是 (5×4096) 浮点，W_Q/W_K/W_V 各是 (4096×4096) 浮点，"
           "输出各是 (5×4096)。</font>", "subtitle")

    # --- 形状对齐的三条铁律 ----------------------------------------------
    y3 = my + 120
    p.text(36, y3 - 60, 1100, 22,
           "<b>第三步：三条形状铁律（记住这三条，看任何实现都不会迷路）</b>",
           "text_b")
    panel(p, 36, y3, 520, "① W 的形状是「输入维 × 输出维」",
          "W_Q 是 4×3：<b>左边一定是输入维</b>（和 X 的列数相等），右边是想输出多少维。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "所以权重文件里 q_proj.weight 的形状是 (输出维, 输入维) 还是反过来？"
          "→ 二者是<b>转置关系</b>，⑱ 专门讲这个坑。</font>", NOTE)
    panel(p, 576, y3, 520, "② 三个投影的输出维可以不同",
          "本页为了好算都取 3（d_head = 3）。真实模型里 "
          "d_head = d_model / n_heads（如 4096/32 = 128），"
          "而 <b>Q/K 的维度决定打分，V 的维度决定输出</b>。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "GQA 里 K/V 的头数比 Q 少（省钱），⑩ 讲。</font>", NOTE)
    panel(p, 1116, y3, 548, "③ 投影后仍保持「行 = token」",
          "X、Q、K、V 的行数都是 5（token 数）——<b>行这一维永远是 token</b>。"
          "唯一变化的是列（特征维）。<br>"
          "<font style=\"font-size:9px;color:#c0392b\">"
          "这条是后面所有形状推理的锚：看到行数变了，说明有人在搬 token，而不是在算特征。</font>",
          ERR)

    # --- 代码锚点 --------------------------------------------------------
    y4 = y3 + 214
    p.box(36, y4, 1628, 100,
          "<b>代码锚点</b><br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "三个投影被合并成一个：vllm/model_executor/layers/linear.py:995"
          "（class QKVParallelLinear——注释里会写它把 q/k/v 拼在一起，"
          "同时按 TP 把「头」切到各卡）<br>"
          "Llama 里的装配：vllm/model_executor/models/llama.py:165"
          "（self.qkv_proj = QKVParallelLinear(...)）→ :175（self.o_proj = RowParallelLinear(...)）<br>"
          "头数/头维的来源：config.json 的 num_attention_heads / num_key_value_heads"
          "（⑮ 逐个字段讲）"
          "</font>", "box", NOTE)

    legend(p, y4 + 136)
