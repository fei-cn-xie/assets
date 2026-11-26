"""⑫ FFN（SwiGLU）：先升维 4 倍，再压回来。"""
import ex
from dg import table
from mth import (ACT, CALC, ERR, NOTE, OUT, W, legend, matrix, op, panel, title)

TITLE = "⑫ 前馈网络 FFN：升维 2.75 倍再压回来（参数最多的那一块）"
SUB = ("注意力负责「token 之间交换信息」，FFN 负责「每个 token 自己深加工」。"
       "它先把每个 token 的向量拉宽，过一道非线性，再压回原宽度。")


def build(doc) -> None:
    p = doc.page("⑫ FFN", pw=1700, ph=1340)
    title(p, TITLE, SUB, w=1620)

    # --- 形状 -------------------------------------------------------------
    y = 150
    panel(p, 36, y, 1628, "形状：5×4 → 5×8 → 5×4（真实：5×4096 → 5×11008 → 5×4096）",
          "中间那一层叫「上投影」（up / gate），把 4 维拉到 8 维；"
          "最后叫「下投影」（down），再压回 4 维。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "Llama-7B：d_model=4096、ffn=11008（≈2.7 倍，不是 4 倍）；"
          "很多模型用 4 倍，具体值看 config.json 的 intermediate_size（⑮）。</font>",
          CALC)

    # --- 三个矩阵 ---------------------------------------------------------
    y2 = y + 110
    p.text(36, y2 - 60, 1200, 22,
           "<b>第一步：两个矩阵乘（都是非方阵，形状一眼能跟上）</b>", "text_b")
    x1, y1, _ = matrix(p, 60, y2 + 6, ex.X, ACT, name="X（5×4）", dim="行=token")
    o1 = op(p, x1 + 12, y2 + 46, "·", 38, 40)
    x2, y2b, _ = matrix(p, o1 + 12, y2 + 6, ex.W1, W, name="W₁（4×8）",
                        dim="上投影：4 维 → 8 维", cw=36)
    o2 = op(p, x2 + 12, y2 + 46, "=", 38, 40)
    x3, y3b, _ = matrix(p, o2 + 12, y2 + 6, ex.H1, OUT, name="H（5×8）",
                        dim="中间层（宽了）", cw=36)
    o3 = op(p, x3 + 12, y2 + 46, "·", 38, 40)
    x4, y4b, _ = matrix(p, o3 + 12, y2 + 6, ex.W2, W, name="W₂（8×4）",
                        dim="下投影：8 维 → 4 维")
    o4 = op(p, x4 + 12, y2 + 46, "=", 38, 40)
    x5, y5b, _ = matrix(p, o4 + 12, y2 + 6, ex.H2, OUT, name="输出（5×4）",
                        dim="回到原宽度")

    p.text(60, max(y1, y3b, y5b) + 16, 1200, 40,
           "<font style=\"font-size:9px\" color=\"#666666\">"
           "本页所有数字都是整数（矩阵乘的结果，代码算出来的，可直接核对）。"
           "真实模型里 X/H 是浮点，W 是小浮点。</font>", "subtitle")

    # --- SwiGLU -----------------------------------------------------------
    y3 = y2 + 250
    p.text(36, y3 - 60, 1200, 22,
           "<b>第二步：现代模型不是「一上一下」，而是「两上一下」（SwiGLU）</b>",
           "text_b")
    a = p.box(36, y3 + 6, 300, 74,
              "<b>X（5×4）</b><br>"
              "<font style=\"font-size:9px\" color=\"#666666\">每个 token 一个向量</font>",
              "box_center", ACT)
    g = p.box(376, y3 + 6, 380, 74,
              "<b>gate 分支：W_gate（4×8）</b><br>"
              "<font style=\"font-size:9px\" color=\"#666666\">"
              "拉宽后过 SiLU（一种平滑的开关）</font>",
              "box_center", W)
    u = p.box(376, y3 + 100, 380, 74,
              "<b>up 分支：W_up（4×8）</b><br>"
              "<font style=\"font-size:9px\" color=\"#666666\">"
              "只是拉宽，不过激活</font>",
              "box_center", W)
    m = p.box(796, y3 + 52, 340, 74,
              "<b>逐元素相乘</b><br>"
              "<font style=\"font-size:9px\" color=\"#666666\">"
              "gate ⊙ up → 5×8</font>",
              "box_center", CALC)
    d = p.box(1176, y3 + 52, 300, 74,
              "<b>down：W_down（8×4）</b><br>"
              "<font style=\"font-size:9px\" color=\"#666666\">压回 5×4</font>",
              "box_center", OUT)
    for s, t in ((a, g), (a, u), (g, m), (u, m), (m, d)):
        p.edge(s, t, "", st="arrow", role="http")

    panel(p, 36, y3 + 190, 810, "为什么要 gate（这个「开关」在干嘛）",
          "SiLU(v) = v · sigmoid(v)：<b>正数基本原样通过，负数被压到接近 0</b>——"
          "相当于「按内容决定放行多少」。<br>"
          "然后与 up 分支逐元素相乘，等于「用 gate 当阀门，去调节 up 的信号」。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "实践上比单路的 ReLU 效果更好，代价是多一个矩阵。</font>",
          CALC)
    panel(p, 866, y3 + 190, 798, "参数量对比（为什么 FFN 是大头）",
          "单路（老式）：W₁(4×8) + W₂(8×4) = 64 个参数。"
          "SwiGLU（现代）：W_gate + W_up + W_down = 32+32+32 = 96 个。<br>"
          "真实 Llama-7B：FFN 大约 <b>4096×11008×3 ≈ 1.35 亿</b>参数/层，"
          "而注意力只有 4096×4096×4 ≈ 6700 万——<b>FFN 是注意力的两倍</b>。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "所以权重文件里 mlp.* 的张量加起来，比 self_attn.* 大得多（⑱ 会看到形状）。</font>",
          NOTE)

    # --- 代码锚点 --------------------------------------------------------
    y4 = y3 + 320
    p.box(36, y4, 1628, 100,
          "<b>代码锚点</b><br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "FFN 本体：vllm/model_executor/models/llama.py:82（class LlamaMLP）<br>"
          "gate 与 up 被合并成一个矩阵乘：同文件 :95"
          "（self.gate_up_proj = MergedColumnParallelLinear(...)）——"
          "两个 4×8 拼成 4×16，一次算完再切开<br>"
          "下投影：:103（self.down_proj = RowParallelLinear(...)）<br>"
          "SiLU 与「逐元素相乘」的融合实现：vllm/model_executor/layers/activation.py:116"
          "（class SiluAndMul —— 名字就说明了：先 silu 再 mul）"
          "</font>", "box", NOTE)

    legend(p, y4 + 132)
