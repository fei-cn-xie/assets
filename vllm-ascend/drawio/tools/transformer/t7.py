"""⑦ 注意力打分：S = Q·Kᵀ，以及为什么它必须是方阵。"""
import ex
from mth import (ACT, CALC, ERR, NOTE, OUT, W, arrow, legend, matrix, op,
                 panel, title)

TITLE = "⑦ 打注意力分数：S = Q · Kᵀ（5×3 乘 3×5 得 5×5）"
SUB = ("这一步回答「每个 token 该看谁」。数学上就是一次矩阵乘：Q 乘 K 的转置。"
       "注意结果的 5×5 —— 全篇唯一必须用方阵的地方，本页会说清为什么。")


def build(doc) -> None:
    p = doc.page("⑦ 注意力打分", pw=1700, ph=1360)
    title(p, TITLE, SUB, w=1620)

    # --- 形状 + 为什么方阵 -------------------------------------------------
    y = 150
    panel(p, 36, y, 810, "为什么这一步的矩阵一定是方阵（其它页都在躲方阵，这里躲不掉）",
          "Q 有 5 行（5 个 token，每个 3 维）、K 也有 5 行。把 K <b>转置</b>成 3×5 后相乘：<br>"
          "<font style=\"font-size:10px\">S = Q(5×3) · Kᵀ(3×5) → 5×5</font><br>"
          "结果的<b>行 = 发问的 token</b>，<b>列 = 被看的 token</b>。"
          "既然是「每个 token 对每个 token」，行数必然等于列数。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "所以 5×5 不是偷懒，而是语义决定的：它是一张「谁注意谁」的表。</font>",
          CALC)
    panel(p, 866, y, 798, "转置是什么意思",
          "K 本来是「每行一个 token」的 5×3；Kᵀ 是它的镜像："
          "第 i 行变第 i 列，形状反过来变成 3×5。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "代码里通常不真的转置，而是让矩阵乘的底层去处理（省一次内存搬运）。</font>",
          NOTE)

    # --- 三个矩阵 ---------------------------------------------------------
    y2 = y + 150
    p.text(36, y2 - 60, 1100, 22,
           "<b>第一步：摆出来算（Q 与 K 来自 ⑥）</b>", "text_b")
    qx, qy, _ = matrix(p, 60, y2 + 10, ex.Q, ACT, name="Q（提问）", dim="5×3")
    ox = op(p, qx + 14, y2 + 50, "·", 40, 40)
    ktx, kty, _ = matrix(p, ox + 14, y2 + 10, ex.T(ex.K), OUT,
                         name="Kᵀ（转置后的标签）", dim="3×5")
    ox2 = op(p, ktx + 14, y2 + 50, "=", 40, 40)
    sx, sy, _ = matrix(p, ox2 + 14, y2 + 10, ex.S, CALC, name="S（分数）",
                       dim="5×5", bold={(0, 0)})

    # --- 手算一格 ---------------------------------------------------------
    y3 = y2 + 250
    p.text(36, y3 - 60, 1100, 22,
           "<b>第二步：手算 S[0][0]（第 0 个 token 看第 0 个 token 的分数）</b>",
           "text_b")
    q0 = ex.Q[0]
    k0 = ex.K[0]
    prod = " + ".join(f"{q0[t]}×{k0[t]}" for t in range(3))
    tot = sum(q0[t] * k0[t] for t in range(3))
    panel(p, 36, y3, 900, "「第 0 行的 Q」点乘「第 0 行的 K」",
          f"<font style=\"font-size:11px\">Q 第 0 行 = {q0}　　"
          f"K 第 0 行（= Kᵀ 第 0 列）= {k0}</font><br>"
          f"<font style=\"font-size:11px\">{prod} = <b>{tot}</b></font><br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "点积越大 → 两个向量方向越一致 → 「越相关」。"
          "这就是把「语义相关度」变成「一个数」的全部秘密。</font>",
          CALC)
    panel(p, 966, y3, 698, "一个 token 看自己（对角线）通常分数很大",
          f"S 的对角线是 {[ex.S[i][i] for i in range(5)]}，"
          "往往比同行其它位置大——因为「自己和自己最像」。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "所以模型很容易只关注自己；训练时靠数据把「该看别人」的情况也学出来。</font>",
          NOTE)

    # --- 缩放 -------------------------------------------------------------
    y4 = y3 + 160
    p.text(36, y4 - 60, 1100, 22,
           "<b>第三步：除以 √d（缩放）——为什么要这一步</b>", "text_b")
    panel(p, 36, y4, 810, "问题：维度越高，点积的数值越大",
          f"本页 d_head = 3，√3 = {ex.SCALE}。S 里的数是 5~24 的量级，"
          f"除以 {ex.SCALE} 后变成 2.9~13.9。<br>"
          "真实模型 d_head = 128，√128 ≈ 11.3；如果不除，softmax 会被"
          "「特别大的分数」直接推到 0/1 的极端，梯度几乎消失。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "所以这一步不是玄学，是为了把分数<b>压回 softmax 舒服的区间</b>。</font>",
          ERR)
    stx, sty, _ = matrix(p, 870, y4 + 4, ex.S_scaled, CALC,
                         name="S / √3", dim="5×5　现在是小数字了",
                         cw=44, ch=26)
    p.text(870, sty + 12, 780, 44,
           "<font style=\"font-size:9px\" color=\"#666666\">"
           "从这一步开始出现<b>小数</b>：真实模型里 S 与 S/√d 全是浮点，"
           "整数只出现在 token id、位置下标、形状这些地方。</font>", "subtitle")

    y5 = y4 + 190
    p.box(36, y5, 1628, 96,
          "<b>代码锚点</b><br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "打分的整体（含缩放、mask、softmax）在 attention backend 里："
          "vllm/v1/attention/backends/ 下各实现（如 flash_attn / ascend 后端）<br>"
          "缩放因子从哪来：config.json 的 head_dim（或 hidden_size / num_heads）"
          "→ 见 vllm/model_executor/models/llama.py:183 附近对 head 维度与 rope 的初始化<br>"
          "工程上不会真的先算完整 5×5 再 softmax：会做 fused kernel（一次核内完成 "
          "QKᵀ → scale → mask → softmax），但数学含义就是本页这三步"
          "</font>", "box", NOTE)

    legend(p, y5 + 130)
