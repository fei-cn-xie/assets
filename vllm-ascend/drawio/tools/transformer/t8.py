"""⑧ Softmax：把分数变成「加起来等于 1」的概率（手算一行）。"""
import math

import ex
from dg import table
from mth import (ACT, CALC, ERR, NOTE, OUT, W, legend, matrix, panel, title)

TITLE = "⑧ Softmax：把分数变成概率（一行一行算给你看）"
SUB = ("上一步得到 5 个分数，这一步把它们变成 5 个概率：都为正、加起来等于 1。"
       "整页只做四则运算和 exp，你会乘除就能跟下来。")

ROW = 0
scores = ex.S_scaled[ROW]
mx = max(scores)
shifted = [round(s - mx, 1) for s in scores]
expd = [round(math.exp(s), 3) for s in shifted]
tot = sum(expd)
probs = [round(e / tot, 3) for e in expd]


def build(doc) -> None:
    p = doc.page("⑧ Softmax", pw=1700, ph=1320)
    title(p, TITLE, SUB, w=1620)

    # --- 为什么要 softmax ------------------------------------------------
    y = 150
    panel(p, 36, y, 810, "为什么不能直接拿分数当权重",
          "分数可能是负数、也可能都很大（比如 19、13、23…），"
          "既不能当「比例」用，加起来也不等于 1。<br>"
          "Softmax 干三件事：① 全部取 exp（变成正数）；② 求和；③ 每个除以总和。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "结果一定 &gt; 0 且 Σ = 1，正好可以当「把注意力分给谁多少」的权重。</font>",
          CALC)
    panel(p, 866, y, 798, "为什么先减最大值（工程细节）",
          "exp(13.9) 约等于 1.1e6，exp(100) 直接溢出成 inf。"
          "所以实现里先减去这一行的最大值（本行 13.9），"
          "最大项变成 exp(0)=1，其余都是小数——<b>数学上完全等价</b>。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "这个技巧叫「数值稳定」，几乎每个 softmax 实现都有。</font>",
          NOTE)

    # --- 四步手算 --------------------------------------------------------
    y2 = y + 170
    p.text(36, y2 - 60, 1200, 22,
           f"<b>第一步：取第 {ROW} 个 token 那一行（5 个分数），逐步算</b>", "text_b")
    rows = []
    for j in range(5):
        rows.append([f"第 {j} 个 token", f"{scores[j]}", f"{shifted[j]}",
                     f"{expd[j]}"])
    rows.append(["<b>合计</b>", "—", "—", f"<b>{round(tot, 3)}</b>"])
    table(p, 36, y2 + 6, [("位置", 150), ("分数 s", 150),
                          ("s − max", 150), ("exp(...)", 170)],
          rows, row_h=36)
    p.text(36, y2 + 6 + 6 * 36 + 10, 640, 40,
           f"<font style=\"font-size:9px\" color=\"#666666\">"
           f"max = {mx}（本行最大值）；exp 后的和 = {round(tot, 3)}</font>",
           "subtitle")

    # --- 结果 -------------------------------------------------------------
    panel(p, 720, y2 + 6, 944, "第二步：每个 exp 值除以总和 → 概率",
          f"概率 = [{', '.join(str(x) for x in probs)}]，和 = {round(sum(probs), 3)}<br>"
          "含义：**第 0 个 token 把 62% 的注意力给了第 4 个 token**，"
          "34% 给了第 2 个，其余几乎没有。<br>"
          "<font style=\"font-size:9px;color:#c0392b\">"
          "注意这里有 0.001：softmax 不会真的给 0，但可以小到几乎为 0——"
          "这就是「长尾」的来源（⑬ 的 top-p 采样就是砍它）。</font>",
          OUT)
    panel(p, 720, y2 + 130, 944, "关于数字的类型（本页特别说明）",
          "① 分数 s 在真实模型里是<b>浮点</b>（本页用 1 位小数便于手算）；<br>"
          "② exp 与概率<b>必须是浮点</b>，而且通常是 fp16/bfloat16 甚至 fp32 计算；<br>"
          "③ 永远不会是整数——整数只出现在 token id、位置、形状这些地方。",
          NOTE)

    # --- 概率矩阵 ---------------------------------------------------------
    y3 = y2 + 300
    p.text(36, y3 - 60, 1200, 22,
           "<b>第三步：5 行都算完，就得到整张概率矩阵 P（5×5）</b>", "text_b")
    mx3, my3, _ = matrix(p, 60, y3 + 4, ex.P, OUT, name="P（注意力权重）",
                         dim="5×5　每行各自加起来 = 1", cw=52, ch=28)
    panel(p, mx3 + 40, y3 + 4, 700, "P 的三个性质（后面全靠它们）",
          "① <b>每行和为 1</b>：每行的含义是「这个 token 把注意力怎么分配」。<br>"
          "② <b>行 = 发问者，列 = 被关注者</b>：所以 P 是方阵（⑦ 解释过）。<br>"
          "③ <b>全部 ≥ 0</b>：注意力的权重不能为负，这让后面的加权求和变成"
          "「凸组合」（结果落在各 V 之间，不会跑到离谱的地方）。",
          NOTE)

    y4 = y3 + 220
    p.box(36, y4, 1628, 92,
          "<b>代码锚点</b><br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "softmax 本体（含数值稳定做法）：vllm/model_executor/layers/... "
          "里各 attention backend 的 softmax 实现；通用实现在 torch.nn.functional.softmax<br>"
          "为什么你在 vllm 里看不到完整的 P 矩阵：实际用 flash-attention 这类 "
          "fused kernel，一边算分块一边归一化，永远不把 5×5（真实是几百×几百）"
          "整块写回显存——数学等价，显存省一个数量级"
          "</font>", "box", NOTE)

    legend(p, y4 + 130)
