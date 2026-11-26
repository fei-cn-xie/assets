"""⑤ 线性层：把「矩阵乘法」亲手算一遍（6×4 · 4×5 = 6×5）。"""
import ex
from mth import (ACT, CALC, ERR, NOTE, OUT, W, arrow, legend, matrix, op,
                 panel, title)

TITLE = "⑤ 线性层（nn.Linear）：整个 Transformer 里唯一的基本运算"
SUB = ("Transformer 有 99% 的浮点运算都是这一件事：y = x·W + b。"
       "学会看这一次乘法，后面的 QKV、注意力、FFN 全都是它的变形。")


def build(doc) -> None:
    p = doc.page("⑤ 线性层", pw=1700, ph=1320)
    title(p, TITLE, SUB, w=1620)

    # --- 形状先行 ---------------------------------------------------------
    y = 152
    p.text(36, y - 60, 1100, 22,
           "<b>第一步：先看形状怎么变（这一步比数字重要）</b>", "text_b")
    panel(p, 36, y, 1628, "形状规则：左矩阵的列数 = 右矩阵的行数，结果的形状取「左行 × 右列」",
          "X（6×4）· W（4×5）= R（6×5）　　"
          "→ 中间那个 4 和 4 必须相等（否则乘不了）；结果的 6 来自左矩阵的行、5 来自右矩阵的列。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "本页故意让 <b>6 ≠ 5</b>、也故意不用方阵——这样你能一眼看出"
          "「结果的每一维是从哪儿继承的」。真实模型里同样是长方形："
          "X 是 (token 数 × 4096)，W 是 (4096 × 12288)，R 是 (token 数 × 12288)。</font>",
          NOTE)

    # --- 三个矩阵 ---------------------------------------------------------
    y2 = y + 108
    p.text(36, y2 - 60, 1100, 22,
           "<b>第二步：摆出三个矩阵（数字都是整数，方便手算）</b>", "text_b")
    mx1, my1, _ = matrix(p, 60, y2 + 10, ex.X5, ACT, name="X（输入）",
                         dim="6×4　6 个 token × 4 维", row_hdr=None)
    ox = op(p, mx1 + 16, y2 + 60, "·", 44, 40)
    mx2, my2, _ = matrix(p, ox + 16, y2 + 10, ex.W5, W, name="W（权重）",
                         dim="4×5　4 维进 → 5 维出")
    ox2 = op(p, mx2 + 16, y2 + 60, "=", 44, 40)
    mx3, my3, _ = matrix(p, ox2 + 16, y2 + 10, ex.R5, OUT, name="R（输出）",
                         dim="6×5　6 个 token × 5 维新特征")

    p.text(ox2 + 16, my3 + 18, 700, 40,
           "<font style=\"font-size:10px\">"
           "真实模型里 W 的元素是<b>很小的浮点</b>（初始化时约 ±0.02），"
           "X 也是浮点；本页用 0/1/2 这种整数只为让你能手算。</font>",
           "subtitle")

    # --- 手算一个格子 ------------------------------------------------------
    y3 = y2 + 300
    p.text(36, y3 - 60, 1100, 22,
           "<b>第三步：手算 R 的左上角那一个格子（顺便看清「行乘列」）</b>",
           "text_b")
    row0 = ex.X5[0]
    col0 = [ex.W5[t][0] for t in range(4)]
    prod = " + ".join(f"{row0[t]}×{col0[t]}" for t in range(4))
    total = sum(row0[t] * col0[t] for t in range(4))
    panel(p, 36, y3, 900,
          "R[0][0] 是「X 的第 0 行」乘「W 的第 0 列」",
          f"<font style=\"font-size:11px\">X 第 0 行 = {row0}　　W 第 0 列 = {col0}</font><br>"
          f"<font style=\"font-size:11px\">{prod} = <b>{total}</b></font><br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "注意到没有：算一个格子需要 4 次乘 + 3 次加（4 = X 的列数 = W 的行数）。"
          "整张 R 有 6×5=30 个格子，所以一共 30×4=120 次乘法。</font>",
          CALC)
    panel(p, 966, y3, 698, "为什么这决定了模型的计算量",
          "参数量 = W 的元素个数 = 4×5 = 20（真实：4096×12288 ≈ 5000 万）。<br>"
          "计算量 ∝ token 数 × 参数量——所以 <b>「上下文越长、越慢」是乘出来的</b>，"
          "不是玄学。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "FFN 为什么占大头：它的 W 是 4096×12288 级别，比注意力的投影宽 3 倍（⑫）</font>",
          NOTE)

    # --- 批归一化/偏置 ----------------------------------------------------
    y4 = y3 + 150
    panel(p, 36, y4, 1628, "别忘了偏置项 b：y = x·W + b",
          "b 是一个长度 5 的向量（每个输出维一个偏置），"
          "它会被<b>广播</b>加到 R 的每一行上——所以 6 个 token 共用同一份 b。<br>"
          "Llama 系为了省参数把 b 去掉了（<font style=\"font-size:9px\">bias=False</font>），"
          "但 Qwen/很多模型仍然保留。这一点直接反映在权重文件里："
          "有 bias 就会有 <font style=\"font-size:9px\">q_proj.bias</font> 这个张量（⑱）。",
          NOTE)

    # --- 代码锚点 --------------------------------------------------------
    y5 = y4 + 122
    p.box(36, y5, 1628, 96,
          "<b>代码锚点</b><br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "线性层的统一基类：vllm/model_executor/layers/linear.py:241（class LinearBase）<br>"
          "最朴素的一份实现：同文件 :324（class ReplicatedLinear）——就是 y = xW + b<br>"
          "工程上的变体（切分权重到多卡）：:430（ColumnParallelLinear，按输出维切）、"
          ":1623（RowParallelLinear，按输入维切后求和）<br>"
          "在 Llama 里被用到的位置：vllm/model_executor/models/llama.py:165（qkv_proj）、"
          ":175（o_proj）、:95（gate_up_proj）、:103（down_proj）"
          "</font>", "box", NOTE)

    legend(p, y5 + 132)
