"""⑪ 残差连接 + RMSNorm：为什么必须加「x + f(x)」，以及归一化到底在算啥。"""
import math

from dg import table
from mth import (ACT, CALC, ERR, NOTE, OUT, W, arrow, legend, matrix, op,
                 panel, strip, title)

# RMSNorm 的手算例子：x = [3, 4, 0, 0]
XV = [3, 4, 0, 0]
MS = sum(v * v for v in XV) / len(XV)          # 6.25
RMS = math.sqrt(MS)                             # 2.5
NORM = [round(v / RMS, 2) for v in XV]          # [1.2, 1.6, 0.0, 0.0]
GAMMA = [1.0, 0.5, 1.5, 1.0]
SCALED = [round(n * g, 2) for n, g in zip(NORM, GAMMA)]

TITLE = "⑪ 残差连接 + RMSNorm：让 32 层能叠起来的两块「胶水」"
SUB = ("每一层 Transformer 都是「归一化 → 注意力 → 加回去 → 归一化 → FFN → 加回去」。"
       "这一页讲清两件事：为什么要「加回去」（残差），以及归一化到底在算什么。")


def build(doc) -> None:
    p = doc.page("⑪ 残差与归一化", pw=1700, ph=1400)
    title(p, TITLE, SUB, w=1620)

    # --- 一层的骨架 -------------------------------------------------------
    y = 150
    p.text(36, y - 60, 1200, 22,
           "<b>第一步：先看一层里四个部件的顺序（记住这张图，⑫ 还要接上）</b>",
           "text_b")
    x = 36
    names = [
        ("输入 x<br><font style='font-size:9px'>5×6</font>", ACT),
        ("RMSNorm<br><font style='font-size:9px'>只改数值，不改形状</font>", W),
        ("注意力<br><font style='font-size:9px'>⑥~⑩ 那一整套</font>", CALC),
        ("+ 残差<br><font style='font-size:9px'>x + 注意力输出</font>", OUT),
    ]
    ids = []
    for i, (t, role) in enumerate(names):
        cid = p.box(x + i * 300, y, 260, 74, t, "box_center", role)
        ids.append(cid)
        if i:
            p.edge(src_pt=(x + i * 300 - 40, y + 37),
                   dst_pt=(x + i * 300 - 6, y + 37), st="arrow", role="http")
    # 残差旁路
    p.edge(src_pt=(166, y), dst_pt=(166, y - 34), st="line", role="http",
           decor=True)
    p.edge(src_pt=(166, y - 34), dst_pt=(x + 3 * 300 + 130, y - 34),
           st="line", role="http", decor=True)
    p.edge(src_pt=(x + 3 * 300 + 130, y - 34),
           dst_pt=(x + 3 * 300 + 130, y - 4), st="arrow", role="http")
    p.text(300, y - 66, 700, 24,
           "<font style=\"font-size:9px\" color=\"#666666\">"
           "残差旁路：把输入原封不动抄一份，加到输出上</font>", "subtitle")

    # --- 为什么残差 -------------------------------------------------------
    y2 = y + 130
    panel(p, 36, y2, 810, "为什么要「加回去」（这是 Transformer 能堆深的关键）",
          "① <b>梯度能抄近路</b>：反向传播时，x 这条直连路径让梯度不用穿过 32 层的矩阵乘，"
          "直接从山顶传到山脚。没有它，深层网络的梯度会衰减到几乎为 0。<br>"
          "② <b>形状必须一致</b>：所以前面每一步都得保持 (n×d) —— 这就是 ⑨ 里"
          "「n×d → n×n → n×d」那个往返的真正意义。<br>"
          "③ <b>语义上是「学增量」</b>：每层只需学「在已有表示上加什么修正」，而不是从零重画。",
          NOTE)
    panel(p, 866, y2, 798, "注意：残差不是「加法层」",
          "它就是一次逐元素加法（element-wise add），<b>没有参数、没有矩阵乘</b>。<br>"
          "<font style=\"font-size:9px;color:#c0392b\">"
          "所以如果你在权重文件里找「residual 的权重」，是找不到的——"
          "它不占任何参数。这一点经常被误解。</font>",
          ERR)

    # --- RMSNorm 手算 -----------------------------------------------------
    y3 = y2 + 190
    p.text(36, y3 - 60, 1200, 22,
           "<b>第二步：RMSNorm 手算一遍（4 维向量，数字刚好整除）</b>", "text_b")
    steps = [
        ("① 输入向量 x", str(XV), "某层输出的一个 token 向量（真实 4096 维浮点）"),
        ("② 每项平方再取平均", f"({'+'.join(f'{v}²' for v in XV)}) / 4 = {MS}",
         "这就是「均方」"),
        ("③ 开根号", f"√{MS} = {RMS}", "均方根 RMS"),
        ("④ 每项除以 RMS", str(NORM), "让整体缩放到「量级为 1」"),
        ("⑤ 再乘学习到的 γ", str(SCALED), "γ 是 4 个可训练参数（真实 4096 个）"),
    ]
    tbl = table(p, 36, y3 + 6, [("步骤", 260), ("计算", 380), ("说明", 560)], steps,
                row_h=38)

    l1 = panel(p, 36, tbl + 44, 810, "和 LayerNorm 的区别（只需记一句）",
               "LayerNorm 会<b>减去均值</b>再除标准差；RMSNorm <b>不减均值</b>，只除均方根。<br>"
               "省掉减均值这一步，计算少一次、效果几乎不变——"
               "所以 Llama 系全用 RMSNorm。<br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "权重文件里对应 input_layernorm.weight，形状就是 [d_model]（⑱）。</font>",
               NOTE)
    l2 = panel(p, 866, tbl + 44, 798, "为什么归一化是必需的",
               "32 层反复做矩阵乘，数值会一层层放大（或缩小）。"
               "归一化把每层输入重新拉回稳定量级，训练才不会发散。<br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "对比一下：如果不做归一化，第 32 层的输入可能比第 1 层大 100 倍，"
               "fp16 直接溢出成 inf。</font>", NOTE)

    # --- 完整一层的形状 ---------------------------------------------------
    y4 = max(l1[1], l2[1]) + 70
    p.text(36, y4 - 60, 1200, 22,
           "<b>第三步：一层的完整形状账（全部是 n×d，从头到尾不变）</b>", "text_b")
    rows = [
        ["x", "5×6", "进入这一层"],
        ["RMSNorm(x)", "5×6", "数值变了，形状没变"],
        ["注意力(...)", "5×6", "⑩ 的最终输出 n_heads×d_head = d_model"],
        ["x + 注意力", "5×6", "残差（逐元素加法）"],
        ["RMSNorm(·)", "5×6", "再归一化一次"],
        ["FFN(...)", "5×6", "⑫ 展开：内部会升到 5×16 再降回"],
        ["+ 残差 → 输出", "5×6", "交给下一层，形状完全一致"],
    ]
    table(p, 36, y4 + 6, [("步骤", 240), ("形状", 120), ("说明", 560)], rows,
          row_h=34)

    # --- 代码锚点 --------------------------------------------------------
    y5 = y4 + 280
    p.box(36, y5, 1628, 96,
          "<b>代码锚点</b><br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "RMSNorm 实现：vllm/model_executor/layers/layernorm.py:37"
          "（class RMSNorm；forward 里就是 x * rsqrt(mean(x²)+eps) * weight）<br>"
          "在一层里的位置：vllm/model_executor/models/llama.py:251"
          "（class LlamaDecoderLayer，注意 forward 里两次 self.input_layernorm / "
          "post_attention_layernorm + 两次残差相加）<br>"
          "最终的归一化：同文件 :395（self.norm = RMSNorm(config.hidden_size, eps=...)）"
          "</font>", "box", NOTE)

    legend(p, y5 + 132)
