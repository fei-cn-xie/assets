"""⑬ 堆 32 层 + LM Head + 采样：从「最后一层输出」到「下一个字」。"""
import math

from dg import table
from mth import (ACT, CALC, ERR, NOTE, OUT, W, arrow, legend, panel, strip,
                 title)

# 5 个候选词的 logits（整数，便于手算；真实是 32000 个浮点）
CAND = ["苹果", "香蕉", "天空", "因为", "的"]
LOGITS = [4, 2, 1, 0, -1]


def softmax_T(logits, T):
    z = [v / T for v in logits]
    mx = max(z)
    e = [math.exp(v - mx) for v in z]
    s = sum(e)
    return [round(x / s, 3) for x in e]


P1 = softmax_T(LOGITS, 1.0)
P05 = softmax_T(LOGITS, 0.5)
P2 = softmax_T(LOGITS, 2.0)

TITLE = "⑬ 堆 32 层 + 输出头 + 采样：最后一个字是怎么被选出来的"
SUB = ("前半页讲「层怎么堆」和「怎么变成词表分数」，后半页用 5 个候选词把"
       "「温度 / top-p / 贪心」的计算过程算给你看。")


def build(doc) -> None:
    p = doc.page("⑬ 堆叠与采样", pw=1700, ph=1420)
    title(p, TITLE, SUB, w=1620)

    # --- 堆叠 -------------------------------------------------------------
    y = 150
    p.text(36, y - 60, 1200, 22,
           "<b>第一步：32 层是怎么堆的（每层权重都不一样）</b>", "text_b")
    xs = 36
    for i in range(4):
        lab = ("第 1 层" if i == 0 else
               ("第 2 层" if i == 1 else ("…" if i == 2 else "第 32 层")))
        role = ACT if i in (0, 3) else NOTE
        p.box(xs + i * 210, y + 6, 180, 66,
              f"<b>{lab}</b><br><font style=\"font-size:9px\" color=\"#666666\">"
              f"{'结构相同' if i < 3 else '结构相同'}</font>",
              "box_center", role)
        if i:
            p.edge(src_pt=(xs + i * 210 - 26, y + 39),
                   dst_pt=(xs + i * 210 - 4, y + 39), st="arrow", role="http")
    p.box(900, y + 6, 300, 66,
          "<b>最后一层输出</b><br>"
          "<font style=\"font-size:9px\" color=\"#666666\">仍是 5×4（形状不变）</font>",
          "box_center", OUT)
    p.edge(src_pt=(846, y + 39), dst_pt=(896, y + 39), st="arrow", role="http")
    p.box(1240, y + 6, 424, 66,
          "<b>关键</b>：32 层结构完全相同，但<b>每层有自己的权重</b>——"
          "所以参数量 = 单层 × 32。",
          "box", NOTE)

    # --- LM Head ---------------------------------------------------------
    y2 = y + 110
    p.text(36, y2 - 60, 1200, 22,
           "<b>第二步：最后做一次矩阵乘，把 4 维变成「词表大小」维的分数</b>",
           "text_b")
    panel(p, 36, y2 + 6, 810, "5×4 · 4×32000 = 5×32000（形状最能说明问题）",
          "行还是 5 个 token，列变成 32000 个候选词的分数（logits）。<br>"
          "<b>只有最后一行的分数有用</b>：因为我们要预测「第 5 个 token 之后是什么」，"
          "前面的行预测的是「第 1~4 个位置的下一个词」，训练时用得上，推理时不需要。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "所以 LM Head 的权重是 (4096×32000)≈1.3 亿参数，和 FFN 一层差不多——"
          "它常常和词嵌入共用同一份权重（③ 讲过）。</font>",
          CALC)
    panel(p, 866, y2 + 6, 798, "为什么要「取最后一个」",
          "自回归生成：前 4 个位置的下一个词是已知的（就是输入里的第 2~5 个 token），"
          "不需要预测。真正要预测的只有「第 5 个之后」。<br>"
          "<font style=\"font-size:9px;color:#c0392b\">"
          "工程实现里这个「取最后一个」是一次索引切片，不额外算矩阵乘——"
          "但为了批处理，通常会取出每个请求各自的最后一个位置。</font>",
          NOTE)

    # --- 采样 -------------------------------------------------------------
    y3 = y2 + 200
    p.text(36, y3 - 60, 1200, 22,
           "<b>第三步：把 5 个候选词的分数变成概率（同一组分数，三种温度）</b>",
           "text_b")
    s1, _, _ = strip(p, 36, y3 + 6, CAND, NOTE, name="候选词（真实 32000 个）",
                     cw=90)
    s2, _, _ = strip(p, 36, y3 + 62, LOGITS, ACT, name="logits（整数，真实是浮点）",
                     cw=90)
    rows = [
        ["T = 0.5（更确定）"] + [str(v) for v in P05],
        ["T = 1.0（原样）"] + [str(v) for v in P1],
        ["T = 2.0（更随机）"] + [str(v) for v in P2],
    ]
    table(p, 36, y3 + 118, [("温度", 240)] + [(c, 92) for c in CAND], rows,
          row_h=34)
    panel(p, 760, y3 + 118, 904, "温度到底在做什么",
          "softmax 前先把每个分数除以 T：<b>T 小 → 分数差距被放大 → 概率更集中</b>；"
          "T 大 → 差距被抹平 → 概率更平均。<br>"
          "所以 T=0.5 时「苹果」几乎必被选中（0.865），T=2.0 时连「的天空」都有 0.16 的机会。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "注意概率永远是小数：这一页出现的 0.865/0.166 在真实模型里是 float16/float32。</font>",
          CALC)

    # --- top-p / 贪心 -----------------------------------------------------
    y4 = y3 + 300
    p.text(36, y4 - 60, 1200, 22,
           "<b>第四步：三种「选词策略」——同一张概率表，不同的抽法</b>", "text_b")
    panel(p, 36, y4 + 6, 520, "贪心（greedy）",
          "直接取概率最大的那个（T=1.0 时选「苹果」）。<br>"
          "优点：稳定、可复现；缺点：容易重复、念经。", NOTE)
    panel(p, 576, y4 + 6, 520, "top-p（核采样）",
          "把概率从大到小累加，累加到 p（如 0.9）就<b>截断</b>，只在剩下的几个词里抽。"
          "本页 T=1.0 时：0.844 + 0.114 = 0.958 &gt; 0.9，"
          "所以只在「苹果/香蕉」里抽——长尾（天空/因为/的）直接被砍掉。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "top-k 是它的兄弟：直接保留前 k 个。</font>", CALC)
    panel(p, 1116, y4 + 6, 548, "为什么需要随机性",
          "如果每次都用贪心，同一句话永远得到同一个回答——"
          "看起来像「死记硬背」。<br>"
          "采样 + 温度让模型有「多样性」，同时 top-p 保证不会抽到离谱的词。<br>"
          "<font style=\"font-size:9px;color:#c0392b\">"
          "这解释了为什么「temperature=0」也不是完全确定的："
          "底层浮点累加顺序也会带来微小差异。</font>", ERR)

    # --- 代码锚点 --------------------------------------------------------
    y5 = y4 + 190
    p.box(36, y5, 1628, 100,
          "<b>代码锚点</b><br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "LM Head：vllm/model_executor/models/llama.py:503（self.lm_head = ParallelLMHead(...)）"
          "→ :510（tie_weights 与 embed_tokens 共享）<br>"
          "层堆叠：同文件 :347（class LlamaModel，self.layers = make_layers(...)）、"
          ":464（class LlamaForCausalLM，把 model + lm_head 串起来）<br>"
          "采样器（温度 / top-p / top-k 的真实实现）："
          "vllm/model_executor/layers/logits_processor.py 与 vllm/v1/sample/ 下的 sampler"
          "</font>", "box", NOTE)

    y6 = y5 + 134
    p.box(36, y6, 1628, 76,
          "<b>到这里，Transformer 的「一整个循环」就走完了</b>："
          "② 分词 → ③ 词嵌入 → ④ 位置 → （⑤~⑫ 一层）× 32 → ⑬ 输出头 → 采样 →"
          "把新 token 接回输入再来一遍。剩下的问题只有一个："
          "<b>这几十亿个浮点数，到底存在硬盘的什么位置、长什么样？</b>（⑭ 开始）",
          "box", NOTE)

    legend(p, y6 + 108)
