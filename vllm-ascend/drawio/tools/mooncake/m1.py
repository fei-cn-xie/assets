"""① 为什么要把 KV 搬到另一台机器上 —— 先把 KV cache 讲清楚。

这一页刻意不出现任何代码锚点：读者是个只学过 attention 的大一新生，
先把"KV 是资产、资产占显存、显存不够要分机器、分机器就得搬"这条因果讲顺。
"""
from dg import Doc, legend_bar, table

TITLE = "① 为什么要把 KV Cache 搬到另一台机器上（从 attention 说起）"
SUB = ("看懂本页只需要两件事：attention 里的 K / V 是什么；矩阵乘法的量级大概怎么估。"
       "其余概念本页现场解释，不预设你懂推理框架。")
X = 36
W = 470
GAP = 42
Y0 = 152


def cx(i: int) -> int:
    return X + i * (W + GAP)


def build(doc: Doc) -> None:
    p = doc.page("① 为什么需要搬 KV", pw=1560, ph=1150)
    p.title_block(TITLE, SUB, x=X, y=20, w=1500)

    # --- 第一行：KV 是什么 -------------------------------------------------
    p.text(X, Y0 - 36, 900, 22, "<b>第一件事：attention 里的 K / V 是什么</b>",
           "text_b")
    c1 = p.box(cx(0), Y0, W, 124,
               "<b>Q / K / V 都是矩阵乘法的产物</b><br>"
               "每个 token 的隐向量 x，乘三个权重矩阵 W_Q / W_K / W_V，"
               "得到它的 query / key / value。<br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "Attention(Q, K, V) = softmax(QKᵀ / √d) · V</font>",
               "box", "neutral")
    c2 = p.box(cx(1), Y0, W, 124,
               "<b>关键观察：K / V 只由「已经见过的 token」决定</b><br>"
               "第 t 步算 attention 时，前 t 个 token 的 K / V <b>已经算过、之后不再变</b>"
               "（它们的输入没变）。<br>只有 Q 是新的：最新那个 token 要发问。",
               "box", "cloudp")
    c3 = p.box(cx(2), Y0, W, 124,
               "<b>所以：把 K / V 存下来，别每次重算</b><br>"
               "这张「历史 token 的 K / V 记账表」就是 <b>KV Cache</b>。<br>"
               "代价是占显存；换来的是省掉重复的矩阵乘法。",
               "box", "cloudd")
    p.edge(c1, c2, "因为不变", st="arrow", role="http")
    p.edge(c2, c3, "所以能存", st="arrow", role="http")

    # --- 第二行：形状 + 算账 + 结论 -----------------------------------------
    y2 = Y0 + 186
    p.text(X, y2 - 36, 1200, 22,
           "<b>第二件事：这张表在显存里有多大（会算这个就够了）</b>", "text_b")

    p.box(cx(0), y2, W, 196,
          "<b>形状：一张 4 维的表</b><br>"
          "层数 × token 位置 × {K, V} × (KV 头数 × 每头维度)<br><br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "层 0 → token 0..t 的 K 、token 0..t 的 V<br>"
          "层 1 → …<br>"
          "层 31 → …</font><br><br>"
          "每个格子里是一个 fp16 数，占 2 字节。",
          "box", "neutral")

    p.text(cx(1) + 4, y2, W - 8, 24,
           "<b>算一笔账（假设 32 层、8 个 KV 头、每头 128 维）</b>", "text_b")
    rows = [
        ["单个 token 的 KV", "32×2×8×128×2 B ≈ <b>128 KB</b>"],
        ["一条 4K token 的请求", "4096 × 128 KB ≈ <b>512 MB</b>"],
        ["128 条并发请求", "128 × 512 MB ≈ <b>64 GB</b>"],
        ["一张 64 GB 的卡", "权重占 ~40 GB，只剩 20 多 GB"],
    ]
    table(p, cx(1) + 4, y2 + 34, [("项目", 176), ("数量", 290)], rows,
          row_h=40, header=False)

    p.box(cx(2), y2, W, 196,
          "<b>结论：显存是硬约束</b><br>"
          "同一张卡上，<b>模型权重</b>和 <b>KV cache</b> 抢同一块显存。"
          "并发一多、上下文一长，KV 就把它吃光。<br><br>"
          "→ 于是把 <b>prefill（算 KV）</b> 和 <b>decode（用 KV）</b> "
          "拆到两拨机器上。",
          "box", "fail")

    # --- 第三行：拆机器 → 必须搬 -------------------------------------------
    y3 = y2 + 250
    p.text(X, y3 - 36, 1200, 22,
           "<b>第三件事：机器拆开后，KV 就躺在「别人的显存」里</b>", "text_b")

    pm = p.box(cx(0), y3, W, 150,
               "<b>P 机（prefill）</b><br>"
               "拿到整段 prompt，一次算完，"
               "所以<b>它显存里有一段对话的完整 KV</b>。<br>"
               "算完它其实就没事了：后面每生成一个字都不需要它再算什么。",
               "box", "cloudp")
    mv = p.box(cx(1), y3, W, 150,
               "<b>于是必须「搬 KV」</b><br>"
               "把 P 机显存里那几十/几百 MB，"
               "经网卡搬到 D 机显存里。<br><br>"
               "<b>这件事就是本仓用 Mooncake 干的活。</b>",
               "box", "data")
    dm = p.box(cx(2), y3, W, 150,
               "<b>D 机（decode）</b><br>"
               "一个字一个字往外蹦，每个字都要用<b>完整历史 KV</b>。<br>"
               "但它手里没有那段 prompt 的 KV——除非有人搬给它。",
               "box", "cloudd")
    p.edge(pm, mv, "", st="arrow", role="data")
    p.edge(mv, dm, "", st="arrow", role="data")

    # --- 底部：一句话总结 --------------------------------------------------
    y4 = y3 + 206
    p.box(X, y4, 1424, 62,
          "<b>一句话记住这一页：</b>"
          "K / V 算一次就不变 → 值得存（KV cache）→ 它很占显存 → "
          "所以把 prefill / decode 拆到两拨机器 → 于是 KV 必须<b>跨机搬</b> → "
          "搬它的工具就是 Mooncake。",
          "box", "neutral")

    legend_bar(p, y4 + 96, extra_fill=("数据面 = 搬 KV 的通道", "data"))
