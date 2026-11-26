"""③ KV cache 在显存里实际长什么样：块（block）与块表。

为什么必须先讲"块"：后面所有传输都以块为单位，"搬 16 个 token 的 KV" 而不是
"搬整个序列"，这个颗粒度决定了传输引擎的请求大小，也决定了 P/D 之间怎么对账。
"""
from dg import Doc, legend_bar

TITLE = "③ KV Cache 在显存里长什么样：为什么按「块」管理"
SUB = ("一句话：显存被切成固定大小的小格子（块），每个块放固定数量 token 的 K/V；"
       "一张块表记录「逻辑第几块 → 物理第几块」。跨机搬 KV 就是以块为单位的。")
X = 36
W = 470
GAP = 42


def cx(i: int) -> int:
    return X + i * (W + GAP)


def build(doc: Doc) -> None:
    p = doc.page("③ 块与块表", pw=1560, ph=1240)
    p.title_block(TITLE, SUB, x=X, y=20, w=1500)

    # --- 左：块是什么 -----------------------------------------------------
    y = 150
    p.text(X, y - 36, 900, 22,
           "<b>第一步：先把显存想成一排小格子</b>", "text_b")
    b1 = p.box(cx(0), y, W, 132,
               "<b>块（block）= 固定大小的格子</b><br>"
               "一个块装 <b>block_size</b> 个 token 的 K/V"
               "（本项目常见 16 或 128，见启动参数）。<br>"
               "不管请求多长，都按整块分配。",
               "box", "neutral")
    b2 = p.box(cx(1), y, W, 132,
               "<b>为什么按块、不按「一整个请求」分</b><br>"
               "① 每个请求长度不同，整段分会产生大量碎片；<br>"
               "② 不同请求的<b>相同前缀</b>可以指到同一块（省显存）；<br>"
               "③ 按块好搬：每块地址固定、长度固定，一次搬一块很干脆。",
               "box", "neutral")
    b3 = p.box(cx(2), y, W, 132,
               "<b>块表（block table）= 逻辑 → 物理 的映射</b><br>"
               "请求自己的第 0、1、2… 块，分别对应显存里哪个物理块。<br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "vllm/v1/worker/block_table.py:57（BlockTable）· :119（slot_mapping）</font>",
               "box", "cloudd")
    p.edge(b1, b2, "所以按块分", st="arrow", role="http")
    p.edge(b2, b3, "再用块表记账", st="arrow", role="http")

    # --- 中：一张示意图（上下两排，竖箭头，避免连线穿过旁边的框）------------
    y2 = y + 196
    p.text(X, y2 - 36, 1000, 22,
           "<b>第二步：一个 40 token 的请求，实际占了 3 个 16-token 的块</b>",
           "text_b")
    p.text(X, y2, 460, 22,
           "请求自己的逻辑块（块表的上半部分）↓", "subtitle")
    logi = []
    for i in range(3):
        cid = p.box(X + i * 158, y2 + 26, 150, 56,
                    f"<b>逻辑块 {i}</b><br>"
                    f"<font style=\"font-size:9px\" color=\"#666666\">"
                    f"token {i*16}..{i*16+15}</font>", "box_center", "neutral")
        logi.append(cid)
    p.text(X, y2 + 88, 470, 22,
           "<font style=\"font-size:9px\" color=\"#666666\">"
           "最后一块只用了 8/16 格：40 = 16 + 16 + 8</font>", "subtitle")

    p.text(X, y2 + 128, 460, 22,
           "块表把它翻译成显存里的物理块 ↓", "subtitle")
    phys = []
    for i, txt in enumerate(["物理块 #7", "物理块 #12", "物理块 #31"]):
        cid = p.box(X + i * 158, y2 + 154, 150, 56,
                    f"<b>{txt}</b><br>"
                    f"<font style=\"font-size:9px\" color=\"#666666\">"
                    f"显存里真实位置</font>", "box_center", "cloudd")
        phys.append(cid)
    for a, b in zip(logi, phys):
        p.edge(a, b, "", st="arrow", role="http")

    # --- 右：这块为什么对"搬"很重要 -----------------------------------------
    y3 = y2 + 262
    p.text(X, y3, 1200, 22,
           "<b>第三步：这条「块」的规则，直接决定了跨机搬 KV 的样子</b>",
           "text_b")
    c1 = p.box(X, y3 + 30, 700, 128,
               "<b>搬的单位是「块」，不是「整段对话」</b><br>"
               "P 机算完 prefill 后，它手上是若干个<b>已经填满的物理块</b>；"
               "D 机要的是「我这 3 个逻辑块的内容」。<br>"
               "于是 D 机说：请把 #7/#12/#31 这三块的内容给我 —— "
               "这就是<b>块级 pull</b>。",
               "box", "data")
    c2 = p.box(X + 740, y3 + 30, 684, 128,
               "<b>也因为按块，P 和 D 必须能对上「哪块是哪块」</b><br>"
               "P 把自己那三块的<b>地址与长度</b>打包成一个「KV 交接单」交给 D；"
               "D 拿着单子逐块去取。<br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "这张单子在代码里叫 kv_transfer_params（下一页会看到它长什么样）</font>",
               "box", "cloudp")
    p.edge(c1, c2, "", st="arrow", role="http")

    # --- 结论 + 锚点 -------------------------------------------------------
    y4 = y3 + 196
    p.box(X, y4, 1424, 96,
          "<b>这一页的代码锚点（想自己去翻的时候用）</b><br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "块的容量与规格：vllm/v1/core/kv_cache_utils.py:668-690（resolve_dcp_kv_block_size / "
          "block_size 解析）<br>"
          "空闲块的分配与回收：vllm/v1/core/block_pool.py:143（BlockPool）、:683（get_new_blocks）、"
          ":789（free_blocks）<br>"
          "按请求分配块：vllm/v1/core/kv_cache_manager.py:402（allocate_slots）<br>"
          "逻辑块 → 物理块 + 写 KV 的槽位：vllm/v1/worker/block_table.py:57、:119（slot_mapping）"
          "</font>",
          "box", "neutral")

    y5 = y4 + 130
    p.box(X, y5, 1424, 58,
          "<b>一句话记住这一页：</b>显存按「块」切分，块表记录「逻辑块 → 物理块」；"
          "所以跨机搬 KV 是<b>按块搬</b>（块级 pull），P 只要把「哪几块、在哪」告诉 D 就够了。",
          "box", "neutral")

    legend_bar(p, y5 + 92)
