"""② 跟着一个 token 走完一层 MoE：门卫、重排、分组矩阵乘、散射回。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc

TITLE = "② 一个 token 走完一层 MoE：六步"
SUB = ("一句话：<b>门卫打分 → 取 top-k → 按专家把 token 重排到一起 → 每个专家各算一段 → "
       "散射回原位 → 按门卫权重加权求和</b>；再和「共享专家」的输出相加。")


def build(doc: Doc) -> None:
    p = doc.page("② token 的 MoE 之旅", pw=1560, ph=1560)
    head(p, TITLE, SUB)
    X = 36

    y = 146
    p.text(X, y - 30, 1300, 22,
           "<b>第一步：六步（前三步决定「去哪」，后三步是真正的计算）</b>",
           "text_b")
    steps = [
        ("①", "门卫打分", "hidden → N 个分数", "cloudp"),
        ("②", "取 top-k", "只留分数最高的 k 个", "cloudp"),
        ("③", "重排", "按专家把 token 聚到一起", "data"),
        ("④", "分组矩阵乘", "每个专家各算自己那段", "cloudd"),
        ("⑤", "散射回原位", "把结果还原到 token 顺序", "data"),
        ("⑥", "加权求和", "k 个结果按门卫权重相加", "cloudd"),
    ]
    cw = 224
    for i, (no, name, note, role) in enumerate(steps):
        cx = X + i * (cw + 12)
        p.box(cx, y, cw, 92,
              f"<b>{no} {name}</b><br>"
              f"<font style=\"font-size:9px;color:#666666\">{note}</font>",
              "box_center", role)
        if i:
            p.edge(src_pt=(cx - 10, y + 46), dst_pt=(cx - 2, y + 46), st="arrow",
                   role="http")

    # --- 关键：为什么需要重排 ---------------------------------------------
    y2 = y + 132
    p.text(X, y2 - 30, 1300, 22,
           "<b>第二步：为什么非要「重排」（这一步是 MoE 实现里最关键的设计）</b>",
           "text_b")
    p.box(X, y2, 700, 188,
          "<b>不重排会怎样</b><br>"
          "<font style=\"font-size:10px\">"
          "假设有 5 个 token、8 个专家，每个 token 要去 2 个。于是有 10 个「(token, 专家)」对，"
          "但它们散落在不同位置。<br>"
          "如果按 token 顺序一个个调专家函数：<br>"
          "· 每次只有 1 行输入 → 矩阵乘退化成向量乘，算力极度浪费；<br>"
          "· 相邻两个 token 可能用同一个专家，却要再调一次。<br>"
          "<font color=\"#c0392b\">结果：同一个专家被调用很多次，每次只算很少的行。</font></font>",
          "box", "fail")
    p.box(X + 730, y2, 694, 188,
          "<b>重排之后</b><br>"
          "<font style=\"font-size:10px\">"
          "把所有「要去专家 i」的行<b>搬到一起、连续存放</b>，形成一张 <b>token 数 × k</b> 行的密集表。<br>"
          "然后<b>每个专家只调一次</b>，输入是「连续的一段行」。<br>"
          "· 每段的行数 = 这个专家分到多少 token（各段不同）；<br>"
          "· 这就是「分组矩阵乘」：一次提交里，每个专家各有一段自己的 M/N/K。<br>"
          "<font color=\"#1c7c54\">算完再散射回原来的位置，保证输出顺序与输入一致。</font></font>",
          "box", "cloudd")

    # --- 两处容易混的细节 -------------------------------------------------
    y3 = y2 + 228
    p.text(X, y3 - 30, 1300, 22,
           "<b>第三步：两个容易混的细节</b>", "text_b")
    panel(p, X, y3, 700, 168, "① 「行数」是 token 数 × k，不是 token 数", [
        "重排后的总行数 = token 数 × k（每个 token 在 k 个专家各占一行）。",
        "推理侧这里<b>不按「容量因子」截断</b>（不做 token drop）——",
        "所以行数就是 token×k，不会有 token 被丢弃。",
        "（容量因子/token drop 是训练期的概念，见 ⑦ 页的说明。）",
    ], "cloudd")
    panel(p, X + 760, y3, 664, 168, "② 加权求和用的权重来自门卫", [
        "门卫给出的 k 个分数，通常会归一化后再作为权重。",
        "所以「同一个 token 在不同专家里的结果」是<b>加权</b>相加，不是简单平均。",
        "这也是「为什么叫混合专家」的字面含义。",
    ], "cloudp")

    # --- 与 dense 的对比 --------------------------------------------------
    y4 = y3 + 208
    p.box(X, y4, 1424, 116,
          "<b>第四步：和 dense FFN 的差别只在「多了门卫与重排」，但代价全在这两处</b><br>"
          "<font style=\"font-size:10px\">"
          "dense：一次 MLP，输入是「全部 token」的连续矩阵。<br>"
          "MoE：先算一次小的线性层（门卫，hidden→N），再做两次数据搬运（重排/散射），"
          "最后才是 k 段 GEMM。<br>"
          "<font color=\"#c0392b\">所以 MoE 在<b>大 batch</b>（prefill）时很划算——GEMM 喂满了；"
          "而<b>小 batch</b>（decode）时，搬运与固定开销占比被放大，这正是 ③ 页要算的账。</font></font>",
          "box", "neutral")

    y5 = y4 + 156
    bridge(p, y5, "本页交接：结构清楚了。下一步（③）算三个账——显存、通信、负载，"
                  "看看 MoE 到底「贵」在哪。")

    y6 = y5 + 76
    anchors(p, y6, [
        "门卫（router）：一个 hidden→N 的线性层，输出 router_logits（token, N）："
        "vllm/model_executor/models/hy_v3.py:155-162、:211",
        "MoE 层装配（专家数 / top-k / 中间维）：同文件 :183-201",
        "重排与散射的昇腾实现：vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py"
        ":592（permute）、:430-433 与 :715-719（unpermute）、:217-227（两者分工的注释）",
        "重排行数 = token×top_k：同文件 :406（active_num=num_tokens*top_k）、:616-620"
        "（num_out_tokens = topk_ids.numel()，无容量截断）",
        "分组矩阵乘的 shape 语义：vllm/_custom_ops.py:1094（problem_sizes 每专家一组 M,N,K）",
        "每专家权重形状：vllm/model_executor/layers/fused_moe/experts/nvfp4_emulation_moe.py:438-439",
    ], h=128)

    y7 = y6 + 162
    concl(p, y7,
          "<b>一句话记住这一页：</b>MoE 一层的计算其实只有「一次小线性 + k 段 GEMM」，"
          "但中间夹了两次数据搬运（重排 / 散射）——"
          "<b>批大时搬运摊薄、批小时搬运变主成本</b>，这是 MoE 推理好坏的枢纽。")

    legend(p, y7 + 92, extra_fill=("数据搬运（重排/散射）", "data"))
