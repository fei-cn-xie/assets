"""⑥ 搬 token 的三种实现：all-gather / MC2 / all-to-all。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc, table

TITLE = "⑥ 搬 token 的三种实现——注意：EP 不必然走 all-to-all"
SUB = ("一句话：专家切到多卡后 token 要跨卡，本仓有<b>三条</b>实现可选："
       "<b>all-gather（默认）</b>、<b>MC2（通信与计算融合）</b>、<b>all-to-all</b>；"
       "选哪条由「是否开 EP、硬件型号、专家密度」共同决定，<b>不开 EP 时根本不用搬</b>。")


def build(doc: Doc) -> None:
    p = doc.page("⑥ 三种通信实现", pw=1560, ph=1540)
    head(p, TITLE, SUB)
    X = 36

    # --- 选择链 -----------------------------------------------------------
    y = 146
    p.text(X, y - 30, 1300, 22,
           "<b>第一步：选择链（按顺序判，先命中先用）</b>", "text_b")
    chain = [
        ("① 不是 MoE", "不涉及", "neutral"),
        ("② 没开 EP", "all-gather", "cloudd"),
        ("③ 开了 EP", "再按下面判", "cloudp"),
    ]
    cw = 300
    for i, (name, note, role) in enumerate(chain):
        cx = X + i * (cw + 16)
        p.box(cx, y, cw, 76,
              f"<b>{name}</b><br>"
              f"<font style=\"font-size:9px;color:#666666\">{note}</font>",
              "box_center", role)
        if i:
            p.edge(src_pt=(cx - 14, y + 38), dst_pt=(cx - 2, y + 38), st="arrow",
                   role="http")
    p.box(X + 3 * (cw + 16), y, 1424 - 3 * (cw + 16), 76,
          "<b>④ 开了 EP 且用了 LoRA</b><br>"
          "<font style=\"font-size:9px\">→ all-to-all（融合算子插不进 LoRA）</font>",
          "box_center", "fail")

    y2 = y + 106
    p.box(X, y2, 1424, 108,
          "<b>第二步（开了 EP、没用 LoRA）继续分岔：看硬件策略与专家密度</b><br>"
          "<font style=\"font-size:10px\">"
          "· 某些硬件：当「每卡专家数 ≤ 24 且 EP 规模 ≥ 16」才启用 MC2（融合算子），"
          "<b>否则回落到 all-gather</b>；<br>"
          "· 另一些硬件：当「EP 规模 ≤ top-k」时也回落 all-gather"
          "（因为每卡能拿到的专家很少，融合不划算）；<br>"
          "· 结论：<b>「开了 EP 就走 all-to-all」是错的</b>——本仓最常见的路径其实是 all-gather。</font>",
          "box", "cloudp")

    # --- 三条实现对比 -----------------------------------------------------
    y3 = y2 + 148
    p.text(X, y3 - 30, 1300, 22,
           "<b>第四步：三条实现各自在干什么（这是本页的核心）</b>", "text_b")
    panel(p, X, y3, 460, 210, "all-gather（默认）", [
        "把<b>所有卡上的 token</b>都收集到每张卡上（每卡都拿到全量 token），",
        "然后每卡只用「本地专家」筛出属于自己的那部分行来算。",
        "优点：实现简单、不需要复杂的收发配对。",
        "缺点：搬的是「全量 token」而不是「该去别卡的那些」——",
        "<b>EP 规模越大越浪费</b>。",
    ], "cloudd")
    panel(p, X + 482, y3, 460, 210, "all-to-all（按需送）", [
        "只把「要去别卡专家」的 token 送过去，接收方也只收自己需要的。",
        "优点：搬的量与「真正跨卡的 token 数」成正比，EP 大时更省。",
        "缺点：要按份数切分/拼装（收发元数据复杂），",
        "且小 batch 时延迟盖过收益；另有实现限制（例如不支持 bias）。",
    ], "data")
    panel(p, X + 964, y3, 460, 210, "MC2（融合）", [
        "把「重排 + 通信 + 分组矩阵乘」<b>压进一个融合算子</b>，通信与计算重叠。",
        "优点：省掉中间搬运、减少一次落盘到显存。",
        "缺点：只支持特定量化组合与特定硬件；专家太少时不划算；",
        "且在<b>未开 EP 时会直接报错</b>（它本身就是为 EP 场景写的）。",
    ], "cloudp")

    # --- 小 batch 的固定开销 ---------------------------------------------
    y4 = y3 + 250
    p.box(X, y4, 1424, 116,
          "<b>第四步：为什么小 batch（decode）时这三条都不便宜</b><br>"
          "<font style=\"font-size:10px\">"
          "通信缓冲要按<b>最坏情况</b>分配：<b>每卡最大批 × EP 规模 × min(top-k, 每卡专家数)</b>——"
          "这个式子与「本次实际有多少 token」无关。<br>"
          "所以 decode 这种「每步只有几十个 token」的场景，仍要付「为最大批准备的开销」"
          "（元数据、屏障、缓冲清零等固定成本）。<br>"
          "<font color=\"#c0392b\">这就是本文档反复强调「MoE 在 decode 阶段更吃亏」的机制之一"
          "（另一条是要读的专家权重零碎）。</font></font>",
          "box", "fail")

    # --- 一致性校验 -------------------------------------------------------
    y5 = y4 + 156
    from dg import table as t2
    p.text(X, y5 - 30, 1300, 22,
           "<b>第五步：一张速查（怎么判断当前用的是哪条）</b>", "text_b")
    rows = [
        ["启动参数", "没开 --enable-expert-parallel", "→ all-gather（或不用搬）"],
        ["启动参数", "开了 EP + LoRA", "→ all-to-all"],
        ["硬件+常量", "EP 规模、每卡专家数、top-k 满足条件", "→ MC2"],
        ["日志", "看「选择哪种 MoE 通信方法」的启动日志（关键字见锚点）", "→ 直接确认"],
    ]
    t2(p, X, y5, [("看哪", 200), ("条件", 720), ("结论", 504)], rows, row_h=34)

    y6 = y5 + 34 * 5 + 34
    bridge(p, y6, "本页交接：通信选好了，但「谁被送到哪个专家」本身也可能极不均衡。"
                  "下一步（⑦）讲负载均衡——训练期靠损失/偏置，推理期靠换位与冗余副本。")

    y7 = y6 + 76
    anchors(p, y7, [
        "选择链（非 MoE→不涉及；未开 EP 或 ep_world==1→ALLGATHER；EP+LoRA→ALLTOALL；"
        "否则按硬件策略）：vllm-ascend/vllm_ascend/ascend_forward_context.py:373-425",
        "A2 策略：每卡专家 ≤24 且 EP≥16 才 MC2，否则回落 ALLGATHER：同文件 :304-315",
        "A5 策略：EP 规模 ≤ top-k 时回落 ALLGATHER：同文件 :361-362",
        "MC2 要求 enable_expert_parallel=True（原文）："
        "vllm-ascend/.../fused_moe/moe_comm_method.py:243-245、:291-293",
        "all-to-all 的实现（torch.distributed.all_to_all_single，group=EP 组）："
        "vllm-ascend/.../fused_moe/moe_utils.py:63-99",
        "MC2 单算子收发：token_dispatcher.py:232、:331（npu_moe_distribute_dispatch_v2/combine_v2）",
        "all-gather 路径只保留本地专家：token_dispatcher.py:396-413（active_expert_range）",
        "all-to-all 路径不支持 bias（原文）：token_dispatcher.py:328、:560",
        "缓冲按最坏情况分配：moe_comm_method.py:416-422；token_dispatcher.py:126-133",
    ], h=152)

    y8 = y7 + 186
    concl(p, y8,
          "<b>一句话记住这一页：</b>搬 token 有三条路：<b>all-gather（默认、简单但搬全量）</b>、"
          "<b>all-to-all（按需、省量但复杂）</b>、<b>MC2（融合、最快但挑硬件与量化）</b>；"
          "选哪条由「是否开 EP + 硬件 + 专家密度」决定，<b>不开 EP 就不需要搬</b>。")

    legend(p, y8 + 92, extra_fill=("按需搬运（all-to-all）", "data"))
