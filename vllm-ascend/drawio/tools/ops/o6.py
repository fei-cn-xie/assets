"""⑥ 通信算子：并行带来的那几种「跨卡动作」。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc, table

TITLE = "⑥ 通信算子：并行之后，卡与卡之间要交换什么"
SUB = ("一句话：本仓的 TP 路径<b>不是「GEMM 后 all_reduce」</b>，而是"
       "「<b>先 all-gather → GEMM → reduce-scatter</b>」；真正的 all_reduce 只剩三处。"
       "而 MoE 的 EP 路径是另一套（dispatch/combine）。")


def build(doc: Doc) -> None:
    p = doc.page("⑥ 通信算子", pw=1560, ph=1560)
    head(p, TITLE, SUB)
    X = 36

    # --- TP 路径 ----------------------------------------------------------
    y = 146
    p.text(X, y - 30, 1300, 22,
           "<b>第一步：TP 的两条路径（列并行 → 行并行），对比「教科书写法」</b>",
           "text_b")
    p.box(X, y, 690, 176,
          "<b>教科书：GEMM + all_reduce</b><br>"
          "<font style=\"font-size:10px\">"
          "列并行 GEMM（各卡算一部分）→ 各卡得到「部分输出」<br>"
          "→ <b>all_reduce</b> 求和 → 完整输出<br><br>"
          "通信量 ≈ 输出张量大小 × 卡数相关系数</font>",
          "box", "neutral")
    p.box(X + 730, y, 694, 176,
          "<b>本仓：先 gather、再算、再 reduce-scatter</b><br>"
          "<font style=\"font-size:10px\">"
          "① <b>all_gather</b> 输入（让每卡拿到全部输入行）<br>"
          "② GEMM（每卡只算自己那段输出，不再需要跨卡求和）<br>"
          "③ <b>reduce_scatter</b> 输出（把结果按行切片分给各卡）<br><br>"
          "<font color=\"#1c7c54\">好处：与「序列并行」天然兼容，且把一次大通信拆成两段可重叠的通信</font></font>",
          "box", "cloudd")

    # --- 只剩三处 all_reduce ----------------------------------------------
    y2 = y + 216
    p.text(X, y2 - 30, 1300, 22,
           "<b>第二步：那「真正的 all_reduce」还剩在哪（就三处）</b>", "text_b")
    rows = [
        ["① 输出头（lm_head）", "词表维并行 → 各卡算一部分词表 → 需要求和得到完整 logits",
         "整条链里最贵的一次"],
        ["② 共享专家输出", "MoE 里共享专家按 TP 切分，算完要归约", "每层一次"],
        ["③ 归一化的统计量", "在某个融合 kernel <b>内部</b>顺手做一次小 all_reduce",
         "省一次独立通信"],
    ]
    table(p, X, y2, [("位置", 380), ("为什么需要", 720), ("备注", 324)], rows,
          row_h=38)

    # --- 另外两种：all_to_all 与 reduce_scatter ----------------------------
    y3 = y2 + 38 * 4 + 30
    panel(p, X, y3, 700, 168, "all_to_all：output 投影的写法", [
        "某些配置下 o_proj 用「按头分给各卡」再「按 token 分回」的写法，",
        "于是出现 <b>all_to_all</b>：每卡把自己该算的头送出去、把别人算好的收回来。",
        "这与 MoE 的 dispatch 用的是同一个 primitive，但意图不同（一个是切头、一个是切专家）。",
    ], "data")
    panel(p, X + 760, y3, 664, 168, "MoE / EP：dispatch 与 combine", [
        "EP 下「token 去找专家」= dispatch，「结果送回」= combine，前后各一次。",
        "实现有三档（all-gather+reduce-scatter / MC2 融合 / 真 all-to-all）——见 MoE 套 ⑥。",
        "所以「MoE 的通信」与「TP 的通信」是两套独立的东西，排障时要分清是哪一套。",
    ], "cloudp")

    # --- MC2 的门槛 -------------------------------------------------------
    y4 = y3 + 208
    p.box(X, y4, 1424, 132,
          "<b>第三步：通信与计算融合（MC2）有个明确的规模门槛</b><br>"
          "<font style=\"font-size:10px\">"
          "本仓把「GEMM + reduce-scatter」做成了<b>一个融合算子</b>（通信与计算在算子内部重叠），"
          "但代码里有一道门：<b>只在 TP 规模 ≤ 8 时启用</b>，注释写的是"
          "「等底层的 mm_reduce_scatter 支持 tp ≥ 16 再去掉这个限制」。<br>"
          "MoE 侧更激进：把「dispatch + 专家计算 + combine」融成<b>单个算子</b>"
          "（那样的路径就<b>不需要多流重叠</b>了）。<br>"
          "<font color=\"#c0392b\">这也解释了一条互斥规则："
          "「融合 MC2」与「多流重叠共享专家」<b>不能同时开</b>——前者已经把重叠做进算子里了。</font></font>",
          "box", "fail")

    # --- 一句话 -----------------------------------------------------------
    y5 = y4 + 172
    p.box(X, y5, 1424, 100,
          "<b>第四步：一句话把「谁和谁通信」记住</b><br>"
          "<font style=\"font-size:10px\">"
          "TP：<b>每层</b>「gather 输入 → 算 → scatter 输出」；块与块之间还会额外做一次 o_proj 的 all_to_all。<br>"
          "EP（MoE）：<b>每层</b>「dispatch → 算 → combine」，且只有 MoE 才有。<br>"
          "DP：推理时几乎不通信（但 MoE 的 DP 例外——它要 wave 同步，见 MoE 套 ⑩）。</font>",
          "box", "neutral")

    y6 = y5 + 140
    bridge(p, y6, "本页交接：通信讲完，算子层的三大块（矩阵乘、小算子、通信）都齐了。"
                  "下一步（⑦）看「回退」是怎么一层层设计的——这决定了你在某台机器上能拿到多少性能。")

    y7 = y6 + 76
    anchors(p, y7, [
        "TP 的 gather→GEMM→reduce-scatter：vllm-ascend/vllm_ascend/ops/linear_op.py:278（all_gather）、"
        ":299（reduce_scatter）、:374（o_proj 的 reduce_scatter）、:365（o_proj 的 all_to_all）",
        "真正的 all_reduce 三处：ops/vocab_parallel_embedding.py:426（logits）、:456；"
        "ops/fused_moe/fused_moe.py:330（共享专家）、:454-538（多流编排）",
        "归一化融合 kernel 内的小 all_reduce：ops/triton/linearnorm/split_qkv_tp_rmsnorm_rope.py:305",
        "MC2 融合（GEMM+reduce-scatter）：ops/linear_op.py:211（npu_quant_mm_reduce_scatter）；"
        "门槛注释 ascend_forward_context.py:182-185（tp_size ≤ 8）；能力检查 :164-181",
        "MC2 base 接口：device/device_op.py:287-313、:1243-1265",
        "MoE 的 dispatch/combine（MC2）：ops/fused_moe/token_dispatcher.py:109、:232、:331；"
        "单算子融合 moe_comm_method.py:591；与多流互斥 ascend_config.py:850-855",
        "异步 all_to_all + 独立通信流：ops/fused_moe/moe_utils.py:63-99",
    ], h=152)

    y8 = y7 + 186
    concl(p, y8,
          "<b>一句话记住这一页：</b>本仓 TP 是「gather → GEMM → reduce-scatter」而不是「GEMM → all_reduce」，"
          "真正的 all_reduce 只剩输出头/共享专家/融合归一化三处；"
          "<b>MoE 的 dispatch/combine 是另一套，别和 TP 的通信混在一起排障。</b>")

    legend(p, y8 + 92, extra_fill=("跨卡搬运", "data"))
