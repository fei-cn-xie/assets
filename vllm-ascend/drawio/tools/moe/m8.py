"""⑧ 昇腾侧主链：从路由算子到分组矩阵乘，以及共享专家怎么重叠。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc

TITLE = "⑧ 昇腾侧：一条被「融合算子」串起来的主链"
SUB = ("一句话：昇腾把「路由 → 重排 → 分组矩阵乘 → 散射」几乎全压成了 <b>NPU 专用算子</b>，"
       "还额外提供「重排时顺便量化」和「共享专家与专家计算并行」两个优化。")


def build(doc: Doc) -> None:
    p = doc.page("⑧ 昇腾 MoE 主链", pw=1560, ph=1560)
    head(p, TITLE, SUB)
    X = 36

    # --- 主链 -------------------------------------------------------------
    y = 146
    p.text(X, y - 30, 1300, 22,
           "<b>第一步：主链（每一步对应一个或一组昇腾算子）</b>", "text_b")
    steps = [
        ("① 门卫", "gating top-k 融合算子", "cloudp"),
        ("② 重排+量化", "init_routing（顺带动态量化）", "data"),
        ("③ 第一段 GEMM", "grouped matmul（gate+up）", "cloudd"),
        ("④ 激活", "可与量化融合成单算子", "cloudd"),
        ("⑤ 第二段 GEMM", "grouped matmul（down）", "cloudd"),
        ("⑥ 散射+加权", "token unpermute（带权重）", "data"),
    ]
    cw = 224
    for i, (name, note, role) in enumerate(steps):
        cx = X + i * (cw + 12)
        p.box(cx, y, cw, 84,
              f"<b>{name}</b><br>"
              f"<font style=\"font-size:9px;color:#666666\">{note}</font>",
              "box_center", role)
        if i:
            p.edge(src_pt=(cx - 10, y + 42), dst_pt=(cx - 2, y + 42), st="arrow",
                   role="http")

    # --- 三个昇腾特有的优化 -----------------------------------------------
    y2 = y + 124
    p.text(X, y2 - 30, 1300, 22,
           "<b>第二步：三个昇腾特有的优化（这是本页的重点）</b>", "text_b")
    panel(p, X, y2, 460, 200, "① 重排时顺便量化", [
        "普通做法：重排（搬一次数据）→ 量化（再搬一次）。",
        "昇腾把量化塞进重排算子：<b>一次搬运完成两件事</b>。",
        "省的是「多搬一遍 token」的带宽——在小 batch 时很值钱。",
    ], "data")
    panel(p, X + 482, y2, 460, 200, "② 融合 kernel：两段 GEMM + 激活", [
        "「gate/up 段 → 激活 → down 段」可以压成<b>一个融合算子</b>，",
        "中间结果不落显存（少一次写+读）。",
        "代价：融合算子对<b>量化组合与硬件型号</b>有要求，不满足就回退到分步。",
    ], "cloudd")
    panel(p, X + 964, y2, 460, 200, "③ 共享专家与专家计算「并行」", [
        "共享专家是 dense 的，和路由专家的计算互不依赖 → 可以<b>放另一条流同时跑</b>。",
        "实现上还用「里程碑/事件」对齐两段之间的依赖（例如激活前必须等到某一段）。",
        "本仓还有一致性校验：<b>拆开算的结果必须等于整体算的结果</b>，不等直接报错。",
    ], "cloudp")

    # --- 一个易错点 -------------------------------------------------------
    y3 = y2 + 240
    p.box(X, y3, 1424, 116,
          "<b>第三步：一个 EP 下的易错点（阅读代码时会遇到）</b><br>"
          "<font style=\"font-size:10px\">"
          "开 EP 后，MoE 层自己的「TP 规模」被改写成了 1（因为专家已经是「每卡一份」），"
          "<b>但共享专家的权重仍然按 TP 切分</b>。<br>"
          "所以判断共享专家布局时，不能看 MoE 层那个被改写过的值，"
          "而要看<b>真正的 TP 组规模</b>——代码里专门为此写了注释，"
          "踩了这个坑会看到「共享专家形状对不上」或「结果不一致」。</font>",
          "box", "fail")

    # --- 与 vLLM 侧的分工 -------------------------------------------------
    y4 = y3 + 156
    p.text(X, y4 - 30, 1300, 22,
           "<b>第四步：vLLM 侧与昇腾侧的分工（避免读代码时迷路）</b>", "text_b")
    from dg import table
    rows = [
        ["vLLM 侧（框架）", "装配 MoE 层、路由策略、专家权重容器、EPLB 状态机、wave 协调",
         "fused_moe/layer.py、runner/moe_runner.py、distributed/eplb/"],
        ["昇腾侧（后端）", "把每一步换成 NPU 算子、选通信实现、管理流与事件、量化实现",
         "vllm_ascend/ops/fused_moe/*"],
        ["两侧的契约", "「准备/执行/收尾」三段 + 专家权重命名 w13/w2 + 量化 scheme 接口",
         "见 ⑩ 页的锚点清单"],
    ]
    table(p, X, y4, [("在哪", 240), ("负责什么", 780), ("看哪里", 404)], rows,
          row_h=38)

    y5 = y4 + 38 * 4 + 36
    bridge(p, y5, "本页交接：主链清楚了。下一步（⑨）回到部署侧——"
                  "本仓到底哪套包用了 MoE、为什么只有它开 EP，以及 MoE 与 PD/MTP 的耦合点。")

    y6 = y5 + 76
    anchors(p, y6, [
        "昇腾 MoE 总装：vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py:85"
        "（AscendMoERunner）、:161（forward）、:498（_forward_impl）",
        "路由：gate_linear.py:27；融合 gating 走 npu_moe_gating_top_k"
        "（device/device_op.py:1225，经 router/fused_topk_router.py:267）；"
        "回落纯 torch：router/grouped_topk_router.py:11、:59",
        "重排/散射：token_dispatcher.py:402（init_routing，带 active_expert_range）、"
        ":430（unpermute）、:592（permute）",
        "分组矩阵乘：routed_experts.py:231（gmm1）、:294（gmm2）；融合版 "
        "grouped_matmul_swiglu_quant（w8a8_dynamic.py:420/429、w4a8.py:495）",
        "「重排时顺便量化」：token_dispatcher.py:377-383（quant_mode 1/3/9）",
        "共享专家多流 + 里程碑：shared_experts.py:99、:358、:743/761；"
        "拆算一致性校验 :152-192（不等即 ValueError :188）",
        "EP 下 MoE 层 tp 被改写为 1 但共享专家按真实 TP 布局：shared_experts.py:219-238（注释 :221-224）",
        "MoE 层总装的 vLLM 侧：vllm/model_executor/layers/fused_moe/layer.py:88"
        "（FusedMoEFactory）、runner/moe_runner.py:681（forward）",
        "四段流水线原文：[Router]→[Quantize-Dispatch]→[Permute-Experts-Unpermute]→[Combine]："
        "vllm/model_executor/layers/fused_moe/modular_kernel.py:48-83",
    ], h=176)

    y7 = y6 + 210
    concl(p, y7,
          "<b>一句话记住这一页：</b>昇腾把 MoE 主链压成了「融合算子串」——"
          "重排顺带量化、两段 GEMM 与激活融合、共享专家放另一条流并行；"
          "<b>代价是这些融合对硬件与量化组合有要求，不满足就回退分步</b>（回退了也正确，只是慢）。")

    legend(p, y7 + 92, extra_fill=("融合/搬运相关", "data"))
