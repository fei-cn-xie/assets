"""⑨ attention 算子：prefill 与 decode 是两个算子（而且本仓实测都落在同一个上）。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc, table

TITLE = "⑨ attention 落点：结构上分两个算子，本仓实测却常落在同一个上"
SUB = ("一句话：理论上 prefill 用「变长拼接」的融合 attention、decode 用「按块表寻址」的 paged attention；"
       "但<b>paged 是可选加速项</b>——本仓部署都没开它，所以<b>decode 实际也走同一个融合算子</b>。")


def build(doc: Doc) -> None:
    p = doc.page("⑨ attention 落点", pw=1560, ph=1580)
    head(p, TITLE, SUB)
    X = 36

    # --- 两个算子 ---------------------------------------------------------
    y = 146
    p.text(X, y - 30, 1300, 22,
           "<b>第一步：两个算子、两种「长度怎么表示」</b>", "text_b")
    p.box(X, y, 690, 180,
          "<b>prefill：变长拼接布局</b><br>"
          "<font style=\"font-size:10px\">"
          "把所有请求的 query <b>首尾拼成一条长序列</b>，"
          "用「前缀和」告诉算子「每个请求从哪到哪」。<br>"
          "所以它能一次算完「很多请求 × 很多 token」。<br>"
          "<font color=\"#c0392b\">限制：每请求的 query 长度不能超过一个上限（本仓是 <b>16</b>）"
          "——这就是投机解码 K+1 撞墙的那道限制。</font></font>",
          "box", "cloudd")
    p.box(X + 730, y, 694, 180,
          "<b>decode：按块表寻址的 paged attention</b><br>"
          "<font style=\"font-size:10px\">"
          "每请求<b>只有 1 个 query token</b>，历史 K/V 在分块缓存里，"
          "靠「块表 + 每请求上下文长度」直接寻址。<br>"
          "它只能表达「单 token 的 decode」，<b>无法</b>表达变长 causal prefill。<br>"
          "<font color=\"#1c7c54\">所以两者不是「新旧关系」，而是「形状能力」不同。</font></font>",
          "box", "cloudp")

    # --- 关键纠正 ---------------------------------------------------------
    y2 = y + 220
    p.box(X, y2, 1424, 132,
          "<b>第二步：一个重要纠正——paged attention 是「可选加速项」，不是默认路径</b><br>"
          "<font style=\"font-size:10px\">"
          "本仓文档原文大意：「融合 attention 是默认算子；在<b>中等并发</b>等场景下它可能表现不佳，"
          "这时可以<b>手动</b>通过 <font style=\"font-size:9px\">pa_shape_list</font> 启用 paged attention。」<br>"
          "也就是说：<b>paged attention 只在「形状命中名单」时才启用</b>，而这份名单<b>默认为空</b>——"
          "本仓十套部署包的启动参数里<b>从未出现</b>它。<br>"
          "<font color=\"#c0392b\">实测结论：本仓 decode 也走融合 attention（不是 paged）。"
          "所以「decode 一定用 paged attention」是个常见误解。</font></font>",
          "box", "fail")

    # --- 布局 -------------------------------------------------------------
    y3 = y2 + 172
    p.text(X, y3 - 30, 1300, 22,
           "<b>第三步：布局与 dtype 决定「能不能用、用哪个变体」</b>", "text_b")
    rows = [
        ["变长拼接（前缀和表示长度）", "prefill 主用", "每请求 query ≤ 16（投机解码撞墙处）"],
        ["块表寻址（块表 + 每请求长度）", "decode 可选", "需要块大小对齐（int8 KV 要求 32 的倍数）"],
        ["KV 在块缓存里的物理布局", "由环境变量/配置决定", "不同布局对应不同的读写算子"],
        ["int8 / fp8 的 KV", "走单独的量化 KV 变体", "需要额外的 scale 张量，且部分变体不支持滑窗/非 causal"],
        ["MLA 类（latent 只有一份 KV）", "KV 头数恒为 1", "TP 间是复制而非切分"],
    ]
    table(p, X, y3, [("维度", 420), ("取值/含义", 500), ("影响", 504)], rows,
          row_h=36)

    # --- 两个必须知道的数字 -----------------------------------------------
    y4 = y3 + 36 * 6 + 32
    p.text(X, y4 - 30, 1300, 22,
           "<b>第四步：两个必须知道的数字（都决定了「能不能跑」）</b>", "text_b")
    panel(p, X, y4, 700, 148, "① 每请求 query 上限 = 16", [
        "这是融合 attention 的变长布局限制。",
        "直接后果：<b>投机解码的 K+1 不能超过 16</b>（MoE/MTP 套里都撞过这个墙）。",
        "本仓部署取 K=1~3，离上限很远；但换模型/改配置时容易忘。",
    ], "fail")
    panel(p, X + 760, y4, 664, 148, "② chunked prefill 的 workspace 会随 token 上限涨", [
        "长上下文模型要把历史 K/V 读进一块 workspace 再算，",
        "这块空间与「token 上限 × 头维度」成正比——",
        "本仓注释给的例子是：同一套公式，<b>288 MB 与 6 GB 的差别只来自头配置不同</b>。",
    ], "data")

    y5 = y4 + 188
    bridge(p, y5, "本页交接：attention 讲完。最后一步（⑩）把「算子怎么被排到设备上」讲清——"
                  "流、事件、同步点，以及 workspace 如何进入显存账。")

    y6 = y5 + 76
    anchors(p, y6, [
        "prefill 走融合 attention（变长布局）：vllm-ascend/vllm_ascend/attention/attention_v1.py:941、"
        ":956（滑窗）、:994（默认）",
        "decode 可选 paged attention：attention_v1.py:1110（非图）、:785（图模式，先取 workspace 再注册任务）",
        "paged 的启用判定：attention_v1.py:1248-1269 + attention/utils.py:284-302"
        "（要求 FULL_DECODE_ONLY 且形状在名单里）",
        "「默认 FIA，PA 是手动调优项」文档原文：vllm-ascend/docs/source/faqs.md:308",
        "「等 FIA 全面胜出就删掉 PA」代码原文：vllm-ascend/vllm_ascend/worker/model_runner_v1.py:4059-4063",
        "每请求 query ≤16 的断言（四处同文）：attention_v1.py:273-277、mla_v1.py:325-329、"
        "sfa_v1.py:534-538、dsa_v1.py:678-682",
        "长度语义：前缀和 attention_v1.py:364（query_start_loc）、每请求长度 :365；"
        "paged 用 context_lens + block_table :1117-1118",
        "布局组合（输入_输出）：mla_v1.py:2868-2871 及 :2899/:2909/:2923/:2931",
        "chunked prefill workspace 算术（288MB→6GB）：attention/utils.py:256-281；"
        "消费 mla_v1.py:728-730",
        "MLA：KV 头数恒 1（TP 复制）：vllm/model_executor/layers/attention/mla_attention.py:467",
        "int8 KV 的块大小约束：attention_v1.py:1561、:1609；C8 变体的滑窗/非 causal 限制："
        "attention_c8_mxfp.py:747/749",
    ], h=192)

    y7 = y6 + 226
    concl(p, y7,
          "<b>一句话记住这一页：</b>attention 有两个算子不是「新旧」而是「形状能力不同」；"
          "paged attention 是<b>要在名单里才启用</b>的加速项，本仓没开 → decode 也走融合 attention。"
          "<b>而「每请求 query ≤ 16」是所有投机解码配置的天花板。</b>")

    legend(p, y7 + 92, extra_fill=("prefill 侧算子", "cloudd"))
