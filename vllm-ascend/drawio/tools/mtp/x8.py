"""⑧ 昇腾侧落地：MTP 层怎么跑、draft 的图与 eager、几道硬上限。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc

TITLE = "⑧ 昇腾侧：MTP 层排在 backbone 之后，draft 默认跑 eager"
SUB = ("一句话：昇腾的 MTP 走 eagle 外壳（<font style=\"font-size:9px\">AscendEagleProposer</font>），"
       "MTP 层是 backbone 之后的额外层、<b>有自己独立的 KV</b>；"
       "而部署里 draft 侧一律 <font style=\"font-size:9px\">enforce_eager: true</font>。")


def build(doc: Doc) -> None:
    p = doc.page("⑧ 昇腾侧 MTP", pw=1560, ph=1520)
    head(p, TITLE, SUB)
    X = 36

    # --- 结构 -------------------------------------------------------------
    y = 152
    p.text(X, y - 34, 1300, 22,
           "<b>第一步：MTP 层在哪、和 target 共享什么</b>", "text_b")
    p.box(X, y, 690, 172,
          "<font style=\"font-size:10px\">"
          "<b>backbone（0 … N-1 层）</b>：target 主体，产出 hidden states<br>"
          "　　　↓<br>"
          "<b>MTP 层（第 N 层起）</b>：吃 hidden + 下一 token 的 embedding，猜第 K 步<br><br>"
          "<font color=\"#c0392b\">共享：embedding 与 lm_head（省显存）<br>"
          "不共享：KV cache —— MTP 层有自己的层号与槽位</font></font>",
          "box", "cloudp")
    panel(p, X + 730, y, 694, 172, "为什么要记「共享权重但不共享 KV」", [
        "共享权重 → 不用再下一份 embedding/lm_head，显存增量小；",
        "不共享 KV → 草案的 KV 要单独占位（PD 传输时也作为额外层单独编号，"
        "但 MTP 多层共用同一份层，所以只传一次）。",
        "搞混这两件事，就会误以为「MTP 不占显存」或「KV 也能共享」。",
    ], "cloudd")

    # --- 起草流程 ---------------------------------------------------------
    y2 = y + 212
    p.text(X, y2 - 34, 1300, 22,
           "<b>第二步：一次起草的内部（MTP 复用同一层 K 次）</b>", "text_b")
    p.box(X, y2, 1424, 128,
          "<font style=\"font-size:10px\">"
          "① target 前向 → 得到本步 logits 与 hidden states；同一函数体里立刻接着起草。<br>"
          "② 第 0 步：用 target 的 hidden + 下一 token 的 embedding 跑一次 MTP 层 → 得到第 1 个草案。<br>"
          "③ 之后串行 K−1 次，<b>复用同一个 MTP 层</b>（层选择 <font style=\"font-size:9px\">"
          "spec_step_idx % num_mtp_layers</font>）→ 得到第 2 … K 个草案。<br>"
          "<font color=\"#c0392b\">所以「MTP 只有一层权重也能猜 K 个」——靠的是反复跑同一层；"
          "这也是官方建议 K=1~3 的原因（越往后越不准）。</font></font>",
          "box", "cloudd")

    # --- 三道硬上限 -------------------------------------------------------
    y3 = y2 + 168
    p.text(X, y3 - 34, 1300, 22,
           "<b>第三步：昇腾侧的三道硬上限（都会直接报错，不是警告）</b>", "text_b")
    panel(p, X, y3, 460, 190, "① TND 布局：K+1 ≤ 16", [
        "昇腾 fused attention 的 TND 布局对 query 长度有上限，",
        "代码里断言 decode_threshold ≤ 16（即 K ≤ 15）；",
        "部署里用 K=3（个别包 K=1），离上限很远。",
    ], "fail")
    panel(p, X + 482, y3, 460, 190, "② padded drafter 与整图互斥", [
        "「整图 + 变长 draft 批」这个组合不支持：",
        "代码直接 NotImplementedError，提示去掉 disable_padded_drafter_batch。",
        "所以要么 batch 补齐、要么别用整图。",
    ], "fail")
    panel(p, X + 964, y3, 460, 190, "③ draft 与某些图/并行特性互斥", [
        "draft 只有在「target 开了整图 + draft 自己没 enforce_eager」时才包 ACLGraphWrapper；",
        "此外 lmhead TP 与 draft 的几种采样方式互斥（4 条 NotImplementedError）；",
        "MLA/MTP 的整图与 CP 不兼容。",
    ], "fail")

    # --- 本仓实际配置 -----------------------------------------------------
    y4 = y3 + 230
    p.text(X, y4 - 34, 1300, 22,
           "<b>第四步：本仓部署脚本里实际怎么配（12 处实测）</b>", "text_b")
    from dg import table
    rows = [
        ["method", "一律 mtp（个别写 qwen3_5_mtp，会被归一成 mtp 并打 deprecation 警告）"],
        ["num_speculative_tokens", "3（fix_1120 那套是 1）"],
        ["enforce_eager", "true —— <b>只作用于 draft</b>，target 仍然走图"],
        ["model（draft 路径）", "不给：默认取 target 的权重目录，MTP 权重就是同目录里的 mtp.* 张量"],
        ["target 的图模式", "P/edge 用 --enforce-eager；D 用 FULL_DECODE_ONLY"],
    ]
    table(p, X, y4, [("配置项", 260), ("本仓实测取值与含义", 1160)], rows, row_h=36)
    p.text(X, y4 + 36 * 6 + 6, 1424, 40,
           "<font style=\"font-size:9px;color:#666666\">"
           "关键理解：<b>speculative-config 里的 enforce_eager 只关 draft 的图</b>，"
           "与 target 的 --enforce-eager 是两件事——所以「target 跑图 + draft 跑 eager」是合法且常见的组合。</font>",
           "subtitle")

    y5 = y4 + 36 * 6 + 54
    bridge(p, y5, "本页交接：昇腾侧讲完，最后一块是本仓特有的「LWD 边云 PD」下 MTP 怎么与 KV 交接共存。"
                  "下一步（⑨）。")

    y6 = y5 + 76
    anchors(p, y6, [
        "MTP 走 eagle 外壳：vllm-ascend/vllm_ascend/spec_decode/__init__.py:54-60、"
        "eagle_proposer.py:10",
        "draft 主体与起草：vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py:1042"
        "（_propose）、:1517-1535（首步）、:1716（串行复用）",
        "MTP 层与共享权重：vllm/model_executor/models/deepseek_mtp.py:142-147、:212-219；"
        "vllm/v1/spec_decode/llm_base_proposer.py:1406、:1517",
        "TND ≤16 断言：vllm-ascend/.../attention/attention_v1.py:273-277（及 mla/sfa/dsa 同构处）",
        "padded drafter × 整图互斥：vllm-ascend/.../spec_decode/llm_base_proposer.py:346-356",
        "draft 是否跑图：同文件 :271、:670-684",
        "部署配置：assets/c_home/bash/pds/30rc_1124/*/cloud_{p,d}.sh（--speculative-config）",
    ], h=128)

    y7 = y6 + 162
    concl(p, y7,
          "<b>一句话记住这一页：</b>昇腾 MTP = 「backbone 之后的一层，反复跑 K 次」；"
          "共享权重、不共享 KV；部署里 draft 走 eager、target 走图；"
          "三道硬上限（TND≤16、padded×整图、draft×部分特性）踩到就直接报错。")

    legend(p, y7 + 92)
