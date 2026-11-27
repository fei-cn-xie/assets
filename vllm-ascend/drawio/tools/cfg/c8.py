"""⑧ 降级链与决策树：引擎什么时候会主动给你降级。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc, table

TITLE = "⑧ 降级链：引擎会「悄悄退一步」，但这次它会打日志"
SUB = ("一句话：当 attention 后端撑不住你要的图模式时，引擎按 "
       "<b>FULL → FULL_AND_PIECEWISE / FULL_DECODE_ONLY → PIECEWISE → NONE</b> 逐级降，"
       "并在日志里说一句；只有「连降级都救不了」才拒绝启动。")


def build(doc: Doc) -> None:
    p = doc.page("⑧ 降级链与决策", pw=1560, ph=1520)
    head(p, TITLE, SUB)
    X = 36

    # --- 降级链 -----------------------------------------------------------
    y = 150
    p.text(X, y - 32, 1300, 22,
           "<b>第一步：五级降级（每级的触发条件与结果）</b>", "text_b")
    rows = [
        ["你要 FULL，后端完全不支持（NEVER）", "直接报错，让你改 PIECEWISE", "拒绝启动"],
        ["你要 FULL，后端只支持部分形状", "降成 FULL_AND_PIECEWISE 或 FULL_DECODE_ONLY",
         "warning"],
        ["decode 要 FULL，后端 NEVER", "降成 PIECEWISE 或 NONE", "warning"],
        ["decode 要 FULL + K>1，后端只支持更弱档", "降成 PIECEWISE 或 NONE",
         "warning（点名 spec-decode）"],
        ["PIECEWISE 也不可用", "去掉 piecewise（→ NONE 或 FULL）", "warning once"],
    ]
    table(p, X, y, [("触发", 520), ("引擎怎么改", 520), ("表现", 380)], rows,
          row_h=40)

    # --- 三条硬拒绝 -------------------------------------------------------
    y2 = y + 40 * 6 + 40
    p.text(X, y2 - 32, 1300, 22,
           "<b>第二步：三条「拒绝启动」（降级也救不了）</b>", "text_b")
    panel(p, X, y2, 460, 160, "① 后端 NEVER + 你非要 FULL", [
        "连降级后的结果里仍带 FULL → 直接 ValueError。",
        "处置：换 attention 后端，或改用 PIECEWISE。",
    ], "fail")
    panel(p, X + 482, y2, 460, 160, "② SP 的 TP 与 K+1 互不整除", [
        "序列并行下桶要取 LCM(K+1, TP)，凑不出来就报错。",
        "处置：调 K、关 SP、或改 TP。",
    ], "fail")
    panel(p, X + 964, y2, 460, 160, "③ Mamba 类模型 max_num_seqs > 块数", [
        "线性注意力需要每序列一个 state 块，块不够直接报错。",
        "处置：降 max_num_seqs 或提 gpu_memory_utilization。",
    ], "fail")

    # --- 决策树 -----------------------------------------------------------
    y3 = y2 + 200
    p.text(X, y3 - 32, 1300, 22,
           "<b>第三步：反过来——遇到症状时该先动哪个参数（决策树）</b>", "text_b")
    p.box(X, y3, 1424, 148,
          "<font style=\"font-size:10px\">"
          "<b>症状 A：启动就 OOM / 报 KV 不够</b> → ① 降 max_model_len 或 max_num_seqs；"
          "② 抬 gpu_memory_utilization（0.90→0.95→0.98）；③ 关掉一项大头（MTP 或图）。<br>"
          "<b>症状 B：单请求很慢（TPOT 高）</b> → ① 确认 D 侧图真的在跑（看 journal 有无 replay 日志）；"
          "② 确认 MTP 接受率（<1.3 就关）；③ 再看是不是桶太小。<br>"
          "<b>症状 C：吞吐上不去</b> → ① 看并发是否被 seqs/KV 卡住；② 看 chunked prefill 是否在切长 prefill；"
          "③ 看是否因形状不整齐掉回 eager。<br>"
          "<b>症状 D：延迟抖动</b> → ① 长 prefill 挤占 decode（调小 batched 或开分块阈值）；"
          "② 图桶不命中造成的偶发回退。</font>",
          "box", "neutral")

    y4 = y3 + 188
    bridge(p, y4, "本页交接：决策树是基于前七页的因果。下一步（⑨）用本仓三套真实部署包，"
                  "把「为什么它这么配」逐条对上。")

    y5 = y4 + 76
    anchors(p, y5, [
        "降级实现：vllm/config/compilation.py:1382-1545（五级：:1402-1417、:1419-1426、"
        ":1429-1453、:1455-1473、:1475-1487、:1489-1501）",
        "后端支持度枚举：vllm/v1/attention/backend.py:559-573",
        "SP × K+1 的 LCM 与硬错：vllm/config/compilation.py:1551-1564",
        "Mamba 块数硬错：vllm/config/compilation.py:1527-1542",
        "昇腾侧再降级：vllm-ascend/vllm_ascend/platform.py:1231-1246、:1346-1352",
        "软件兜底（connector/pooling/encoder-decoder 各自改成 PIECEWISE）："
        "vllm/config/vllm.py:1796-1821、:1773-1781、:1782-1794",
    ], h=120)

    y6 = y5 + 154
    concl(p, y6,
          "<b>一句话记住这一页：</b>降级链是这条联动脉络里<b>唯一会主动说话</b>的部分——"
          "看到「falling back / not supported with … backend」就说明你的图被降了；"
          "而「静默掉出」（形状不命中桶）永远不会有日志。")

    legend(p, y6 + 92)
