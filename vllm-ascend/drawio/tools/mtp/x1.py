"""① 什么是 draft / MTP：先猜几个，再一次验证。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc

TITLE = "① 什么是 draft（草案）/ MTP：先猜几个，再一次验证"
SUB = ("一句话：<b>用小代价先猜出接下来的 N 个 token（draft），"
       "再让大模型一次前向把这 N 个一起验证</b>；猜对的直接白拿，猜错就从错的地方截断。")


def build(doc: Doc) -> None:
    p = doc.page("① draft 与 MTP 是什么", pw=1560, ph=1400)
    head(p, TITLE, SUB)
    X = 36

    # --- 两个方案对照 -----------------------------------------------------
    y = 152
    p.text(X, y - 34, 1300, 22,
           "<b>第一步：没有 draft 时，每生成一个字都要跑一次完整前向</b>", "text_b")
    p.box(X, y, 1424, 108,
          "<font style=\"font-size:10px\">"
          "生成 4 个字 = <b>4 次完整前向</b>，每次前向都要把全部权重从显存读一遍"
          "（这就是「decode 是带宽瓶颈」的直接后果）。<br>"
          "注意：每次前向其实<b>只看 1 个新 token</b>，却要付整份权重的访存成本——"
          "这就是为什么 decode 的算术强度只有 ~2 FLOP/字节（见入门套 ⑤）。<br>"
          "于是问题变成：<b>能不能让一次前向多干点活？</b></font>",
          "box", "fail")

    y2 = y + 148
    p.text(X, y2 - 34, 1300, 22,
           "<b>第二步：加一个「便宜的起草者」之后</b>", "text_b")
    panel(p, X, y2, 690, 176, "① 起草（draft）", [
        "用一个便宜的来源先猜出接下来的 N 个 token：",
        "· 同模型的额外小头（MTP 层）、· 独立小模型、· 直接查 n-gram。",
        "成本远低于「让大模型跑 N 次」——因为只跑一次小东西（或干脆不跑模型）。",
        "得到 N 个候选 token，例如 [北京, 是, 中国, 的]。",
    ], "cloudd")
    panel(p, X + 730, y2, 694, 176, "② 验证（verify）", [
        "把 N 个候选<b>拼在输入里</b>，让大模型<b>一次前向</b>算出这 N 个位置的 logits。",
        "然后逐位置比对：这个位置上，大模型自己会选哪个 token？",
        "· 一致 → 接受（白拿一个字）；· 不一致 → 从这里截断，用它自己的结果。",
        "所以「一次前向」最多能产出 N+1 个 token（N 个接受 + 1 个 bonus）。",
    ], "cloudp")

    # --- 关键性质 ---------------------------------------------------------
    y3 = y2 + 216
    p.box(X, y3, 1424, 120,
          "<b>为什么这不改变「输出分布」（这是投机解码的立身之本）</b><br>"
          "<font style=\"font-size:10px\">"
          "直觉上的担心：草案是便宜模型猜的，接受它会不会让结果变差？<br>"
          "答案：<b>不会</b>——因为接受与否由<b>大模型自己的概率</b>决定："
          "只有当大模型「也认为这个 token 够合理」时才接受，否则就按大模型的分布重新采样一个补上。<br>"
          "严格版本叫<b>拒绝采样（rejection sampling）</b>，"
          "保证最终输出的分布与「老老实实逐字生成」完全一致（本仓代码注释直接引用了论文 arXiv:2211.17192）。</font>",
          "box", "neutral")

    # --- MTP 这个名字 -----------------------------------------------------
    y4 = y3 + 160
    p.text(X, y4 - 34, 1300, 22,
           "<b>第三步：「MTP」这个词在两种语境下出现，别混</b>", "text_b")
    panel(p, X, y4, 690, 150, "训练期的 MTP（多 token 预测辅助头）", [
        "在训练时额外挂几个「预测未来第 2、3 个 token」的头，",
        "让模型学到「往后看几步」的能力（DeepSeek-V3 风格）。",
        "好处：训练信号更密；同时给推理期留下了一个可用的草稿头。",
    ], "neutral")
    panel(p, X + 730, y4, 694, 150, "推理期的 MTP（= 用这个头来起草）", [
        "推理时把上面那个头当作 <b>draft 层</b>：吃 target 的隐状态，自回归猜 N 个。",
        "本仓代码里 <font style=\"font-size:9px\">method='mtp'</font> 会被当成 "
        "<font style=\"font-size:9px\">eagle</font> 一类处理（共享同一套 proposer）。",
        "所以「MTP 加速」= 「用 MTP 头起草 + 大模型验证」= 投机解码的一种特例。",
    ], "cloudd")

    y5 = y4 + 190
    bridge(p, y5, "本页交接：这一页只说了「能白拿」。下一步（②）把「为什么划算」算成数："
                  "一次前向算 1+N 个位置，比算 1 个贵多少？")

    y6 = y5 + 76
    anchors(p, y6, [
        "「一次前向处理 1+N 个 query」：vllm/config/vllm.py:660-679"
        "（uniform_decode_query_len = 1 + num_speculative_tokens）",
        "草稿 N 的配置入口：vllm/config/speculative.py:381（num_speculative_tokens）",
        "验证与拒绝采样：vllm/v1/sample/rejection_sampler.py:45-65（术语表与论文引用）、"
        ":188-200（入口）",
        "本步要验证的草稿集合：vllm/v1/core/sched/output.py:334-337"
        "（scheduled_spec_decode_tokens）",
    ], h=104)

    y7 = y6 + 136
    concl(p, y7,
          "<b>一句话记住这一页：</b>投机解码 = <b>先猜（draft）+ 一次前向验证</b>；"
          "猜对的直接白拿、猜错从错处截断；因为接受规则由大模型自己的概率决定，"
          "<b>输出分布不变</b>——这就是它敢上线的原因。")

    legend(p, y7 + 92)
