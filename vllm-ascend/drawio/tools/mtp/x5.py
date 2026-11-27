"""⑤ 验证规则：贪心看 argmax、随机看 p/q（拒绝采样）。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc

TITLE = "⑤ 验证规则：贪心看 argmax、随机比 p/q —— 为什么这样就不改变分布"
SUB = ("一句话：验证不是「草案看着像就收」，而是<b>用 target 自己的概率做裁判</b>："
       "贪心模式下比 argmax 是否一致；随机模式下按 min(1, p_target/p_draft) 的概率接受。")


def build(doc: Doc) -> None:
    p = doc.page("⑤ 验证规则", pw=1560, ph=1480)
    head(p, TITLE, SUB)
    X = 36

    # --- 两条分支 ---------------------------------------------------------
    y = 152
    p.text(X, y - 34, 1300, 22,
           "<b>第一步：两条分支（同一个 sampler 里按 temperature 是否 0 分开）</b>",
           "text_b")
    panel(p, X, y, 700, 200, "贪心模式（temperature = 0）", [
        "对每个候选位置：target 的 argmax 是多少？",
        "· 草案 token == target 的 argmax → 接受，继续看下一位",
        "· 不相等 → 拒绝，并把 target 的 argmax 作为这一位的输出，后面全作废",
        "全中 → 额外给 1 个 bonus token（也从 target 采）",
        "",
        "本质：等价于「取草案与真实输出的最长公共前缀」。",
    ], "cloudd")
    panel(p, X + 760, y, 664, 200, "随机模式（temperature > 0）", [
        "对每个候选位置，取一个均匀随机数 u ∈ [0,1)：",
        "· 若 <b>u < p_target(token) / p_draft(token)</b> → 接受",
        "· 否则拒绝；被拒的这位不是「丢掉」，而是从<b>残差分布</b>里重新采一个：",
        "　 p_residual ∝ max(p_target − p_draft, 0)",
        "",
        "本质：接受 + 补位的组合，使最终分布严格等于 target 的分布（论文 arXiv:2211.17192）。",
    ], "cloudp")

    # --- 为什么无损 -------------------------------------------------------
    y2 = y + 240
    p.text(X, y2 - 34, 1300, 22,
           "<b>第二步：为什么这套规则是「无损」的（只需三层直觉）</b>", "text_b")
    p.box(X, y2, 1424, 128,
          "<font style=\"font-size:10px\">"
          "① <b>草案只是「提议」</b>：它不决定输出，只决定「我们多看几个位置」。<br>"
          "② <b>接受有概率上限</b>：p_target/p_draft 比值小的地方（= target 不喜欢的 token）就难被接受；"
          "比值 ≥1 的地方（target 也喜欢）必然接受。<br>"
          "③ <b>拒绝处补位</b>：被拒的位置用 max(p_t − p_d, 0) 归一化后重采，"
          "正好补上「草案少给的那部分概率质量」。<br>"
          "<font color=\"#c0392b\">接受 + 补位合起来，恰好等于「直接用 target 分布采样」——"
          "所以输出分布不变，只是变快了。</font></font>",
          "box", "neutral")

    # --- 三种边界 ---------------------------------------------------------
    y3 = y2 + 168
    p.text(X, y3 - 34, 1300, 22,
           "<b>第三步：三个必须知道的边界（代码里都有专门处理）</b>", "text_b")
    panel(p, X, y3, 460, 176, "① 没有概率的草案（n-gram）", [
        "查表出来的草案没有 q 可算，于是代码里令 p_draft = 1，",
        "接受条件退化成 <b>u < p_target(token)</b>——「target 觉得够可能就收」。",
    ], "fail")
    panel(p, X + 482, y3, 460, 176, "② 草案概率为 0", [
        "若 p_draft == 0，比值会变成 inf/NaN。",
        "代码里显式判为「拒绝」，避免除零带来的数值问题。",
    ], "fail")
    panel(p, X + 964, y3, 460, 176, "③ padding 的位置", [
        "为了凑形状，scheduler 会填 -1 占位；验证时这些位置一律按「拒绝」处理，",
        "不会污染结果（同时保证图捕获要求的固定形状）。",
    ], "fail")

    # --- bonus -----------------------------------------------------------
    y4 = y3 + 216
    p.box(X, y4, 1424, 100,
          "<b>bonus token：为什么全中时要多送一个</b><br>"
          "<font style=\"font-size:10px\">"
          "一次前向算了 N+1 行（N 个草案位置 + 1 个「下一位置」）。"
          "如果 N 个草案全被接受，那么第 N+1 行 logits 本来就摆在手上——"
          "直接用它采一个 token 是<b>免费的</b>。<br>"
          "所以「全中」的产出是 N+1 个 token。也正因如此，"
          "平均接受长度（mean acceptance length）的统计口径是 <b>1 + 接受数/草案数</b>。</font>",
          "box", "cloudd")

    y5 = y4 + 140
    bridge(p, y5, "本页交接：被拒的那些位置，KV 已经写进去了——怎么处理？"
                  "下一步（⑥）就是这套机制最容易被误解的一环。")

    y6 = y5 + 76
    anchors(p, y6, [
        "两条分支与入口：vllm/v1/sample/rejection_sampler.py:188-200（入口）、"
        ":719-773（贪心 kernel，:761 比较 argmax）、:822-893（随机 kernel）",
        "接受条件与除零保护：rejection_sampler.py:864-877"
        "（accepted = draft_prob > 0 and target_prob/draft_prob >= uniform_prob）",
        "无概率草案 p_draft=1：rejection_sampler.py:864-865",
        "残差补位：rejection_sampler.py:1043-1077（prob = max(target_prob - draft_prob, 0)）",
        "bonus token：rejection_sampler.py:55-65、:887-893",
        "pad 位置按 -1 拒绝：rejection_sampler.py:856-859；scheduler.py:1330-1334",
    ], h=120)

    y7 = y6 + 154
    concl(p, y7,
          "<b>一句话记住这一页：</b>验证的裁判是 target 自己的概率——"
          "贪心比 argmax、随机比 p/q 并按时被拒后从残差补位；"
          "<b>因此投机解码不改变输出分布，只改变「一次前向能推进几个 token」。</b>")

    legend(p, y7 + 92)
