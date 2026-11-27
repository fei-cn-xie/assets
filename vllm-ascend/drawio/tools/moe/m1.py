"""① 什么是 MoE：从「一个 FFN」到「很多专家，每个 token 只走几个」。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc

TITLE = "① 什么是 MoE：把一层 FFN 换成「很多个 FFN + 一个门卫」"
SUB = ("一句话：dense 模型每层只有一个 FFN，所有 token 都走它；"
       "MoE 把这一层换成 <b>N 个 FFN（专家）+ 一个路由器（门卫）</b>，"
       "每个 token 只被送进其中 <b>top-k 个</b>。于是<b>总参数变多，但每个 token 的计算量几乎不变</b>。")


def build(doc: Doc) -> None:
    p = doc.page("① 什么是 MoE", pw=1560, ph=1440)
    head(p, TITLE, SUB)
    X = 36

    # --- 对照 -------------------------------------------------------------
    y = 150
    p.text(X, y - 32, 1300, 22,
           "<b>第一步：一张对照图（左边 dense，右边 MoE）</b>", "text_b")
    p.box(X, y, 690, 168,
          "<b>dense 层</b><br><br>"
          "<font style=\"font-size:10px\">"
          "　　输入 x（每个 token 一个向量）<br>"
          "　　　　↓<br>"
          "　　升维 → 非线性 → 降维（一个大 FFN）<br>"
          "　　　　↓<br>"
          "　　输出 y<br><br>"
          "<b>所有 token 都走同一条路</b>；参数量 = 这一个 FFN 的大小。</font>",
          "box", "cloudd")
    p.box(X + 730, y, 694, 168,
          "<b>MoE 层</b><br><br>"
          "<font style=\"font-size:10px\">"
          "　　输入 x<br>"
          "　　　　↓<br>"
          "　　<b>门卫</b>：算出「该去哪些专家」+「权重多少」<br>"
          "　　　　↓<br>"
          "　　只把 x 送进 top-k 个专家（其余专家这一步不参与）<br>"
          "　　　　↓<br>"
          "　　把 k 个结果按门卫给的权重<b>加权相加</b> → y<br><br>"
          "<b>不同 token 走不同专家</b>；参数量 = 全部专家之和。</font>",
          "box", "cloudp")

    # --- 三个关键量 -------------------------------------------------------
    y2 = y + 208
    p.text(X, y2 - 32, 1300, 22,
           "<b>第二步：三个必须分清的量（MoE 全部反直觉都来自这里）</b>", "text_b")
    cards = [
        ("总参数量", "N 个专家之和\n（模型“看起来很大”）", "neutral"),
        ("激活参数量", "每 token 只走 k 个\n（计算量“其实不大”）", "cloudd"),
        ("显存占用", "全部专家都要驻留\n（推理端最痛的一条）", "fail"),
    ]
    cw = 460
    for i, (name, body, role) in enumerate(cards):
        cx = X + i * (cw + 12)
        p.box(cx, y2, cw, 116, "", "box", role)
        p.text(cx + 10, y2 + 10, cw - 20, 24, f"<b>{name}</b>", "text_b")
        p.text(cx + 10, y2 + 36, cw - 20, 74,
               "<font style=\"font-size:10px\">" + body.replace("\n", "<br>") + "</font>",
               "text")

    # --- 一个算例 ---------------------------------------------------------
    y3 = y2 + 156
    p.box(X, y3, 1424, 132,
          "<b>第三步：一个算例（用相对量级，不做绝对参数承诺）</b><br>"
          "<font style=\"font-size:10px\">"
          "设 dense 模型某层 FFN 的参数为 <b>1 份</b>。换成 <b>64 个专家、每 token 走 2 个</b>：<br>"
          "· <b>总参数</b> ≈ 64 份 → 模型“看起来”大了 64 倍；<br>"
          "· <b>每 token 的计算量</b> ≈ 2 份 → 只比原来多 1 倍（而不是 64 倍）；<br>"
          "· <b>显存</b> 仍要放下 64 份 → 所以 MoE 是「<b>用显存换参数量</b>」的一条路。<br>"
          "<font color=\"#c0392b\">这就是为什么 MoE 模型的「参数量」与「能跑多快」几乎无关，"
          "而和「显存够不够」强相关。</font></font>",
          "box", "neutral")

    # --- 为什么有用 -------------------------------------------------------
    y4 = y3 + 172
    panel(p, X, y4, 700, 152, "它解决了什么", [
        "在参数总量不变的前提下，想让模型「知道得更多」→ 加专家；",
        "在计算量不变的前提下，想让推理「不变慢」→ 每 token 只走 k 个。",
        "所以 MoE 是「容量/算力比」最高的一种结构（这也是它在大模型上流行的原因）。",
    ], "cloudp")
    panel(p, X + 760, y4, 664, 152, "它带来了什么新问题", [
        "① 显存：全部专家都要驻留 → 单卡放不下 → 要切（专家并行）；",
        "② 通信：token 要去别的卡上的专家 → 要搬 token（all-to-all）；",
        "③ 负载：热门专家忙死、冷门专家闲着 → 要均衡（训练期辅助损失、推理期换位）。",
        "后面 9 页就是把这三个问题逐个讲清。",
    ], "fail")

    y5 = y4 + 192
    bridge(p, y5, "本页交接：MoE 的结构说清了。下一步（②）跟着一个 token 走一遍——"
                  "它在一层 MoE 里到底被怎么处理。")

    y6 = y5 + 76
    anchors(p, y6, [
        "MoE 层的实现入口：vllm/model_executor/layers/fused_moe/layer.py（FusedMoE 类）",
        "配置字段（专家数 / top-k / 中间维度）：见 ② 页与 moe-math 页汇总；"
        "本仓模型的 config.json 里对应 n_routed_experts / num_experts_per_tok / moe_intermediate_size",
        "「dense FFN vs MoE」在模型代码里的分叉：各模型文件按 config 决定用 "
        "MLP 还是 FusedMoE（例如 deepseek 系模型文件）",
    ], h=104)

    y7 = y6 + 136
    concl(p, y7,
          "<b>一句话记住这一页：</b>MoE = 很多专家 + 一个门卫 + 每 token 只走 top-k；"
          "<b>总参数按「专家数」涨，计算量按「top-k」涨，显存也按「专家数」涨</b>——"
          "后两条就是推理侧所有麻烦的来源。")

    legend(p, y7 + 92)
