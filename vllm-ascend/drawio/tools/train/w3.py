"""③ 损失与 logits：从 hidden 到词表分布，再分成「采样」与「打分」。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc

TITLE = "③ 损失与 logits：模型输出其实是一个「词表上的分布」"
SUB = ("一句话：一次前向的产物是 <b>[token 数, 词表大小]</b> 的分数矩阵；"
       "推理拿它<b>挑一个</b>（采样），训练拿它<b>打分</b>（−log p(正确词)，即交叉熵）。"
       "同一个量，两种用法。")


def build(doc: Doc) -> None:
    p = doc.page("③ 损失与 logits", pw=1560, ph=1460)
    head(p, TITLE, SUB)
    X = 36

    # --- 一条链 -----------------------------------------------------------
    y = 150
    p.text(X, y - 32, 1300, 22,
           "<b>第一步：一条链（前四步仓内都能查到，最后一步是训练侧用法）</b>",
           "text_b")
    chain = [
        ("hidden", "每层输出的隐藏向量<br>[token 数, hidden]", "cloudd"),
        ("lm_head", "乘到词表维度<br>[token 数, 词表大小]", "cloudd"),
        ("log_softmax", "取对数概率<br>仍是 [token 数, 词表大小]", "cloudd"),
        ("推理：挑一个", "argmax / 按概率采样", "cloudp"),
        ("训练：−log p(y)", "取正确词那一列取负 → 一个标量", "fail"),
    ]
    cw = 270
    for i, (name, note, role) in enumerate(chain):
        cx = X + i * (cw + 11)
        p.box(cx, y, cw, 96,
              f"<b>{name}</b><br>"
              f"<font style=\"font-size:9px\">{note}</font>",
              "box_center", role)
        if i:
            p.edge(src_pt=(cx - 9, y + 48), dst_pt=(cx - 1, y + 48), st="arrow",
                   role="http")
    p.text(X, y + 106, 1424, 24,
           "<font style=\"font-size:9px;color:#666666\">"
           "前四步（含 log_softmax）在仓内有实现；最右一格是训练侧独有的用法，"
           "仓内没有它的代码——它只出现在训练框架里。</font>", "subtitle")

    # --- 为什么是「分布」 -------------------------------------------------
    y2 = y + 148
    p.text(X, y2 - 32, 1300, 22,
           "<b>第二步：为什么叫「分布」（三个必须理解的细节）</b>", "text_b")
    panel(p, X, y2, 460, 180, "① 词表有多大", [
        "每一行都是「下一个词是词表里第 i 个词」的分数。",
        "词表通常几万到十几万 → 这个矩阵很宽。",
        "所以 lm_head 是全模型里「输出维度最大」的那一层。",
    ], "neutral")
    panel(p, X + 482, y2, 460, 180, "② 为什么要 log_softmax 而不是 softmax", [
        "概率连乘会下溢，取对数后变成相加，数值稳定；",
        "而且 log 概率单调 → 排序不变（仓内注释明说这一点）。",
        "仓内 softmax/log_softmax 一律用 fp32 算，避免精度问题。",
    ], "cloudd")
    panel(p, X + 964, y2, 460, 180, "③ 训练要「每个位置」，推理只要「最后一个」", [
        "训练：每个位置都有一个「正确的下一个词」，所以每个位置都算损失；",
        "推理：只关心接下来那一个词，所以只取最后一个位置的分布。",
        "→ 这就是「推理比训练省一大截」的第一层原因。",
    ], "fail")

    # --- 交叉熵 -----------------------------------------------------------
    y3 = y2 + 220
    p.box(X, y3, 1424, 116,
          "<b>第三步：交叉熵到底在算什么（一句话 + 一个直觉）</b><br>"
          "<font style=\"font-size:10px\">"
          "交叉熵 = <b>−log p(正确词)</b>：模型给正确词的概率越高，损失越小；给得越低，损失越大。<br>"
          "直觉：它衡量的不是「答案对不对」，而是「模型有多确信正确答案」——"
          "所以哪怕 argmax 对了，只要概率很低，损失依然大，梯度依然会推着它变得更确信。<br>"
          "<font color=\"#c0392b\">这一步在 vLLM 仓里<b>不存在</b>：vLLM 只把 log 概率算出来"
          "（用于返回 logprobs），不拿它跟标签比。</font></font>",
          "box", "neutral")

    # --- 推理侧的加工 -----------------------------------------------------
    y4 = y3 + 156
    p.text(X, y4 - 32, 1300, 22,
           "<b>第四步：推理会在「分布」上再做四件事（训练侧不做或做法不同）</b>",
           "text_b")
    from dg import table
    rows = [
        ["温度 temperature", "把 logits 除以 T 再 softmax", "T→0 变贪心，T 大更随机"],
        ["top-k / top-p", "只保留前 k 个 / 累计概率到 p 的词", "裁掉长尾，避免抽到离谱词"],
        ["罚分（重复/存在/频率）", "对已出现过的词降分", "抑制复读"],
        ["logprobs 输出", "把 log 概率返回给调用方", "用于打分、置信度、RL 训练信号"],
    ]
    table(p, X, y4, [("加工", 300), ("怎么做", 560), ("目的", 564)], rows,
          row_h=36)

    y5 = y4 + 36 * 5 + 36
    bridge(p, y5, "本页交接：logits 这条链解释了「训练的损失从哪来」。"
                  "下一步（④）算显存账：训练为什么要多存三份东西，推理为什么不用。")

    y6 = y5 + 76
    anchors(p, y6, [
        "logits 的产生：vllm/model_executor/layers/logits_processor.py:108；"
        "fp32 输出 :163-170",
        "softmax 求概率：vllm/v1/sample/ops/topk_topp_sampler.py:168",
        "log_softmax（logprobs）：vllm/v1/sample/sampler.py:306-307；fp32 注释 :96-97；"
        "单调性注释 :213-214",
        "形状文档：vllm/docs/design/logits_processors.md:13（（num_requests) × (vocab_size)）",
        "「只算采样位」的切片：vllm/v1/worker/gpu/model_runner.py:915",
        "仓内明说没有反向：vllm/third_party/flash_linear_attention/ops/layernorm_guard.py:514-516",
    ], h=120)

    y7 = y6 + 154
    concl(p, y7,
          "<b>一句话记住这一页：</b>模型输出是「词表上的分数」；"
          "推理<b>挑一个</b>、训练<b>用正确答案给它打分</b>。"
          "理解这一点，才能理解为什么训练要每个位置的分布、推理只要最后一个。")

    legend(p, y7 + 92, extra_fill=("训练侧用法（仓内无实现）", "fail"))
