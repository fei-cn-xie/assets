"""② 一次训练 step 的完整数据流（并标出哪一步在推理里也发生）。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc

TITLE = "② 一次训练 step：九步数据流，其中只有四步会出现在推理里"
SUB = ("一句话：训练一步 = <b>切 token → 前向 → 算损失 → 反向 → 更新</b>；"
       "推理只保留「切 token → 前向 → 采样」，把中间三步全砍掉——"
       "这就是为什么推理能省下大半显存。")


def build(doc: Doc) -> None:
    p = doc.page("② 训练 step 数据流", pw=1560, ph=1520)
    head(p, TITLE, SUB)
    X = 36

    y = 148
    p.text(X, y - 32, 1300, 22,
           "<b>第一步：九步（蓝＝训练独有，绿＝推理也有）</b>", "text_b")
    steps = [
        ("①", "取一条文本", "数据集采样 → 一段字符串", "neutral"),
        ("②", "分词", "字符串 → token id 序列", "cloudd"),
        ("③", "拼成 batch", "多条序列补齐或打包（二维/一维）", "cloudd"),
        ("④", "前向 forward", "逐层算 → 每个位置的 logits", "cloudd"),
        ("⑤", "算损失 loss", "与「下一个词」比对 → 一个标量", "cloudp"),
        ("⑥", "反向 backward", "从 loss 回传 → 每个参数的梯度", "cloudp"),
        ("⑦", "更新参数", "优化器按梯度走一小步", "cloudp"),
        ("⑧", "清梯度", "梯度清零，进入下一步", "cloudp"),
        ("⑨", "周期存档", "checkpoint：权重 + 优化器状态", "data"),
    ]
    cw = 156
    for i, (no, title, note, role) in enumerate(steps):
        cx = X + i * (cw + 6)
        p.box(cx, y, cw, 104,
              f"<b>{no}</b> {title}<br>"
              f"<font style=\"font-size:9px;color:#666666\">{note}</font>",
              "box_center", role)
        if i:
            p.edge(src_pt=(cx - 5, y + 52), dst_pt=(cx - 1, y + 52), st="arrow",
                   role="http")

    # --- 推理只做哪几步 ---------------------------------------------------
    y2 = y + 144
    p.text(X, y2 - 32, 1300, 22,
           "<b>第二步：推理把 ④ 之后换成什么（砍掉 ⑤~⑧，加上采样）</b>", "text_b")
    p.box(X, y2, 1424, 116,
          "<font style=\"font-size:10px\">"
          "推理：① 文本 → ② 分词 → ③ 组批 → ④ <b>前向</b> → "
          "<b>⑤′ 取最后一个位置的 logits → 采样一个 token</b> → 把它接回输入，重复。<br>"
          "区别的关键：<b>训练要「每个位置」的 logits</b>（因为要跟每个位置的正确下一个词比对），"
          "而<b>推理只要「最后一个位置」的 logits</b>（因为只需要下一个词）。<br>"
          "<font color=\"#c0392b\">注意 ⑤′ 之后没有损失、没有梯度、没有更新——"
          "权重从加载那一刻起就不变了（唯一的例外是图捕获时的格式重排）。</font></font>",
          "box", "neutral")

    # --- 三个「看起来一样其实不同」 ---------------------------------------
    y3 = y2 + 156
    p.text(X, y3 - 32, 1300, 22,
           "<b>第三步：三处「名字一样、语义不一样」的地方（最容易混）</b>", "text_b")
    panel(p, X, y3, 460, 176, "batch 的含义", [
        "训练：为了「梯度更稳、算力更满」把样本堆起来；",
        "推理：为了「吞吐」把不同请求拼一起，但会牺牲单请求延迟。",
        "→ 训练 batch 越大越好；推理 batch 是个权衡（入门套 ⑥）。",
    ], "fail")
    panel(p, X + 482, y3, 460, 176, "序列长度的处理", [
        "训练：整条样本一次算完（长度固定后 padding）；",
        "推理：先算 prompt（一次），再逐 token 续写 —— 这就是 prefill/decode 之分。",
        "→ 同一个 forward，被切成两种形态调用（入门套 ④）。",
    ], "fail")
    panel(p, X + 964, y3, 460, 176, "精度", [
        "训练：前向用低精度、但主权重与梯度累加常保持高精度（数值稳定）；",
        "推理：整份权重就是低精度（甚至量化），因为不需要数值稳定性。",
        "→ 这是「量化」在推理侧能成立的前提（见 ⑧）。",
    ], "fail")

    y4 = y3 + 216
    bridge(p, y4, "本页交接：九步里最关键的一步是 ⑤ 算损失——它定义「模型在学什么」。"
                  "下一步（③）把损失与 logits 讲清：为什么它就是一个词表上的概率分布。")

    y5 = y4 + 76
    anchors(p, y5, [
        "推理侧的 logits 计算与采样：vllm/v1/sample/sampler.py:21（Sampler）、:73（forward）",
        "词表维度的分布：vllm/model_executor/layers/logits_processor.py（按 vocab 维算）",
        "「只取最后一个位置」的体现：gpu_model_runner 里 logits_indices 的构造"
        "（vllm/v1/worker/gpu_model_runner.py:4478-4499 附近）",
        "训练侧的五步（损失/反向/更新/清梯度/存档）在 vLLM 仓内<b>不存在</b>："
        "见 ① 页的说明",
    ], h=104)

    y6 = y5 + 136
    concl(p, y6,
          "<b>一句话记住这一页：</b>训练一步九步、推理一步四步；"
          "差别集中在「要不要对每个位置算损失、要不要保留中间激活」——"
          "<b>这两件事决定了显存里有没有梯度与激活这两块。</b>")

    legend(p, y6 + 92, extra_fill=("训练独有", "cloudp"))
