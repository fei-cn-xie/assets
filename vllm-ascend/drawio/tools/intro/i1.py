"""① 训练在做什么：一条链看清「权重是怎么来的」。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc

TITLE = "① 训练在做什么：权重是怎么「学」出来的"
SUB = ("一句话：训练 = <b>前向算一个预测 → 用损失函数给它的错打分 → 反向传播算出"
       "「每个参数往哪边调」→ 优化器真的调一下</b>，重复几百万次。"
       "推理只是把这条链的第一段单独拿出来用。")


def build(doc: Doc) -> None:
    p = doc.page("① 训练在做什么", pw=1560, ph=1400)
    head(p, TITLE, SUB)
    X = 36

    # --- 四步链条 ---------------------------------------------------------
    y = 152
    p.text(X, y - 34, 1200, 22,
           "<b>第一步：训练的四步循环（每步都对应代码里真实存在的东西）</b>",
           "text_b")
    steps = [
        ("①", "前向 forward", "输入一批文本 → 逐层算 → 输出每个位置的 logits", "前向与推理时完全一样", "cloudd"),
        ("②", "损失 loss", "拿 logits 和「正确下一个词」比 → 一个标量（越小越准）", "交叉熵", "neutral"),
        ("③", "反向 backward", "从 loss 往回求梯度：<b>每个参数该往哪边调、调多少</b>", "只有训练才有", "cloudp"),
        ("④", "更新 optimizer", "按梯度把参数推开一小步（学习率决定步子大小）", "AdamW 等", "neutral"),
    ]
    cw = 340
    for i, (no, title, body, note, role) in enumerate(steps):
        cx = X + i * (cw + 21)
        p.box(cx, y, cw, 118,
              f"<b>{no} {title}</b><br>"
              f"<font style=\"font-size:10px\">{body}</font><br>"
              f"<font style=\"font-size:9px;color:#666666\">{note}</font>",
              "box", role)
        if i:
            p.edge(src_pt=(cx - 19, y + 59), dst_pt=(cx - 3, y + 59), st="arrow",
                   role="http")
    p.edge(src_pt=(X + 3 * (cw + 21) + cw / 2, y + 118),
           dst_pt=(X + cw / 2, y + 138), st="arrow_dash", role="ctrl",
           label="重复上百万次", label_pos=0)

    # --- 训练 vs 推理 -----------------------------------------------------
    y2 = y + 186
    p.text(X, y2 - 34, 1200, 22,
           "<b>第二步：为什么「训练完」的权重，在推理时是只读的常量</b>", "text_b")
    panel(p, X, y2, 700, 176, "训练时：权重是「变量」", [
        "· 每个参数都有两份内存开销：值 + 梯度（还有优化器的动量等状态）；",
        "· 前向要保留中间激活，供反向链式求导用（显存大户）；",
        "· batch 越大越好：梯度更稳、算力利用率更高；",
        "· 精度常常混合：fp16/bf16 算、fp32 存主权重。",
    ], "cloudp")
    panel(p, X + 760, y2, 664, 176, "推理时：权重是「常量」", [
        "· 只要值，不要梯度、不要优化器状态、不保留反向用的激活；",
        "· 权重加载进显存后<b>只读</b>（唯一的例外是图捕获时的格式重排）；",
        "· batch 不再是「越大越好」：它换的是吞吐，牺牲的是单请求延迟；",
        "· 显存主要被「权重 + KV cache」占掉，而不是激活。",
    ], "cloudd")

    y3 = y2 + 216
    panel(p, X, y3, 700, 148, "一句话把两者接起来", [
        "训练产出的是<b>一份权重文件</b>（如 safetensors）；",
        "推理系统的第一件事就是<b>把这份文件读进显存</b>；",
        "之后所有权重的计算都是「同一份常量张量」参与矩阵乘。",
    ], "data")
    panel(p, X + 760, y3, 664, 148, "所以 vLLM 里没有的东西（别去找）", [
        "没有 backward、没有 optimizer、没有 loss（它不做训练）；",
        "没有梯度的显存开销 → 同样一张卡，推理能放下比训练大得多的并发；",
        "但它多了训练时不存在的东西：<b>KV cache</b>（见 ⑦）。",
    ], "fail")

    y4 = y3 + 188
    bridge(p, y4, "本页交接：训练那四步里，<b>只有第 ① 步（前向）会出现在推理里</b>；"
                  "下一步（②）看这份权重从文件变成显存里的常量的完整路径。")

    y5 = y4 + 76
    anchors(p, y5, [
        "vLLM 的自我定位是推理服务：vllm/README.md:1（Easy, fast, and cheap LLM serving）",
        "推理侧「加载完就只读」的证据：加载写入 layer.weight 后由 forward 直接读 —— "
        "参见 tools/proc/t16.py 与 vllm/model_executor/layers/utils.py:84-90",
        "权重与激活/梯度的取舍在推理侧只剩权重：vllm_ascend/worker/worker.py:601-634"
        "（profile 时只算权重 + 激活峰值，不含梯度/优化器）",
    ], h=96)

    y6 = y5 + 128
    concl(p, y6,
          "<b>一句话记住这一页：</b>训练 = 前向 + 损失 + 反向 + 更新，循环到收敛，产出权重文件；"
          "推理 = 只用前向，且权重是只读常量。"
          "<b>这条差别决定了后面所有推理优化的方向。</b>")

    legend(p, y6 + 92)
