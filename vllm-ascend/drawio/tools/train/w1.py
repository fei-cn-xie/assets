"""① vLLM 到底做不做训练：它站在训练流水线的哪一段。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc

TITLE = "① 先划清边界：vLLM 不做训练，但它和训练共享三样东西"
SUB = ("一句话：vLLM 是<b>推理引擎</b>——它没有反向传播、没有优化器、没有损失函数；"
       "但它的<b>模型定义、权重文件格式、配置约定</b>与训练侧是同一套。"
       "搞清这一点，才不会在仓里白找训练代码。")


def build(doc: Doc) -> None:
    p = doc.page("① 训练与推理的边界", pw=1560, ph=1420)
    head(p, TITLE, SUB)
    X = 36

    # --- 训练流水线全景 ---------------------------------------------------
    y = 150
    p.text(X, y - 32, 1300, 22,
           "<b>第一步：一条完整的「造模型」流水线（vLLM 只站在最右端）</b>", "text_b")
    stages = [
        ("数据", "清洗 / 去重 / 配比", "neutral"),
        ("训练", "前向 + 损失 + 反向 + 更新", "cloudp"),
        ("产出", "checkpoint（权重 + 优化器状态）", "data"),
        ("转换", "合并分片 / 格式转换 / 量化", "cloudd"),
        ("部署", "vLLM 加载并服务", "fail"),
    ]
    cw = 274
    for i, (name, note, role) in enumerate(stages):
        cx = X + i * (cw + 10)
        p.box(cx, y, cw, 86,
              f"<b>{name}</b><br>"
              f"<font style=\"font-size:9px;color:#666666\">{note}</font>",
              "box_center", role)
        if i:
            p.edge(src_pt=(cx - 9, y + 43), dst_pt=(cx - 1, y + 43), st="arrow",
                   role="http")
    p.text(X, y + 96, 1424, 24,
           "<font style=\"font-size:9px;color:#666666\">"
           "本仓两份材料的分工：① 训练侧（本套）只在“产出 → 转换”这一段；"
           "② 其余八套讲最右边的“部署”以及它与前面各段的接口。</font>", "subtitle")

    # --- 三样共享 ---------------------------------------------------------
    y2 = y + 140
    p.text(X, y2 - 32, 1300, 22,
           "<b>第二步：训练与推理共享的三样东西（这才是本套的意义）</b>", "text_b")
    panel(p, X, y2, 460, 186, "① 模型结构定义", [
        "同一个 `deepseek_mtp` / `qwen` 类既描述了「训练时怎么算」，",
        "也描述了「推理时怎么算」——只是推理不挂 backward。",
        "所以读 vLLM 的 models/ 目录，本质上是在读训练代码的 forward 部分。",
    ], "cloudp")
    panel(p, X + 482, y2, 460, 186, "② 权重文件格式", [
        "训练输出 safetensors 分片 + index.json；",
        "推理按同一份 index 只读权重，忽略优化器/训练状态文件。",
        "名字、形状、分片约定必须完全一致，否则加载报形状不匹配。",
    ], "data")
    panel(p, X + 964, y2, 460, 186, "③ 配置约定", [
        "`config.json` 里的层数、头数、隐藏维度既决定训练参数量，",
        "也决定推理的显存与并行切分。",
        "还有像 `num_nextn_predict_layers` 这种「训练期定义、推理期复用」的字段。",
    ], "cloudd")

    # --- 仓里找不到什么 ---------------------------------------------------
    y3 = y2 + 226
    p.box(X, y3, 1424, 128,
          "<b>第三步：在 vLLM 仓里找不到的四样东西（省下你的搜索时间）</b><br>"
          "<font style=\"font-size:10px\">"
          "① <b>反向传播</b>（全仓 0 处 `loss.backward`）；"
          "② <b>优化器</b>（0 处 `AdamW`/梯度清零）；"
          "③ <b>数据加载</b>（包内 0 处 `DataLoader`）；"
          "④ <b>训练循环</b>（0 处 `model.train()`，只有 `model.eval()`）。<br>"
          "但它<b>确实有一块与训练打交道的代码</b>：一套跑在<b>训练进程侧</b>的「权重同步客户端」"
          "（把新权重推给正在服务的推理引擎），这是为强化学习准备的——详见第 ⑩ 页。<br>"
          "<font color=\"#c0392b\">它多出来的是训练侧没有的东西：<b>调度器、KV cache、批量管理与图捕获</b>"
          "——因为只有推理才需要「持续服务很多长短不一的请求」。</font></font>",
          "box", "fail")

    y4 = y3 + 148
    bridge(p, y4, "本页交接：边界划清了。下一步（②）把训练那一格展开——"
                  "一次训练 step 到底在算什么，以及它和推理的那一步差在哪。")

    y5 = y4 + 76
    anchors(p, y5, [
        "vLLM 的自我定位（serving）：vllm/README.md:1",
        "模型定义被推理复用：vllm/model_executor/models/（每个模型一个 forward）",
        "权重加载只吃权重、忽略训练状态：vllm/model_executor/model_loader/weight_utils.py:1002、"
        ":1253-1271",
        "训练期定义但推理期不用的字段示例："
        "vllm/transformers_utils/configs/hy_v4.py:53-54（mtp_loss_factor 注释写明 unused at inference）",
    ], h=104)

    y6 = y5 + 136
    concl(p, y6,
          "<b>一句话记住这一页：</b>vLLM 不做训练，但它与训练共享<b>模型结构、权重格式、配置</b>；"
          "所以「理解训练」的价值不是去改 vLLM，而是<b>理解那些权重数字是怎么来的、哪些字段只在训练期有意义</b>。")

    legend(p, y6 + 92)
