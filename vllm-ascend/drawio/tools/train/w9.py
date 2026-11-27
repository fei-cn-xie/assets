"""⑨ 加载器视角：训练产出的目录里，它只挑走哪些文件。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc, table

TITLE = "⑨ 训练目录 → 推理加载：五个文件被明确丢弃"
SUB = ("一句话：训练产出的目录里通常有<b>权重 + 优化器状态 + 调度器状态 + 缩放器状态 + 训练参数</b>；"
       "推理只吃权重，其余五个在加载时被<b>按文件名直接过滤掉</b>——这一条在仓里写得很清楚。")


def build(doc: Doc) -> None:
    p = doc.page("⑨ 加载器挑文件", pw=1560, ph=1500)
    head(p, TITLE, SUB)
    X = 36

    # --- 一个目录 ---------------------------------------------------------
    y = 148
    p.text(X, y - 32, 1300, 22,
           "<b>第一步：一个训练产出的目录（左：推理要的；右：推理丢的）</b>",
           "text_b")
    p.box(X, y, 690, 208,
          "<b>推理要的</b><br>"
          "<font style=\"font-size:10px\">"
          "· <b>model-*.safetensors</b>（分片权重）<br>"
          "· <b>*.safetensors.index.json</b>（哪个张量在哪个分片）<br>"
          "· <b>config.json</b>（结构参数）<br>"
          "· <b>tokenizer 相关文件</b>（分词器与模板）<br>"
          "· （若量化）<b>量化描述文件</b>：每层用哪个量化方案</font>",
          "box", "cloudd")
    p.box(X + 730, y, 694, 208,
          "<b>推理明确丢弃的五个</b><br>"
          "<font style=\"font-size:10px\">"
          "· <b>optimizer.bin / optimizer.pt</b>（Adam 动量与方差）<br>"
          "· <b>scheduler.pt</b>（学习率调度状态）<br>"
          "· <b>scaler.pt</b>（fp16 梯度缩放器状态）<br>"
          "· <b>training_args.bin</b>（训练参数快照）<br>"
          "<font color=\"#c0392b\">仓内有一份<b>黑名单函数</b>，就按这五个文件名过滤——"
          "这是「推理不关心训练状态」最直接的代码证据。</font></font>",
          "box", "fail")

    # --- 加载六步 ---------------------------------------------------------
    y2 = y + 248
    p.text(X, y2 - 32, 1300, 22,
           "<b>第二步：加载器只挑权重时的六步</b>", "text_b")
    steps = [
        ("①", "过滤文件", "黑名单剔掉五个训练文件", "fail"),
        ("②", "按扩展名白名单", "只取 safetensors / bin", "fail"),
        ("③", "校验分片完整", "索引里列的文件必须都在", "neutral"),
        ("④", "逐张量读盘", "读成 CPU 张量，不占设备显存", "cloudp"),
        ("⑤", "名字映射", "改名 / 合并（q,k,v → qkv）/ 丢弃（None）", "cloudp"),
        ("⑥", "写进参数", "按 TP 切片后 copy_ 到设备", "cloudd"),
    ]
    cw = 224
    for i, (no, title, note, role) in enumerate(steps):
        cx = X + i * (cw + 12)
        p.box(cx, y2, cw, 88,
              f"<b>{no} {title}</b><br>"
              f"<font style=\"font-size:9px;color:#666666\">{note}</font>",
              "box_center", role)
        if i:
            p.edge(src_pt=(cx - 10, y2 + 44), dst_pt=(cx - 2, y2 + 44), st="arrow",
                   role="http")

    # --- 三种「丢弃/改名」 -------------------------------------------------
    y3 = y2 + 128
    p.text(X, y3 - 32, 1300, 22,
           "<b>第三步：第 ⑤ 步里的三种操作（第三种是本套反复出现的关键机制）</b>",
           "text_b")
    rows = [
        ["改名", "文件里的名字与代码里的名字不同（例如加前缀）",
         "一张「旧名 → 新名」的映射表", "普通过滤"],
        ["合并", "文件里 q/k/v 三个权重，代码里是一个融合层",
         "把三份拼成一份", "形状变化"],
        ["<b>丢弃</b>", "映射成「空」（None）", "<b>这个键直接不加载</b>",
         "MTP 层、训练遗留张量都靠它"],
    ]
    table(p, X, y3, [("操作", 160), ("什么时候用", 500), ("怎么实现", 420),
                     ("备注", 344)], rows, row_h=36)

    # --- 两个兼容性开关 ---------------------------------------------------
    y4 = y3 + 36 * 4 + 36
    panel(p, X, y4, 700, 160, "① 训练框架遗留的张量也会被丢掉", [
        "有些早期训练框架（或某些量化工具）会在权重文件里多写几个旋转位置编码的缓存张量，",
        "推理根本不用它们 → 仓内用同一套「映射成 None」的机制丢弃。",
        "否则会报「意外的键」。",
    ], "fail")
    panel(p, X + 760, y4, 664, 160, "② 训练中期就能热更新权重", [
        "仓内支持「按层热加载」：不重启引擎，把新权重逐层换进去；",
        "并且特意保持张量地址不变——因为图捕获记录的是地址，换地址会让图失效。",
        "→ 这是给强化学习训练循环用的（见 ⑩）。",
    ], "cloudp")

    y5 = y4 + 200
    bridge(p, y5, "本页交接：加载器告诉我们「推理只吃权重」。反过来问：训练真的一直在改权重，"
                  "那推理怎么跟上？下一步（⑩）就是这个接口。")

    y6 = y5 + 76
    anchors(p, y6, [
        "五个被丢弃的文件（黑名单函数原文）："
        "vllm/model_executor/model_loader/weight_utils.py:639-655；调用点 default_loader.py:254",
        "扩展名白名单（allow_patterns）：vllm/model_executor/model_loader/default_loader.py:184-208",
        "索引与分片校验：vllm/model_executor/model_loader/weight_utils.py:625-635",
        "名字映射与「映射成 None 即丢弃」：vllm/model_executor/models/utils.py:46-57；"
        "训练遗留张量的丢弃 :215-224",
        "逐张量加载与 TP 切片：weight_utils.py:1253-1271（default_weight_loader）",
        "按层热加载：vllm/model_executor/model_loader/reload/layerwise.py:84；"
        "保地址的原因 vllm/model_executor/utils.py:94-99（图捕获需要地址不变）",
    ], h=128)

    y7 = y6 + 162
    concl(p, y7,
          "<b>一句话记住这一页：</b>加载器只挑权重：五个训练状态文件按文件名丢弃、"
          "多余的键按「映射成空」丢弃、分片按索引校验；"
          "<b>它唯一在意的兼容性是「名字对得上、形状对得上、地址能稳定」。</b>")

    legend(p, y7 + 92, extra_fill=("被丢弃的训练状态", "fail"))
