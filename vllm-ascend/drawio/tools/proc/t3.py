"""③ 三种 executor：同一个「引擎」，进程形态可以完全不同。"""
from common import head, concl, anchors, legend
from dg import Doc, table

TITLE = "③ 三种 executor：worker 到底是不是独立进程，由它决定"
SUB = ("一句话：`--distributed-executor-backend`（以及 world_size 大小）决定 worker 跑在哪——"
       "同进程、同机多进程、还是远端 ray 集群。**边端之所以只有 3 个进程，就是因为走了 uni**。")
X = 36


def build(doc: Doc) -> None:
    p = doc.page("③ 三种 executor", pw=1560, ph=1380)
    head(p, TITLE, SUB)

    # --- 判定链 -----------------------------------------------------------
    y = 148
    p.text(X, y - 32, 1200, 22, "<b>第一步：它怎么被选中（一条判定链）</b>", "text_b")
    d1 = p.box(X, y, 340, 96,
               "<b>world_size = TP × PP × DP</b><br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "先算出这次要用几张卡（跨 DP 一起算）</font>",
               "box", "neutral")
    d2 = p.box(X + 362, y, 340, 96,
               "<b>= 1 → uni 后端</b><br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "只有一张卡：worker 直接与 EngineCore 同进程</font>",
               "box", "cloudp")
    d3 = p.box(X + 724, y, 340, 96,
               "<b>&gt; 1 且没指定后端 → mp</b><br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "本机多进程：每 rank 一个 VllmWorker 进程</font>",
               "box", "cloudd")
    d4 = p.box(X + 1086, y, 338, 96,
               "<b>跨机 / 显式指定 → ray</b><br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "每个 rank 一个 ray actor（远端进程）</font>",
               "box", "data")
    p.edge(d1, d2, "", st="arrow", role="http")
    p.edge(d2, d3, "大于 1", st="arrow", role="http")
    p.edge(d3, d4, "要跨机", st="arrow", role="http")
    p.text(X, y + 108, 1200, 24,
           "<font style=\"font-size:9px\" color=\"#666666\">"
           "本仓部署里：edge TP=1/DP=1 → world_size=1 → <b>uni</b>；"
           "云端 TP=2 或 DP=2 → &gt;1 → <b>mp</b>；没人用 ray。</font>",
           "subtitle")

    # --- 三种形态 ---------------------------------------------------------
    y2 = y + 176
    p.text(X, y2 - 32, 1000, 22,
           "<b>第二步：三种形态下的进程图（认清「一个方框 = 一个进程」）</b>",
           "text_b")

    # uni
    p.box(X, y2, 470, 232, "", "lane", "cloudp")
    p.text(X + 10, y2 + 8, 450, 22, "<b>uni：全部塞在一个进程里</b>", "text_b")
    ub = p.box(X + 40, y2 + 40, 390, 76,
               "<b>EngineCore 进程</b><br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "调度 + 忙循环 + 模型前向 + 权重，都在这里</font>", "box", "cloudp")
    ub2 = p.box(X + 40, y2 + 130, 390, 60,
               "<b>（同进程内）worker</b><br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "没有独立进程，ps 看不到</font>", "box", "neutral")
    p.edge(ub, ub2, "", st="arrow", role="http")

    # mp
    p.box(X + 500, y2, 470, 232, "", "lane", "cloudd")
    p.text(X + 510, y2 + 8, 450, 22, "<b>mp：EngineCore + N 个 worker 进程</b>",
           "text_b")
    mp1 = p.box(X + 540, y2 + 40, 390, 52,
                "<b>EngineCore 进程</b>（调度 + 下发）", "box", "cloudd")
    w1 = p.box(X + 540, y2 + 108, 118, 84, "<b>VllmWorker-0</b>", "box_center",
               "neutral")
    w2 = p.box(X + 672, y2 + 108, 118, 84, "<b>VllmWorker-1</b>", "box_center",
               "neutral")
    w3 = p.box(X + 804, y2 + 108, 118, 84, "<b>…</b>", "box_center", "neutral")
    p.edge(mp1, w1, "", st="arrow", role="http")
    p.text(X + 540, y2 + 198, 390, 22,
           "<font style=\"font-size:9px\" color=\"#666666\">"
           "通信：消息队列 RPC（collective_rpc → 广播队列）</font>", "subtitle")

    # ray
    p.box(X + 1000, y2, 424, 232, "", "lane", "data")
    p.text(X + 1010, y2 + 8, 400, 22, "<b>ray：进程可以不在本机</b>", "text_b")
    r1 = p.box(X + 1040, y2 + 40, 344, 52, "<b>EngineCore 进程</b>", "box", "data")
    r2 = p.box(X + 1040, y2 + 108, 344, 60,
               "<b>ray actor（远端进程）× N</b><br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "由 ray 集群调度，可能跨机器</font>", "box", "neutral")
    p.edge(r1, r2, "", st="arrow", role="data")

    # --- 对照表 -----------------------------------------------------------
    y3 = y2 + 272
    p.text(X, y3 - 32, 1000, 22, "<b>第三步：一张表看清差异</b>", "text_b")
    rows = [
        ["uni", "0（同进程）", "函数调用", "单卡 / 边端，最省进程"],
        ["mp", "TP×PP 个", "消息队列 RPC（同机）", "默认；多卡单机"],
        ["ray", "TP×PP 个 actor", "ray RPC（可跨机）", "多机；需要 ray 集群"],
        ["external_launcher", "由外部启动器给定", "环境变量 + 外部进程", "多机手工编排（少见）"],
    ]
    table(p, X, y3, [("后端", 200), ("worker 进程数", 220), ("EngineCore↔worker 怎么通信", 420),
                     ("什么时候用", 480)], rows, row_h=34)

    # --- 本仓特例 ---------------------------------------------------------
    y4 = y3 + 34 * 5 + 40
    p.box(X, y4, 1424, 104,
          "<b>本仓的两个特例（读 LWD 代码时会遇到）</b><br>"
          "① <b>边端 world_size=1 → uni</b>：所以边端「模型加载」其实发生在 EngineCore 进程里，"
          "ps 里看不到 Worker。<br>"
          "② <b>LWD host 数据面还会换掉 executor</b>："
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "host 模式 + world_size==1 时用 LwdHostUniProcExecutor"
          "（vllm/config/vllm.py:1306-1316）</font>——同样是「不开子进程」，"
          "但多了 host 数据面通道的初始化。",
          "box", "neutral")

    y5 = y4 + 138
    anchors(p, y5, [
        "world_size 与后端缺省：vllm/config/parallel.py:982-1024；uni 判定 :1026-1027",
        "uni 实现：vllm/v1/executor/uniproc_executor.py:52-68",
        "mp 实现：vllm/v1/executor/multiproc_executor.py:190-206（spawn）、"
        ":1088-1118（worker 循环）",
        "LWD 特例：vllm/config/vllm.py:1306-1316",
    ], h=90)

    y6 = y5 + 122
    concl(p, y6,
          "<b>一句话记住这一页：</b>worker 是不是独立进程，取决于 world_size 与后端——"
          "world_size=1 就同进程（uni），多卡默认同机多进程（mp），要跨机才用 ray；"
          "所以「进程数」这件事，一半由 TP/PP/DP 决定，一半由 executor 决定。")

    legend(p, y6 + 92, extra_fill=("数据面 / 跨机", "data"))
