"""⑦ 启动期捕获流程：捕获发生在「权重之后、KV 之后」。"""
from common import head, concl, anchors, legend
from dg import Doc, table

TITLE = "⑦ 启动期的捕获流程：它排在「权重 → profile → KV」之后"
SUB = ("一句话：<b>加载权重 → profile 量激活峰值 → 切 KV 块池 → 估算图显存 → 逐桶捕获</b>。"
       "所以启动日志里「Graph capturing finished」永远出现在「Available KV cache memory」之后。")


def seg(p, x, y, w, h, no, title, lines, role):
    p.box(x, y, w, h, "", "box", role)
    p.text(x + 12, y + 8, w - 24, 22, f"<b>{no}　{title}</b>", "text_b")
    body = "<br>".join(f"<font style=\"font-size:9px\">{ln}</font>" for ln in lines)
    p.text(x + 12, y + 32, w - 24, h - 40, body, "text")


def build(doc: Doc) -> None:
    p = doc.page("⑦ 启动期捕获流程", pw=1560, ph=1560)
    head(p, TITLE, SUB)
    X = 36

    # --- 时间线 -----------------------------------------------------------
    y = 156
    p.text(X, y - 32, 1200, 22,
           "<b>第一步：启动时间线（前四步都是「为图做准备的账」）</b>", "text_b")
    steps = [
        ("①", "加载权重", "权重先吃掉显存", "neutral"),
        ("②", "profile 空跑", "eager 跑一遍，量激活峰值", "neutral"),
        ("③", "切 KV 块池", "剩余显存才给 KV", "cloudd"),
        ("④", "估算图显存", "可选；失败不影响正确性", "fail"),
        ("⑤", "逐桶捕获", "从大到小真跑一遍", "cloudp"),
        ("⑥", "锁 workspace", "回放期不许改尺寸", "neutral"),
    ]
    cw = 224
    for i, (no, title, note, role) in enumerate(steps):
        cx = X + i * (cw + 12)
        p.box(cx, y, cw, 78,
              f"<b>{no} {title}</b><br>"
              f"<font style=\"font-size:9px\" color=\"#666666\">{note}</font>",
              "box_center", role)
        if i:
            p.edge(src_pt=(cx - 10, y + 39), dst_pt=(cx - 2, y + 39), st="arrow",
                   role="http")

    # --- 关键点 -----------------------------------------------------------
    y2 = y + 118
    p.text(X, y2 - 32, 1200, 22,
           "<b>第二步：四个容易搞错的点</b>", "text_b")
    seg(p, X, y2, 700, 132, "①", "捕获必须在 profile 与 KV 之后",
        ["因为图本身要占显存（进程级图池），且要用这个池去估算。",
         "先抓图再算 KV，会把 KV 预算算错、甚至启动 OOM。"], "fail")
    seg(p, X + 760, y2, 664, 132, "②", "图池是「进程级」的，不是每桶一份",
        ["所有桶共用一块池，占用 ≈ 各桶峰值的最大值。",
         "所以必须<b>从大到小</b>捕获：大桶先占下够用的池，小桶复用它。"], "fail")

    y3 = y2 + 172
    seg(p, X, y3, 700, 132, "③", "每个桶都要 warmup 几遍再录",
        ["warmup 会触发算子库/通信库的 lazy 初始化；不 warmup 直接录，",
         "图里会混进「第一次才发生」的初始化动作，回放时行为不一致。"], "cloudp")
    seg(p, X + 760, y3, 664, 132, "④", "启动成本是可量化的",
        ["日志会打「Graph capturing finished in N secs, took X GiB」，",
         "还会同时打印「估算 vs 实际」的图显存对比，方便判断要不要缩桶。"], "cloudd")

    # --- 代价表 -----------------------------------------------------------
    y4 = y3 + 172
    p.text(X, y4 - 32, 1200, 22, "<b>第三步：这份「启动保险」的价格</b>", "text_b")
    rows = [
        ["启动时间", "典型 5~20 秒（随桶数与模型深度增长）", "桶越多、层越深越慢"],
        ["显存", "进程级图池 ≈ 各桶峰值取 max（不是相加）", "可从 KV 预算里扣掉这部分"],
        ["显存不够时", "捕获段<b>没有 try/except 回退</b>，OOM 直接失败启动", "所以先调小桶或降 gpu-memory-utilization"],
        ["稳态收益", "decode 每步少发上千次命令 → TPOT 下降", "prefill 大 batch 基本不受益"],
    ]
    table(p, X, y4, [("项", 220), ("本仓口径", 760), ("应对", 420)], rows, row_h=36)

    y5 = y4 + 36 * 5 + 36
    p.box(X, y5, 1424, 92,
          "<b>想省这笔钱，有三个旋钮（按代价从小到大）</b><br>"
          "<font style=\"font-size:10px\">"
          "① <b>缩小桶列表</b>：<font style=\"font-size:9px\">--cudagraph-capture-sizes "
          "[1,2,4,8,16]</font>——桶少了，池变小、启动变快，代价是更多请求落到 eager。<br>"
          "② <b>降模式</b>：只留 decode 的图（FULL_DECODE_ONLY），别抓 mixed 的 PIECEWISE。<br>"
          "③ <b>关掉</b>：--enforce-eager / -O0——启动最快，稳态 decode 变慢。</font>",
          "box", "neutral")

    y6 = y5 + 126
    anchors(p, y6, [
        "启动顺序（profile→KV→编译/捕获）：vllm/v1/engine/core.py:254、:308、:354、:356；"
        "vllm/v1/worker/gpu_worker.py:773-816",
        "图显存估算：vllm/v1/worker/gpu_worker.py:580-592；"
        "vllm/v1/worker/gpu_model_runner.py:6674-6710",
        "捕获主循环（从大到小、逐桶 warmup）：vllm/v1/worker/gpu_model_runner.py:6905-6914、"
        ":6949-6995",
        "锁 workspace：vllm/v1/worker/gpu_model_runner.py:6934-6936",
        "耗时与显存日志：vllm/v1/worker/gpu_model_runner.py:6942-6946；"
        "估算 vs 实际 vllm/v1/worker/gpu_worker.py:818-832",
        "缩桶的两个开关：vllm/engine/arg_utils.py:1660-1665（--cudagraph-capture-sizes）、"
        "vllm/docs/configuration/conserving_memory.md:48-65",
    ], h=112)

    y7 = y6 + 146
    concl(p, y7,
          "<b>一句话记住这一页：</b>捕获排在「权重 → profile → KV」之后；"
          "图池是进程级共用、必须从大到小抓；桶越多启动越慢、显存越多；"
          "想省就缩桶或降模式，别指望它自己回退。")

    legend(p, y7 + 92)
