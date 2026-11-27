"""① 全景：四个机制各自抢什么资源。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc, table

TITLE = "① 四个机制一起开时，它们其实在抢同四样东西"
SUB = ("一句话：PD 分离、Mooncake 搬 KV、图捕获、MTP 投机解码——"
       "这四个机制各自的收益写在不同的指标上，"
       "但它们花的是<b>同一份显存、同一份算力、同一份启动时间和同一份「形状自由度」</b>。")


def build(doc: Doc) -> None:
    p = doc.page("① 四机制抢什么", pw=1560, ph=1460)
    head(p, TITLE, SUB)
    X = 36

    y = 152
    p.text(X, y - 34, 1300, 22,
           "<b>第一步：每个机制「买」什么、「付」什么</b>", "text_b")
    rows = [
        ["PD 分离", "把算力密集（prefill）与带宽密集（decode）分到不同实例，各用各的卡",
         "整机数量翻倍；KV 要跨机搬；调度的尾段要对齐"],
        ["Mooncake 搬 KV", "把 decode 侧的 KV 从 P 侧拉过来，省掉重算",
         "带宽与延迟；额外的传输缓冲；失败要兜底（fail-closed）"],
        ["图捕获（CUDA/ACL Graph）", "削减每步的 CPU 提交开销 → TPOT 下降",
         "显存（进程级图池）；启动时间（5~20 s）；<b>形状必须固定</b>"],
        ["MTP / 投机解码", "一次前向多产出几个 token → 单请求更快",
         "显存（draft 权重 + lookahead 槽）；算力（被拒的草案白算）；<b>形状变成 1+K</b>"],
    ]
    table(p, X, y, [("机制", 220), ("买到什么", 620), ("付出什么", 580)], rows,
          row_h=46)

    # --- 四种稀缺资源 -----------------------------------------------------
    y2 = y + 46 * 5 + 40
    p.text(X, y2 - 34, 1300, 22,
           "<b>第二步：四样稀缺资源——冲突都发生在这里</b>", "text_b")
    cards = [
        ("显存", "权重 / KV / 激活峰值 / 图池 / draft。\n这五个抽屉总和固定，谁多谁少。", "data"),
        ("算力", "decode 本来就闲着；MTP 把闲着的算力换成 token，\n但被拒的草案是纯浪费。", "cloudp"),
        ("带宽", "decode 卡在显存带宽；跨机搬 KV 又吃网络带宽。\n两条带宽不是同一根管子，但都稀缺。", "cloudd"),
        ("形状自由度", "图捕获要求形状固定；MTP 把 1 变成 1+K；\nchunked prefill 让形状每步都变。三者互相拉扯。", "fail"),
    ]
    cw = 340
    for i, (name, body, role) in enumerate(cards):
        cx = X + i * (cw + 12)
        p.box(cx, y2, cw, 118, "", "box", role)
        p.text(cx + 10, y2 + 8, cw - 20, 22, f"<b>{name}</b>", "text_b")
        p.text(cx + 10, y2 + 30, cw - 20, 80,
               "<font style=\"font-size:9px\">" +
               body.replace("\n", "<br>") + "</font>", "text")

    # --- 为什么必须先理解这个 ---------------------------------------------
    y3 = y2 + 158
    panel(p, X, y3, 1424, 116, "为什么「同时开」比「一个个开」难得多", [
        "单独读每个机制的文档，你都会觉得它很合理；但它们的约束会互相冲突："
        "MTP 要形状 1+K，图捕获要形状固定，chunked prefill 要形状可变，"
        "PD 又要两边形状对齐。",
        "而且冲突的表现方式几乎都不是「报错」，而是「某条优化静默失效」——"
        "你会看到性能没达到预期，却没有任何日志提示。",
        "所以这一套图的目的是：<b>把「改一个参数会连累谁」显式画出来</b>，而不是给出万能配方。",
    ], "fail")

    y4 = y3 + 156
    bridge(p, y4, "本页交接：先建立「四种资源 + 四个机制」的坐标。"
                  "下一步（②）把启动命令里的参数按这张坐标分类，看它们各自落在哪一格。")

    y5 = y4 + 76
    anchors(p, y5, [
        "显存五抽屉：vllm/v1/worker/gpu_worker.py:528-642（determine_available_memory）；"
        "入门套 ⑦",
        "图池是进程级共用：vllm/compilation/cuda_graph.py:200；图捕获套 ⑦",
        "MTP 的形状影响：vllm/config/vllm.py:679（uniform_decode_query_len）；MTP 套 ⑦",
        "PD 的两侧形状对齐：vllm/v1/lwd_control/pd/envelope.py:215-227；MTP 套 ⑨",
        "chunked prefill 让形状可变：vllm/config/scheduler.py:116；进程/线程 ⑫",
    ], h=104)

    y6 = y5 + 136
    concl(p, y6,
          "<b>一句话记住这一页：</b>四个机制各自买不同的指标，却花同一份显存/算力/带宽/形状自由度；"
          "所以「都打开」不是叠加收益，而是<b>重新分配稀缺资源</b>。")

    legend(p, y6 + 92)
