"""⑦ 显存去哪了：一本账（权重 / KV / 激活 / 图池 / 碎片）。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc, table

TITLE = "⑦ 显存去哪了：五个抽屉 + 一个上限旋钮"
SUB = ("一句话：推理的显存 = <b>权重 + KV cache + 激活峰值 + 图池 + 碎片</b>；"
       "其中只有 KV 会随「并发 × 上下文」长大，所以它是主要变量。")


def build(doc: Doc) -> None:
    p = doc.page("⑦ 显存账", pw=1560, ph=1480)
    head(p, TITLE, SUB)
    X = 36

    # --- 五个抽屉 ---------------------------------------------------------
    y = 152
    p.text(X, y - 34, 1300, 22,
           "<b>第一步：把这五个抽屉摆出来（每个都有「谁决定它多大」）</b>",
           "text_b")
    drawers = [
        ("权重", "模型结构与精度决定；加载后只读", "固定，加载时一次占定", "neutral"),
        ("KV cache", "并发数 × 上下文长度 × 每 token 大小", "唯一会持续变化的", "data"),
        ("激活峰值", "一步里最大那批的中间张量", "由 profile 量出来，短暂占用", "cloudd"),
        ("图池", "抓了多少桶、桶多大（进程级共用）", "进程级一块，≈各桶峰值取 max", "cloudp"),
        ("碎片", "分配器行为、块大小、反复增删", "可优化但消不掉", "fail"),
    ]
    cw = 274
    for i, (name, who, note, role) in enumerate(drawers):
        cx = X + i * (cw + 10)
        p.box(cx, y, cw, 128,
              f"<b>{name}</b><br>"
              f"<font style=\"font-size:10px\">{who}</font><br>"
              f"<font style=\"font-size:9px;color:#666666\">{note}</font>",
              "box", role)

    # --- 上限旋钮 ---------------------------------------------------------
    y2 = y + 168
    p.text(X, y2 - 34, 1300, 22,
           "<b>第二步：一个旋钮决定「总量」，剩下的是分配问题</b>", "text_b")
    p.box(X, y2, 1424, 104,
          "<font style=\"font-size:10px\">"
          "<b>gpu_memory_utilization</b>（本仓部署常用 0.90~0.98）："
          "告诉引擎「我最多用这张卡多少比例」。<br>"
          "引擎按这个上限做三件事：<b>① 装权重</b> → <b>② 跑 profile 量激活峰值</b> → "
          "<b>③ 把剩下的全切成 KV 块</b>。<br>"
          "所以日志里那句 <font style=\"font-size:9px\">Available KV cache memory: X GiB</font> "
          "就是「扣完前三项之后还能给 KV 多少」，它直接决定<b>能同时服务多少请求</b>。</font>",
          "box", "neutral")

    # --- KV 怎么算 --------------------------------------------------------
    y3 = y2 + 144
    p.text(X, y3 - 34, 1300, 22,
           "<b>第三步：KV cache 的账怎么算（会算这个就能预估并发）</b>", "text_b")
    rows = [
        ["每 token 的 KV 大小", "层数 × 2（K 和 V）× KV 头数 × 每头维度 × 每元素字节",
         "32 层、8 KV 头、128 维、bf16 → 约 128 KB/token"],
        ["一条请求的 KV", "每 token 大小 × 上下文长度", "4K → 约 512 MB"],
        ["总需求", "并发请求数 × 每请求 KV", "16 条 4K 请求 → 约 8 GB"],
        ["能开多少并发", "可用 KV 显存 ÷ 每请求 KV", "可用 8 GB 时 ≈ 16 条 4K 请求"],
    ]
    table(p, X, y3, [("要算的量", 300), ("公式", 620), ("举例（按上面假设）", 480)],
          rows, row_h=38)

    # --- 三件事由此而来 ---------------------------------------------------
    y4 = y3 + 38 * 5 + 34
    panel(p, X, y4, 700, 148, "为什么 KV 要「分块管理」而不是整段预留", [
        "整段预留：长度未知 → 只能按最大长度占，浪费巨大；",
        "分块：按需分配、前缀可共享、结束后立刻还回池子。",
        "这就是 block_size（默认 16 token 一块）与块表存在的原因。",
    ], "data")
    panel(p, X + 760, y4, 664, 148, "为什么 KV 还要「跨机搬」", [
        "当「一台机器的显存」已经装不下「想要的并发 × 上下文」时，",
        "剩下的选择只有两个：降并发（掉吞吐）或把 KV 放到别的机器上（PD 分离 / KV 池）。",
        "这就是 Mooncake 那套图存在的前提。",
    ], "cloudp")

    y5 = y4 + 188
    bridge(p, y5, "本页交接：显存这本账解释了「为什么要有 KV 管理和 PD 分离」。"
                  "还有另一条路——把一张卡变成多张卡。下一步（⑧）讲并行的四种切法。")

    y6 = y5 + 76
    anchors(p, y6, [
        "上限旋钮：vllm/config/cache.py:103（gpu_memory_utilization 默认 0.92）、"
        ":71（block_size 默认 16）",
        "三步定容量：vllm/v1/worker/gpu_worker.py:528-642（determine_available_memory）、"
        "vllm/v1/engine/core.py:354（按剩余显存切块池）",
        "块池与块表：vllm/v1/core/block_pool.py:143、vllm/v1/core/kv_cache_manager.py:402",
        "图池占用也会被扣掉：vllm/v1/worker/gpu_worker.py:613-624、:850-854",
    ], h=104)

    y7 = y6 + 136
    concl(p, y7,
          "<b>一句话记住这一页：</b>显存 = 权重 + KV + 激活峰值 + 图池 + 碎片；"
          "上限由 gpu_memory_utilization 定，KV 拿走剩下的全部；"
          "所以「权重/图池变大 → KV 变少 → 并发变低」，三者是一条链。")

    legend(p, y7 + 92, extra_fill=("会持续变化的那个抽屉", "data"))
