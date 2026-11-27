"""⑥ 连续批处理：为什么「凑一个 batch」这件事本身很难。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc

TITLE = "⑥ 连续批处理（continuous batching）：推理系统里最难的那件小事"
SUB = ("一句话：把多个请求「捆成一批」能提高算术强度，"
       "但每个请求的长度、结束时间、要的 token 数都不一样 —— "
       "于是必须<b>每一轮重新组批</b>，而不是「一批跑到底」。")


def build(doc: Doc) -> None:
    p = doc.page("⑥ 连续批处理", pw=1560, ph=1480)
    head(p, TITLE, SUB)
    X = 36

    # --- 两种方式对照 -----------------------------------------------------
    y = 152
    p.text(X, y - 34, 1300, 22,
           "<b>第一步：静态批处理 vs 连续批处理（同一段时间轴）</b>", "text_b")
    p.text(X, y, 700, 20, "<b>静态批（static batching）：一批跑到底</b>", "text_b")
    p.box(X, y + 24, 700, 118,
          "<font style=\"font-size:10px\">"
          "请求 A、B 一起进 → <b>等最慢的那个结束</b> → 才收下一批。<br>"
          "B 早早答完了，它占的位置却一直空着（<b>空转</b>）；<br>"
          "新请求 C 只能在外面排队等到这一批整体结束。<br>"
          "<font color=\"#c0392b\">GPU 利用率与尾延迟都被「最慢的那个」拖住。</font></font>",
          "box", "fail")
    p.text(X + 760, y, 664, 20, "<b>连续批处理：每一轮重新组批</b>", "text_b")
    p.box(X + 760, y + 24, 664, 118,
          "<font style=\"font-size:10px\">"
          "每一「步」（生成一轮 token）都重新决定：<b>这一步算谁、各算几个 token</b>。<br>"
          "B 答完了 → 下一步就把它移出批，位置立刻给排队中的 C；<br>"
          "新请求可以在<b>下一步</b>就进来，不用等整批结束。<br>"
          "<font color=\"#1c7c54\">代价：必须有一个调度器，而且每步都要重算批的形状。</font></font>",
          "box", "cloudd")

    # --- 三个必须解决的问题 -----------------------------------------------
    y2 = y + 186
    p.text(X, y2 - 34, 1300, 22,
           "<b>第二步：一旦「每步重组批」，三个问题立刻冒出来</b>", "text_b")
    panel(p, X, y2, 460, 176, "① 形状每步都变", [
        "这一步批里有 7 个请求，下一步可能 9 个；",
        "每个请求这一步要算的 token 数也能不同"
        "（一个还在 prefill 大块、另一个已是 decode 1 个）。",
        "→ 于是要有「每步的 token 预算」与「分块 prefill」；",
        "→ 也是图捕获必须按「桶」抓、且会静默掉出的根源（图捕获 ⑥）。",
    ], "neutral")
    panel(p, X + 482, y2, 460, 176, "② 显存要能「随时借还」", [
        "新请求进来要立刻能分到 KV 块；",
        "请求结束时块必须立刻还回池子，否则并发上不去。",
        "→ 于是要有块池 + 块表 + 引用计数（生命周期 ⑤）；",
        "→ 块不够时要么等、要么抢占（调度器 ⑫）。",
    ], "data")
    panel(p, X + 964, y2, 460, 176, "③ 公平与延迟", [
        "如果一直让「跑得快的短请求」插队，长请求会被饿死；",
        "反过来先来先服务（FCFS），短请求的尾延迟又变差。",
        "→ 于是要有排队策略（FCFS / priority）、有 chunked prefill 的配额、",
        "→ 还要有准入控制（排队上限，超了直接 503）。",
    ], "cloudp")

    # --- 本仓的对应实现 ---------------------------------------------------
    y3 = y2 + 216
    p.text(X, y3 - 34, 1300, 22,
           "<b>第三步：vLLM 里这三件事分别落在哪里（这就是后面几套图的目录）</b>",
           "text_b")
    rows = [
        ["每步组批", "Scheduler.schedule()：先 running 再 waiting，两个预算",
         "进程/线程 ⑪⑫"],
        ["KV 借还", "KVCacheManager + BlockPool + 块表（block_size 默认 16）",
         "生命周期 ⑤"],
        ["分块 prefill", "enable_chunked_prefill（默认开）+ max_num_batched_tokens",
         "本页 & 进程/线程 ⑫"],
        ["批形状固定化", "图捕获的桶 + padding", "图捕获 ⑥"],
        ["跨机借显存", "PD 分离 + KV 搬运（Mooncake）", "生命周期 ⑥ & Mooncake 全set"],
    ]
    from dg import table
    table(p, X, y3, [("问题", 300), ("vLLM 里的对应物", 780), ("在哪套图里细讲", 340)],
          rows, row_h=34)

    y4 = y3 + 34 * 6 + 34
    bridge(p, y4, "本页交接：batching 把「算力利用率」拉起来了，但它的代价是「吃显存」。"
                  "下一步（⑦）就把显存这本账摊开算。")

    y5 = y4 + 76
    anchors(p, y5, [
        "每步组批：vllm/v1/core/sched/scheduler.py:568-1520（running 段 :617、waiting 段 :866）",
        "两个预算：scheduler.py:588-591（token / input budget）、:124-130（上限）",
        "分块 prefill：vllm/config/scheduler.py:116（默认 True）、:70（阈值）、:49"
        "（max_num_batched_tokens 默认 2048）",
        "块大小与显存上限：vllm/config/cache.py:71（16）、:103（0.92）",
    ], h=104)

    y6 = y5 + 136
    concl(p, y6,
          "<b>一句话记住这一页：</b>连续批处理 = 「每一步重新组批」；"
          "它换来算力利用率，代价是「形状每步变、显存要随时借还、还要处理公平性」，"
          "这三件事正好对应调度器、KV 管理器与图捕获的桶机制。")

    legend(p, y6 + 92)
