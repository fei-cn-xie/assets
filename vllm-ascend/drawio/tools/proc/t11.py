"""⑪ 调度器是什么：四个容器、六种状态、每步挑一批。"""
from common import head, concl, anchors, legend
from dg import Doc, table

TITLE = "⑪ Scheduler 是什么：一个「每一步挑一批请求去算」的决定器"
SUB = ("一句话：调度器不碰 NPU、不搬 KV，它只做三件事——"
       "决定这一步<b>算哪些请求</b>、每个请求<b>算几个 token</b>、给它<b>分哪些 KV 块</b>。"
       "它跑在 EngineCore 主线程的忙循环里（第 ④ 页）。")
X = 36


def build(doc: Doc) -> None:
    p = doc.page("⑪ Scheduler 是什么", pw=1560, ph=1420)
    head(p, TITLE, SUB)

    # --- 四个容器 ---------------------------------------------------------
    y = 150
    p.text(X, y - 32, 1000, 22,
           "<b>第一步：它只有四个容器（读代码时先认这四个）</b>", "text_b")
    c1 = p.box(X, y, 340, 132,
               "<b>requests（字典）</b><br>"
               "<font style=\"font-size:10px\">req_id → Request：<b>所有没完成的请求</b>都在这里，"
               "不管它在哪个队列。</font><br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "scheduler.py:194</font>", "box", "neutral")
    c2 = p.box(X + 362, y, 340, 132,
               "<b>waiting（队列）</b><br>"
               "<font style=\"font-size:10px\">还没开始算的请求。默认 FCFS（就是个 deque），"
               "换 priority 策略就变小顶堆。</font><br>"
               "<font style=\"font-size:9px\" color=\"#666666\">:203 · request_queue.py:75</font>",
               "box", "cloudp")
    c3 = p.box(X + 724, y, 340, 132,
               "<b>skipped_waiting（队列）</b><br>"
               "<font style=\"font-size:10px\">这一步被跳过的等待请求：等语法、等远端 KV、"
               "等流式输入。<b>它和 waiting 是两个队</b>。</font><br>"
               "<font style=\"font-size:9px\" color=\"#666666\">:205 · :2367-2373</font>",
               "box", "cloudd")
    c4 = p.box(X + 1086, y, 338, 132,
               "<b>running（列表）</b><br>"
               "<font style=\"font-size:10px\">正在跑的请求，<b>是 list 不是 queue</b>——"
               "顺序就是调度顺序，抢占时从末尾挑。</font><br>"
               "<font style=\"font-size:9px\" color=\"#666666\">:206</font>",
               "box", "data")

    # --- 关键字段 ---------------------------------------------------------
    y2 = y + 176
    p.text(X, y2 - 32, 1100, 22,
           "<b>第二步：调度器眼里一个请求有哪些关键数（决定「还能不能算」）</b>",
           "text_b")
    rows = [
        ["num_tokens", "这个请求现在总共有多少 token（prompt + 已生成）", "request.py:288"],
        ["num_computed_tokens", "已经算过多少个 token —— 与总数一减就是「还差多少」",
         "request.py:182"],
        ["status", "WAITING / WAITING_FOR_REMOTE_KVS / RUNNING / PREEMPTED / FINISHED_*",
         "request.py:364-380"],
        ["block_hashes", "前缀缓存用的块哈希（构造时算一次，每次生成再增量补）",
         "request.py:219,278"],
        ["num_preemptions", "被抢占过几次（诊断用）", "request.py:210"],
        ["is_prefill_chunk", "这一段是不是「非最后一块」的 prefill 分块", "request.py:198"],
    ]
    table(p, X, y2, [("字段", 250), ("含义", 620), ("锚点", 200)], rows, row_h=34)
    p.text(X, y2 + 34 * 7 + 6, 1424, 40,
           "<font style=\"font-size:9px\" color=\"#666666\">"
           "注意：<b>Request 上没有 block_ids 字段</b>——KV 块表存在 KVCacheManager 里"
           "（single_type_kv_cache_manager.py:127），调度器只在装配输出时把 block_ids 取出来。</font>",
           "subtitle")

    # --- 状态机 -----------------------------------------------------------
    y3 = y2 + 34 * 7 + 58
    p.text(X, y3 - 32, 1000, 22,
           "<b>第三步：请求的状态机（调度器就是在这几个状态之间搬它）</b>", "text_b")
    s1 = p.box(X, y3, 260, 62, "<b>WAITING</b><br>排队中", "box_center", "cloudp")
    s3 = p.box(X + 380, y3, 260, 62, "<b>RUNNING</b><br>正在算", "box_center",
               "cloudd")
    s5 = p.box(X + 760, y3, 300, 62, "<b>FINISHED_*</b><br>结束（多种原因）",
               "box_center", "neutral")
    p.edge(s1, s3, "分到 KV 块", st="arrow", role="http")
    p.edge(s3, s5, "生成完 / 中止", st="arrow", role="http")

    # 两个"侧态"放在下面，用短竖线连接，避免连线互相穿过
    s2 = p.box(X + 380, y3 + 110, 260, 74,
               "<b>WAITING_FOR_REMOTE_KVS</b><br>"
               "<font style=\"font-size:9px\">PD 场景：等远端的 KV 到齐</font>",
               "box_center", "fail")
    s4 = p.box(X + 760, y3 + 110, 300, 74,
               "<b>PREEMPTED</b><br>"
               "<font style=\"font-size:9px\">块不够被抢占 → 回 WAITING 队头重算</font>",
               "box_center", "fail")
    p.edge(src_pt=(X + 510, y3 + 62), dst_pt=(X + 510, y3 + 110), st="arrow",
           role="ctrl")
    p.edge(src_pt=(X + 910, y3 + 62), dst_pt=(X + 910, y3 + 110), st="arrow",
           role="fail")
    p.text(X + 380, y3 + 194, 700, 40,
           "<font style=\"font-size:9px\" color=\"#666666\">"
           "两个侧态都能回到主链：KV 到齐 → RUNNING；被抢占 → 回 WAITING 队头，"
           "下次重新 prefill（不是「换出到 CPU」，本版本没有 SWAP 模式）。</font>",
           "subtitle")

    # --- 一句话职责 -------------------------------------------------------
    y4 = y3 + 252
    p.box(X, y4, 1424, 106,
          "<b>它每步要回答的三个问题（整个 P 页的骨架）</b><br>"
          "① <b>算谁</b>：先把 running 里的老请求排一遍，再把 waiting 里的新请求收进来"
          "（前提：这一步没发生抢占）。<br>"
          "② <b>算几个 token</b>：两个预算——token 预算（这一步能发多少 token）"
          "与 input 预算（这一步能喂多少 input token）；超出就切块（chunked prefill）。<br>"
          "③ <b>分哪些块</b>：问 KVCacheManager 要 KV 槽位；要不到就挑个倒霉蛋抢占。",
          "box", "neutral")

    y5 = y4 + 140
    anchors(p, y5, [
        "四个容器与状态：vllm/v1/core/sched/scheduler.py:194-206；"
        "RequestStatus vllm/v1/request.py:364-380",
        "请求关键字段：vllm/v1/request.py:182（num_computed_tokens）、:288（num_tokens）、"
        ":219（block_hashes）",
        "KV 块表的真实位置：vllm/v1/core/single_type_kv_cache_manager.py:127；"
        "取块号 scheduler.py:1394",
        "统计与日志出口：scheduler.py:2824-2860（make_stats）；"
        "vllm/v1/metrics/loggers.py:260-292（Running/Waiting 日志）",
    ], h=96)

    y6 = y5 + 128
    concl(p, y6,
          "<b>一句话记住这一页：</b>调度器 = 四个容器（requests / waiting / skipped_waiting / "
          "running）+ 一个状态机；它每步只产出一份「执行计划」（SchedulerOutput），"
          "自己不碰模型也不碰 KV 数据。")

    legend(p, y6 + 92)
