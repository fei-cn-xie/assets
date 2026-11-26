"""⑫ 一次 schedule() 的完整流程：先跑老的、再收新的、块不够就抢占。"""
from common import head, concl, anchors, legend
from dg import Doc, table

TITLE = "⑫ 一次 schedule() 里到底发生了什么（五段流程）"
SUB = ("一句话：<b>先排老的（running）→ 要块 → 块不够就抢占 → 再收新的（waiting）"
       "→ 产出执行计划</b>。顺序不能反：本步一旦抢占，就不再收新请求。")
X = 36


def seg(p, y, no, title, lines, role, h=150):
    p.box(X, y, 1424, h, "", "box", role)
    p.text(X + 12, y + 8, 1380, 24, f"<b>{no}　{title}</b>", "text_b")
    body = "<br>".join(f"<font style=\"font-size:10px\">{ln}</font>" for ln in lines)
    p.text(X + 12, y + 34, 1400, h - 42, body, "text")


def build(doc: Doc) -> None:
    p = doc.page("⑫ schedule 内部流程", pw=1560, ph=1620)
    head(p, TITLE, SUB)

    y = 156
    seg(p, y, "①", "开场：数预算、开新的一步", [
        "token 预算 = max_num_scheduled_tokens（没配就回落到 max_num_batched_tokens）；"
        "input 预算 = max_num_batched_tokens。",
        "给 KV 块池发「新一步开始」的通知（new_step_starts），并算出这一步是否要"
        "<b>推迟 prefill</b>（DP 波次节流）。",
        "⚠ 两个预算同时存在：一个管「发多少 token」，一个管「喂多少 input token」。",
    ], "neutral", h=120)

    seg(p, 292, "②", "先排老的：running 里的请求挨个算「这一步给它几个 token」",
        [
            "每个请求：还差多少 = num_tokens（含 spec）− num_computed_tokens；"
            "再被 long_prefill_token_threshold 截断、被剩余预算压小、"
            "再按 max_model_len / LWD 窗口 / MTP lookahead 收紧。",
            "算出来是 0 就<b>跳过这个、看下一个</b>（continue，不是 break）。",
            "拿到数字后立刻去要 KV 槽位（下一段）。",
        ], "cloudp", h=140)

    seg(p, 448, "③", "要 KV：要不到就抢占（本版本只有「重算」一种）",
        [
            "要块：allocate_slots(请求, num_new_tokens)。块不够返回 None。",
            "抢占谁：FCFS 直接挑 <b>running 队尾</b>；priority 策略挑"
            "「优先级最低 / 最晚到」的那个。",
            "被抢占的代价：<b>释放它的 KV 块、num_computed_tokens 归零、"
            "插回 waiting 队头</b>——下次重新 prefill（没有「换出到 CPU」这种模式）。",
            "⚠ 一旦本步发生抢占，第 ④ 段直接跳过：这一步不再接新请求。",
        ], "fail", h=156)

    seg(p, 624, "④", "再收新的：从 waiting 里挑人，先查前缀缓存",
        [
            "准入条件：还有 token 预算、running 数没到 max_num_seqs、本步没抢占。",
            "先看 skipped_waiting（上一步被跳过的），再看 waiting。",
            "若请求 num_computed_tokens == 0：<b>查前缀缓存</b>"
            "（按 block_hashes 找最长命中，最多命中到 num_tokens − 1，"
            "因为最后一个 token 必须重算才能出 logits）。",
            "本地命中 + 远端命中（connector）合并成 num_computed_tokens；本步只算剩下的。",
            "PD 场景：若 connector 说「要异步拉 KV」，则<b>只分块不算 token</b>，"
            "状态置 WAITING_FOR_REMOTE_KVS，放回 skipped_waiting。",
        ], "cloudd", h=178)

    seg(p, 824, "⑤", "收尾：产出执行计划，并推进「已算多少」",
        [
            "组装 SchedulerOutput：每请求 num_scheduled_tokens + 新块的 block_ids"
            "（<b>注意 SchedulerOutput 里没有 slot_mapping</b>，那是 worker 自己算的）。",
            "把本步的 token 数累加进 num_computed_tokens（异步调度下还要维护 placeholder），"
            "清掉 finished / preempted 集合。",
            "这份计划通过消息队列广播给所有 TP worker —— <b>同一份对象、4 个 worker 各拿一份</b>。",
        ], "data", h=140)

    # --- 抢占示意图 -------------------------------------------------------
    y2 = 986
    p.text(X, y2 - 32, 1000, 22,
           "<b>抢占到底做了什么（顺序很重要，代码就是这么写的）</b>", "text_b")
    p.box(X, y2, 1424, 92,
          "<font style=\"font-size:10px\">"
          "释放这个请求的 KV 块（含 encoder cache）→ 从 _inflight_prefills 移除 → "
          "status = PREEMPTED → <b>num_computed_tokens = 0</b> → 清掉 spec draft token → "
          "处理掉在飞的旧输出 → num_preemptions += 1 → "
          "<b>prepend 到 waiting 队头</b>（下次最先被重新调度）。<br>"
          "所以「抢占」在 v1 里等于「这笔 prefill 白算，从头再来」——"
          "这也是为什么块不够时要先尽量别超发。</font>", "box", "fail")

    # --- 预算表 -----------------------------------------------------------
    y3 = y2 + 126
    p.text(X, y3 - 32, 1000, 22, "<b>三个上限各管一段（别混）</b>", "text_b")
    rows = [
        ["max_num_seqs", "每个 EngineCore 同时 running 的请求数", "只在 waiting 准入时检查"],
        ["max_num_batched_tokens（input 预算）", "一步能喂多少 input token",
         "chunked prefill 就是被它切出来的"],
        ["max_num_scheduled_tokens（token 预算）", "一步能发多少 token",
         "没配时 = max_num_batched_tokens"],
        ["max_num_queued_reqs（前端准入）", "全局排队上限（跨 DP）", "超了直接 503，不进入调度"],
    ]
    table(p, X, y3, [("上限", 340), ("管什么", 420), ("注意", 620)], rows, row_h=34)

    y4 = y3 + 34 * 5 + 40
    anchors(p, y4, [
        "schedule() 主体与两段循环：vllm/v1/core/sched/scheduler.py:568-1520"
        "（running 段 :617-821；waiting 段 :866-1329）",
        "要块与抢占：scheduler.py:739-807（allocate_slots 循环）、:1541-1582（_preempt_request）",
        "前缀缓存命中：scheduler.py:929-1018、vllm/v1/core/kv_cache_manager.py:266-326",
        "PD 挂起/放行：scheduler.py:1273-1303（挂起）、:3026-3041 + :3062-3085（放行）",
        "产出计划：scheduler.py:1471-1493；结构 vllm/v1/core/sched/output.py:318-405",
    ], h=104)

    y5 = y4 + 136
    concl(p, y5,
          "<b>一句话记住这一页：</b>先跑老的、再收新的；要块失败就从队尾抢占"
          "（代价是那条请求全部重算）；一旦抢占，这一步不再收新请求；"
          "最后把「每请求算几个 token + 用哪些块」打成一份 SchedulerOutput 广播出去。")

    legend(p, y5 + 92)
