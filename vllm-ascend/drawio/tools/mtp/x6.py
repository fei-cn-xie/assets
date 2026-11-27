"""⑥ 被拒的 KV 怎么办：不回滚数据，只回退计数。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc

TITLE = "⑥ 被拒的 KV 怎么办：不回滚数据，只回退「算到哪了」"
SUB = ("一句话：验证前向会把 1+N 行 KV <b>全部写进去</b>（包括后来被拒的位置）；"
       "被拒时不擦数据，而是把 num_computed_tokens 退回去 —— 下一轮从同一批槽位重新覆盖。")


def build(doc: Doc) -> None:
    p = doc.page("⑥ KV 回退", pw=1560, ph=1440)
    head(p, TITLE, SUB)
    X = 36

    # --- 三个事实 ---------------------------------------------------------
    y = 152
    p.text(X, y - 34, 1300, 22,
           "<b>第一步：三个事实（顺序很重要）</b>", "text_b")
    panel(p, X, y, 460, 190, "① 验证前向「全都写」", [
        "slot_mapping 是按 positions 全量算出来的，",
        "所以 1+N 行都会写进 KV cache，",
        "包括那些后来被拒的位置。",
        "",
        "→ 也就是说：被拒的 KV 确实存在过。",
    ], "fail")
    panel(p, X + 482, y, 460, 190, "② 回退的是「计数」", [
        "num_rejected = 草案数 − 接受数；",
        "num_computed_tokens -= num_rejected；",
        "",
        "→ 「这个请求算到第几个 token」被退回到被拒位置。",
        "（注意：这一步在调度器侧做，不在 worker）",
    ], "cloudp")
    panel(p, X + 964, y, 460, 190, "③ 不擦数据，靠「同槽覆盖」", [
        "下一轮 positions 从回退后的位置重新算，",
        "compute_slot_mapping 会算出<b>同一批槽位</b>，",
        "于是新数据直接覆盖掉旧的（脏数据自愈）。",
        "",
        "→ 所以没有「擦除」操作，也不需要额外开销。",
    ], "cloudd")

    # --- 一张时间线 -------------------------------------------------------
    y2 = y + 230
    p.text(X, y2 - 34, 1300, 22,
           "<b>第二步：一轮的时间线（第 2、3 个草案被拒的情形）</b>", "text_b")
    p.box(X, y2, 1424, 108,
          "<font style=\"font-size:10px\">"
          "上一轮末尾：算出 4 行（1 个新 token + 3 个草案）→ 全部写入 KV 槽位 #a #b #c #d<br>"
          "验证：第 1 个接受、第 2 个被拒 → 接受数 = 1<br>"
          "回退：num_rejected = 3 − 1 = 2 → num_computed_tokens 退回 2 个位置<br>"
          "<font color=\"#c0392b\">下一轮：positions 从 #c 重新开始，"
          "#c #d … 被新数据覆盖（不需要任何擦除/释放操作）</font></font>",
          "box", "neutral")

    # --- 异步调度的额外一层 -----------------------------------------------
    y3 = y2 + 148
    p.text(X, y3 - 34, 1300, 22,
           "<b>第三步：异步调度（async scheduling）还要多一层「乐观账」</b>",
           "text_b")
    panel(p, X, y3, 700, 176, "问题", [
        "异步模式下，调度器不等 worker 回报就先把「已算 token 数」按乐观值推进",
        "（假设全接受，因为多数情况确实接受）。",
        "但实际可能少接受几个 → 必须事后校正。",
    ], "fail")
    panel(p, X + 760, y3, 664, 176, "做法", [
        "worker 回传「本步真实接受数」（valid_sampled_token_count），",
        "调度器算 correction = 乐观值 − 实际值，再把 num_computed_tokens 扣回来；",
        "同时同步 num_output_placeholders（占位计数）。",
    ], "cloudd")

    # --- 为什么要多留 KV --------------------------------------------------
    y4 = y3 + 216
    p.box(X, y4, 1424, 116,
          "<b>顺带解释一个容易困惑的点：为什么「给 drafter 多留 N 个 KV 槽」</b><br>"
          "<font style=\"font-size:10px\">"
          "草案阶段要往 target 之外的位置写 KV（例如第 N+1 个草案位置），"
          "所以分配时除了给 1+N 个 query 的槽，还要额外预留 N 个 lookahead 槽。<br>"
          "如果不预留 → 写越界 / 需要临时扩容 → 既慢又可能碎。"
          "这也是「开投机解码会让可用 KV 变少」的直接原因之一（见 ⑦ 与 ⑨ 的实测）。</font>",
          "box", "cloudp")

    y5 = y4 + 156
    bridge(p, y5, "本页交接：KV 的账（回退 + lookahead）就是投机解码与「显存」这条约束的接口。"
                  "下一步（⑦）把三条系统约束（形状/显存/调度）一次性摊开。")

    y6 = y5 + 76
    anchors(p, y6, [
        "全量写 KV：slot_mapping 由 positions 算出 vllm/v1/worker/gpu_model_runner.py:2196-2209",
        "回退计数：vllm/v1/core/sched/scheduler.py:2041-2059"
        "（num_rejected 与 num_computed_tokens 扣减）",
        "同槽覆盖：下一轮 positions/slot 重算同文件 :2196-2209",
        "异步乐观/校正：gpu_model_runner.py:1383-1408（乐观）、:1554-1584（校正）、"
        "vllm/v1/spec_decode/utils.py:822-851",
        "lookahead 预留：vllm/config/vllm.py:633-658；scheduler.py:741-745",
        "hybrid/Mamba 只保留末位 state：gpu_model_runner.py:1588-1644",
    ], h=120)

    y7 = y6 + 154
    concl(p, y7,
          "<b>一句话记住这一页：</b>被拒的 KV 不擦、不释放，只把「算到哪了」的计数退回去，"
          "下一轮覆盖同一批槽位；异步调度还要再补一层乐观校正。"
          "<b>代价是：为了给 drafter 留槽，可用 KV 会变少。</b>")

    legend(p, y7 + 92)
