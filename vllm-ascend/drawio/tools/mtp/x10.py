"""⑩ 指标、日志与排障：接受率怎么看、什么时候该关掉。"""
from common import head, concl, anchors, legend, bridge
from dg import Doc, table

TITLE = "⑩ 指标与排障：先看接受率，再决定要不要开"
SUB = ("一句话：投机解码的<b>唯一核心指标是平均接受长度</b>；"
       "它低于 1.3 左右就说明「草案基本没人要」，此时应减小 K 或直接关掉。")


def build(doc: Doc) -> None:
    p = doc.page("⑩ 指标与排障", pw=1560, ph=1520)
    head(p, TITLE, SUB)
    X = 36

    # --- 指标 -------------------------------------------------------------
    y = 152
    p.text(X, y - 34, 1300, 22,
           "<b>第一步：四个指标（前两个决定要不要开）</b>", "text_b")
    rows = [
        ["平均接受长度 mean acceptance length", "1 + 接受数/草案数", "∈[1, K+1]；越大越好"],
        ["草案接受率 draft acceptance rate", "接受数/草案数", "0~1；配合 K 看"],
        ["逐位接受率 per-position", "每个位置各自的接受率向量", "看「第几位开始塌」→ 定 K"],
        ["接受吞吐 / 起草吞吐", "接受 token/s ÷ 起草 token/s", "看浪费了多少算力"],
    ]
    table(p, X, y, [("指标", 400), ("定义", 460), ("怎么读", 560)], rows, row_h=36)

    # --- 怎么判读 ---------------------------------------------------------
    y2 = y + 36 * 5 + 40
    p.text(X, y2 - 34, 1300, 22,
           "<b>第二步：判读（三个典型区间）</b>", "text_b")
    p.box(X, y2, 460, 150,
          "<b>K=3、平均接受长度 ≈ 2.5~3</b><br>"
          "<font style=\"font-size:10px\">草案很准，接近理论上限。"
          "此时可以尝试加 K，或保持现状<br>"
          "（典型场景：代码、格式化输出、大段引用）</font>",
          "box", "cloudd")
    p.box(X + 482, y2, 460, 150,
          "<b>平均接受长度 ≈ 1.5~2.5</b><br>"
          "<font style=\"font-size:10px\">正常收益区。注意看<b>逐位接受率</b>："
          "如果第 2 位就掉到很低，说明 K 该减到 1~2。</font>",
          "box", "neutral")
    p.box(X + 964, y2, 460, 150,
          "<b>平均接受长度 ≈ 1.0~1.3</b><br>"
          "<font style=\"font-size:10px\">草案基本被全拒：你多付了 draft 的成本 + lookahead 的显存，"
          "换来的只有 padding 的浪费。<b>应该关掉 MTP 或换策略。</b></font>",
          "box", "fail")

    # --- 三个常见坑 -------------------------------------------------------
    y3 = y2 + 190
    p.text(X, y3 - 34, 1300, 22,
           "<b>第三步：三个常见「开了但没用」的原因</b>", "text_b")
    p.box(X, y3, 1424, 118,
          "<font style=\"font-size:10px\">"
          "① <b>并发太高把 K 的优势吃掉</b>：验证时有效 batch 变成「请求数 ×(1+K)」，"
          "大 batch 下被拒的那些草案全是白算的算力。<br>"
          "② <b>长上下文后期接受率塌陷</b>：越到后面越难猜准，投机反而变成开销。<br>"
          "③ <b>形状不整齐导致退回 eager</b>：批里混了长短不一的请求 → 丢了整图，"
          "投机省下的时间又被 eager 的提交开销吃掉。</font>",
          "box", "fail")

    # --- 日志对照 ---------------------------------------------------------
    y4 = y3 + 158
    p.text(X, y4 - 34, 1300, 22, "<b>第四步：日志对照表（照这个 grep）</b>", "text_b")
    rows2 = [
        ["SpecDecoding metrics: Mean acceptance length: …", "主指标一行，含逐位接受率"],
        ["[spec_decode/base] Initializing spec decode proposer: method=…", "确认 draft 真的在跑"],
        ["… Sharing target model embedding weights …", "确认 MTP 复用了 embedding（省显存）"],
        ["Enabling num_speculative_tokens > 1 will run multiple times of forward …",
         "K>1 的官方警告（接受率可能下降）"],
        ["GLM series models with speculative decoding … switched to eager mode",
         "某些族被强制 eager（draft 不跑图）"],
        ["decode_threshold exceeded npu_fused_infer_attention_score TND layout's limit of 16",
         "K 太大被硬上限拦住（K+1 ≤ 16）"],
        ["[Lwd][edge-worker] UNEMBED seqno=… accepted=…", "LWD 下每次 UNEMBED 实际吐了几个 token"],
    ]
    table(p, X, y4, [("日志关键字", 800), ("含义", 620)], rows2, row_h=32)

    # --- 调优顺序 ---------------------------------------------------------
    y5 = y4 + 32 * 8 + 34
    p.box(X, y5, 1424, 96,
          "<b>调优顺序（按代价从小到大）</b><br>"
          "<font style=\"font-size:10px\">"
          "① 先看接受率够不够（不够就减 K，别加）；"
          "② 再看并发与显存（MTP 会把可用 KV 吃掉一块 → 调 max_num_seqs / gpu-memory-utilization）；"
          "③ 再看批形状（是否因不 uniform 掉出整图）；"
          "④ 最后才考虑换 draft 策略（n-gram / 换小头 / 自适应 K）。</font>",
          "box", "neutral")

    y6 = y5 + 130
    bridge(p, y6, "本套到此结束。回到 `学习地图.md` 的六站顺序，可以把这一套插在"
                  "「vLLM图捕获」之后读——因为投机解码同时踩到了形状与调度两条约束。")

    y7 = y6 + 76
    anchors(p, y7, [
        "指标定义与聚合：vllm/v1/spec_decode/metrics.py:107-118、:120-136（日志原文）",
        "Prometheus 指标名与 PromQL：metrics.py:180-196、:231",
        "K>1 警告：vllm/config/speculative.py:1356-1365",
        "TND 上限断言：vllm-ascend/vllm_ascend/attention/attention_v1.py:273-277",
        "draft 是否跑图：vllm-ascend/.../spec_decode/llm_base_proposer.py:271、:670-684",
        "LWD UNEMBED 日志：vllm-ascend/worker/lwd_edge_model/lwd_edge_worker.py:159",
    ], h=112)

    y8 = y7 + 146
    concl(p, y8,
          "<b>一句话记住这一页：</b>先量平均接受长度再谈优化："
          ">2.5 可以加 K，1.5~2.5 看逐位衰减，<1.3 就直接关；"
          "同时别忘了它吃掉的显存与「形状不整齐就掉图」的连带代价。")

    legend(p, y8 + 92)
