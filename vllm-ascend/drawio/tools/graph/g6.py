"""⑥ 形状桶与 dispatch：一次 decode step 怎么选中一张图。"""
from common import head, concl, anchors, legend, panel
from dg import Doc, table

TITLE = "⑥ 形状桶与 dispatch：这一步到底用哪张图"
SUB = ("一句话：启动时按「桶」抓了一批图；运行时先把本步的 batch 形状<b>向上取整到某个桶</b>，"
       "再按 (token 数, 是否整齐 decode, LoRA 数) 去查表 —— 查不到就用 eager。")


def build(doc: Doc) -> None:
    p = doc.page("⑥ 桶与 dispatch", pw=1560, ph=1460)
    head(p, TITLE, SUB)
    X = 36

    # --- 桶长什么样 -------------------------------------------------------
    y = 152
    p.text(X, y - 32, 1200, 22,
           "<b>第一步：桶不是连续的，是一串阶梯（本仓默认栅格）</b>", "text_b")
    p.box(X, y, 1424, 96,
          "<font style=\"font-size:10px\">"
          "默认栅格 = <b>1, 2, 4</b> + <b>8, 16, 24, …, 248</b>（步长 8）+ "
          "<b>256, 272, …, 上限</b>（步长 16），上限默认 <b>512</b>（Blackwell 1024），"
          "且不超过 max_num_batched_tokens。<br>"
          "还会补上 max_num_batched_tokens 本身；开 spec decode 时按 "
          "uniform_decode_query_len 的倍数补桶。<br>"
          "<font color=\"#c0392b\">含义：batch 越小越「细」，越大越「粗」——"
          "因为小 batch 的延迟最敏感。</font></font>",
          "box", "neutral")

    # --- padding 示例 -----------------------------------------------------
    y2 = y + 130
    p.text(X, y2 - 32, 1200, 22,
           "<b>第二步：一个 5 个请求、共 5 个 token 的 decode step 会发生什么</b>",
           "text_b")
    panel(p, X, y2, 700, 158, "① 先算本步形状", [
        "num_tokens = 5（5 个请求各 1 个新 token）",
        "uniform_decode = True（大家都只算 1 个 token）",
        "查表：5 → 落到桶 8（向上取整）",
    ], "neutral")
    panel(p, X + 760, y2, 664, 158, "② 把形状 padding 到 8", [
        "多出来的 3 个位置填占位：",
        "· block_table 填 NULL_BLOCK_ID（不指向真实块）",
        "· slot_mapping 填 -1（跳过写 KV）",
        "· seq_lens / query_start_loc 补成合法值",
        "输出只取前 5 个位置的结果（其余丢弃）。",
    ], "cloudd")

    # --- dispatch 决策 ----------------------------------------------------
    y3 = y2 + 198
    p.text(X, y3 - 32, 1200, 22,
           "<b>第三步：dispatch 的查表顺序（三条路，最后一条是保底）</b>", "text_b")
    d1 = p.box(X, y3, 430, 92,
               "<b>① 先查 FULL 的精确键</b><br>"
               "<font style=\"font-size:9px\">"
               "(padded num_tokens, uniform, 活跃 LoRA 数) 完全匹配</font>",
               "box", "cloudp")
    d2 = p.box(X + 470, y3, 460, 92,
               "<b>② 再查 PIECEWISE 的放宽键</b><br>"
               "<font style=\"font-size:9px\">"
               "PIECEWISE 不关心 num_reqs / uniform，键更宽松</font>", "box", "cloudd")
    d3 = p.box(X + 960, y3, 464, 92,
               "<b>③ 都不中 → 返回 NONE（eager）</b><br>"
               "<font style=\"font-size:9px;color:#c0392b\">"
               "⚠ 这一步<b>没有任何日志</b>，是最隐蔽的「图没了」</font>", "box", "fail")
    p.edge(d1, d2, "没中", st="arrow", role="http")
    p.edge(d2, d3, "没中", st="arrow", role="http")

    y4 = y3 + 132
    p.box(X, y4, 1424, 118,
          "<b>四个「提前退出、直接 eager」的条件（都会让图静默失效）</b><br>"
          "<font style=\"font-size:10px\">"
          "① 模式是 NONE（本来就没图）；② 桶还没初始化（还没捕获完）；"
          "③ <b>num_tokens 超过最大桶</b>（并发突然变高最容易踩）；"
          "④ 只允许 NONE 的特殊场景（例如某些 attention 后端 / encoder-decoder 不支持 FULL）。<br>"
          "排查手法：看 KV cache usage 与 Running 请求数是否刚好越过了最大桶；"
          "或者直接开 <font style=\"font-size:9px\">--cudagraph-metrics</font> 看命中率。</font>",
          "box", "fail")

    # --- 桶大小怎么定 -----------------------------------------------------
    y5 = y4 + 152
    p.text(X, y5 - 32, 1200, 22,
           "<b>第四步：桶的上下界怎么被卡住（三个数决定）</b>", "text_b")
    rows = [
        ["max_num_seqs", "同时跑的请求数上限", "决定「最大桶大约多大」"],
        ["max_num_batched_tokens", "一步的 token 预算", "再给一个上界（chunked prefill 会打到这里）"],
        ["max_cudagraph_capture_size", "显式上限（默认 512 / Blackwell 1024）",
         "超过就一律 eager"],
        ["--cudagraph-capture-sizes", "手工指定桶列表", "想省显存就缩小它（如 [1,2,4,8,16]）"],
    ]
    table(p, X, y5, [("参数", 320), ("管什么", 460), ("对桶的影响", 600)], rows,
          row_h=34)

    y6 = y5 + 34 * 5 + 36
    anchors(p, y6, [
        "栅格与上限推导：vllm/config/vllm.py:2196-2463（:2262-2269 上限、:2376-2388 栅格、"
        ":2348-2349 取 min、:2398-2400 spec 倍数）",
        "桶键 BatchDescriptor：vllm/forward_context.py:29-58；"
        "bs→padded 映射 vllm/v1/cudagraph_dispatcher.py:72-91",
        "注册桶：cudagraph_dispatcher.py:166-233；dispatch 决策与四条早退：:235-324（早退 :274-281、"
        "回落 NONE :320-324）",
        "padding 落点：vllm/v1/worker/gpu_model_runner.py:2353-2355（NULL_BLOCK_ID）、"
        ":4150-4152（slot_mapping=-1）",
        "按 attention 后端降级：vllm/config/compilation.py:1382-1545",
    ], h=112)

    y7 = y6 + 146
    concl(p, y7,
          "<b>一句话记住这一页：</b>桶是一串「阶梯」，运行时向上取整 + padding；"
          "dispatch 先查 FULL 精确键、再查 PIECEWISE 放宽键、都不中就静默回 eager。"
          "<b>「并发一高就变慢」十有八九是越过了最大桶。</b>")

    legend(p, y7 + 92)
