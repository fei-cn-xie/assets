"""④ 推理的两个阶段：prefill 与 decode（形态完全不同）。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc

TITLE = "④ 推理的两个阶段：prefill 一次算一大段，decode 每步只算一个字"
SUB = ("一句话：<b>prefill</b> 把整段 prompt 一次算完（矩阵很宽、算力吃满）；"
       "<b>decode</b> 每步给每个请求只补 1 个 token（矩阵很窄、算力闲着）。"
       "这一个差别能解释后面 80% 的优化。")


def token_strip(p, x, y, w, label, n_new, n_total, role, last_role=None):
    """Draw `n_total` cells; the last `n_new` are coloured with `last_role`."""
    p.text(x, y, w, 20, f"<b>{label}</b>", "text_b")
    bw = w / max(n_total, 1)
    for i in range(n_total):
        r = (last_role or role) if i >= n_total - n_new else "neutral"
        p.box(x + i * bw, y + 24, max(bw - 2, 2), 30, "", "box_center", r)


def build(doc: Doc) -> None:
    p = doc.page("④ prefill 与 decode", pw=1560, ph=1400)
    head(p, TITLE, SUB)
    X = 36

    # --- 形状对比 ---------------------------------------------------------
    y = 156
    p.text(X, y - 34, 1200, 22,
           "<b>第一步：同一句话，两种阶段的「矩阵形状」完全不同</b>", "text_b")
    token_strip(p, X, y, 1424, "prompt 有 12 个 token：prefill 一次把 12 个位置全算了（新算 12）",
                12, 12, "cloudp")
    token_strip(p, X, y + 84, 1424,
                "decode 第 1 步：只有第 13 个位置是新的（新算 1，但要读全部 12 个的历史 KV）",
                1, 13, "cloudd")
    token_strip(p, X, y + 168, 1424,
                "decode 第 2 步：第 14 个位置是新的（同样只算 1 个）", 1, 14, "cloudd")

    # --- 四维对比 ---------------------------------------------------------
    y2 = y + 250
    p.text(X, y2 - 34, 1200, 22,
           "<b>第二步：四个维度对比（记住这张表，后面全靠它）</b>", "text_b")
    panel(p, X, y2, 700, 176, "prefill：宽矩阵，算力是瓶颈", [
        "· 一次处理的 token 数 = prompt 长度（几十到几万）；",
        "· 每算一个 token，都要做 O(n) 次乘加（n = 长度）→ 总计算量 O(n²)；",
        "· 计算量 / 访存量 比例高 → <b>compute-bound</b>（算力吃满）；",
        "· 只发生一次，所以「把 prompt 算完」的延迟叫 TTFT。",
    ], "cloudp")
    panel(p, X + 760, y2, 664, 176, "decode：窄矩阵，显存带宽是瓶颈", [
        "· 一次只算 1 个 token（spec decode 时 1+N）；",
        "· 但每步都要把<b>全部权重</b>从显存读一遍，还要读历史 KV；",
        "· 计算量 / 访存量 比例极低 → <b>memory-bound</b>（带宽吃满、算力闲着）；",
        "· 每生成一个 token 都要一轮，所以它的速度叫 TPOT。",
    ], "cloudd")

    y3 = y2 + 216
    panel(p, X, y3, 1424, 116, "为什么这两个名字值得单独记：它们的所有优化手段都不一样", [
        "· prefill 想的是「怎么把宽矩阵喂满」→ 分块（chunked prefill）、大 batch 的 prefill、"
        "甚至和 decode 拼一批；<br>"
        "· decode 想的是「怎么少读一点、少等一点」→ KV 复用（前缀缓存）、量化、"
        "把多个请求捆一起（batching）、把下发次数压掉（图捕获）、甚至把 KV 搬到别的机器（PD 分离）。",
    ], "data")

    y4 = y3 + 156
    bridge(p, y4, "本页交接：「算力 vs 带宽」这句话需要一点更硬的解释，否则容易变成口号。"
                  "下一步（⑤）用「算术强度」把它讲成可以自己算的数。")

    y5 = y4 + 76
    anchors(p, y5, [
        "「每步给每个请求补 1 个 token」：vllm/config/vllm.py:661-679"
        "（decode 提交 1 个 query，spec 时 1+N）",
        "uniform decode 判定：vllm/v1/worker/gpu_model_runner.py:3888-3907",
        "分块 prefill 的开关：vllm/config/scheduler.py:116（enable_chunked_prefill 默认 True）、"
        ":70（long_prefill_token_threshold）、:49（max_num_batched_tokens）",
        "两个阶段共用一个调度入口：vllm/v1/core/sched/scheduler.py:568",
    ], h=104)

    y6 = y5 + 136
    concl(p, y6,
          "<b>一句话记住这一页：</b>prefill 是「一次算一大段」的算力密集型，"
          "decode 是「每步算 1 个但要把全部权重读一遍」的带宽密集型；"
          "两个阶段的问题不同，所以要分开优化，甚至分到不同机器上（PD 分离）。")

    legend(p, y6 + 92, extra_fill=("本步新算的 token", "cloudp"))
