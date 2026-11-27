"""② 为什么划算：一次前向算 1+N 个位置，权重只搬一次。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc, table

TITLE = "② 为什么划算：把「1 次前向」的边际成本算清楚"
SUB = ("一句话：decode 每步的<b>大头是搬权重</b>（与 query 数几乎无关），"
       "所以把 1+N 个位置塞进同一次前向，权重的搬运费只付一遍——这就是投机解码的经济学基础。")


def build(doc: Doc) -> None:
    p = doc.page("② 为什么划算", pw=1560, ph=1440)
    head(p, TITLE, SUB)
    X = 36

    # --- 成本结构 ---------------------------------------------------------
    y = 152
    p.text(X, y - 34, 1300, 22,
           "<b>第一步：一次 decode 前向的成本，分两块看</b>", "text_b")
    rows = [
        ["搬权重", "把全部参数从显存读一遍", "与 query 数<b>无关</b>（只读一次）",
         "每步都要付，且是主要开销"],
        ["搬 KV", "读历史 KV 参与 attention", "随 query 数线性增长",
         "N 个位置就要读 N 份"],
        ["算矩阵", "GEMM 的浮点运算", "随 query 数线性增长",
         "算力本来就闲着（带宽瓶颈）"],
    ]
    table(p, X, y, [("成本项", 200), ("是什么", 420), ("随 1+N 怎么变", 420),
                    ("结论", 380)], rows, row_h=40)

    y2 = y + 40 * 4 + 40
    p.text(X, y2 - 34, 1300, 22,
           "<b>第二步：于是「多算几个位置」的边际成本很低</b>", "text_b")
    panel(p, X, y2, 460, 176, "只算 1 个（普通 decode）", [
        "搬权重：1 份（大头）",
        "搬 KV：1 份",
        "算矩阵：1 份",
        "",
        "产出：1 个 token",
        "→ 每 token 都要付一整份权重搬运费",
    ], "fail")
    panel(p, X + 482, y2, 460, 176, "算 1+N 个（验证阶段）", [
        "搬权重：<b>仍是 1 份</b>（同一批 GEMM）",
        "搬 KV：1+N 份（线性涨）",
        "算矩阵：1+N 份（算力本来闲着）",
        "",
        "最好情况产出：N+1 个 token（全中 + bonus）",
    ], "cloudd")
    panel(p, X + 964, y2, 460, 176, "所以算术强度被拉起来了", [
        "算术强度 = 运算量 ÷ 访存量：",
        "· 1 个位置：≈ 2 FLOP/字节（带宽死锁）",
        "· 1+N 个位置：约 (1+N)×2（权重那部分被摊薄）",
        "→ 这正是 ①「一次前向多干活」的意义，",
        "也是 batching 的同一个道理（入门套 ⑤⑥）。",
    ], "cloudp")

    # --- 加速比公式 -------------------------------------------------------
    y3 = y2 + 216
    p.text(X, y3 - 34, 1300, 22,
           "<b>第三步：加速比怎么估（通用公式，仓内只给各分量）</b>", "text_b")
    p.box(X, y3, 1424, 108,
          "<font style=\"font-size:10px\">"
          "设第 i 个位置的<b>条件接受率</b>为 αᵢ（越往后越低），draft 长度 K，则<br>"
          "<b>期望产出长度 E[L] = 1 + α₁ + α₁α₂ + … + α₁α₂…α_K</b>（含最后那个 bonus token）<br>"
          "<b>加速比 ≈ E[L] ÷ (1 + 验证额外开销比)</b>　—— 注意分母：draft 自己也要花时间，"
          "K 越大验证越贵。<br>"
          "<font color=\"#c0392b\">所以「K 越大越好」是错的：E[L] 增长会饱和，而开销线性涨。</font></font>",
          "box", "neutral")

    # --- 两个极端 ---------------------------------------------------------
    y4 = y3 + 142
    panel(p, X, y4, 700, 150, "理想情况：草案很准（α 都接近 1）", [
        "一次前向产出接近 1+K 个 token，加速比接近 K+1（再扣掉 draft 开销）。",
        "什么时候准：文本有强规律（代码、格式化输出、RAG 里大量原文引用、n-gram 重复）。",
    ], "cloudd")
    panel(p, X + 760, y4, 664, 150, "糟糕情况：草案不准（α 很快衰减）", [
        "第一个位置就被拒 → 白算了一遍 draft，还多读了 KV，纯亏。",
        "什么时候不准：开放式创作、长上下文后期（接受率会明显下降）。",
        "这就是为什么需要「按位置统计接受率」并用它来调 K（见 ⑧ 的指标）。",
    ], "fail")

    y5 = y4 + 190
    bridge(p, y5, "本页交接：既然「草案准不准」决定一切，那草案从哪来就成了关键问题。"
                  "下一步（③）看 draft 的四种来源，以及 vLLM 里各自的实现类。")

    y6 = y5 + 76
    anchors(p, y6, [
        "「一次 decode 提交 1 + N 个 query」：vllm/config/vllm.py:661-679；"
        "vllm/v1/worker/gpu_model_runner.py:905",
        "memory-bound 与「闲着算力」的本仓原文："
        "vllm/docs/features/speculative_decoding/README.md:3、"
        "adaptive_verification.md:3（the extra draft tokens are close to free）",
        "验证成本随 K 增长的说明：vllm/docs/features/speculative_decoding/"
        "adaptive_verification.md:3；vllm-ascend/docs/.../speculative_decoding.md:785",
        "逐位接受率衰减：adaptive_verification.md:5；指标定义 vllm/v1/spec_decode/metrics.py:107-118",
    ], h=112)

    y7 = y6 + 146
    concl(p, y7,
          "<b>一句话记住这一页：</b>投机的收益来自「权重只搬一次」，"
          "上限是 1+K 个 token；实际加速 ≈ 期望接受长度 ÷ 额外开销。"
          "<b>K 不是越大越好，接受率决定一切。</b>")

    legend(p, y7 + 92)
