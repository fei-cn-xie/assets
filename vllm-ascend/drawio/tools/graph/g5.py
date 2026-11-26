"""⑤ vLLM 的四种图模式：NONE / PIECEWISE / FULL / FULL_AND_PIECEWISE。"""
from common import head, concl, anchors, legend, panel
from dg import Doc, table

TITLE = "⑤ 四种图模式：图从哪里开始、到哪里结束"
SUB = ("一句话：<b>PIECEWISE</b> 把 attention 挖出去（attention 之间那些层进图），"
       "<b>FULL</b> 连 attention 一起进图（但只给「形状整齐的 decode」用）。"
       "vLLM 默认是两者都要（FULL_AND_PIECEWISE），按 batch 形状挑。")


def layer_strip(p, x, y, w, label, segs, role_by_seg):
    """A horizontal strip: label + coloured segments."""
    p.text(x, y, w, 20, f"<b>{label}</b>", "text_b")
    cx = x
    for name, width, role in segs:
        p.box(cx, y + 24, width, 34, name, "box_center", role)
        cx += width + 4


def build(doc: Doc) -> None:
    p = doc.page("⑤ 四种图模式", pw=1560, ph=1440)
    head(p, TITLE, SUB)
    X = 36

    # --- 一张对照图 -------------------------------------------------------
    y = 154
    p.text(X, y - 32, 1200, 22,
           "<b>第一步：同一段 forward 的三种切法（沙色＝逐个 eager 下发，"
           "绿＝分段图里跑的段，橙＝整张图一起跑）</b>", "text_b")
    layer_strip(p, X, y, 1424, "① eager（NONE）：全部逐个算子下发",
                [("ln", 90, "neutral"), ("qkv", 90, "neutral"),
                 ("attn", 90, "neutral"), ("o_proj", 90, "neutral"),
                 ("ln", 80, "neutral"), ("MLP", 110, "neutral"),
                 ("…下一层…", 780, "neutral")],
                {"neutral": "neutral"})
    layer_strip(p, X, y + 84, 1424,
                "② PIECEWISE：attention 留 eager、其余进图（层数越多，图段越多）",
                [("ln", 90, "cloudd"), ("qkv", 90, "cloudd"),
                 ("attn", 90, "neutral"), ("o_proj", 90, "cloudd"),
                 ("ln", 80, "cloudd"), ("MLP", 110, "cloudd"),
                 ("…下一层…", 780, "cloudd")],
                {"cloudd": "cloudd"})
    layer_strip(p, X, y + 168, 1424,
                "③ FULL：整层连 attention 一起进图（只允许「整齐的 decode」形状）",
                [("ln", 90, "cloudp"), ("qkv", 90, "cloudp"),
                 ("attn", 90, "cloudp"), ("o_proj", 90, "cloudp"),
                 ("ln", 80, "cloudp"), ("MLP", 110, "cloudp"),
                 ("…下一层…", 780, "cloudp")],
                {"cloudp": "cloudp"})

    # --- 模式表 -----------------------------------------------------------
    y2 = y + 268
    p.text(X, y2 - 32, 1200, 22,
           "<b>第二步：四种模式的定义与本仓用法</b>", "text_b")
    rows = [
        ["NONE", "完全不用图（= --enforce-eager 的效果）", "启动最快、稳态最慢", "P 侧、edge、调试"],
        ["PIECEWISE", "只把「attention 之间」的部分进图，attention 留在 eager",
         "形状要求宽松，能覆盖 prefill；图段数随层数增长", "O1 默认；prefill/mixed 批次"],
        ["FULL", "整模型一张图，含 attention",
         "最快，但要求 batch 形状规整（uniform decode）且地址固定", "只在 decode 用"],
        ["FULL_DECODE_ONLY", "decode 用 FULL，其余一律 eager",
         "P/D 分离里 decode 实例的标准配置", "本仓 D 侧脚本就是这么配的"],
        ["FULL_AND_PIECEWISE", "decode 用 FULL，其它用 PIECEWISE",
         "默认（O2）；兼顾两类流量，显存与启动开销最大", "vLLM 默认"],
    ]
    table(p, X, y2, [("模式", 200), ("什么进图", 560), ("代价/特点", 420),
                     ("典型用途", 260)], rows, row_h=38)

    # --- 为什么 attention 在图外 ------------------------------------------
    y3 = y2 + 38 * 6 + 36
    panel(p, X, y3, 700, 168, "为什么 PIECEWISE 要把 attention 挖出去", [
        "attention 是不透明的 custom op，且它的形状随「KV 长度」变化——",
        "而 KV 长度每个请求都不同、每步都在变，无法在捕获时固定。",
        "所以编译时在 attention 处「打个断点」：前面进图、attention eager、后面再进图。",
        "这也解释了为什么 PIECEWISE 的图段数 ≈ 层数 ×2。",
    ], "cloudd")
    panel(p, X + 760, y3, 664, 168, "那 FULL 凭什么能把 attention 也放进去", [
        "因为 decode 阶段的形状是「整齐」的：每请求 1 个 query token"
        "（spec decode 时 1+N），batch 大小又被 padding 到固定的桶。",
        "此时 attention 的 dynamic 部分只剩「KV 长度数组」，"
        "而它可以像别的输入一样被 copy_ 进固定 buffer。",
        "所以 FULL 只对「uniform decode」开放——这行判断在 dispatcher 里。",
    ], "cloudp")

    y4 = y3 + 208
    p.box(X, y4, 1424, 92,
          "<b>怎么选（实用建议）</b><br>"
          "<font style=\"font-size:10px\">"
          "① 想最快跑通/调试 → <b>NONE</b>（--enforce-eager）：启动最快、报错最直白。<br>"
          "② 单机混跑 prefill+decode → 默认 <b>FULL_AND_PIECEWISE</b>（O2）。<br>"
          "③ P/D 分离 → P 侧 <b>NONE 或 PIECEWISE</b>、D 侧 <b>FULL_DECODE_ONLY</b>"
          "（本仓部署脚本正是这样：P 与 edge 加 --enforce-eager，D 配 FULL_DECODE_ONLY）。</font>",
          "box", "neutral")

    y5 = y4 + 126
    anchors(p, y5, [
        "模式枚举与语义：vllm/config/compilation.py:53-110、:614-649",
        "默认值来自 -O 级别：vllm/config/vllm.py:277（O0→NONE）、:300（O1→PIECEWISE）、"
        ":323/:346（O2/O3→FULL_AND_PIECEWISE）、:447（默认 O2）",
        "PIECEWISE 的切点 = attention 类 op：vllm/config/compilation.py:772-790、"
        ":1169（set_splitting_ops_for_v1）",
        "attention 是「捕获期要打断」的 custom op："
        "vllm/model_executor/layers/attention/attention.py:758-794（eager_break_during_capture）",
        "两种 wrapper 的嵌套：vllm/v1/worker/gpu_model_runner.py:5453-5493",
    ], h=104)

    y6 = y5 + 136
    concl(p, y6,
          "<b>一句话记住这一页：</b>PIECEWISE＝把 attention 挖空的「分段图」，形状宽松、"
          "段数多；FULL＝整张图，最快但只吃「整齐的 decode」；"
          "vLLM 默认两种都抓，运行时按 batch 形状挑一种。")

    legend(p, y6 + 92,
           extra_fill=("FULL 的图范围", "cloudp"),
           extra_line=("虚线/灰 = 仍在 eager 执行", "neutral"))
