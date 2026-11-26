"""④ 三个硬约束：为什么图必须是「固定形状 + 固定地址 + 不能 CPU 同步」。"""
from common import head, concl, anchors, legend, panel
from dg import Doc

TITLE = "④ 三个硬约束：图为什么「挑食」"
SUB = ("一句话：录下来的图里记的是<b>具体的地址和具体的形状</b>，"
       "所以回放时必须一模一样；图里也不能有「回头问 CPU」的操作。"
       "三条约束各自对应一类常见故障。")


def build(doc: Doc) -> None:
    p = doc.page("④ 三个硬约束", pw=1560, ph=1500)
    head(p, TITLE, SUB)
    X = 36

    y = 152
    p.text(X, y - 32, 1200, 22,
           "<b>约束一：形状固定 —— 所以要「按桶捕获 + padding」</b>", "text_b")
    panel(p, X, y, 700, 168, "为什么必须固定", [
        "图里的循环次数、reshape、kernel 的 grid 大小都由形状决定。",
        "形状变了 = 图里那些数都对不上 → 只能换一张图。",
        "而形状在真实流量里千变万化（1 个请求 / 7 个请求 / 44 个 token…），"
        "不可能每个都抓。",
    ], "fail")
    panel(p, X + 760, y, 664, 168, "本仓怎么解决：有限个桶 + 向上取整", [
        "只抓有限的一组尺寸（cudagraph_capture_sizes，如 1,2,4,8,16,…,512）。",
        "运行时的 batch 先「对齐到最近的桶」＝ padding（多出来的位置填占位）。",
        "padding 的具体落点：block_table 填 NULL_BLOCK_ID、slot_mapping 填 -1。",
        "副作用：padding 带来的空算力，也算在图的开销里（有 num_paddings 统计）。",
    ], "cloudd")

    y2 = y + 208
    p.text(X, y2 - 32, 1200, 22,
           "<b>约束二：地址固定 —— 所以要用「持久 buffer + copy_」</b>", "text_b")
    panel(p, X, y2, 700, 176, "为什么必须固定", [
        "图里记的是张量的<b>显存地址</b>，不是「第几个参数」。",
        "回放时如果输入换了一块新显存，图还是往老地址读 → 读到的是垃圾。",
        "所以每次进图前必须把新数据写进<b>捕获时那几块地址</b>。",
    ], "fail")
    panel(p, X + 760, y2, 664, 176, "本仓怎么解决：预分配 + 每步 copy_", [
        "input_ids / positions / query_start_loc / seq_lens 等一次性按"
        "最大长度分配好（持久 buffer）。",
        "每步只做 copy_ 覆写这些 buffer 的前 N 个位置。",
        "代码里甚至有开关：debug 模式下会比对「捕获期与回放期的输入地址是否一致」，"
        "不一致直接抛 <font style=\"font-size:9px\">Input addresses for cudagraphs are "
        "different during replay</font>。",
    ], "cloudd")

    y3 = y2 + 216
    p.text(X, y3 - 32, 1200, 22,
           "<b>约束三：图里不能有「回头问 CPU」的操作</b>", "text_b")
    panel(p, X, y3, 700, 176, "为什么", [
        "回放时 CPU 只发一条命令，之后不再参与。",
        "所以图里任何需要「读回一个值再决定下一步」的操作都没法工作："
        ".item() / print / 动态分配 / 条件分支里的 Python 判断。",
        "更糟的是它们不会报错——<b>只会把捕获那一刻的值「烤」进图里</b>，"
        "于是你拿到一个永远算出旧结果、但一点也不报错的图。",
    ], "fail")
    panel(p, X + 760, y3, 664, 176, "本仓的对应处理", [
        "捕获闸门：在非捕获窗口触发捕获会直接抛"
        "<font style=\"font-size:9px\">CUDA graph capturing detected at an inappropriate "
        "time</font>。",
        "捕获期禁止重新分配：attention 后端看 forward_context.capturing 决定"
        "「只登记任务、不申请 workspace」。",
        "一次性准备全在图外：dummy run 里把 metadata 都构造好再进图。",
    ], "cloudd")

    y4 = y3 + 216
    p.box(X, y4, 1424, 96,
          "<b>三张「故障 → 原因」速查（对得上就是这三条之一）</b><br>"
          "<font style=\"font-size:10px\">"
          "① 输出全是重复/错乱的旧结果 → 某处 .item() 或分支被烤进图（约束三）。<br>"
          "② 报 Input addresses ... different during replay → 有代码在回放时换了输入张量"
          "（约束二）。<br>"
          "③ 并发一上来反而变慢、图「没了」→ 形状超了最大桶，静默回落 eager（约束一）。"
          "</font>", "box", "neutral")

    y5 = y4 + 130
    anchors(p, y5, [
        "桶与 padding：vllm/config/compilation.py:655-658（capture_sizes）、"
        "vllm/forward_context.py:29-58（BatchDescriptor）、"
        "vllm/v1/worker/gpu_model_runner.py:2353-2355 / :4150-4152（padding 落点）",
        "持久 buffer 与 copy_：vllm/v1/worker/gpu/input_batch.py:19-40、"
        "vllm/v1/worker/gpu_model_runner.py:4435、:1815",
        "地址校验断言：vllm/compilation/cuda_graph.py:348-357；"
        "breakable_cudagraph.py:422",
        "闸门与捕获期禁止分配：vllm/compilation/monitor.py:87-104；"
        "vllm-ascend/vllm_ascend/ascend_forward_context.py:177（capturing 标志）",
        "「.item() 被烤进图」的仓内注释：vllm/model_executor/models/cohere_compass.py:1633、"
        "vllm/v1/attention/backends/rocm_aiter_fa.py:778",
    ], h=112)

    y6 = y5 + 146
    concl(p, y6,
          "<b>一句话记住这一页：</b>形状固定 → 按桶捕获 + padding；地址固定 → 持久 buffer + "
          "copy_；不能问 CPU → 一切动态准备都放在图外。三条里破一条，"
          "要么报错、要么静默算错、要么悄悄退回 eager。")

    legend(p, y6 + 92)
