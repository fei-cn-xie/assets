"""① 为什么需要图捕获：一次 decode step 的两条时间线。"""
from common import head, concl, anchors, legend, panel
from dg import Doc

TITLE = "① 为什么需要图捕获：CPU 在「下发」，NPU 在「等」"
SUB = ("一句话：推理的每一步都要把整层的成百上千个算子<b>逐个提交</b>给设备。"
       "decode 时每步只算很少的 token，于是<b>提交的时间开始和计算的时间同一个量级</b>——"
       "图捕获就是把「逐个提交」压成「一次提交」。")
X = 36


def build(doc: Doc) -> None:
    p = doc.page("① 为什么需要图捕获", pw=1560, ph=1340)
    head(p, TITLE, SUB)

    # --- 两条时间线 -------------------------------------------------------
    y = 156
    p.text(X, y - 34, 1200, 22,
           "<b>第一步：先看清一步推理里其实有两条时间线</b>", "text_b")

    p.text(X, y, 700, 22, "<b>普通执行（eager / 逐算子下发）</b>", "text_b")
    p.box(X, y + 30, 700, 150,
          "<font style=\"font-size:10px\">"
          "CPU：<b>算子1</b> 提交 → <b>算子2</b> 提交 → <b>算子3</b> 提交 → …（每层几十个，几十层）<br>"
          "NPU：　　　　　　　执行1　　　　　执行2　　　　　执行3 …<br><br>"
          "两条线在<b>同一个流</b>上交替：CPU 提交一个，NPU 才能算一个。"
          "当「执行」很短（decode 只算 1 个 token）时，"
          "<b>NPU 有大量时间在等 CPU 提交下一个算子</b>。</font>",
          "box", "fail")

    p.text(X + 760, y, 664, 22, "<b>图捕获之后（回放）</b>", "text_b")
    p.box(X + 760, y + 30, 664, 150,
          "<font style=\"font-size:10px\">"
          "CPU：<b>replay()</b> —— 只提交一次<br>"
          "NPU：　　▷ 整段计算连着跑（中间不再等 CPU）<br><br>"
          "整张图的算子序列、依赖、参数在<b>启动时就录好了</b>；"
          "回放时 CPU 只发一条命令，剩下的交给驱动/硬件按录好的顺序跑。</font>",
          "box", "cloudd")
    p.edge(src_pt=(X + 350, y + 190), dst_pt=(X + 1092, y + 190), st="arrow",
           role="http", label="捕获这一步做完之后", label_pos=0)

    # --- decode 为什么最吃亏 ---------------------------------------------
    y2 = y + 240
    p.text(X, y2 - 34, 1200, 22,
           "<b>第二步：为什么 decode 阶段最吃亏（prefill 反而还好）</b>", "text_b")
    panel(p, X, y2, 700, 176, "prefill（一次算整段 prompt）", [
        "一次要算 N 个 token（几百到几万），矩阵很「宽」。",
        "device 上的计算时间长 → 提交开销被摊薄，CPU 追得上。",
        "所以 prefill 不做整图捕获也影响不大。",
    ], "neutral")
    panel(p, X + 760, y2, 664, 176, "decode（一次只算新生成的 1 个 token）", [
        "每请求 1 个 query token（spec decode 时是 1+N）。",
        "计算量极小 → 每步的时间几乎由「下发次数」决定。",
        "而层数 × 每层算子数 一个都不能少 → 提交开销成为主导。",
        "所以「图」几乎只为 decode 而生。",
    ], "fail")

    # --- 一句话动机 + 代价 ------------------------------------------------
    y3 = y2 + 216
    p.box(X, y3, 1424, 118,
          "<b>代价与收益（本仓文档的原话）</b><br>"
          "<font style=\"font-size:10px\">"
          "① <b>收益</b>：把 host 侧的上千次下发压成一次，"
          "换来的是「稳态 decode 性能」（vllm/docs/configuration/optimization.md:25 原文："
          "跳过图捕获可以最快启动，<b>代价是稳态 decode 性能</b>）。<br>"
          "② <b>代价</b>：启动变慢（要真跑一遍来录）+ 显存变多（每张图要占一份内存池），"
          "而且形状必须固定（见 ④ 的三条硬约束）。<br>"
          "③ Ascend 侧的设计目标与 CUDA 一致："
          "<font style=\"font-size:9px\">"
          "reduce host launch overhead for small and medium runtime shapes</font>"
          "（vllm-ascend/docs/.../ACL_Graph.md:7）。</font>",
          "box", "neutral")

    y4 = y3 + 152
    anchors(p, y4, [
        "收益/代价的定性描述：vllm/docs/configuration/optimization.md:25、"
        "vllm/docs/design/cuda_graphs.md:43-46、vllm/docs/configuration/conserving_memory.md:48-65",
        "decode 每请求只有 1 个 query token：vllm/config/vllm.py:661-679",
        "「uniform decode」判定：vllm/v1/cudagraph_dispatcher.py:37、"
        "vllm/v1/worker/gpu_model_runner.py:3888-3907",
        "host 侧同步被当作热路径敌人：vllm/envs.py:635-643（VLLM_GPU_SYNC_CHECK）、"
        "vllm/utils/gpu_sync_debug.py:3",
        "Ascend 的设计目标原文：vllm-ascend/docs/source/developer_guide/Design_Documents/"
        "ACL_Graph.md:7",
    ], h=104)

    y5 = y4 + 136
    concl(p, y5,
          "<b>一句话记住这一页：</b>图捕获不是「让算子变快」，而是<b>把下发的次数变少</b>——"
          "prefill 计算量大、摊得开；decode 每步只有 1 个 token，"
          "省下的 CPU 提交时间直接变成端到端的 token 间隔（TPOT）。")

    legend(p, y5 + 92)
