"""② 「一个算子」≠「一次 kernel 下发」：融合为什么值钱。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc

TITLE = "② 一个算子 ≠ 一次下发：融合省的是「中间结果落显存再读回来」"
SUB = ("一句话：数学上写成三步（比如「残差相加 → 归一化 → 量化」），"
       "如果不融合，设备上就是<b>三次 kernel 下发 + 两次中间结果落显存</b>；"
       "融合成一个算子后只剩<b>一次下发</b>——省的是带宽和启动开销，不是算力。")


def build(doc: Doc) -> None:
    p = doc.page("② 算子与下发", pw=1560, ph=1520)
    head(p, TITLE, SUB)
    X = 36

    # --- 两种写法的对比 ---------------------------------------------------
    y = 148
    p.text(X, y - 30, 1300, 22,
           "<b>第一步：同一个语义，两种落地（对比「三次下发」与「一次下发」）</b>",
           "text_b")
    p.box(X, y, 690, 190,
          "<b>不融合（三个算子）</b><br><br>"
          "<font style=\"font-size:10px\">"
          "① 读 x、读残差 → 算加法 → <b>写回显存</b>（中间结果 A）<br>"
          "② 读 A → 算归一化 → <b>写回显存</b>（中间结果 B）<br>"
          "③ 读 B → 算量化 → 写入目标<br><br>"
          "<font color=\"#c0392b\">3 次下发 + 2 次「写回再读出」= 3 倍启动开销 + 2 倍额外带宽</font></font>",
          "box", "fail")
    p.box(X + 730, y, 694, 190,
          "<b>融合（一个算子）</b><br><br>"
          "<font style=\"font-size:10px\">"
          "① 读 x、读残差 → 加法 → 归一化 → 量化 → <b>一次写出</b><br><br>"
          "<font color=\"#1c7c54\">1 次下发 + 0 次额外读写<br>"
          "中间结果留在片上（寄存器/片上缓存），不落显存</font><br><br>"
          "代价：这种算子是<b>专门写的</b>，对形状/dtype/头数有要求，不满足就得回退。</font>",
          "box", "cloudd")

    # --- 三个收益 ---------------------------------------------------------
    y2 = y + 230
    p.text(X, y2 - 30, 1300, 22,
           "<b>第二步：融合到底省了什么（三条，按收益排）</b>", "text_b")
    panel(p, X, y2, 460, 176, "① 省带宽（decode 时最值钱）", [
        "中间结果不用写回显存再读出来。",
        "decode 本来就卡在带宽上（入门套 ⑤），",
        "所以「少读写一次」直接换成速度。",
    ], "data")
    panel(p, X + 482, y2, 460, 176, "② 省下发开销", [
        "每次 kernel 下发都有固定成本（提交、校验、排队）。",
        "算子越碎，固定成本占比越高——小 batch 时尤其明显。",
        "这也是图捕获（站 6）要解决的同一个问题。",
    ], "cloudp")
    panel(p, X + 964, y2, 460, 176, "③ 省片上内存搬运", [
        "部分融合算子还能把「读一次、多处用」的中间量留在片上",
        "（例如归一化里的方差，既要做除法又要做缩放）。",
        "这类优化纯靠手写 kernel 才能做到。",
    ], "cloudd")

    # --- 代价与回退 -------------------------------------------------------
    y3 = y2 + 216
    p.box(X, y3, 1424, 124,
          "<b>第三步：融合的代价（这就是为什么本仓到处是「回退」代码）</b><br>"
          "<font style=\"font-size:10px\">"
          "一个融合算子要固定死很多东西：<b>dtype 组合、形状约束、头数、布局、硬件代次</b>。"
          "任何一条不满足，就只能走分步实现。<br>"
          "所以本仓的代码里你会看到<b>一层套一层的开关</b>：先看自定义算子库加载成功没，"
          "再看这个硬件有没有这个能力，再看本次的形状/dtype 是否符合，最后才决定用融合还是分步。<br>"
          "<font color=\"#c0392b\">理解这一点，就能解释「为什么同一份代码在不同机器上性能差很多」——"
          "不是配置差异，是<b>融合路径被逐级关掉了</b>。</font></font>",
          "box", "neutral")

    # --- 一张决策图 -------------------------------------------------------
    y4 = y3 + 164
    p.text(X, y4 - 30, 1300, 22,
           "<b>第四步：一次算子调用的决策顺序（六道闸门，任一不过就降级）</b>",
           "text_b")
    gates = ["自定义算子库\n加载成功？", "硬件支持\n这个能力？", "dtype/形状\n符合？",
             "布局符合？", "融合路径\n可用？", "用融合算子"]
    cw = 224
    for i, g in enumerate(gates):
        cx = X + i * (cw + 12)
        role = "cloudd" if i == len(gates) - 1 else ("cloudp" if i < 3 else "neutral")
        p.box(cx, y4, cw, 84,
              f"<font style=\"font-size:10px\">{g.replace(chr(10), '<br>')}</font>",
              "box_center", role)
        if i:
            p.edge(src_pt=(cx - 10, y4 + 42), dst_pt=(cx - 2, y4 + 42), st="arrow",
                   role="http")
    p.text(X, y4 + 92, 1424, 40,
           "<font style=\"font-size:9px;color:#666666\">"
           "任一闸门不过 → 走「分步实现」：功能一样、就是慢。"
           "所以「结果对但慢」十有八九是这条链上某一环被关掉了。</font>", "subtitle")

    y5 = y4 + 148
    bridge(p, y5, "本页交接：融合的价值说清了。下一步（③）看最大的一类算子——"
                  "矩阵乘（linear / GEMM），它占了绝大部分计算量。")

    y6 = y5 + 76
    anchors(p, y6, [
        "自定义算子库加载开关（失败则全部禁用并告警）："
        "vllm-ascend/vllm_ascend/utils.py:544-603（enable_custom_op）；"
        "日志 \"Failed to register custom ops, all custom ops will be disabled.\" :590/:597",
        "单点回退示例（融合版 vs 基础版）：ops/layernorm.py:92（融合 add+rmsnorm+bias）→ :118（回退）",
        "硬件能力门控：device/hardware_profile.py（能力枚举），调用示例 "
        "ops/rotary_embedding.py:712-719、sample/sampler.py:163-167",
        "融合路径前置条件检查：ops/linear_op.py:164-181（unsupported_reason）、:229-260（_can_fuse）",
        "图级融合 pass（在图捕获前面再融一次）：compilation/passes/norm_quant_fusion_pass.py:385-393、"
        "qknorm_rope_fusion_pass.py:91",
    ], h=128)

    y7 = y6 + 162
    concl(p, y7,
          "<b>一句话记住这一页：</b>融合算子省的是<b>下发次数与中间结果的读写</b>，不是算力；"
          "而它挑形状/ dtype/硬件，所以本仓到处是「能融就融、不能融就分步」的六道闸门——"
          "<b>「结果对但慢」通常是闸门被关掉了。</b>")

    legend(p, y7 + 92)
