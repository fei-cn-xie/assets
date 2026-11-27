"""① 一张前向图上的每个方块，在昇腾上到底是谁在算。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc, table

TITLE = "① 从「模型图」到「NPU 算子」：每个方块最后由谁执行"
SUB = ("一句话：前十一套讲的是「算什么」，这套讲<b>「谁来算」</b>——"
       "同一个数学操作，在昇腾上可能对应一个专用融合算子，也可能是几步通用算子。"
       "本文把这条映射摊开，并说明<b>哪一步最贵、为什么不融合会慢</b>。")


def build(doc: Doc) -> None:
    p = doc.page("① 算子层总览", pw=1560, ph=1560)
    head(p, TITLE, SUB)
    X = 36

    # --- 一张分层图 -------------------------------------------------------
    y = 146
    p.text(X, y - 30, 1300, 22,
           "<b>第一步：三层视角（越往下越接近硬件）</b>", "text_b")
    layers = [
        ("模型层", "「一层 Transformer」：embedding → attention → MLP → lm_head", "neutral"),
        ("框架层", "vLLM 把一层拆成「线性 / 归一化 / attention / 激活 / 采样」这些算子调用", "cloudp"),
        ("算子层", "每个算子落到<b>哪个 NPU kernel</b>、要不要融合、走哪条流", "cloudd"),
    ]
    cw = 460
    for i, (name, body, role) in enumerate(layers):
        cy = y + i * 92
        p.box(X, cy, 1424, 76, "", "box", role)
        p.text(X + 12, cy + 8, 200, 24, f"<b>{name}</b>", "text_b")
        p.text(X + 200, cy + 10, 1200, 56,
               f"<font style=\"font-size:10px\">{body}</font>", "text")
        if i:
            p.edge(src_pt=(X + 712, cy - 14), dst_pt=(X + 712, cy - 2), st="arrow",
                   role="http")

    # --- 一张映射表 -------------------------------------------------------
    y2 = y + 3 * 92 + 16
    p.text(X, y2 - 30, 1300, 22,
           "<b>第二步：本文要建立的那条映射（每一步的「谁在算」）</b>", "text_b")
    rows = [
        ["词嵌入查表", "通用算子（按索引取值）", "只管搬一行数据，没有矩阵乘"],
        ["RMSNorm", "专用归一化算子（也可与残差相加融合）", "一层里出现两次"],
        ["QKV 投影", "量化/非量化 GEMM 算子", "列并行，不通信"],
        ["attention", "<b>prefill 与 decode 是两个不同算子</b>", "布局不同，见 ③④"],
        ["o_proj", "GEMM + 行并行 all-reduce", "每层一次通信"],
        ["gate/up + SiLU + down", "可融合的向量/矩阵算子；MoE 则是分组矩阵乘", "MLP 主力"],
        ["lm_head", "GEMM（词表维度很大）", "只对需要判 token 的位置算"],
        ["采样", "通用算子（排序/累加）+ 少量专用算子", "越往后越吃延迟"],
    ]
    table(p, X, y2, [("模型层的一步", 300), ("算子层的落点", 700), ("要点", 424)], rows,
          row_h=38)

    # --- 为什么值得单独讲 --------------------------------------------------
    y3 = y2 + 38 * 9 + 32
    p.box(X, y3, 1424, 116,
          "<b>第三步：为什么「算子层」值得单独一套图（三个理由）</b><br>"
          "<font style=\"font-size:10px\">"
          "① <b>它决定了性能上限</b>：同一份模型，融合算子用上与否能差出一大截——"
          "因为省掉的是「中间结果落显存再读回来」。<br>"
          "② <b>它决定了限制条件</b>：很多「必须满足 X 才能跑」的要求（布局、头数、dtype、形状）"
          "不是框架规定的，而是<b>算子本身只支持这些</b>。<br>"
          "③ <b>它解释了跨平台差异</b>：vLLM 在 GPU 上的一套写法，到昇腾上要换成专用算子，"
          "所以本仓与上游的差异大量集中在这一层。</font>",
          "box", "neutral")

    # --- 阅读建议 ---------------------------------------------------------
    y4 = y3 + 156
    p.text(X, y4 - 30, 1300, 22,
           "<b>第四步：这一套的阅读顺序（与前面几站的接口）</b>", "text_b")
    p.box(X, y4, 1424, 108,
          "<font style=\"font-size:10px\">"
          "站 1（Transformer 数学：每个算子算什么）× 站 4 ⑯（权重怎么参与计算）"
          "→ <b>本套（这些计算落到哪个 kernel）</b> → 站 6（图捕获：怎么把一串 kernel 录成图）"
          "→ 站 8（参数：哪些开关决定用哪个 kernel）。<br>"
          "所以本套适合在「已经懂模型怎么算、也懂进程结构」之后读；"
          "反之，如果一上来就读算子名，会不知道它在算哪一步。</font>",
          "box", "cloudp")

    y5 = y4 + 148
    bridge(p, y5, "本页交接：总览有了。下一步（②）先讲清一件最容易混的事——"
                  "「一个算子」和「一次 kernel 下发」不是一回事。")

    y6 = y5 + 76
    anchors(p, y6, [
        "昇腾算子的统一封装入口：vllm-ascend/vllm_ascend/device/device_op.py"
        "（`torch_npu.*` 与 `torch.ops._C_ascend.*` 的集中封装，见 ⑧ 页的算子总表）",
        "linear 与融合算子的实现：vllm-ascend/vllm_ascend/ops/（linear / layernorm / rotary_embedding 等）",
        "attention 的落点见 ③④ 页；MoE 的落点见「MoE 套」⑧ 页",
        "权重如何参与计算（GEMM 读的那份张量从哪来）：见「进程与线程」套 ⑯ 页",
    ], h=112)

    y7 = y6 + 146
    concl(p, y7,
          "<b>一句话记住这一页：</b>模型图上的每一步，在昇腾上都要落到<b>一个具体的 kernel</b>；"
          "融合算子省的是「落显存再读回来」；"
          "<b>而所有「必须满足 X 才能跑」的限制，绝大多数来自「算子只支持这些」。</b>")

    legend(p, y7 + 92)
