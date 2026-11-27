"""⑦ 量化：训练的产物怎么变成「一字节一个数」。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc, table

TITLE = "⑦ 量化：从「2 字节一个数」到「1 字节 + 一把尺子」"
SUB = ("一句话：量化 = <b>把每个数换成更少的位 + 一个尺度（scale）</b>；"
       "粒度越细越准、但尺子越多。本仓全部是<b>训练后量化（PTQ）</b>的产物——"
       "仓库只负责「按描述文件正确加载」，不做量化本身。")


def build(doc: Doc) -> None:
    p = doc.page("⑦ 量化", pw=1560, ph=1540)
    head(p, TITLE, SUB)
    X = 36

    # --- 四种粒度 ---------------------------------------------------------
    y = 148
    p.text(X, y - 32, 1300, 22,
           "<b>第一步：四种量化粒度（仓内有严格判据，不是随口叫的）</b>", "text_b")
    rows = [
        ["per-tensor", "整个张量一个尺度", "<b>1 个</b>", "最省额外开销，最不准"],
        ["per-channel", "每个输出通道一个", "输出通道数", "权重常用（仓内 w8a8 的权重）"],
        ["per-token", "每个 token 一个", "token 数", "激活常用（动态量化）"],
        ["per-group", "每 N 个元素一组", "元素数 ÷ 组大小", "最准；scale 元数据约增 3%"],
    ]
    table(p, X, y, [("粒度", 220), ("尺度按什么分", 420), ("scale 个数", 340),
                    ("特点", 444)], rows, row_h=36)

    # --- 昇腾 scheme ------------------------------------------------------
    y2 = y + 36 * 5 + 38
    p.text(X, y2 - 32, 1300, 22,
           "<b>第二步：昇腾侧的实际方案（每个的名字就说明了粒度组合）</b>",
           "text_b")
    rows2 = [
        ["W8A8（静态）", "权重 per-channel int8；激活 <b>静态</b> per-tensor",
         "精度优先，需要校准数据"],
        ["W8A8_DYNAMIC", "权重 per-channel int8；激活 <b>逐 token 动态</b>",
         "最常用（本仓 w8a8 主干）"],
        ["W8A8_MXFP8", "fp8 + <b>per-group</b> 尺度（组大小默认 32）",
         "更省、更依赖硬件支持"],
        ["W4A16 / W4A8", "权重压到 4 bit，激活保留 bf16 / int8",
         "权重更小，适合带宽瓶颈"],
    ]
    table(p, X, y2, [("方案", 260), ("粒度组合", 620), ("适用", 544)], rows2,
          row_h=36)

    # --- 加载时的四件事 ---------------------------------------------------
    y3 = y2 + 36 * 5 + 38
    p.text(X, y3 - 32, 1300, 22,
           "<b>第三步：加载一个量化 checkpoint 时会发生的四件事</b>", "text_b")
    p.box(X, y3, 1424, 132,
          "<font style=\"font-size:10px\">"
          "① <b>读描述文件</b>：模型目录里有一份「每层用哪个方案」的清单，"
          "加载器按它逐层选实现（找不到就<b>直接报错</b>，并说明「你加了这个参数但目录不是量化产物」）。<br>"
          "② <b>声明额外张量</b>：除了权重，还要为 scale / offset / bias 分配参数；"
          "其中一部分<b>不从文件读、而是加载后算出来</b>（仓内有一份「算出来」的名字名单）。<br>"
          "③ <b>格式重排</b>：昇腾的矩阵算子只吃 NZ（分形）布局，所以要把权重就地重排（见 ⑧ 的耗时）。<br>"
          "④ <b>决定是否反量化</b>：默认<b>保持低精度</b>，反量化在算子内部完成；"
          "只有原生 fp8 产物在某些硬件上会被摊成 bf16（代价：显存约翻倍）。</font>",
          "box", "neutral")

    # --- 为什么省 ---------------------------------------------------------
    y4 = y3 + 172
    panel(p, X, y4, 700, 152, "为什么量化在推理侧特别划算", [
        "decode 是带宽瓶颈：每步都要把全部权重从显存读一遍（入门套 ⑤）。",
        "位宽减半 → 要搬的字节减半 → 单步时间近似减半。",
        "省的不只是容量，更是<b>每步的搬运量</b>——这才是量化的主收益。",
    ], "cloudd")
    panel(p, X + 760, y4, 664, 152, "为什么训练侧不能直接这么干", [
        "训练要更新参数：梯度是「微小的增量」，低精度会把增量抹掉；",
        "所以训练必须留高精度主权重（这是 QAT 存在的理由之一）。",
        "→ 量化基本是「训练完之后」的动作，所以叫训练后量化（PTQ）。",
    ], "fail")

    # --- 本仓实测 ---------------------------------------------------------
    y5 = y4 + 192
    p.box(X, y5, 1424, 108,
          "<b>本仓实测：十个部署包里只有一个包用量化</b><br>"
          "<font style=\"font-size:10px\">"
          "· 两个脚本（P 侧与 D 侧）带 <font style=\"font-size:9px\">--quantization ascend</font>，"
          "模型目录名里直接写着 w8a8；<br>"
          "· 只有 D/P 的那个包显式给了 <font style=\"font-size:9px\">indexer_kv_dtype: int8</font>；<br>"
          "· <b>没有任何包设置 <font style=\"font-size:9px\">--kv-cache-dtype</font></b>"
          "→ 其余包的 KV cache 仍是模型精度（bf16）。"
          "<font color=\"#c0392b\">也就是说：那个包省的是「索引器侧的小 K cache」，主 KV 容量没变。</font></font>",
          "box", "data")

    y6 = y5 + 148
    bridge(p, y6, "本页交接：量化只是「权重文件形态」的一种。下一步（⑧）回到权重本身——"
                  "顺便把训练产出的其他文件（优化器状态等）为什么被丢弃讲清。")
    p.text(X, y6 + 52, 1424, 24,
           "<font style=\"font-size:9px;color:#666666\">"
           "注：本页「训练侧不能直接低精度更新」与 PTQ/QAT 的划分属通用知识；"
           "仓内没有任何 QAT（伪量化）实现，只有 PTQ 产物的加载。</font>", "subtitle")

    y7 = y6 + 92
    axes = ["粒度判据：vllm/model_executor/layers/quantization/utils/quant_utils.py:111-137"
            "（per-tensor :122 / per-token :125 / per-channel :128 / per-group :131）",
            "昇腾方案注册：(quant_type, layer_type) → class："
            "vllm-ascend/vllm_ascend/quantization/methods/registry.py:21-52",
            "W8A8 静态与动态的粒度原文：w8a8/w8a8_static.py:34-39、w8a8/w8a8_dynamic.py:61-66；"
            "MXFP8 :55-61（per-group，组大小 32）",
            "额外张量声明：methods/.../method_adapters.py:100-191；"
            "「算出来而非读文件」名单 :197-207",
            "描述文件与分派：configs/modelslim_config.py:109（文件名）、:911-918（读取）、"
            ":660-771（get_quant_method）",
            "本仓实测：assets/c_home/bash/pds/fix_1120/prefill.sh:113-115、decode.sh:113-115"]
    anchors(p, y7, axes, h=124)

    y8 = y7 + 158
    concl(p, y8,
          "<b>一句话记住这一页：</b>量化 = 更少的位 + 一把尺子；粒度决定「尺子多不多」，"
          "本仓只负责按描述文件正确加载 PTQ 产物。"
          "<b>它省的主要不是容量，而是 decode 每一步要搬的字节数。</b>")

    legend(p, y8 + 92, extra_fill=("量化产物/数据面", "data"))
