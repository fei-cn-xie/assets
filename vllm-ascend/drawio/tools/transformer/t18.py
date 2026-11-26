"""⑱ 权重名字 → 形状：把文件里的字符串和前面的矩阵对上号。"""
from dg import table
from mth import (ACT, CALC, ERR, NOTE, OUT, W, arrow, legend, panel, title)

TITLE = "⑱ 权重名字与形状：把文件里的字符串翻译成前面的矩阵"
SUB = ("文件里全是 model.layers.7.self_attn.q_proj.weight 这种名字。"
       "这一页把每个名字对应的形状列出来，你会发现它们和前 13 页的矩阵一一对应。")


def build(doc) -> None:
    p = doc.page("⑱ 权重名与形状", pw=1700, ph=1440)
    title(p, TITLE, SUB, w=1620)

    # --- 命名规则 ---------------------------------------------------------
    y = 150
    p.text(36, y - 60, 1200, 22,
           "<b>第一步：名字就是路径（用点号分层，一眼能看出属于哪个部件）</b>",
           "text_b")
    segs = [
        ("model", "整个模型", NOTE),
        ("layers", "第几层（0~31）", ACT),
        ("7", "具体层号", CALC),
        ("self_attn", "哪个子模块（注意力 / mlp / 归一化）", ACT),
        ("q_proj", "哪个矩阵（q/k/v/o、gate/up/down）", W),
        ("weight", "是权重还是偏置（weight / bias）", OUT),
    ]
    x = 36
    for seg, why, role in segs:
        w = max(120, 26 + len(seg) * 12)
        p.box(x, y, w, 56, f"<b>{seg}</b>", "box_center", role)
        p.text(x, y + 62, w, 20,
               f"<font style=\"font-size:9px\" color=\"#666666\">{why}</font>",
               "subtitle")
        x += w + 26
    p.box(36, y + 92, 1628, 50,
          "<font style=\"font-size:10px\">"
          "完整名字：<b>model.layers.7.self_attn.q_proj.weight</b>"
          "　＝　第 7 层的注意力里，query 投影矩阵的权重。"
          "　<font style=\"font-size:9px\" color=\"#666666\">"
          "第 0 层的形状和第 31 层完全一样，只有数值不同。</font></font>",
          "box", NOTE)

    # --- 形状对照表 -------------------------------------------------------
    y2 = y + 170
    p.text(36, y2 - 60, 1200, 22,
           "<b>第二步：Llama-2-7B 的全部权重形状（这就是「7B」的真身）</b>",
           "text_b")
    rows = [
        ["model.embed_tokens.weight", "[32000, 4096]", "词嵌入表（③ 的 E）", "1.31 亿"],
        ["model.layers.N.input_layernorm.weight", "[4096]", "归一化 γ（⑪）", "4 千"],
        ["model.layers.N.self_attn.q_proj.weight", "[4096, 4096]", "W_Q（⑥）", "1678 万"],
        ["model.layers.N.self_attn.k_proj.weight", "[4096, 4096]", "W_K（⑥）", "1678 万"],
        ["model.layers.N.self_attn.v_proj.weight", "[4096, 4096]", "W_V（⑥）", "1678 万"],
        ["model.layers.N.self_attn.o_proj.weight", "[4096, 4096]", "W_O（⑩ 的拼接投影）", "1678 万"],
        ["model.layers.N.post_attention_layernorm.weight", "[4096]", "第二个 γ（⑪）", "4 千"],
        ["model.layers.N.mlp.gate_proj.weight", "[11008, 4096]", "gate 升维（⑫）", "4509 万"],
        ["model.layers.N.mlp.up_proj.weight", "[11008, 4096]", "up 升维（⑫）", "4509 万"],
        ["model.layers.N.mlp.down_proj.weight", "[4096, 11008]", "down 降维（⑫）", "4509 万"],
        ["model.norm.weight", "[4096]", "最后一层归一化（⑪）", "4 千"],
        ["lm_head.weight", "[32000, 4096]", "输出头（⑬；与 embed 共享时不存在）", "1.31 亿"],
    ]
    table(p, 36, y2, [("文件名（N = 0~31）", 470), ("形状 [out, in]", 200),
                      ("对应图里的哪一页", 400), ("参数个数", 140)], rows,
          row_h=34)

    # --- 三个坑 -----------------------------------------------------------
    y3 = y2 + 34 * 13 + 40
    p.text(36, y3 - 60, 1200, 22,
           "<b>第三步：三个几乎人人踩过的坑</b>", "text_b")
    panel(p, 36, y3, 520, "坑 ① 形状是 [out, in]，和公式反着",
          "公式写 y = xW（W 是 in×out），但 <b>PyTorch 的 Linear.weight 存成 "
          "[out, in]</b>——正好是转置。<br>"
          "所以 gate_proj.weight 是 [11008, 4096] 而不是 [4096, 11008]。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "判断方法：看那个 11008 在哪一维——它在第 0 维，就是 [out, in]。</font>", ERR)
    panel(p, 576, y3, 520, "坑 ② 文件里是分着的，代码里是合并的",
          "文件里 k_proj / v_proj 各自独立；"
          "vLLM 会把它们<b>拼成 qkv_proj</b> 一次算完（⑥ 提过）。<br>"
          "所以你在代码里搜 qkv_proj.weight 是搜不到文件的——"
          "加载时会做名字映射与拼接。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "改模型时如果忘了这层映射，会出现「张量找不到」或「形状不匹配」。</font>", ERR)
    panel(p, 1116, y3, 548, "坑 ③ 量化后多出一堆 .scales / .qweight",
          "4bit 权重不再是 [4096, 4096] 的 fp16，而是「打包的整数 + 每组的缩放系数」"
          "（⑳ 讲）。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "此时名字会变成 qweight / scales / qzeros 之类，形状也完全不同——"
          "看到这些名字说明这是量化模型。</font>", NOTE)

    # --- 代码锚点 --------------------------------------------------------
    y4 = y3 + 210
    p.box(36, y4, 1628, 100,
          "<b>代码锚点</b><br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "名字映射表（把 HF 的 q_proj/k_proj/v_proj 映射成 vllm 的 qkv_proj 等）："
          "各模型文件里的 stacked_params_mapping / WeightsMapper，"
          "见 vllm/model_executor/models/llama.py 的 load_weights 与 "
          "vllm/model_executor/models/utils.py<br>"
          "逐层装配：vllm/model_executor/models/llama.py:347（LlamaModel）、"
          ":251（LlamaDecoderLayer）、:165/:175（注意力）、:82/:95/:103（FFN）<br>"
          "权重加载器：vllm/model_executor/model_loader/default_loader.py"
          "</font>", "box", NOTE)

    legend(p, y4 + 132)
