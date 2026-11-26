"""㉑ 收尾：从文件到推理（三张表的对齐）+ 全套 21 页索引 + 自检清单。"""
from dg import table
from mth import (ACT, CALC, ERR, NOTE, OUT, W, arrow, legend, panel, title)

TITLE = "㉑ 收尾：从硬盘到第一个 token，以及 21 页的总索引"
SUB = ("最后一页做两件事：把「config ↔ 权重名 ↔ 计算图」三张表对齐，"
       "再给一张随时能翻回上一页的索引。")


def build(doc) -> None:
    p = doc.page("㉑ 收尾与索引", pw=1700, ph=1500)
    title(p, TITLE, SUB, w=1620)

    # --- 三张表对齐 -------------------------------------------------------
    y = 150
    p.text(36, y - 60, 1200, 22,
           "<b>第一步：同一个矩阵的三张「身份证」（对上了，你就真的入门了）</b>",
           "text_b")
    rows = [
        ["config.json 里的字段", "num_attention_heads=32, head_dim=128", "决定它有多大"],
        ["权重文件里的名字", "model.layers.7.self_attn.q_proj.weight", "决定它存在哪"],
        ["权重文件里的形状", "[4096, 4096]", "out × in（⑱ 的坑①）"],
        ["代码里的层", "llama.py:165 self.qkv_proj", "算的时候怎么用"],
        ["图里的位置", "⑥ Q/K/V 投影（本套 ⑥ 页）", "为什么要它"],
    ]
    table(p, 36, y, [("从哪个角度", 280), ("具体长什么样", 620), ("它回答什么", 300)],
          rows, row_h=36)

    # --- 加载到第一个 token ----------------------------------------------
    y2 = y + 6 * 36 + 40
    p.text(36, y2 - 60, 1200, 22,
           "<b>第二步：从「按下回车」到「吐出第一个 token」的完整链路</b>",
           "text_b")
    steps = [
        ("读 config", "决定网络结构（⑮）", CALC),
        ("读 index / 分片", "知道每个张量在哪（⑰）", CALC),
        ("按名字装权重", "名字映射 + 形状校验（⑱）", W),
        ("切分 / 量化", "按 TP 切、按方案反量化（⑰⑳）", W),
        ("建 KV cache 池", "按 config 与显存算能放多少块（③）", OUT),
        ("分词 + 模板", "文本 → id 序列（②⑲）", ACT),
        ("前向 32 层", "⑤~⑫ 重复 32 次", ACT),
        ("输出头 + 采样", "logits → 概率 → 抽一个（⑬）", OUT),
    ]
    cx = 36
    for i, (t, why, role) in enumerate(steps):
        p.box(cx, y2, 190, 78,
              f"<b>{t}</b><br><font style=\"font-size:9px\" color=\"#666666\">{why}</font>",
              "box_center", role)
        if i:
            p.edge(src_pt=(cx - 14, y2 + 39), dst_pt=(cx - 2, y2 + 39),
                   st="arrow", role="http")
        cx += 204

    # --- 自检清单 ---------------------------------------------------------
    y3 = y2 + 130
    p.text(36, y3 - 60, 1200, 22,
           "<b>第三步：拿到一个新模型时的五步自检（照着念一遍就能定位问题）</b>",
           "text_b")
    checks = [
        ("1. 参数规模对得上吗？", "hidden² × 4 + hidden × ffn × 3 再乘层数 ≈ 模型名的 B 数（⑮）", CALC),
        ("2. 文件总大小对得上吗？", "参数量 × 每参数字节（fp16=2、q4≈0.5）（⑯⑳）", CALC),
        ("3. 词表对得上吗？", "vocab_size = 词表条目数 = embed_tokens 第 0 维（⑲）", ACT),
        ("4. 头数对得上吗？", "num_heads × head_dim = hidden_size；GQA 时 kv 头更少（⑩⑮）", ACT),
        ("5. 模板对得上吗？", "chat_template 里有没有 assistant 前缀；BOS/EOS 是否重复（⑲）", ERR),
    ]
    cy = y3
    for t, body, role in checks:
        p.box(36, cy, 460, 52, f"<b>{t}</b>", "box", role)
        p.box(508, cy, 1156, 52,
              f"<font style=\"font-size:10px\">{body}</font>", "box", NOTE)
        cy += 60

    # --- 索引 -------------------------------------------------------------
    y4 = cy + 30
    p.text(36, y4 - 60, 1200, 22,
           "<b>第四步：21 页索引（想复习哪一块就翻回去）</b>", "text_b")
    idx = [
        ["A · Transformer 怎么算", "① 总览　② 分词　③ 词嵌入　④ 位置编码 RoPE　⑤ 矩阵乘（线性层）"
                                "　⑥ QKV 投影　⑦ 打分 Q·Kᵀ　⑧ softmax　⑨ 加权求和 P·V"
                                "　⑩ 多头　⑪ 残差+RMSNorm　⑫ FFN/SwiGLU　⑬ 堆叠+输出头+采样"],
        ["B · 模型文件结构", "⑭ 目录全景　⑮ config.json　⑯ safetensors 字节布局　⑰ 分片与 index"
                          "　⑱ 权重名→形状　⑲ tokenizer 与模板　⑳ GGUF 与量化　㉑ 本页"],
        ["三个「一以贯之」的数字", "本套全程用 5 个 token × 4 维（⑩ 换成 6 维）演示；"
                              "换成真实模型只需把 4 换成 4096、5 换成你的 token 数——"
                              "<b>形状规则完全不变</b>"],
        ["数字类型的规律", "整数：token id、层号、形状、块号｜ 小数：权重、激活、分数、概率"],
    ]
    for t, body in idx:
        p.box(36, y4, 1628, 66,
              f"<b>{t}</b><br><font style=\"font-size:10px\">{body}</font>",
              "box", NOTE)
        y4 += 76

    p.box(36, y4 + 6, 1628, 74,
          "<b>最后一句</b>：Transformer 没有魔法——它只是「查表、乘矩阵、按权重求和、"
          "再乘矩阵」重复几十次；模型文件也没有魔法——它只是「一份 JSON 说明书 + "
          "一大堆按名字排好的浮点数」。把这两句话落实到形状和字节上，你就已经超过"
          "大多数「会用但说不清」的人了。",
          "box", NOTE)

    legend(p, y4 + 108)
