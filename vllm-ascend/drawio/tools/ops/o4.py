"""④ 小而碎的一族：归一化 / 激活 / 位置编码——最容易被融合，也最容易回退。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc, table

TITLE = "④ 归一化、激活、位置编码：三个「每层都要跑好几遍」的小算子"
SUB = ("一句话：它们单次计算量小，但**每层要跑好几次**，所以是融合收益最高的一族；"
       "而它们的实现按「模型族」分叉——同一件事（例如位置编码）在不同模型上落到了不同算子。")


def build(doc: Doc) -> None:
    p = doc.page("④ 小算子家族", pw=1560, ph=1580)
    head(p, TITLE, SUB)
    X = 36

    # --- 归一化：三层融合 ------------------------------------------------
    y = 146
    p.text(X, y - 30, 1300, 22,
           "<b>第一步：归一化是「三层融合」的典型（同一件事有三档实现）</b>",
           "text_b")
    tiers = [
        ("第三档：图级再融合", "在编译 pass 里把「相加+归一化+量化」合成一个算子",
         "最快，但只在特定 dtype/形状下启用", "cloudd"),
        ("第二档：厂商融合算子", "「相加 + 归一化 + bias」一次完成", "通用性好", "cloudp"),
        ("第一档：基础归一化", "只有归一化本身", "最保险，最慢", "neutral"),
    ]
    for i, (name, body, note, role) in enumerate(tiers):
        cy = y + i * 84
        p.box(X, cy, 1424, 68, "", "box", role)
        p.text(X + 12, cy + 6, 260, 24, f"<b>{name}</b>", "text_b")
        p.text(X + 280, cy + 8, 1140, 56,
               f"<font style=\"font-size:10px\">{body}　"
               f"<font color=\"#666666\">{note}</font></font>", "text")

    # --- 激活 -------------------------------------------------------------
    y2 = y + 3 * 84 + 12
    p.text(X, y2 - 30, 1300, 22,
           "<b>第二步：激活函数的「变体爆炸」（看起来都叫 SwiGLU，算子却不止一个）</b>",
           "text_b")
    rows = [
        ["普通 SwiGLU", "门控激活（SiLU × 另一路）", "dense MLP 与 MoE 都用"],
        ["带 clamp 的 SwiGLU", "激活前先截断，防溢出", "某些模型/量化方案需要"],
        ["激活 + 反量化 + 再量化", "激活完直接产出量化张量，省一次往返", "量化 MLP 主用"],
        ["分组激活 + 量化", "MoE 里每组专家一段", "MoE 量化路径"],
        ["GELU 变体", "另一种激活（BERT 系）", "少量模型"],
        ["门控 + 其它（SiTU 等）", "不同模型族的门控写法", "按模型选"],
    ]
    table(p, X, y2, [("算子形态", 460), ("它在算什么", 560), ("用在哪", 404)], rows,
          row_h=32)

    # --- 位置编码：按模型分叉 ---------------------------------------------
    y3 = y2 + 32 * 7 + 30
    p.text(X, y3 - 30, 1300, 22,
           "<b>第三步：位置编码最「按模型分叉」——四个模型族四种落点</b>", "text_b")
    p.box(X, y3, 1424, 150,
          "<font style=\"font-size:10px\">"
          "① <b>普通 GQA 模型</b>：一个通用的旋转位置编码算子（内部已处理「全旋」与「部分旋」），"
          "prefill 与 decode <b>同一个算子</b>。<br>"
          "② <b>MLA 类（DeepSeek 系）</b>：prefill 与 decode <b>不是一个</b>——"
          "decode 走「归一化 + 旋转 + 写 KV 缓存」<b>三合一的融合算子</b>，prefill 走交错旋转。<br>"
          "③ <b>稀疏 MLA（带 indexer）</b>：用「原地部分旋转」，只转一部分维度，省一次数据拷贝。<br>"
          "④ <b>老硬件（310P）</b>：走一个只支持「头维 64 或 128」的算子，"
          "所以老平台上换模型容易踩限制。<br>"
          "<font color=\"#c0392b\">读法：想知道某个模型的位置编码落到哪个算子，"
          "先看它是 GQA 还是 MLA、有没有稀疏/索引器，再看硬件代次。</font></font>",
          "box", "cloudd")

    # --- 一个隐藏前提 -----------------------------------------------------
    y4 = y3 + 190
    p.box(X, y4, 1424, 116,
          "<b>第四步：一个容易忽略的隐藏前提——「cos/sin 缓存」要预处理</b><br>"
          "<font style=\"font-size:10px\">"
          "框架里位置编码的表是一整块「cos 与 sin 交错」的缓存，"
          "但算子通常要<b>两份分离的 cos、sin</b>，且要求它们<b>不参与梯度追踪</b>。<br>"
          "所以本仓在加载后会把这块缓存<b>切分并重新登记</b>（拆成交错的 cos/sin），"
          "还特意把它注册成「非持久缓冲」——原因是唤醒/重建后指针会失效。<br>"
          "<font color=\"#c0392b\">这就是为什么「换一个位置编码实现」不是改一行的事："
          "缓存的布局、生命周期、地址稳定性都要跟着改。</font></font>",
          "box", "fail")

    y5 = y4 + 156
    bridge(p, y5, "本页交接：小算子家族讲完。下一步（⑤）看一个反直觉的取舍——"
                  "采样那一步<b>主动不用</b>硬件提供的融合算子。")

    y6 = y5 + 76
    anchors(p, y6, [
        "归一化三档：ops/layernorm.py:92（自研「相加+归一化+bias」）、:97（基础归一化）、"
        ":118（回退到厂商「相加+归一化」）；图级融合 compilation/passes/norm_quant_fusion_pass.py:385-393",
        "归一化 + 动态量化融合：attention/dsa_v1.py:1956（自研融合）；"
        "图级 norm_quant_fusion_pass.py:328",
        "激活变体：ops/activation.py:37（SwiGLU）、:47（clamp 版）、:31（GELU 变体）；"
        "激活+反量化+量化 w8a8/w8a8_dynamic.py:474；分组激活+量化 shared_experts.py:607、:651",
        "位置编码：ops/rotary_embedding.py:244（通用旋转，内部处理全旋/部分旋）、:736（旋转乘）；"
        "MLA decode 三合一 attention/mla_v1.py:1865、prefill attention/mla_v1.py:2800；"
        "原地部分旋转 attention/dsa_v1.py:1915；老硬件 _310p/ops/rotary_embedding.py:146、:223"
        "（只支持 rotary 64/128）",
        "cos/sin 缓存预处理：ops/rotary_embedding.py:47-64（注释为什么要预处理）、"
        ":141-171（切分与「非持久缓冲」的原因）、:67-91（预分配固定地址缓冲）",
        "QKNorm + RoPE 图级融合（head_dim 必须 128）：compilation/passes/qknorm_rope_fusion_pass.py:67-96、"
        ":315",
    ], h=160)

    y7 = y6 + 194
    concl(p, y7,
          "<b>一句话记住这一页：</b>归一化/激活/位置编码单次很便宜但每层跑好几遍，"
          "所以融合收益最高；<b>而它们全都按「模型族 + 硬件代次」分叉</b>——"
          "想知道某个模型落到哪个算子，先看模型类型（GQA/MLA/稀疏）再看硬件。")

    legend(p, y7 + 92)
