"""② 分词：一段文本怎么变成一串整数 id。"""
from mth import (ACT, CALC, ERR, NOTE, OUT, W, arrow, callout, legend, strip,
                 title)

TITLE = "② 分词（tokenizer）：文本 → 整数 id"
SUB = ("神经网络只会算数，不会读字。所以第一步是把文本切成「token」并查表换成整数。"
       "这一页只回答三件事：切成什么、查到什么、特殊 token 是什么。")


def build(doc) -> None:
    p = doc.page("② 分词", pw=1700, ph=1200)
    title(p, TITLE, SUB, w=1620)

    # --- 切分 -------------------------------------------------------------
    y = 150
    p.text(36, y - 34, 900, 22, "<b>第一步：把文本切成 token（不一定是「字」也不一定是「词」）</b>",
           "text_b")
    p.box(36, y, 340, 96,
          "<b>原始文本</b><br>"
          "<font style=\"font-size:13px\">「我爱吃苹果」</font><br>"
          "<font style=\"font-size:9px\" color=\"#666666\">5 个汉字</font>",
          "box", NOTE)
    p.box(406, y, 400, 96,
          "<b>切成 token</b><br>"
          "<font style=\"font-size:13px\">我 | 爱 | 吃 | 苹果</font><br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "4 个 token：常见词「苹果」占一个 token</font>",
          "box", CALC)
    p.box(836, y, 420, 96,
          "<b>查词表换成 id</b><br>"
          "<font style=\"font-size:13px\">12841 | 3027 | 882 | 6132</font><br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "每个 id 是「词表里的第几行」</font>",
          "box", ACT)
    p.box(1286, y, 378, 96,
          "<b>切开方式由算法定</b><br>"
          "BPE（byte-pair encoding）：从单字节开始，"
          "反复把<b>最常一起出现</b>的两段合并<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "所以 token 粒度是「统计出来的」</font>",
          "box", NOTE)
    arrow(p, 376, y + 48, 406, y + 48)
    arrow(p, 806, y + 48, 836, y + 48)
    arrow(p, 1256, y + 48, 1286, y + 48)

    # --- 一串 id ---------------------------------------------------------
    y2 = y + 150
    p.text(36, y2 - 34, 1100, 22,
           "<b>第二步：这串 id 就是模型的输入（注意：它是整数，不是浮点）</b>",
           "text_b")
    sx, sy, _ = strip(p, 36, y2, [12841, 3027, 882, 6132], ACT,
                      name="input_ids（长度 5）", cw=110)
    p.box(36, y2 + 50, 640, 116,
          "<b>形状：5（没有第二维）</b><br>"
          "严格说它是一个长度 5 的整数数组；"
          "在框架里常写成 5×1 或 1×5 只是为了凑成矩阵。<br><br>"
          "<font style=\"font-size:9px;color:#c0392b\">"
          "这条很重要：<b>整数 id 不能直接做矩阵乘</b>，必须先变成浮点向量（③ 词嵌入）。"
          "</font>",
          "box", ERR)
    p.box(716, y2 + 50, 948, 116,
          "<b>真实规模</b>：Llama-2 词表 32000；Qwen2.5 词表 152064（含大量中文字与多字节）。"
          "一句中文通常切成 1~2 字/token，英文常见词约 1 词/token，代码里的空格和符号各占 token。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "词表大小 = ⑮ config.json 的 vocab_size；词表本身 = ⑲ tokenizer 三件套里的 vocab 文件"
          "</font>",
          "box", NOTE)

    # --- 特殊 token ------------------------------------------------------
    y3 = y2 + 200
    p.text(36, y3 - 34, 1100, 22,
           "<b>第三步：除了「字」，词表里还住着几个特殊 token</b>", "text_b")
    sp = [
        ("<s> / <|bos|>", "BOS：句子开头", "告诉模型「从这里开始算」"),
        ("</s> / <|eos|>", "EOS：句子结束", "模型生成到它就停，这是它学会「说完」的方式"),
        ("<|pad|>", "PAD：填充", "批量推理时把短句补齐到同一长度（补的部分不参与计算）"),
        ("<|im_start|>", "角色分隔（chat 模板）", "区分 system / user / assistant 三种说话人"),
    ]
    cy = y3
    for tok, name, why in sp:
        p.box(36, cy, 300, 54, f"<b>{tok}</b>", "box_center", ACT)
        p.box(350, cy, 420, 54,
              f"<b>{name}</b><br><font style=\"font-size:9px\">{why}</font>",
              "box", NOTE)
        cy += 62

    # 第三列内容改成一句话说明，避免空白
    p.box(784, y3, 880, 4 * 62 - 8,
          "<b>为什么特殊 token 重要</b><br>"
          "① <b>BOS/EOS</b> 决定「什么时候开始/停止生成」；没学好的模型会一直说下去。<br>"
          "② <b>PAD</b> 让「多条请求一起算」成为可能（批量推理的前提）。<br>"
          "③ <b>角色分隔符</b>是现代 chat 模型区分说话人的唯一手段——"
          "同一个模型，模板不同，回答风格就完全不同。<br><br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "这些 token 的 id 固定写在 tokenizer_config.json / special_tokens_map.json 里（⑲）"
          "</font>",
          "box", NOTE)

    # --- 代码锚点 --------------------------------------------------------
    y4 = cy + 40
    p.box(36, y4, 1628, 92,
          "<b>代码锚点（在 vllm 仓里能直接翻到）</b><br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "词嵌入层本体：vllm/model_executor/layers/vocab_parallel_embedding.py:205"
          "（class VocabParallelEmbedding）——它的 forward 就是「按 id 取行」<br>"
          "词表与 tokenizer 的加载：vllm/transformers_utils/tokenizer.py"
          "（get_tokenizer / 特殊 token 处理）<br>"
          "把 id 变成向量：vllm/model_executor/models/llama.py:382"
          "（self.embed_tokens = VocabParallelEmbedding(...)）"
          "</font>", "box", NOTE)

    legend(p, y4 + 130)
