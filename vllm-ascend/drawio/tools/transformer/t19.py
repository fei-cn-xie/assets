"""⑲ tokenizer 三件套 + generation_config + chat_template：文本侧的全部配置。"""
from dg import table
from mth import (ACT, CALC, ERR, NOTE, OUT, W, arrow, legend, panel, title)

TITLE = "⑲ tokenizer 与模板：文本侧的三个文件 + 一个模板"
SUB = ("权重决定「模型懂什么」，这几个文件决定「文本怎么进去、怎么出来」。"
       "它们加起来不到 10 MB，却是最常见的踩坑来源。")


def build(doc) -> None:
    p = doc.page("⑲ tokenizer 与模板", pw=1700, ph=1420)
    title(p, TITLE, SUB, w=1620)

    # --- 三个文件 ---------------------------------------------------------
    y = 150
    p.text(36, y - 60, 1200, 22,
           "<b>第一步：三个文件各管一件事（缺一个就跑不起来）</b>", "text_b")
    panel(p, 36, y, 520, "① tokenizer.json（或 vocab.json + merges.txt）",
          "<b>规则本体</b>：词表（每行一个 token 片段）+ BPE 合并优先级表。<br>"
          "新版 HF 把它打包成一个 tokenizer.json；"
          "老模型（GPT-2 系）还是两个文本文件。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "它就是 ② 里「切成 token 再查 id」的那张表。</font>", ACT)
    panel(p, 576, y, 520, "② tokenizer_config.json",
          "<b>行为配置</b>：用什么分词器类、大小写、特殊 token 的名字、"
          "要不要加 BOS、以及 <b>chat_template</b>（可内嵌也可单独 .jinja）。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "这一份最容易出事：字段名改了，tokenizer 就会把文本切错。</font>", CALC)
    panel(p, 1116, y, 548, "③ special_tokens_map.json",
          "<b>名字 ↔ 特殊 token 字符串</b>：bos_token / eos_token / pad_token / "
          "unk_token 各是哪个字符串。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "它们的 <b>id</b> 不在这个文件里，而在 tokenizer 本体里（查表得到）。</font>",
          NOTE)

    # --- 一个请求走一遍 --------------------------------------------------
    y2 = y + 160
    p.text(36, y2 - 60, 1200, 22,
           "<b>第二步：一句话从字符串到 id 的完整过程（含特殊 token）</b>",
           "text_b")
    steps = [
        ("原始文本", "「你好，帮我写首诗」", NOTE),
        ("套模板（chat_template）",
         "&lt;|im_start|&gt;user⏎你好，帮我写首诗&lt;|im_end|&gt;⏎"
         "&lt;|im_start|&gt;assistant⏎", CALC),
        ("切 token（BPE）", "&lt;|im_start|&gt; ｜ user ｜ 你好 ｜ ， ｜ 帮我 ｜ 写 ｜ 首诗 ｜ …",
         ACT),
        ("查词表换 id", "[151644, 872, 108386, 3922, 101058, ...]", OUT),
        ("加 BOS/EOS（视配置）", "开头加 bos_token_id，结尾留给模型生成", NOTE),
    ]
    cy = y2
    for name, body, role in steps:
        p.box(36, cy, 330, 56, f"<b>{name}</b>", "box", role)
        p.box(378, cy, 1286, 56,
              f"<font style=\"font-size:10px\">{body}</font>", "box", NOTE)
        cy += 64

    # --- 模板为什么关键 ---------------------------------------------------
    y3 = cy + 30
    panel(p, 36, y3, 810, "chat_template 为什么能改变「模型性格」",
          "同一个模型权重，用不同模板拼出来的字符串不一样 → token 序列不一样 → "
          "输出风格明显不同。<br>"
          "比如少写 <b>&lt;|im_start|&gt;assistant</b>，模型就不知道「该轮到我说话了」，"
          "可能继续扮演用户。<br>"
          "<font style=\"font-size:9px;color:#c0392b\">"
          "这就是「同一个模型在 A 框架好用、在 B 框架胡说」的最常见原因。</font>",
          ERR)
    panel(p, 866, y3, 798, "generation_config.json 在这里的角色",
          "它不参与分词，而是给「生成」提供默认参数：temperature、top_p、top_k、"
          "max_new_tokens、eos_token_id、repetition_penalty。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "⑬ 里那些默认值就来自它；命令行/接口传参会覆盖它。</font>", CALC)

    # --- 形状与 id 的关系 -------------------------------------------------
    y4 = y3 + 190
    p.text(36, y4 - 60, 1200, 22,
           "<b>第三步：和 config.json 的唯一约定——词表大小必须对得上</b>",
           "text_b")
    rows = [
        ["config.json: vocab_size", "32000", "网络输出头的宽度（⑬）"],
        ["tokenizer 词表实际条目数", "32000", "必须 ≤ vocab_size"],
        ["embed_tokens.weight 第 0 维", "32000", "必须 = vocab_size（③⑱）"],
        ["lm_head.weight 第 0 维", "32000", "必须 = vocab_size（若不共享）"],
    ]
    table(p, 36, y4, [("项", 420), ("值", 180), ("说明", 620)], rows, row_h=34)
    panel(p, 36, y4 + 6 * 34 + 30, 1628, "对不上会怎样",
          "词表比 vocab_size 大 → 分词后出现越界 id（运行时崩）；"
          "比 vocab_size 小 → 有一批 id 永远用不到（浪费显存，但不崩）。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "换 tokenizer（比如给模型加新词）时，必须同时改权重里的 embed_tokens / lm_head，"
          "这就是「词表扩充」要做的事。</font>", ERR)

    # --- 代码锚点 --------------------------------------------------------
    y5 = y4 + 6 * 34 + 160
    p.box(36, y5, 1628, 100,
          "<b>代码锚点</b><br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "tokenizer 的加载与特殊 token 处理：vllm/transformers_utils/tokenizer.py"
          "（get_tokenizer / get_cached_tokenizer）<br>"
          "chat template 的应用：同目录 tokenizer.py 里的 apply_chat_template 相关逻辑，"
          "以及 vllm/entrypoints/openai/ 下的对话接口<br>"
          "GGUF 模型的 tokenizer 特殊处理：vllm/transformers_utils/gguf_utils.py"
          "（GGUF 把分词器也塞在同一个文件里，需要另走一条路，见 ⑳）"
          "</font>", "box", NOTE)

    legend(p, y5 + 132)
