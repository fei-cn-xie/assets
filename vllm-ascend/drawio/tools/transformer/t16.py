"""⑯ safetensors 文件格式：字节层面长什么样（为什么它比 .bin 安全）。"""
from dg import table
from mth import (ACT, CALC, ERR, NOTE, OUT, W, arrow, legend, panel, title)

TITLE = "⑯ safetensors 文件格式：权重是怎么一个字节一个字节存起来的"
SUB = ("safetensors 只有两部分：一段 JSON 目录 + 一大块裸数据。"
       "没有代码、没有 pickle，所以加载快且安全——这一页把它拆到你敢自己解析。")


def build(doc) -> None:
    p = doc.page("⑯ safetensors 格式", pw=1700, ph=1400)
    title(p, TITLE, SUB, w=1620)

    # --- 三段落布局 -------------------------------------------------------
    y = 150
    p.text(36, y - 60, 1200, 22,
           "<b>第一步：整个文件只有三段（从左到右就是字节顺序）</b>", "text_b")
    a = p.box(36, y, 200, 90,
              "<b>① 8 字节</b><br>"
              "头部长度 N<br>"
              "<font style=\"font-size:9px\" color=\"#666666\">小端 uint64</font>",
              "box_center", CALC)
    b = p.box(252, y, 700, 90,
              "<b>② N 字节：JSON 头部（目录）</b><br>"
              "每个张量的名字、dtype、形状、<b>数据在文件里的起止偏移</b><br>"
              "<font style=\"font-size:9px\" color=\"#666666\">"
              "纯文本 JSON，按 8 字节对齐补空格</font>",
              "box_center", NOTE)
    c = p.box(968, y, 696, 90,
              "<b>③ 剩下的全部：裸数据区</b><br>"
              "几十亿个 fp16 紧挨着排，没有任何额外结构<br>"
              "<font style=\"font-size:9px\" color=\"#666666\">"
              "张量的偏移都是相对这一段的开头</font>",
              "box_center", W)
    p.edge(src_pt=(236, y + 45), dst_pt=(252, y + 45), st="arrow", role="http")
    p.edge(src_pt=(952, y + 45), dst_pt=(968, y + 45), st="arrow", role="http")

    # --- JSON 示例 --------------------------------------------------------
    y2 = y + 120
    p.text(36, y2 - 60, 1200, 22,
           "<b>第二步：那个 JSON 头部到底写了什么（真实结构，缩短过）</b>",
           "text_b")
    p.box(36, y2, 900, 250,
          "<font style=\"font-size:9px\" face=\"Courier New\" color=\"#333333\">"
          "{{<br>"
          "&nbsp;&nbsp;\"model.layers.0.self_attn.q_proj.weight\": {{<br>"
          "&nbsp;&nbsp;&nbsp;&nbsp;\"dtype\": \"F16\",<br>"
          "&nbsp;&nbsp;&nbsp;&nbsp;\"shape\": [4096, 4096],<br>"
          "&nbsp;&nbsp;&nbsp;&nbsp;\"data_offsets\": [0, 33554432]<br>"
          "&nbsp;&nbsp;}},<br>"
          "&nbsp;&nbsp;\"model.layers.0.self_attn.o_proj.weight\": {{<br>"
          "&nbsp;&nbsp;&nbsp;&nbsp;\"dtype\": \"F16\",<br>"
          "&nbsp;&nbsp;&nbsp;&nbsp;\"shape\": [4096, 4096],<br>"
          "&nbsp;&nbsp;&nbsp;&nbsp;\"data_offsets\": [33554432, 67108864]<br>"
          "&nbsp;&nbsp;}},<br>"
          "&nbsp;&nbsp;\"__metadata__\": {{\"format\": \"pt\"}}<br>"
          "}}</font>",
          "box", NOTE)
    panel(p, 966, y2, 698, "怎么从这一段算出「读多少字节」",
          "4096×4096 个 fp16 = 16,777,216 × 2 字节 = <b>33,554,432</b> 字节，"
          "正好是 offsets 的差值（33554432−0）。<br>"
          "所以加载器只需要：读 JSON → 拿到 (offset, shape, dtype) → "
          "<b>seek 到 8+N+offset，读 length 个字节</b>，直接 memcpy 到显存。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "这就是为什么 safetensors 加载快：没有解析、没有反序列化，纯内存拷贝。</font>",
          CALC)
    panel(p, 966, y2 + 170, 698, "为什么它不是「一整块连续」",
          "张量的顺序由保存时的遍历顺序决定，加载器<b>不应该假设</b>任何顺序——"
          "一切以 JSON 里的 offsets 为准。<br>"
          "<font style=\"font-size:9px;color:#c0392b\">"
          "这也解释了为什么「把两个 safetensors 直接拼起来」不叫合并："
          "你必须重算所有 offsets（⑰ 的分片就是这么产生的）。</font>", ERR)

    # --- 与 .bin 的对比 ---------------------------------------------------
    y3 = y2 + 290
    p.text(36, y3 - 60, 1200, 22,
           "<b>第三步：和老的 PyTorch .bin（pickle）比，好在哪</b>", "text_b")
    rows = [
        ["加载速度", "需要 Python 反序列化，慢", "纯内存拷贝，快一个数量级"],
        ["安全性", "pickle 能执行代码，恶意模型可以拿到你的 shell",
         "只有数据，没有可执行内容"],
        ["内存峰值", "可能先整份读进内存再转", "可以 mmap / 流式按张量读"],
        ["随机访问", "要先整体反序列化", "按 offsets 直接 seek，只读需要的张量"],
        ["并行加载", "难", "按张量分给多个线程/进程（vLLM 默认就这么干）"],
    ]
    table(p, 36, y3, [("维度", 200), ("PyTorch .bin / pickle", 420),
                      ("safetensors", 560)], rows, row_h=34)

    # --- 代码锚点 --------------------------------------------------------
    y4 = y3 + 6 * 34 + 40
    p.box(36, y4, 1628, 100,
          "<b>代码锚点</b><br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "vLLM 里读 safetensors 的地方：vllm/model_executor/model_loader/weight_utils.py:26"
          "（from safetensors.torch import safe_open / load_file）→ :866（safetensors_weights_iterator，"
          "它用 safe_open 打开文件、按张量名惰性读取）<br>"
          "多线程/多进程预取：同文件 :1012（multi_thread_safetensors_weights_iterator）、"
          ":33-34（DEFAULT_SAFETENSORS_PREFETCH_* 两个常量）<br>"
          "上层装配：vllm/model_executor/model_loader/default_loader.py"
          "（拿到迭代器后逐个 load_weights）"
          "</font>", "box", NOTE)

    legend(p, y4 + 132)
