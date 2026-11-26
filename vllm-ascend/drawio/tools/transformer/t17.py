"""⑰ 分片与 index.json：一个 14 GB 的文件怎么被切成 4 份。"""
from dg import table
from mth import (ACT, CALC, ERR, NOTE, OUT, W, arrow, legend, panel, title)

TITLE = "⑰ 分片与 index.json：14 GB 的权重怎么拆成 4 个文件"
SUB = ("单个 safetensors 文件太大会撑爆内存、也无法并行下载，所以按张量切成几片，"
       "再用一份 index.json 记录「谁在哪」。这一页讲清它和「张量并行切分」的区别。")


def build(doc) -> None:
    p = doc.page("⑰ 分片与索引", pw=1700, ph=1360)
    title(p, TITLE, SUB, w=1620)

    # --- 分片 -------------------------------------------------------------
    y = 150
    p.text(36, y - 60, 1200, 22,
           "<b>第一步：切成 4 片（按张量切，不切开任何一个张量）</b>", "text_b")
    for i in range(4):
        p.box(36 + i * 300, y, 280, 84,
              f"<b>model-0000{i+1}-of-00004.safetensors</b><br>"
              f"<font style=\"font-size:9px\" color=\"#666666\">≈ 3.4 GB</font>",
              "box_center", W)
    p.box(1236, y, 428, 84,
          "<b>每条分片自己是完整的 safetensors</b><br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "各自有 8 字节长度 + JSON 头 + 数据区（⑯）</font>",
          "box", NOTE)
    panel(p, 36, y + 106, 810, "切分规则：按张量累加，到阈值就换新文件",
          "保存时按顺序累加每个张量的字节数，超过 max_shard_size（HF 默认 5 GB）就开新文件。"
          "<b>不会把一个张量拆到两个文件里</b>。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "所以 4 片的大小不一定完全相等（3.4/3.4/3.4/3.3 之类）。</font>",
          CALC)
    panel(p, 866, y + 106, 798, "为什么不用一个 14 GB 的文件",
          "① 下载/校验可以并行、断点续传；② 加载时可以只读需要的张量；"
          "③ 单个文件过大在某些文件系统/传输协议上有上限；④ 便宜：多进程能同时读不同分片。",
          NOTE)

    # --- index.json -------------------------------------------------------
    y2 = y + 300
    p.text(36, y2 - 60, 1200, 22,
           "<b>第二步：index.json 就是一张「张量名 → 文件」的字典</b>", "text_b")
    p.box(36, y2, 860, 236,
          "<font style=\"font-size:9px\" face=\"Courier New\" color=\"#333333\">"
          "{{\"metadata\": {{\"total_size\": 13476877312}},<br>"
          "&nbsp;\"weight_map\": {{<br>"
          "&nbsp;&nbsp;&nbsp;\"model.embed_tokens.weight\": \"model-00001-of-00004.safetensors\",<br>"
          "&nbsp;\"model.layers.0.self_attn.q_proj.weight\": \"model-00001-of-00004.safetensors\",<br>"
          "&nbsp;&nbsp;&nbsp;...<br>"
          "&nbsp;&nbsp;&nbsp;\"model.layers.31.mlp.down_proj.weight\": \"model-00004-of-00004.safetensors\",<br>"
          "&nbsp;&nbsp;&nbsp;\"lm_head.weight\": \"model-00004-of-00004.safetensors\"<br>"
          "&nbsp;}}}}</font>",
          "box", NOTE)
    panel(p, 926, y2, 738, "加载器怎么用它",
          "① 读 index.json 拿到 weight_map；② 按「当前层需要哪些张量」把请求分组到文件；"
          "③ 对每个文件只 seek 需要的偏移（⑯）。<br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "所以加载速度取决于「同时打开几个文件」和「是否顺序读」，"
          "而不是文件个数本身。</font>",
          CALC)
    panel(p, 926, y2 + 116, 738, "常见误解：分片 ≠ 张量并行切分",
          "分片（shard）只是<b>物理存储</b>的切法，模型还是一份；"
          "张量并行（TP）是把同一个矩阵<b>按维度切开分发到多张卡</b>，每张卡只持有一部分。<br>"
          "<font style=\"font-size:9px;color:#c0392b\">"
          "所以「4 个分片」和「4 张卡」毫无关系——一个 4 分片的模型可以在 1 张卡上跑，"
          "也可以在 8 张卡上用 TP=8 跑（这时每张卡只加载自己那份切片）。</font>",
          ERR)

    # --- 两种切法对照 -----------------------------------------------------
    y3 = y2 + 290
    p.text(36, y3 - 60, 1200, 22,
           "<b>第三步：两种「切」的本质区别（一张表记住）</b>", "text_b")
    rows = [
        ["切什么", "张量之间的边界（整块搬）", "张量内部的某一维（按行/列切）"],
        ["谁决定", "保存模型的人（max_shard_size）", "推理/训练时的 TP 配置（tp_size=4）"],
        ["影响计算吗", "不影响，模型完全一样", "影响：每卡只算一部分，最后通信汇总"],
        ["权重文件里能看到吗", "能看到：多个 .safetensors 文件 + index.json",
         "看不到：文件还是完整的，加载时才切"],
        ["典型数字", "7B fp16 → 4 片 × 3.4 GB", "7B 在 4 卡 TP=4 → 每卡约 3.4 GB"],
    ]
    table(p, 36, y3, [("维度", 220), ("分片 shard（存储层）", 620),
                      ("张量并行 TP（计算层）", 700)], rows, row_h=42)

    # --- 代码锚点 --------------------------------------------------------
    y4 = y3 + 6 * 42 + 40
    p.box(36, y4, 1628, 96,
          "<b>代码锚点</b><br>"
          "<font style=\"font-size:9px\" color=\"#666666\">"
          "按 index.json 分文件迭代权重：vllm/model_executor/model_loader/weight_utils.py:866"
          "（safetensors_weights_iterator，内部用 safe_open 逐张量取）<br>"
          "分片感知的上层：vllm/model_executor/model_loader/default_loader.py"
          "（决定读哪些文件、什么顺序、是否多进程）<br>"
          "TP 的切分发生在层内：vllm/model_executor/layers/linear.py:430"
          "（ColumnParallelLinear，按输出维切）/ :1623（RowParallelLinear，按输入维切后 all-reduce）"
          "</font>", "box", NOTE)

    legend(p, y4 + 132)
