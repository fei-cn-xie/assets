"""⑮ 模型加载：权重从磁盘文件一步步变成 NPU 上的张量。"""
from common import head, concl, anchors, legend
from dg import Doc, table

TITLE = "⑮ 模型加载：从权重文件到 NPU 显存（8 步）"
SUB = ("一句话：<b>先读 config 知道模型长什么样 → 在 NPU 上搭一个空骨架 → "
       "逐个张量从磁盘读出来、按 TP 切好、拷进对应的参数里 → 再做格式/量化后处理</b>。"
       "加载完才轮到 profile 显存、切 KV 块池。")
X = 36


def step(p, x, y, w, h, no, title, body, role):
    p.box(x, y, w, h, "", "box", role)
    p.text(x + 10, y + 8, w - 20, 22, f"<b>{no}　{title}</b>", "text_b")
    p.text(x + 10, y + 32, w - 20, h - 40,
           f"<font style=\"font-size:9px\" color=\"#333333\">{body}</font>", "text")


def build(doc: Doc) -> None:
    p = doc.page("⑮ 模型加载", pw=1560, ph=1560)
    head(p, TITLE, SUB)

    # --- 第一排：磁盘侧 ---------------------------------------------------
    y = 150
    p.text(X, y - 32, 1100, 22,
           "<b>第一排：先看清磁盘上有什么（不读权重也能知道很多）</b>", "text_b")
    step(p, X, y, 466, 132, "①", "config.json：模型的结构与 dtype",
         "层数、hidden、head 数、dtype、是否量化都在这里。<br>"
         "读不到会直接报 Invalid repository ID or local directory。", "neutral")
    step(p, X + 490, y, 466, 132, "②", "权重分片：*.safetensors",
         "大模型切成多个文件（shard）。每片里是一堆<b>名字 → 张量</b>。<br>"
         "读盘用 <font style=\"font-size:9px\">safe_open</font>，拿到的是 <b>CPU 张量</b>。",
         "cloudd")
    step(p, X + 980, y, 444, 132, "③", "index.json：名字 → 在哪个分片",
         "有它在才能只读该读的片；<br>"
         "索引点名的文件缺失会报 Weight files referenced in index but missing。",
         "data")

    # --- 第二排：三步进 NPU ------------------------------------------------
    y2 = y + 172
    p.text(X, y2 - 32, 1100, 22,
           "<b>第二排：进 NPU 的三步（关键：参数是「先分配、后填充」）</b>",
           "text_b")
    step(p, X, y2, 466, 156, "④", "在 NPU 上搭空骨架",
         "在 device 上下文里建模型 → 每个 <font style=\"font-size:9px\">nn.Parameter</font> "
         "<b>一开始就分配在 NPU 上</b>（还没填数）。<br>"
         "所以后面不是「把权重搬到 NPU」，而是「往 NPU 上已有的参数里写数」。",
         "cloudp")
    step(p, X + 490, y2, 466, 156, "⑤", "名字映射 + 选 weight_loader",
         "checkpoint 的名字（q_proj/k_proj/v_proj）→ 拼成模型的 qkv_proj；<br>"
         "每个参数自己带一个 weight_loader，决定「怎么切、怎么拷」。",
         "cloudp")
    step(p, X + 980, y2, 444, 156, "⑥", "切片 + 拷进去（每张量一次）",
         "TP&gt;1 时：从全量张量里 <b>narrow</b> 出本 rank 的那一段，"
         "然后 <font style=\"font-size:9px\">param.data.copy_(切片)</font> —— "
         "这一步才是真正的 <b>CPU→NPU 拷贝</b>。", "cloudp")

    # --- 第三排：后处理 + 显存顺序 -----------------------------------------
    y3 = y2 + 196
    p.text(X, y3 - 32, 1100, 22,
           "<b>第三排：加载完还要「改写格式」，然后才轮到 KV cache</b>", "text_b")
    step(p, X, y3, 700, 168, "⑦", "后处理：量化 / 布局转换",
         "process_weights_after_loading：<br>"
         "· 把 fp32 参数换成 fp16/bf16（如需要）<br>"
         "· 昇腾上做 <b>ND → NZ</b> 布局重排（<font style=\"font-size:9px\">"
         "npu_format_cast(..., ACL_FORMAT_FRACTAL_NZ)</font>），让矩阵乘更快<br>"
         "· 权重绑定 / 融合<br>"
         "⚠ 这一步是「就地换掉 weight.data」，不是再拷一次。", "fail")
    step(p, X + 724, y3, 700, 168, "⑧", "此时才 profile 显存 → 切 KV 块池",
         "顺序是硬性的：<br>"
         "权重先吃掉显存 → 跑一次 profile 量出激活峰值 → "
         "<b>剩余显存才切成 KV 块</b>。<br>"
         "所以「权重变大 = KV 块变少 = 并发变低」，三者是一条链。", "cloudd")

    # --- 日志与失败 -------------------------------------------------------
    y4 = y3 + 208
    p.text(X, y4 - 32, 1000, 22,
           "<b>加载期间的日志顺序（对着日志能判断卡在哪一步）</b>", "text_b")
    rows = [
        ["Starting to load model /path/to/weights...", "开始加载（worker 里）"],
        ["Loading safetensors checkpoint shards: 100%", "正在逐片读盘（有进度条）"],
        ["Filesystem type for checkpoints: ...", "在探磁盘/网络盘速度"],
        ["Loading weights took 12.34 seconds", "读盘 + 拷贝完成"],
        ["Loading model weights took 53.4 GB", "权重实际占了显存"],
        ["Available KV cache memory: 18.2 GiB", "扣掉权重后剩下的 KV 预算"],
    ]
    table(p, X, y4, [("日志", 700), ("说明", 720)], rows, row_h=32)

    y5 = y4 + 32 * 7 + 34
    p.box(X, y5, 1424, 96,
          "<b>常见失败与一句话定位</b><br>"
          "<font style=\"font-size:10px\">"
          "① <b>目录里没权重</b>：Cannot find any model weights with ... → 路径或 shard 命名不对<br>"
          "② <b>形状不匹配</b>：Attempted to load weight (…) into parameter (…) → "
          "多为 TP 数与权重分片不匹配<br>"
          "③ <b>有参数没被赋值</b>：Following weights were not initialized from checkpoint → "
          "名字映射漏了<br>"
          "④ <b>显存不够</b>：Free memory on device (…) is less than desired GPU memory "
          "utilization → 权重+KV 超预算，先降 gpu-memory-utilization 或换更小的权重</font>",
          "box", "neutral")

    y6 = y5 + 130
    anchors(p, y6, [
        "骨架与 target_device：vllm/model_executor/model_loader/base_loader.py:55-72",
        "选 loader：vllm/config/load.py:30、model_loader/__init__.py:50-66、"
        "default_loader.py:163-256",
        "逐张量读盘：model_loader/weight_utils.py:866-1009（safe_open:1002）",
        "切片与拷贝：layers/linear.py:572-589（列并行）、:1730-1746（行并行）、"
        "weight_utils.py:1253-1271（default_weight_loader）",
        "后处理：model_loader/utils.py:97-163；昇腾 NZ：vllm_ascend/ops/linear.py:107-123",
        "显存顺序：vllm_ascend/worker/worker.py:601-634（profile→KV 预算）、"
        "vllm/v1/engine/core.py:354（切块池）",
    ], h=112)

    y7 = y6 + 146
    concl(p, y7,
          "<b>一句话记住这一页：</b>加载 = 读 config（结构）→ 在 NPU 上搭空骨架（参数先分配）→ "
          "逐张量「读盘 → 按 TP 切片 → copy_ 进参数」→ 格式/量化后处理 → "
          "profile 出激活峰值 → 剩余显存切成 KV 块。")

    legend(p, y7 + 92)
