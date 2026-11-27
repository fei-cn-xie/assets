"""③ 权重从文件到显存：与前置图的接口（不重复细节，只给路径与三个坑）。"""
from common import head, concl, anchors, legend, panel, bridge
from dg import Doc

TITLE = "③ 权重从文件到显存：三个坑与前置页的接口"
SUB = ("一句话：<b>磁盘上的 safetensors → 读成 CPU 张量 → 按 TP 切片 → 拷进 NPU 上已分配的参数</b>。"
       "细节在另外两套图里；本页只标出「在哪一步最容易错」，并说明为什么要先看那两套。")


def build(doc: Doc) -> None:
    p = doc.page("③ 权重到显存", pw=1560, ph=1380)
    head(p, TITLE, SUB)
    X = 36

    # --- 五步 -------------------------------------------------------------
    y = 152
    p.text(X, y - 34, 1200, 22,
           "<b>第一步：五步（每步都指向已有图页，不在这里重复展开）</b>", "text_b")
    steps = [
        ("①", "看目录与 config.json", "知道模型有几个参数、多深、多宽", "Transformer ⑮"),
        ("②", "读分片与 index.json", "只读该读的文件；名字→文件靠索引", "Transformer ⑰"),
        ("③", "名字映射 + 选 weight_loader", "q_proj/k_proj/v_proj 合成 qkv_proj", "Transformer ⑱"),
        ("④", "按 TP 切片 + copy_ 进 NPU", "narrow 出本 rank 那段，写入参数", "进程/线程 ⑮"),
        ("⑤", "格式后处理 + profile + 切 KV", "NZ 重排；然后才轮到 KV 块池", "进程/线程 ⑮⑯"),
    ]
    for i, (no, title, body, ref) in enumerate(steps):
        cx = X + i * 288
        p.box(cx, y, 274, 116,
              f"<b>{no} {title}</b><br>"
              f"<font style=\"font-size:10px\">{body}</font><br>"
              f"<font style=\"font-size:9px;color:#666666\">详见：{ref}</font>",
              "box", "neutral")
        if i:
            p.edge(src_pt=(cx - 12, y + 58), dst_pt=(cx - 2, y + 58), st="arrow",
                   role="http")

    # --- 三个坑 -----------------------------------------------------------
    y2 = y + 156
    p.text(X, y2 - 34, 1200, 22,
           "<b>第二步：三个坑（都能用一句话解释为什么）</b>", "text_b")
    panel(p, X, y2, 700, 150, "坑一：把「分片」当成「张量并行」", [
        "分片（shard）＝训练/发布时把权重切成几个文件；",
        "张量并行（TP）＝运行时把一层矩阵横着/竖着切到几张卡上。",
        "两者毫无关系：TP=4 也可以只有一个权重文件（读进来再切）。",
    ], "fail")
    panel(p, X + 760, y2, 664, 150, "坑二：忘了「[out, in] 的转置」", [
        "权重文件里的 Linear 通常是 [out_features, in_features]，",
        "而很多公式写成 y = xW 时 W 是 [in, out]。",
        "报错形态就是形状不匹配（Attempted to load weight (…) into parameter (…)）。",
    ], "fail")

    y3 = y2 + 190
    panel(p, X, y3, 1424, 128, "坑三：先权重、后 KV —— 顺序不能反", [
        "加载权重 → 跑一次 profile 量出激活峰值 → <b>剩余显存才切成 KV 块</b>。",
        "如果反过来（先切 KV）会导致 KV 预算算错、甚至启动 OOM；"
        "这也是为什么日志里 Available KV cache memory 出现在加载权重之后。",
        "副作用（很重要）：<b>权重变大 = KV 块变少 = 能同时服务的请求变少</b>。这就是 ⑦ 那张显存账的起点。",
    ], "cloudp")

    y4 = y3 + 168
    bridge(p, y4, "本页交接：权重就位后，模型就能「算」了。"
                  "但推理不是「算一次」，而是<b>先算一大段、再一个字一个字算</b>——下一步（④）讲这两个阶段。")

    y5 = y4 + 76
    anchors(p, y5, [
        "读盘与切片的实现：vllm/model_executor/model_loader/weight_utils.py:1002（safe_open）、"
        ":1253-1271（default_weight_loader）",
        "TP 切片落地：vllm/model_executor/layers/linear.py:572-589（列并行）、:1730-1746（行并行）",
        "先权重后 KV 的顺序：vllm/v1/worker/gpu_worker.py:528-642（profile→KV 预算）、"
        "vllm/v1/engine/core.py:354（切块池）",
        "默认块大小与显存上限：vllm/config/cache.py:71（block_size=16）、:103"
        "（gpu_memory_utilization=0.92）",
    ], h=104)

    y6 = y5 + 136
    concl(p, y6,
          "<b>一句话记住这一页：</b>加载只有五步，坑只有三个——分片≠TP、转置、以及"
          "「先权重再 profile 再 KV」这个不可换的顺序；顺序决定了后面所有显存档。")

    legend(p, y6 + 92)
