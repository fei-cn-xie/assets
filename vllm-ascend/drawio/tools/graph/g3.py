"""③ 一次捕获里发生了什么：10 步（照着代码顺序走）。"""
from common import head, concl, anchors, legend
from dg import Doc

TITLE = "③ 一次捕获里发生了什么（10 步，按代码顺序）"
SUB = ("一句话：<b>包一层 wrapper → 开闸门 → 切到侧流 → 从大到小逐个桶："
       "warmup 几遍 → 同步 → 在 torch.cuda.graph 里真跑一遍</b> → 关闸门、锁 workspace。"
       "注意：捕获是在「profile 量完显存、KV 块池也建好」之后才做的。")


def step(p, x, y, w, h, no, title, body, role):
    p.box(x, y, w, h, "", "box", role)
    p.text(x + 10, y + 8, w - 20, 22, f"<b>{no}　{title}</b>", "text_b")
    p.text(x + 10, y + 32, w - 20, h - 40,
           f"<font style=\"font-size:9px\" color=\"#333333\">{body}</font>", "text")


def build(doc: Doc) -> None:
    p = doc.page("③ 捕获内部 10 步", pw=1560, ph=1620)
    head(p, TITLE, SUB)
    X, W = 36, 690

    y = 158
    p.text(X, y - 32, 1300, 22,
           "<b>左列：准备阶段（1-5）　右列：逐桶录制（6-10）</b>", "text_b")

    step(p, X, y, W, 118, "①", "包 wrapper（决定「图从哪开始到哪结束」）",
         "FULL：在整模型外面包一层 CUDAGraphWrapper。<br>"
         "PIECEWISE：由编译后端给<b>每个子图</b>各包一层（每个 attention 之间一段）。<br>"
         "<font color=\"#c0392b\">这一步决定了图的粒度，也决定了 attention 在不在图里。</font>",
         "neutral")
    step(p, X, y + 148, W, 118, "②", "开「捕获闸门」",
         "全局标志位打开：此刻允许捕获。若在别的时候触发捕获，"
         "会直接抛 <font style=\"font-size:9px\">CUDA graph capturing detected at an "
         "inappropriate time</font>。", "neutral")
    step(p, X, y + 296, W, 132, "③", "切到侧流 + 冻住 GC + 清缓存量基线",
         "必须在<b>非默认 stream</b> 上捕获，这样只录到本次的算子（不然会把别人的也录进去）。<br>"
         "然后 synchronize + empty_cache，记下此刻的空闲显存 → 用来事后算「图占了多少钱」。",
         "neutral")
    step(p, X, y + 458, W, 118, "④", "拿到「进程级」的 graph 显存池",
         "整个进程只有一个 pool（不是每张图一个）。所以多张图共用一块池，"
         "占用≈各桶峰值的最大值。", "fail")
    step(p, X, y + 606, W, 132, "⑤", "排好捕获顺序：从大到小",
         "代码注释写得很直白：先抓大形状，小形状可以复用大形状已经分配过的池内存。<br>"
         "<font color=\"#c0392b\">反过来（先小后大）会多占显存，甚至 OOM。</font>",
         "fail")

    X2 = X + 742
    step(p, X2, y, 682, 118, "⑥", "每个桶先 warmup 几遍",
         "在<b>无图模式</b>下把这一步空跑 N 遍（cudagraph_num_of_warmups，V1 默认 1）。"
         "目的是让算子库/通信库把 lazy 初始化都做完。", "cloudp")
    step(p, X2, y + 148, 682, 118, "⑦", "warmup 完必须同步一次",
         "注释原因：warmup 可能在辅助流上跑，不等它完成就开始捕获，会录到一半。", "cloudp")
    step(p, X2, y + 296, 682, 146, "⑧", "把输入「钉」到固定地址（dummy run 里做）",
         "input_ids / positions / slot_mapping / attn_metadata 全部取<b>预先分配好的持久 buffer "
         "的切片</b>；<br>"
         "捕获时 slot_mapping 全填 -1（跳过真正写 KV），attn_metadata 用 "
         "for_cudagraph_capture 专用构造。<br>"
         "这就是「地址固定」约束的落地方式。", "cloudp")
    step(p, X2, y + 472, 682, 146, "⑨", "真正的录制：在 torch.cuda.graph 里跑一遍 forward",
         "<font style=\"font-size:9px\">with torch.cuda.graph(cudagraph, pool=全局pool, "
         "stream=当前侧流): output = self.runnable(...)</font><br>"
         "这一跑被录成图；output 只留<b>弱引用</b>（省显存，因为它是池里的同一块）。<br>"
         "昇腾侧对应 <font style=\"font-size:9px\">torch.npu.graph(...)</font>。", "cloudd")
    step(p, X2, y + 648, 682, 118, "⑩", "收尾：关闸门、锁 workspace、打日志",
         "关掉捕获闸门；把通信 workspace 的尺寸<b>锁死</b>（回放期间改尺寸会让图里的指针失效）；"
         "打印 <font style=\"font-size:9px\">Graph capturing finished in N secs, took X GiB</font>。",
         "neutral")

    # --- 位置说明 ---------------------------------------------------------
    y2 = y + 800
    p.box(X, y2, 1424, 96,
          "<b>它在启动流程里的位置（很容易记错）</b><br>"
          "<font style=\"font-size:10px\">"
          "正确的顺序是：<b>加载权重 → profile（eager 空跑一遍量激活峰值）→ 按剩余显存切 KV 块池 → "
          "然后才捕获图</b>。<br>"
          "原因：捕获本身也要占显存，而且要用「图显存池」去估；"
          "所以必须先把权重和 KV 的账算清，才知道还能不能抓图、抓多大。</font>",
          "box", "neutral")

    y3 = y2 + 130
    anchors(p, y3, [
        "捕获入口（V1）：vllm/v1/worker/gpu_model_runner.py:6843 → :6997 → :6949 → :5851",
        "捕获入口（V2，本分支默认）：vllm/v1/worker/gpu/model_runner.py:998-1071 → "
        "vllm/v1/worker/gpu/cudagraph_utils.py:398-488",
        "启动时序（profile→KV→捕获）：vllm/v1/engine/core.py:254、:308、:354、:356；"
        "vllm/v1/worker/gpu_worker.py:773-816",
        "闸门：vllm/compilation/monitor.py:87-104；侧流：vllm/distributed/parallel_state.py:1637-1666",
        "全局 pool：vllm/compilation/cuda_graph.py:200、:305-308；"
        "平台单例 vllm/platforms/interface.py:1174-1181",
        "录制那一行：vllm/compilation/cuda_graph.py:315-321；"
        "从大到小：vllm/v1/worker/gpu_model_runner.py:6858-6860、"
        "vllm/v1/cudagraph_dispatcher.py:326-350",
    ], h=112)

    y4 = y3 + 146
    concl(p, y4,
          "<b>一句话记住这一页：</b>捕获 = 开闸门 + 侧流 + 逐桶（warmup → 同步 → "
          "在 graph 上下文里真跑一遍）」；输入必须来自预分配的持久 buffer；"
          "顺序必须从大到小；结束要锁 workspace。")

    legend(p, y4 + 92)
