"""⑩ 排障与坑：图「看起来开了，其实没生效」的六种情况。"""
from common import head, concl, anchors, legend
from dg import Doc, table

TITLE = "⑩ 排障与坑：六种「以为开了图、其实没有」"
SUB = ("一句话：图失效几乎都不报错——要么<b>静默回落 eager</b>，要么<b>开着但算错</b>。"
       "这页是排查顺序：先看有没有捕获，再看有没有命中，最后看是不是越了桶。")


def case(p, y, no, title, symptom, cause, fix, role="fail"):
    p.box(36, y, 1424, 84, "", "box", role)
    p.text(48, y + 6, 1400, 22, f"<b>{no}　{title}</b>", "text_b")
    p.text(48, y + 28, 1400, 52,
           f"<font style=\"font-size:9px\" color=\"#333333\">"
           f"现象：{symptom}<br>原因：{cause}　→　处置：{fix}</font>", "text")
    return y + 84 + 10


def build(doc: Doc) -> None:
    p = doc.page("⑩ 排障与坑", pw=1560, ph=1560)
    head(p, TITLE, SUB)
    X = 36

    y = 152
    p.text(X, y - 32, 1300, 22,
           "<b>六种情况（按「先怀疑谁」排序）</b>", "text_b")

    y = case(p, y, "①", "启动日志里根本没有捕获",
             "没有 Graph capturing finished / Capturing ACL graph",
             "被 --enforce-eager 或 -O0 一票否决；或 cudagraph_mode=NONE",
             "看有没有 Enforce eager set, disabling torch.compile and CUDAGraphs 这条 warning")
    y = case(p, y, "②", "捕获了但并发一高就变慢",
             "低并发时快、并发上到某个数就退回原速",
             "<b>num_tokens 超过最大桶</b>（默认 512）→ dispatcher 直接回 NONE，且无日志",
             "看最大桶与 max_num_seqs 的关系；必要时 --max-cudagraph-capture-size 调大或压并发")
    y = case(p, y, "③", "捕获时就 OOM / 启动直接失败",
             "Graph capturing ... 之后进程退出",
             "图池显存放不下（捕获段没有 try/except 回退）",
             "缩桶（--cudagraph-capture-sizes 1,2,4,8,16）或降 gpu-memory-utilization")
    y = case(p, y, "④", "所有请求都拿到同一个旧结果",
             "输出重复、内容像是缓存的",
             "某处 .item() / 分支被「烤」进图（约束三）",
             "把该逻辑挪到图外；或先 --enforce-eager 验证是不是图的问题")
    y = case(p, y, "⑤", "报 Input addresses ... different during replay",
             "回放时抛地址不一致",
             "有代码在回放期换了输入张量（没走持久 buffer）",
             "改成 copy_ 进固定 buffer；查这段日志的栈找到改地址的地方")
    y = case(p, y, "⑥", "昇腾上捕获报错 207008 / Alloc sq cq fail",
             "RuntimeError，提示流资源不足或旧 HDK",
             "capture sizes 太大 / HDK 版本旧",
             "调小 capture sizes；或升级 HDK（仓内文案给的处理建议）")

    # --- 排查顺序 ---------------------------------------------------------
    y2 = y + 10
    p.text(X, y2 - 32, 1300, 22,
           "<b>四步排查法（照着走，不要一上来就改代码）</b>", "text_b")
    p.box(X, y2, 1424, 92,
          "<font style=\"font-size:10px\">"
          "① <b>确认模式</b>：看启动日志的 cudagraph 模式与桶数（有没有被 eager 否决、"
          "有没有被 NPU 降级）。<br>"
          "② <b>确认捕获</b>：有没有 Graph capturing finished in N secs, took X GiB；"
          "显存是否符合预期。<br>"
          "③ <b>确认命中</b>：开指标（--cudagraph-metrics）看命中率；"
          "或临时移除 --disable-log-stats 观察。<br>"
          "④ <b>确认形状</b>：对比 Running 请求数 × 每请求 token 数与最大桶，"
          "判断是不是「越桶掉出」。</font>",
          "box", "neutral")

    # --- 日志表 -----------------------------------------------------------
    y3 = y2 + 126
    p.text(X, y3 - 32, 1300, 22, "<b>必背的六条日志</b>", "text_b")
    rows = [
        ["Enforce eager set, disabling torch.compile and CUDAGraphs...", "WARNING", "eager 一票否决"],
        ["Capturing ACL graph (FULL, BatchDescriptor(...))", "DEBUG", "昇腾每抓一张图（需开 debug 日志）"],
        ["Graph capturing finished in N secs, took X GiB", "INFO", "捕获完成：耗时与图显存"],
        ["CUDA graph pool memory: X GiB (actual), Y GiB (estimated)", "INFO", "估算 vs 实际"],
        ["ACL graph replay is active (logged once).", "INFO", "journal 里确认图真的在用"],
        ["cudagraph_mode X is unsupported on NPU; falling back to NONE.", "INFO", "被平台降级了"],
    ]
    table(p, X, y3, [("日志原文（关键字）", 760), ("级别", 120), ("含义", 400)], rows,
          row_h=32)

    y4 = y3 + 32 * 7 + 34
    anchors(p, y4, [
        "静默回落 NONE（无日志）：vllm/v1/cudagraph_dispatcher.py:274-282",
        "eager 警告与降级：vllm/config/vllm.py:1591-1597；"
        "vllm-ascend/vllm_ascend/platform.py:1218-1222、:1346-1352",
        "捕获失败的昇腾文案：vllm-ascend/vllm_ascend/compilation/acl_graph.py:228-243",
        "地址不一致断言：vllm/compilation/cuda_graph.py:348-357",
        "日志原文：vllm/v1/worker/gpu_model_runner.py:6942-6946（耗时/显存）、"
        "vllm-ascend/vllm_ascend/compilation/acl_graph.py:182、:301",
        "指标开关：vllm-ascend/docs/source/developer_guide/Design_Documents/ACL_Graph.md:105",
    ], h=112)

    y5 = y4 + 146
    concl(p, y5,
          "<b>一句话记住这一页：</b>图的问题九成不报错——"
          "先看「有没有捕获」，再看「有没有命中」，最后看「形状是否越桶」；"
          "任何一次怀疑，都用 --enforce-eager 做对照实验来二分。")

    legend(p, y5 + 92)
