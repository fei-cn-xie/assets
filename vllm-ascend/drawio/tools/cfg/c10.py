"""⑩ 一页纸：因果速查 + 自测。"""
from common import head, concl, anchors, legend
from dg import Doc, table

TITLE = "⑩ 一页纸：改 A 会连累谁（速查）+ 五道自测"
SUB = ("用途：调参或排障时先扫这张表，确认你改的那个参数<b>是不是还会顺手改掉别人</b>。")


def build(doc: Doc) -> None:
    p = doc.page("⑩ 速查与自测", pw=1560, ph=1580)
    head(p, TITLE, SUB)
    X = 36

    y = 146
    p.text(X, y - 32, 1300, 22,
           "<b>第一张表：静默改写（有 warning 或干脆没日志）</b>", "text_b")
    rows = [
        ["设了 K", "query 长度 1+K → 图桶全部向上取整到 K+1 的倍数", "启动无感"],
        ["调小 max-num-batched-tokens", "图桶上限跟着缩；max-num-seqs 也可能被反推变小", "可能 warning"],
        ["关 chunked prefill", "max-num-batched-tokens 被抬到 ≥ max-model-len", "无日志"],
        ["性能模式设 throughput", "batched 与 seqs 的默认值各 ×2（仅当你没显式设）", "无日志"],
        ["开序列并行 + TP>1", "桶里非 TP 倍数的项被删；batched 上取整到 TP×PCP", "warning"],
        ["block_size 改大", "KV 块数按比例减少；PD 两侧块粒度对不上会失败", "启动期报错"],
    ]
    table(p, X, y, [("你改的", 420), ("被连带改的", 720), ("提示", 284)], rows,
          row_h=36)

    y2 = y + 36 * 7 + 32
    p.text(X, y2 - 32, 1300, 22,
           "<b>第二张表：强制覆盖（你写的不算）</b>", "text_b")
    rows2 = [
        ["给了 --kv-cache-memory-bytes", "gpu-memory-utilization 被忽略，跳过显存画像",
         "info 日志明说"],
        ["给了 kv-offloading-size", "kv_connector 被改写，kv_role 被锁成 kv_both", "info"],
        ["P 侧收到 PD 请求", "max_tokens=1 / ignore_eos / 抑制用户输出", "无日志"],
        ["target 加了 --enforce-eager", "编译与图全 NONE；draft 的 enforce_eager 变冗余", "warning once"],
        ["GLM 族开投机", "draft 被强制切成 eager", "warning"],
    ]
    table(p, X, y2, [("你改的", 420), ("被覆盖的", 720), ("提示", 284)], rows2,
          row_h=36)

    y3 = y2 + 36 * 6 + 32
    p.text(X, y3 - 32, 1300, 22,
           "<b>第三张表：直接拒绝启动（会点名原因）</b>", "text_b")
    rows3 = [
        ["batched < max_model_len 且关了分块", "一步装不下整条序列"],
        ["SP 的 TP 与 K+1 互不整除", "桶取不到公倍数"],
        ["attention 后端 NEVER 却要 FULL 图", "换后端或改 PIECEWISE"],
        ["Mamba 类模型 seqs > 块数", "每序列要一个 state 块"],
        ["PD 两侧块大小不一致", "KV 无法按同一粒度对齐"],
        ["K > MTP 层数且不整除", "同一层复用次数对不上"],
        ["max_model_len 的 KV 超过可用", "报错会给「本卡最多支持多长」"],
    ]
    table(p, X, y3, [("情形", 620), ("为什么", 804)], rows3, row_h=32)

    # --- 自测 -------------------------------------------------------------
    y4 = y3 + 32 * 8 + 34
    p.text(X, y4 - 32, 1300, 22,
           "<b>五道自测（答不上就回 ③④⑤⑥⑦ 对应页）</b>", "text_b")
    p.box(X, y4, 1424, 132,
          "<font style=\"font-size:10px\">"
          "① 你把 K 从 0 改成 3，为什么「图捕获突然不生效了」？先查哪两处？<br>"
          "② 同一个模型，为什么有的包敢报 max_model_len=65536、有的只敢报 512？<br>"
          "③ 你没设 max_num_batched_tokens，它凭什么变成 8192？谁会跟着它变？<br>"
          "④ 「调大 gpu-memory-utilization」是不是等于「能开更多并发」？为什么？<br>"
          "⑤ 两个 enforce-eager，哪一个会让 target 的图一起没掉？为什么？</font>",
          "box", "neutral")

    y5 = y4 + 172
    p.box(X, y5, 1424, 96,
          "<b>读完之后你应该能回答的一句总结</b><br>"
          "<font style=\"font-size:10px\">"
          "「本仓的每个参数都能用三句话解释清：<b>它花的是谁的钱、它被谁悄悄改过、"
          "它会把哪个机制弄失效</b>。」——这就是这一套图想训练的能力。</font>",
          "box", "cloudp")

    y6 = y5 + 130
    anchors(p, y6, [
        "静默改写：vllm/engine/arg_utils.py:2966-3016（默认值推导）；"
        "vllm/config/vllm.py:2348-2349、:2404-2412（桶裁剪）；"
        "vllm/config/compilation.py:1547-1592（K+1 对齐）",
        "强制覆盖：vllm/config/cache.py:228-235 + vllm/v1/worker/gpu_worker.py:542-564；"
        "vllm/config/vllm.py:1052-1073（kv_offloading）；"
        "vllm/v1/lwd_control/pd/cloud_runtime.py:358-383（P 侧改写）",
        "拒绝启动：vllm/config/scheduler.py:296-329；vllm/config/compilation.py:1402-1417、"
        ":1527-1542、:1551-1564；vllm/v1/core/kv_cache_utils.py:872-909",
        "eager 两个作用域：vllm/config/vllm.py:1591-1597、:1823-1829；"
        "vllm/config/speculative.py:378-379、:1289",
    ], h=112)

    y7 = y6 + 146
    concl(p, y7,
          "<b>一句话记住这一套：</b>参数不是孤立的——"
          "改一个数，引擎可能替你改三个，也可能让某个机制静默失效；"
          "<b>所以每次改完，都要问：桶还对不对、KV 还够不够、图还在不在。</b>")

    legend(p, y7 + 92)
