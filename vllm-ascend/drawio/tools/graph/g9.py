"""⑨ 昇腾侧：ACL Graph 是什么、与 CUDA Graph 差在哪、本仓为什么这么配。"""
from common import head, concl, anchors, legend, panel
from dg import Doc, table

TITLE = "⑨ 昇腾侧：ACL Graph，以及本仓「D 用图、P 和 edge 不用」的选择"
SUB = ("一句话：昇腾上叫 <b>ACL Graph</b>（PyTorch 里是 <code>torch.npu.NPUGraph</code>），"
       "用法与 CUDA Graph 同构；<b>最大差别是回放前要先在 host 侧刷新 attention 参数</b>。"
       "本仓部署里只有 D 侧开图。")


def build(doc: Doc) -> None:
    p = doc.page("⑨ 昇腾 ACL Graph", pw=1560, ph=1480)
    head(p, TITLE, SUB)
    X = 36

    # --- 同构 -------------------------------------------------------------
    y = 154
    p.text(X, y - 32, 1200, 22,
           "<b>第一步：两者「同一套抽象，两处实现」</b>", "text_b")
    rows = [
        ["图对象", "torch.cuda.CUDAGraph", "torch.npu.NPUGraph（vllm-ascend 里做了别名）"],
        ["捕获上下文", "torch.cuda.graph(...)", "torch.npu.graph(...)（acl_graph.py 主线）"],
        ["内存池", "graph_pool_handle（平台钩子透传）", "同一个钩子最终也落到 torch.npu.graph_pool_handle"],
        ["包装器", "CUDAGraphWrapper", "ACLGraphWrapper（平台注册给 vLLM）"],
        ["模式枚举", "CUDAGraphMode：NONE/PIECEWISE/FULL…", "完全复用同一套枚举"],
        ["回放前", "直接 replay()", "<b>必须先刷 host 侧参数</b>再 replay（见下）"],
    ]
    table(p, X, y, [("维度", 260), ("CUDA 侧", 560), ("昇腾侧", 600)], rows,
          row_h=34)

    # --- 关键差异 ---------------------------------------------------------
    y2 = y + 34 * 7 + 40
    p.text(X, y2 - 32, 1200, 22,
           "<b>第二步：昇腾独有的「回放前刷新」——这是最该记住的一条</b>", "text_b")
    panel(p, X, y2, 700, 186, "问题：attention 的这几个量每步都在变", [
        "· actual_seq_lengths（query 累计长度）",
        "· actual_seq_lengths_kv（每个请求当前的 KV 长度）",
        "· block_table（本步用哪些块）",
        "它们在昇腾上是以 <b>host 侧 Python list</b> 的形式随 metadata 传的，"
        "而图里只认地址——所以不能只 replay，必须先把这些值刷进去。",
    ], "fail")
    panel(p, X + 760, y2, 664, 186, "解法：register_task + graph_task_update", [
        "把 attention 算子用 register_task 注册成图里的任务，"
        "并附一个 ParamProvider（负责取上面的动态值）；",
        "回放前用 <font style=\"font-size:9px\">torch.npu.graph_task_update_begin/end"
        "</font> 包住一次算子调用 → 把这些参数更新进图；",
        "然后才 replay。对应实现：UpdatableGraph / graph_task_* API。",
    ], "cloudd")

    # --- 本仓部署选择 -----------------------------------------------------
    y3 = y2 + 226
    p.text(X, y3 - 32, 1200, 22,
           "<b>第三步：本仓部署脚本怎么配（这一条能解释很多现象）</b>", "text_b")
    d1 = p.box(X, y3, 460, 112,
               "<b>D 侧（decode）：开图</b><br>"
               "<font style=\"font-size:10px\">"
               "--compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}'，"
               "有的包还显式给 cudagraph_capture_sizes</font>"
               "<br><font style=\"font-size:9px;color:#666666\">"
               "sidecar/cloud_d.sh:31 · mooncake_pd_195/decode.sh:34</font>",
               "box", "cloudd")
    d2 = p.box(X + 480, y3, 460, 112,
               "<b>P 侧（prefill）：--enforce-eager</b><br>"
               "<font style=\"font-size:10px\">"
               "prefill 是长宽矩阵、形状差异大；且 P 只跑一次就交接，"
               "省启动时间更划算</font>"
               "<br><font style=\"font-size:9px;color:#666666\">"
               "cloud_p.sh:26 · prefill.sh:33</font>",
               "box", "cloudp")
    d3 = p.box(X + 960, y3, 464, 112,
               "<b>edge：--enforce-eager，而且代码里直接跳过捕获</b><br>"
               "<font style=\"font-size:10px\">"
               "LwdEdgeWorker 覆写了 compile_or_warm_up_model，直接返回 0 秒——"
               "因为 edge 的 forward 被 LWD 劫持，只做 embed/unembed</font>"
               "<br><font style=\"font-size:9px;color:#666666\">"
               "lwd_edge_worker.py:122-131（日志 [lwd-edge] skip model warmup/capture）</font>",
               "box", "fail")

    y4 = y3 + 152
    p.box(X, y4, 1424, 104,
          "<b>两个重要澄清</b><br>"
          "<font style=\"font-size:10px\">"
          "① <b>「LWD 下必须 enforce-eager」并不存在于代码里</b>——"
          "全仓 grep 找不到这条强制；它只是部署脚本的约定（P/edge 加 flag、D 开 FULL_DECODE_ONLY）。<br>"
          "② <b>eager 与 --compilation-config 同时给不会报错</b>：eager 会静默胜出"
          "（一条 warning_once + 一条 info），用户给的 cudagraph_mode 被丢弃、永远不捕获。</font>",
          "box", "neutral")

    # --- 已知问题 ---------------------------------------------------------
    y5 = y4 + 138
    p.text(X, y5 - 32, 1200, 22,
           "<b>第四步：昇腾侧已知限制（遇到别当成 bug 猜）</b>", "text_b")
    rows2 = [
        ["PIECEWISE 需要编译", "必须 VLLM_COMPILE 或开 breakable，否则启动即 AssertionError"],
        ["DP 要取最小", "cudagraph_mode 跨 DP rank 取最小：一个 rank 是 NONE 就全体 NONE"],
        ["TP × query_len 对齐", "FULL decode 的桶要对齐到 LCM(query_len, TP)（临时 workaround）"],
        ["流资源耗尽", "报 207008 / Alloc sq cq fail → 调小 capture sizes 或升级 HDK"],
        ["超桶静默掉出", "num_tokens 超过最大桶 → 直接回 eager，<b>没有日志</b>"],
    ]
    table(p, X, y5, [("限制", 320), ("表现 / 应对", 1060)], rows2, row_h=32)

    y6 = y5 + 32 * 6 + 34
    anchors(p, y6, [
        "ACL 图包装器与捕获/回放：vllm-ascend/vllm_ascend/compilation/acl_graph.py:156-302"
        "（torch.npu.graph 在 :210；replay :297-302；捕获失败文案 :228-243）",
        "参数刷新：vllm-ascend/vllm_ascend/compilation/updatable_graph.py:72-76、:147-174",
        "平台接入（wrapper 与静态图支持）：vllm-ascend/vllm_ascend/platform.py:159-160、:218-222；"
        "模式降级 :1171-1246、:1346-1352",
        "CUDA→NPU 名字别名：vllm-ascend/vllm_ascend/worker/model_runner_v1.py:6687-6739；"
        "V2 版 worker/v2/utils.py:162-171",
        "edge 跳过捕获：vllm-ascend/vllm_ascend/worker/lwd_edge_model/lwd_edge_worker.py:122-131",
        "部署侧 flag：assets/c_home/bash/pds/30rc_1124/sidecar/cloud_d.sh:31（D 开图）、"
        "cloud_p.sh:26 与 edge.sh:30（enforce-eager）",
    ], h=120)

    y7 = y6 + 154
    concl(p, y7,
          "<b>一句话记住这一页：</b>昇腾的图叫 ACL Graph，用法与 CUDA 同构，"
          "但<b>回放前要多一步「host 侧刷 attention 参数」</b>；"
          "本仓只在 D 侧开 FULL_DECODE_ONLY，P 与 edge 一律 eager（edge 还在代码里直接跳过捕获）。")

    legend(p, y7 + 92)
