"""⑬ dp=2 × tp=4 的 PD 分离：进程/线程拓扑（27 个进程）。"""
from common import head, concl, anchors, legend
from dg import Doc, table

TITLE = "⑬ dp=2 × tp=4 的边云 PD 分离：一共 27 个进程，谁和谁通信"
SUB = ("配置：prefill 实例 dp=2/tp=4（8 卡），decode 实例 dp=2/tp=4（8 卡），"
       "edge tp=1/dp=1（1 卡），外加 1 个 PD Proxy。<b>dense 模型在 PD 里每个 DP rank 是独立 EngineCore</b>。")
X = 36


def build(doc: Doc) -> None:
    p = doc.page("⑬ dp2tp4 拓扑", pw=1660, ph=1520)
    head(p, TITLE, SUB, w=1600)

    # --- 三列拓扑 ---------------------------------------------------------
    y = 152
    p.text(X, y - 32, 700, 22,
           "<b>① edge（1 卡，uni 后端 → worker 与引擎同进程）</b>", "text_b")
    e1 = p.box(X, y, 460, 74,
               "<b>API server（asyncio 主线程）</b><br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "收请求、发 attempt 号、最后把 token 用 SSE 吐回去</font>",
               "box", "cloudp")
    e2 = p.box(X + 40, y + 104, 420, 96,
               "<b>LwdEdgeEngineCore</b> + 同进程 LwdEdgeWorker<br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "PD runtime（信封、bind、claim、tick）· lwd-edge-c2e 预收线程<br>"
               "单 worker：只做 UNEMBED(lm_head)，不跑整模型</font>",
               "box", "cloudp")
    p.edge(e1, e2, "", st="arrow", role="http")
    p.text(X, y + 216, 460, 24,
           "<b>edge 合计 3 个进程</b>", "text_b")

    p.text(X + 500, y - 32, 760, 22,
           "<b>② P 实例（dp=2 × tp=4 = 8 卡，12 个进程）</b>", "text_b")
    p1 = p.box(X + 500, y, 700, 66,
               "<b>API server 主进程</b>　+ 每 DP 一个 coord listener 线程（2 个 coord_port）"
               "<br><font style=\"font-size:9px\" color=\"#666666\">"
               "lwd-cloud-ctrl · lwd-ctrl-dispatch</font>",
               "box", "cloudp")
    p2 = p.box(X + 500, y + 96, 330, 66,
               "<b>EngineCore_DP0</b><br>"
               "<font style=\"font-size:9px\">调度器 + 忙循环 + 2 IO 线程</font>",
               "box", "cloudp")
    p3 = p.box(X + 870, y + 96, 330, 66,
               "<b>EngineCore_DP1</b><br>"
               "<font style=\"font-size:9px\">独立调度器（互不共享队列）</font>",
               "box", "cloudp")
    p4 = p.box(X + 500, y + 192, 330, 86,
               "<b>VllmWorker-0..3（TP=4）</b><br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "卡 0-3 · 每卡算 1/4 个 head</font>", "box", "cloudd")
    p5 = p.box(X + 870, y + 192, 330, 86,
               "<b>VllmWorker-4..7（TP=4）</b><br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "卡 4-7 · 每卡算 1/4 个 head</font>", "box", "cloudd")
    pc = p.box(X + 500, y + 308, 700, 56,
               "<b>DP 协调进程 ×1</b>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "　dense 下会起，但<b>不发波次</b>（只有 MoE 才 enable_wave_coordination）</font>",
               "box", "fail")
    p.edge(p1, p2, "", st="arrow", role="http")
    p.edge(p1, p3, "", st="arrow", role="http")
    p.edge(p2, p4, "广播", st="arrow", role="http")
    p.edge(p3, p5, "广播", st="arrow", role="http")
    p.text(X + 500, y + 376, 700, 24,
           "<b>P 合计 12 个进程</b>（1 API + 1 协调 + 2 引擎 + 8 worker）", "text_b")

    # --- D 实例（同构，压缩画） -------------------------------------------
    y2 = y + 420
    p.box(X + 500, y2, 700, 108,
          "<b>③ D 实例：与 P 完全同构（dp=2 × tp=4，12 个进程）</b><br>"
          "<font style=\"font-size:10px\">"
          "区别只在角色：D 的每个 TP worker 各自去 <b>P 的对应 TP rank</b> 拉自己那 1/4 份 KV"
          "（P/D 的 TP 相等时是一对一映射，不需要拼装）；"
          "拉齐后 num_computed_tokens 必须正好等于 r_pos 才允许算 tail。</font>",
          "box", "cloudd")
    p.box(X, y2, 460, 108,
          "<b>④ PD Proxy（1 个进程）</b><br>"
          "<font style=\"font-size:10px\">"
          "uvicorn 单进程。用 <b>round-robin</b> 从静态目标表里挑 P 的一个 DP 和 D 的一个 DP"
          "（两个独立计数器），再分别敲它们的 coord 端口。</font>",
          "box", "proxy")

    y3 = y2 + 148
    p.box(X, y3, 1600, 66,
          "<b>全链路合计 ≈ 27 个进程</b>"
          "　·　edge 3　+　P 12　+　D 12　+　Proxy 1　"
          "+　每主机 1 个 ucx_daemon（若用 sidecar 形态）",
          "box", "neutral")

    # --- 通信矩阵 ---------------------------------------------------------
    y4 = y3 + 106
    p.text(X, y4 - 32, 1200, 22,
           "<b>谁和谁通信、走什么（这张表是排障的第一手材料）</b>", "text_b")
    rows = [
        ["客户端 ↔ edge", "HTTP / SSE", "OpenAI 兼容接口"],
        ["edge ↔ PD Proxy", "HTTP（aiohttp 流式）", "信封：attempt/绑定/导出事件"],
        ["Proxy ↔ P/D coord", "HTTP", "指定 target_dp，注册/导出/结果"],
        ["云 coord → 本进程 EngineCore", "进程内 mp.Queue + 唤醒线程", "命令投递"],
        ["EngineCore → 4 个 worker", "共享内存消息队列（广播）", "同一份 SchedulerOutput"],
        ["4 个 worker 之间", "<b>HCCL 集合通信</b>", "行并行 all_reduce / logits gather"],
        ["P worker ↔ D worker", "Mooncake（ZMQ 握手 + 搬运引擎）", "KV 块级 pull（TP rank 一对一）"],
        ["edge ↔ 每个云 DP", "LWD 控制面 ROUTER/DEALER", "bind/claim/notify/c2e"],
        ["edge ↔ 云（KV 之外的数据）", "LWD host 数据面", "UP/DOWN 帧（隐向量、token）"],
    ]
    table(p, X, y4, [("两端", 360), ("走什么", 420), ("承载内容", 660)], rows,
          row_h=32)

    # --- 三个易错点 -------------------------------------------------------
    y5 = y4 + 32 * 10 + 34
    p.box(X, y5, 1600, 118,
          "<b>这个配置下最容易踩的三点</b><br>"
          "① <b>kv_connector_extra_config 里的 dp_size 要写「部署级 DP」=2</b>，不是塌缩后的 1；"
          "且 prefill/decode 两边都要写全 dp+tp。<br>"
          "② <b>KV 握手端口要用 data_parallel_index 算</b>"
          "（kv_port + dp_index×tp×pp×pcp）；用 data_parallel_rank 会让两个 DP 都去绑同一个端口，"
          "第二个进程 address in use。<br>"
          "③ <b>P 与 D 的 TP 必须能整除对齐</b>（本例都是 4，一对一）；"
          "不对称配置会在<b>首个请求</b>时报 KV 分片断言，而不是启动时报。",
          "box", "fail")

    y6 = y5 + 152
    anchors(p, y6, [
        "进程数推导：vllm/v1/engine/utils.py:1279-1291（每 DP 一个 EngineCore）、"
        ":362-425（DP rank → 物理卡切分）、"
        "vllm/v1/executor/multiproc_executor.py:190-206（每 rank 一个 worker）",
        "dense 塌缩（每 DP 独立引擎）：vllm/v1/engine/core.py:1372-1380；"
        "wave 仅 MoE：vllm/v1/engine/utils.py:1199",
        "DP 由 Proxy 选：vllm/v1/lwd_control/pd/proxy.py:131-174（选择 :159-162）",
        "KV 端口与一对一映射：kv_p2p/mooncake_connector.py:1761-1770、:2193-2204、"
        ":4028-4029、:4078-4087",
        "extra_config 校验：vllm_ascend/utils.py:1580-1612",
    ], h=112)

    y7 = y6 + 146
    concl(p, y7,
          "<b>一句话记住这一页：</b>dp=2×tp=4 时一个实例 = 1 API + 1 协调 + <b>2 个独立引擎</b> "
          "+ <b>8 个 worker</b>；请求的 DP 由 Proxy 轮询选定（dense 下没有 wave 锁步），"
          "P 内部靠「广播 + HCCL all_reduce」让 4 卡协同，P→D 的 KV 是 TP rank 之间一对一拉。")

    legend(p, y7 + 92)
