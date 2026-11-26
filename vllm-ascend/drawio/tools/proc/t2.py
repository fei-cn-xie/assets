"""② 进程个数怎么算：三个参数决定一张进程树。"""
from common import head, concl, anchors, legend
from dg import Doc, table

TITLE = "② 进程个数怎么算：参数 → 进程树的一步步推导"
SUB = ("一句话：进程数不是玄学，是四个数的加法——"
       "API server 个数（跟 DP 走）+ EngineCore 个数（跟 DP 走）+ worker 个数（跟 TP×PP 走）"
       "+ 若干「因为并行才多出来」的进程。")
X = 36
W = 340
GAP = 22


def step_box(p, x, y, w, h, no, title, body, role):
    p.box(x, y, w, h, "", "box", role)
    p.text(x + 10, y + 8, w - 20, 22, f"<b>{no}　{title}</b>", "text_b")
    p.text(x + 10, y + 34, w - 20, h - 42,
           f"<font style=\"font-size:9px\" color=\"#333333\">{body}</font>", "text")


def build(doc: Doc) -> None:
    p = doc.page("② 进程数怎么算", pw=1560, ph=1420)
    head(p, TITLE, SUB)

    # --- 四个加法项 -------------------------------------------------------
    y = 148
    p.text(X, y - 32, 1200, 22,
           "<b>第一步：四个加法项（每一项都由一个具体参数决定）</b>", "text_b")
    step_box(p, X, y, W, 196, "①", "API server 进程",
             "由 <b>--api-server-count</b> 决定；"
             "不显式给时，内部负载均衡模式下<b>默认 = data_parallel_size</b>。<br><br>"
             "&gt;1 时走多 API server：主进程只当「工头」，"
             "另外 fork 出 N 个子进程，用 SO_REUSEPORT 抢同一个端口。", "neutral")
    step_box(p, X + (W + GAP), y, W, 196, "②", "EngineCore 进程",
             "由 <b>--data-parallel-size</b> 决定：本机每个 DP 分片一个 EngineCore，"
             "进程名形如 <font style=\"font-size:9px\">EngineCore_DP0</font>。<br><br>"
             "它才是「引擎」：跑调度 + 忙循环，前端通过 ZMQ 和它说话。", "cloudp")
    step_box(p, X + 2 * (W + GAP), y, W, 196, "③", "worker 进程",
             "由 <b>TP × PP</b>（以及是否 DP&gt;1）决定：<br>"
             "· TP×PP×DP == 1 → <b>uni 后端，worker 不另起进程</b>（与 EngineCore 同进程）<br>"
             "· 否则 → mp 后端，每 rank 一个 <font style=\"font-size:9px\">VllmWorker-N</font>",
             "cloudd")
    step_box(p, X + 3 * (W + GAP), y, W, 196, "④", "因为并行才多出来的",
             "+ <b>DP 协调进程</b>（DP&gt;1 且 rank0 时才有，负责把请求分给各 DP）<br>"
             "+ <b>sidecar/daemon 进程</b>（LWD host 数据面，每主机最多 1 个，"
             "先连后拉、flock 去重）<br>"
             "+ <b>proxy 进程</b>（PD 场景的入口，与引擎无关）", "data")

    # --- 四种实例形态 -----------------------------------------------------
    y2 = y + 240
    p.text(X, y2 - 32, 1200, 22,
           "<b>第二步：把这四个数代进本仓四种真实形态（可直接对照 ps）</b>", "text_b")
    rows = [
        ["edge（TP=1, DP=1）", "1（主进程自带）", "1", "0（uni，同进程）",
         "0", "1（daemon）", ("3", "cloudp")],
        ["sidecar 云端（TP=2, DP=1）", "1（主进程自带）", "1", "2（mp）", "0",
         "1（共享）", ("4", "cloudd")],
        ["dp2_tp1 云端（TP=1, DP=2）", "1 主 + 2 子 = 3", "2", "2（每核 1 个）", "1",
         "1（共享）", ("8", "fail")],
        ["PD proxy", "1", "0", "0", "0", "0", ("1", "neutral")],
    ]
    table(p, X, y2, [("实例形态", 250), ("API", 150), ("EngineCore", 110),
                     ("Worker", 150), ("DP 协调", 90), ("daemon", 110),
                     ("合计进程", 110)], rows, row_h=34)
    p.text(X, y2 + 34 * 5 + 6, 1424, 40,
           "<font style=\"font-size:9px\" color=\"#666666\">"
           "读法：edge 那行走的是 uni 后端——世界大小 TP×PP×DP=1，"
           "所以 worker 就在 EngineCore 进程里跑，ps 看不到 Worker 进程，"
           "这是本仓边端「看起来只有 3 个进程」的原因。"
           "dp2_tp1 那行 API 是 3 个：主进程 + 2 个子进程（SO_REUSEPORT 共听 8600）。</font>",
           "subtitle")

    # --- 一句口诀 ---------------------------------------------------------
    y3 = y2 + 34 * 5 + 66
    p.box(X, y3, 1424, 92,
          "<b>口诀（背这三句，看到 ps 就不会懵）</b><br>"
          "① <b>DP 决定「几个引擎」</b>：DP=2 就有 2 个 EngineCore 进程，各自独立调度。<br>"
          "② <b>TP/PP 决定「一个引擎里几个 worker 进程」</b>；只有 1 个 rank 时不另起进程。<br>"
          "③ <b>API server 个数默认跟着 DP 走</b>：DP=2 就 2 个 API 子进程抢同一个端口。",
          "box", "neutral")

    y4 = y3 + 126
    p.box(X, y4, 1424, 116,
          "<b>三个容易搞错的边界条件（都有代码依据）</b><br>"
          "① <b>PCP 也要乘进去</b>：一个 EngineCore 内的 worker 数 = TP × PP × PCP"
          "（PCP 是 prefill context parallel，一般都等于 1，所以平时忽略它）。<br>"
          "② <b>--enable-expert-parallel 不改变进程数</b>：EP 只是把专家分组从 TP 扩到 "
          "TP×PCP×DP，进程数不受影响——所以「开了 EP 进程没变多」不是 bug。<br>"
          "③ <b>还能把 EngineCore 也塞回前端进程</b>："
          "<font style=\"font-size:9px\">VLLM_ENABLE_V1_MULTIPROCESSING=0</font> "
          "时用进程内客户端（但 AsyncLLM / 在线服务不允许这么干）。",
          "box", "fail")

    y5 = y4 + 150
    concl(p, y5,
          "<b>一句话记住这一页：</b>进程数 = API(跟 DP) + EngineCore(跟 DP) + "
          "Worker(跟 TP×PP，=1 时同进程) + 协调/daemon/proxy；"
          "所以「我明明只写了 tp=2，怎么有 4 个进程」通常是把 coordinator 和 API 子进程忘了。")

    y6 = y5 + 92
    anchors(p, y6, [
        "API server 个数与多进程启动：vllm/entrypoints/cli/serve.py:118-135（默认 = DP）、"
        ":156-161（单 API）、:309（SO_REUSEPORT）",
        "EngineCore 进程管理：vllm/v1/engine/utils.py:1104（local_engine_count）、"
        ":1279-1291（每个 rank 一个 Process）、:241（进程名 EngineCore_DP{i}）",
        "DP 协调进程：vllm/v1/engine/utils.py:1178-1210、vllm/v1/engine/coordinator.py:111-124",
        "worker 进程数 world_size：vllm/config/parallel.py:982-1024；"
        "uni 判定 :1026-1027；mp spawn vllm/v1/executor/multiproc_executor.py:190-206",
    ], h=104)

    legend(p, y6 + 136, extra_fill=("数据面/daemon 相关", "data"))
