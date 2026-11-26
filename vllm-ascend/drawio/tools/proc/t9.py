"""⑨ 本仓 LWD 部署的真实进程树（以 dp2_tp1_194_195 为例）。"""
from common import head, concl, anchors, legend
from dg import Doc, table

TITLE = "⑨ 本仓 LWD 部署的真实进程树：194 起 3 个，195 起 18 个"
SUB = ("以 assets/c_home/bash/pds/30rc_1124/dp2_tp1_194_195 为例（sidecar 形态的差异见页末）。"
       "这页就是「我实际 ps 出来会看到什么」。")
X = 36


def node(p, x, y, w, h, label, role, st="box"):
    return p.box(x, y, w, h, label, st, role)


def build(doc: Doc) -> None:
    p = doc.page("⑨ LWD 真实进程树", pw=1560, ph=1520)
    head(p, TITLE, SUB)

    # --- 左：194 ---
    y = 146
    p.text(X, y - 32, 700, 22,
           "<b>主机 194：边端（TP=1, DP=1 → 走 uni 后端）</b>", "text_b")
    e1 = node(p, X, y, 680, 76,
              "<b>① 主进程 = API server（uvicorn + asyncio）</b><br>"
              "<font style=\"font-size:9px\" color=\"#666666\">"
              "listen :8510　·　DP=1 所以不另起 API 子进程</font>",
              "cloudp")
    e2 = node(p, X + 40, y + 106, 600, 82,
              "<b>② EngineCore（同进程内跑 worker）</b><br>"
              "<font style=\"font-size:9px\" color=\"#666666\">"
              "进程名 EngineCore_DP0　·　TP×PP×DP=1 → uni，worker 不单起进程</font>",
              "cloudp")
    e3 = node(p, X + 40, y + 218, 600, 82,
              "<b>③ ucx_daemon（host 数据面 daemon）</b><br>"
              "<font style=\"font-size:9px\" color=\"#666666\">"
              "进程标题 lwd_host_comm · 每主机 1 个 · 首个建链引擎按需拉起</font>",
              "data")
    p.edge(e1, e2, "spawn", st="arrow", role="http")
    p.edge(e2, e3, "按需", st="arrow", role="data")
    p.text(X, y + 316, 680, 24,
           "<b>合计 3 个进程</b>（边端没有 Worker 进程，也没有 DP 协调进程）",
           "text_b")

    # --- 右：195 ---
    p.text(X + 740, y - 32, 700, 22,
           "<b>主机 195：云端两实例（TP=1, DP=2 → 走 mp 后端）+ proxy</b>",
           "text_b")
    c1 = node(p, X + 740, y, 684, 72,
              "<b>cloud0 = P（prefill）：8 个进程</b><br>"
              "<font style=\"font-size:9px\" color=\"#666666\">"
              "listen :8600（2 个 API 子进程 SO_REUSEPORT）· coord :8100/:8101 · "
              "ctrl :6100/:6101 · data :7200/:7201 · kv :14570/:14571</font>",
              "cloudp")
    c2 = node(p, X + 776, y + 96, 612, 120,
              "<b>主进程</b>（launcher：算进程数、拉子进程、绑 coord HTTP、绑 DP 握手）<br>"
              "+ <b>API server 子进程 ×2</b>（跟 DP 数走，抢 8600）<br>"
              "+ <b>EngineCore_DP0 / EngineCore_DP1</b>（每个 DP 分片一个）<br>"
              "+ <b>VllmWorker-0 ×2</b>（每个 EngineCore 里 1 个）<br>"
              "+ <b>VLLM_DP_Coordinator ×1</b>（DP&gt;1 才有）",
              "cloudp")
    c3 = node(p, X + 740, y + 236, 684, 72,
              "<b>cloud1 = D（decode）：同样 8 个进程</b><br>"
              "<font style=\"font-size:9px\" color=\"#666666\">"
              "listen :8601 · coord :8110/:8111 · ctrl :6110/:6111 · "
              "data :7210/:7211 · kv :14580（只连不绑）</font>",
              "cloudd")
    c4 = node(p, X + 740, y + 328, 684, 60,
              "<b>PD proxy：1 个进程</b>"
              "<font style=\"font-size:9px\" color=\"#666666\">"
              "　uvicorn :8102，扇出到 4 个 coord 端口；无 NPU、无 UCX、无模型</font>",
              "proxy")
    c5 = node(p, X + 740, y + 400, 684, 60,
              "<b>ucx_daemon：主机级 1 个</b>"
              "<font style=\"font-size:9px\" color=\"#666666\">"
              "　两个实例共用；ctl socket 在 /tmp/lwd-host-comm-&lt;uid&gt;.sock</font>",
              "data")
    p.text(X + 740, y + 468, 684, 24,
           "<b>合计 ≈ 18 个进程</b>（8 + 8 + proxy + daemon）", "text_b")

    # --- 端口表 -----------------------------------------------------------
    y2 = y + 520
    p.text(X, y2 - 32, 1200, 22,
           "<b>端口表（这台机器上谁在听、谁在连）</b>", "text_b")
    rows = [
        ["8510", "edge 主进程", "外部 client", "边端 OpenAI API 入口"],
        ["8600 / 8601", "两个实例的 API 子进程", "外部 client", "P / D 的 OpenAI API"],
        ["8100/8101 · 8110/8111", "各实例的 coord listener（主进程里的线程）",
         "PD proxy", "PD 注册 / 导出 / 结果（HTTP）"],
        ["6100/6101 · 6110/6111", "各 EngineCore（ZMQ ROUTER）", "edge 引擎", "控制面 ctrl"],
        ["7200/7201 · 7210/7211", "ucx_daemon（或 in_process 时引擎自己）", "edge 引擎",
         "数据面 UP/DOWN"],
        ["14570+dp / 14580+dp", "cloud0 各 EngineCore（producer）", "cloud1（consumer）",
         "Mooncake KV 握手"],
        ["8102", "PD proxy", "edge 引擎", "PD 统一入口（扇出到 4 个 coord）"],
        ["29600 / 29601", "launcher（ZMQ ROUTER）", "本机 EngineCore 子进程", "DP 引擎握手"],
        ["/tmp/lwd-host-comm-&lt;uid&gt;.sock", "ucx_daemon（AF_UNIX）", "各引擎进程",
         "daemon 控制面（ATTACH / 传 fd）"],
    ]
    table(p, X, y2, [("端口", 250), ("谁 listen", 380), ("谁 connect", 180),
                     ("用途", 400)], rows, row_h=32)

    # --- sidecar 差异 + 结论 ----------------------------------------------
    y3 = y2 + 32 * 10 + 40
    p.box(X, y3, 1424, 118,
          "<b>与 sidecar 形态的差别（同一台机器、同样的卡）</b><br>"
          "<font style=\"font-size:10px\">"
          "sidecar 包是 TP=2 / DP=1：<b>没有 API 子进程</b>（DP=1 → 主进程自己当 API server）、"
          "<b>没有 DP 协调进程</b>、<b>没有 296xx 握手端口</b>，"
          "worker 变成 2 个（TP=2）。所以云端每个实例只有 <b>4 个进程</b>，195 上合计 ≈ 10。<br>"
          "另外两个「会多/会少一个进程」的开关："
          "① 开 <b>EPLB</b> 时每个 worker 里会多一个 planner 子进程；"
          "② vllm-ascend 把 worker 的 daemon 标志改成了 <b>False</b>"
          "（为了让 worker 能再拉子进程），代价是父进程崩了 worker 不会自动被回收，"
          "残留必须靠 terminate/kill 链清。</font>",
          "box", "neutral")

    y4 = y3 + 152
    anchors(p, y4, [
        "入口脚本与顺序：assets/c_home/bash/pds/30rc_1124/dp2_tp1_194_195/"
        "cloud_p.sh:29-43、cloud_d.sh:24-39、start_proxy.sh:13-16、edge.sh:23-34",
        "API server 进程管理：vllm/v1/utils.py:167-224（APIServerProcessManager）",
        "EngineCore / 协调进程：vllm/v1/engine/utils.py:1178-1210、:1279-1291",
        "daemon 的 spawn 与去重：vllm/v1/lwd_comm/data_comm/ucx_port.py:173-242；"
        "空闲 30s 自退 ucx_daemon.py:67,389-399",
    ], h=96)

    y5 = y4 + 128
    concl(p, y5,
          "<b>一句话记住这一页：</b>本仓这套部署，边端 3 个进程、云端每个实例 8 个、"
          "proxy 与 daemon 各 1 个；<b>进程数最多的机器是云端</b>，"
          "而「最容易卡住不 ready 的」却是边端——因为它要逐条等 4 条链路。")

    legend(p, y5 + 92, extra_fill=("数据面/daemon", "data"))
