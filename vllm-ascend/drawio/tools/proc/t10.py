"""⑩ 排障地图：进程/线程/端口/日志，先看哪里。"""
from common import head, concl, anchors, legend
from dg import Doc, table

TITLE = "⑩ 排障地图：进程、线程、端口、日志各自该看什么"
SUB = ("这一页是工具页：线上/本机出问题时，按「先看有几层进程 → 再看线程 → 再看端口 → 最后看日志」的顺序，"
       "能快速把问题定位到某一层。")


def build(doc: Doc) -> None:
    p = doc.page("⑩ 排障地图", pw=1560, ph=1460)
    head(p, TITLE, SUB)
    X = 36

    # --- 第一段：四条命令 -------------------------------------------------
    y = 146
    p.text(X, y - 32, 900, 22,
           "<b>第一步：用四条命令把「进程树 / 线程 / 端口」看清</b>", "text_b")
    p.box(X, y, 1424, 150,
          "<font style=\"font-size:9px\" color=\"#333333\">"
          "<b># 1. 有几个进程（Linux）</b><br>"
          "ps -ef | grep -E \"vllm|EngineCore|WorkerProc\" | grep -v grep<br>"
          "<b># 2. 进程树的父子关系（谁 spawn 了谁）</b><br>"
          "pstree -p &lt;pid&gt;　或　ps -o pid,ppid,cmd -e | grep vllm<br>"
          "<b># 3. 这个进程里有几个线程</b><br>"
          "top -H -p &lt;pid&gt;　或　cat /proc/&lt;pid&gt;/status | grep Threads　或　"
          "py-spy dump --pid &lt;pid&gt;（看每个线程卡在哪个 Python 栈）<br>"
          "<b># 4. 端口谁在听、谁在连</b><br>"
          "ss -lntp | grep -E \"8510|8600|8601|8102|14570|14580\"　（端口表见 ⑨ 页）"
          "</font>", "box", "neutral")

    # --- 第二段：日志锚点 -------------------------------------------------
    y2 = y + 190
    p.text(X, y2 - 32, 1000, 22,
           "<b>第二步：日志里对应的「阶段」关键词（按出现顺序）</b>", "text_b")
    rows = [
        ["前端起来了", "Starting vLLM API server / uvicorn running"],
        ["EngineCore 进程起来了", "Starting EngineCore / init engine"],
        ["worker 进程起来了", "WorkerProc / init_device / load_model"],
        ["握手完成（三通道）", "handshake / HELLO / ready"],
        ["KV cache 分配完成", "KV cache usage / num_gpu_blocks"],
        ["请求在跑", "Running: N reqs / Engine 000: Avg prompt throughput"],
        ["进程死了", "EngineDeadError / Engine core proc died / "
                     "WorkerProc 退出码 / was terminated"],
        ["等太久超时", "timeout / iteration timeout / Timeout waiting for"],
    ]
    table(p, X, y2, [("阶段", 260), ("日志里的样子（关键词）", 620)], rows,
          row_h=32)

    # --- 第三段：常见死法 -------------------------------------------------
    y3 = y2 + 32 * len(rows) + 56
    p.text(X, y3 - 32, 1000, 22,
           "<b>第三步：常见「死法」对照表（现象 → 先查哪一层）</b>", "text_b")
    cases = [
        ("前端报 EngineDeadError，进程还在", "EngineCore 进程死了",
         "先 ps 看 EngineCore 还在不在；再翻它退出前的日志（OOM / assert）", "fail"),
        ("卡在启动，永远不 ready", "worker 里有进程没起来或握手没完成",
         "看 worker 日志最后一行；ss 看端口有没有 listen；等的是哪个 ready", "fail"),
        ("请求全部排队不动", "线程池/后台线程卡住或 ZMQ 队列堵了",
         "py-spy dump 看 frontend 与 EngineCore 主线程栈；看输出通道是否积压", "fail"),
        ("显存慢慢涨到 OOM", "KV 块没释放（钉住没解）或进程泄漏",
         "看 KV cache usage 是否只涨不落；对端是否一直没回执", "fail"),
        ("吞吐突然掉一半", "某个 worker/DP 分片掉队",
         "top -H 看各进程 CPU；对比每个 worker 的日志时间戳", "cloudp"),
    ]
    cy = y3
    for sym, layer, first, role in cases:
        p.box(X, cy, 1424, 62,
              f"<b>{sym}</b><br>"
              f"<font style=\"font-size:10px\">"
              f"通常是：<b>{layer}</b>　→　先做：{first}</font>",
              "box", role)
        cy += 74

    # --- 第四段：速查公式 -------------------------------------------------
    y4 = cy + 16
    p.box(X, y4, 1424, 92,
          "<b>速查：这些数对不上就是配置和实际不一致</b><br>"
          "<font style=\"font-size:10px\">"
          "· worker 进程数（mp 模式）≈ tp × pp × （每个 EngineCore 的 DP 分片数）<br>"
          "· EngineCore 进程数 = DP 外部分片的数量（internal 模式由引擎自己起多份）<br>"
          "· 一个实例的通信端口数 ≈ 控制面 1 个基址 + 每 worker 1 个握手端口<br>"
          "· 线程数最多的进程通常是 worker（模型执行 + 通信库 + connector 后台线程）"
          "</font>", "box", "neutral")

    y5 = y4 + 126
    anchors(p, y5, [
        "进程/线程的钩子所在：vllm/v1/engine/core.py（EngineCoreProc / run_busy_loop）",
        "三通道与 worker 名册：vllm/v1/engine/utils.py（CoreEngineProcManager / "
        "ZMQ 地址与端口分配）",
        "worker 侧：vllm/v1/worker/worker_base.py、gpu_worker.py",
        "前端：vllm/entrypoints/openai/api_server.py、vllm/v1/engine/async_llm.py",
    ], h=90)

    y6 = y5 + 122
    concl(p, y6,
          "<b>一句话记住这一页：</b>先确认「有几层进程、各自活没活」，"
          "再看「线程卡在哪」，最后才看业务日志——"
          "顺序反了，就会在正确的日志里找错误的原因。")

    legend(p, y6 + 92)
