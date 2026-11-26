"""④ EngineCore 进程内部：1 个忙循环 + 2 个 IO 线程 + 1 个监控线程。"""
from common import head, concl, anchors, legend
from dg import Doc, table

TITLE = "④ EngineCore 进程内部：一个忙循环，两个 IO 线程"
SUB = ("这是整条链路里最关键的一个进程：调度、决策、下发都在这里。"
       "看明白它内部的 4 个线程，就明白「为什么 vLLM 能一边算一边收请求」。")
X = 36


def build(doc: Doc) -> None:
    p = doc.page("④ EngineCore 内部", pw=1560, ph=1400)
    head(p, TITLE, SUB)

    # --- 主线程 -----------------------------------------------------------
    y = 148
    p.text(X, y - 32, 900, 22, "<b>主线程：busy loop（唯一碰 NPU 的线程）</b>",
           "text_b")
    m = p.box(X, y, 700, 150,
              "<b>run_busy_loop：一直转，直到该退出</b><br>"
              "<font style=\"font-size:10px\">"
              "① <b>_process_input_queue()</b>：把排队中的命令取出来执行"
              "（ADD / ABORT / WAKEUP / UTILITY）<br>"
              "② <b>_maybe_publish_request_counts()</b>：把队列长度等统计发出去<br>"
              "③ <b>_process_engine_step()</b>：<b>跑一步</b>"
              "（调度 → 执行模型 → 收集输出）</font><br>"
              "<font style=\"font-size:9px\" color=\"#666666\">"
              "退出条件：处理到 shutdown 命令 → 返回 False → 抛 SystemExit</font>",
              "box", "cloudp")
    io = p.box(X + 760, y, 664, 150,
               "<b>为什么是「忙循环」而不是「事件驱动」</b><br>"
               "<font style=\"font-size:10px\">"
               "推理是<b>连续不停地算</b>：只要有请求在跑，每一步都不能等。<br>"
               "所以主线程不做任何可能阻塞的事——"
               "收包、发包、等待都被赶到了两个 IO 线程里。<br><br>"
               "代价：主线程空转时会吃满一个核（这是设计取舍，不是 bug）。</font>",
               "box", "neutral")
    p.edge(m, io, "", st="arrow", role="http")

    # --- 两个 IO 线程 -----------------------------------------------------
    y2 = y + 190
    p.text(X, y2 - 32, 900, 22,
           "<b>两个 IO 线程：存在的唯一理由就是「别让主线程等」</b>", "text_b")
    t_in = p.box(X, y2, 700, 160,
                 "<b>input 线程（daemon）</b><br><font style=\"font-size:10px\">"
                 "从 DEALER socket 收 multipart 帧 → 首帧是命令类型字节"
                 "（ADD / ABORT / UTILITY…）→ msgpack 解码 → "
                 "<b>塞进 input_queue</b> 就完事。</font><br><br>"
                 "<font style=\"font-size:9px\" color=\"#666666\">"
                 "主线程随后从 input_queue 里取。这样「网络收包」和「前向计算」并行。<br>"
                 "启动期如果这个线程先死，主线程会直接报 "
                 "Input socket thread died during startup。</font>",
                 "box", "cloudd")
    t_out = p.box(X + 760, y2, 664, 160,
                  "<b>output 线程（daemon）</b><br><font style=\"font-size:10px\">"
                  "从 output_queue 取 EngineCoreOutputs → msgpack 编码 → "
                  "PUSH 给前端（或多 API server 时给对应客户端）。</font><br><br>"
                  "<font style=\"font-size:9px\" color=\"#666666\">"
                  "编码后会复用 bytearray，用 ZeroMQ 的 MessageTracker 回收，避免反复申请内存。<br>"
                  "收到 ENGINE_CORE_DEAD 标记时：发完最后一帧就退出。</font>",
                  "box", "cloudd")

    # --- 监控线程 ---------------------------------------------------------
    y3 = y2 + 200
    p.text(X, y3 - 32, 900, 22,
           "<b>第 4 个线程：worker 进程监控（真线程，不是 Task）</b>", "text_b")
    mon = p.box(X, y3, 1424, 104,
                "<b>MultiprocWorkerMonitor</b>"
                "　<font style=\"font-size:9px\" color=\"#666666\">"
                "（mp 后端才有：uni 时没有子进程可监控）</font><br>"
                "<font style=\"font-size:10px\">"
                "它阻塞在「等 worker 进程句柄」，一旦某个 worker 死了 → 调回调 → "
                "往 input_queue 里塞一个 <b>EXECUTOR_FAILED</b> 命令 → "
                "主线程下一轮取到它就 <b>raise RuntimeError(\"Executor failed.\")</b> → "
                "进程 cd 收尾并发 ENGINE_CORE_DEAD 给前端。</font>",
                "box", "fail")

    # --- 线程表 -----------------------------------------------------------
    y4 = y3 + 144
    p.text(X, y4 - 32, 1200, 22,
           "<b>一张表收尾：EngineCore 进程里的线程清单</b>", "text_b")
    rows = [
        ["MainThread", "busy loop：取命令 → 跑一步 → 收输出", "唯一碰 NPU；一步不能停"],
        ["input 线程（无显式名）", "收前端命令，解包后放 input_queue", "让收包与计算重叠"],
        ["output 线程（无显式名）", "取输出、编码、PUSH 给前端", "让发包与计算重叠"],
        ["MultiprocWorkerMonitor", "监听 worker 进程句柄，死了塞 EXECUTOR_FAILED",
         "不轮询内存，靠内核句柄"],
        ["（LWD 才有）lwd-ctrl-io-*", "LWD 控制面收发与唤醒", "只在 PD 场景"],
    ]
    row_roles = ["neutral", "cloudd", "cloudd", "fail", "data"]
    table(p, X, y4, [("线程", 300), ("干什么", 520), ("为什么单独开", 380)],
          rows, row_h=36,
          role_matrix=lambda ri, ci: row_roles[ri] if ci == 0 else "neutral")

    y5 = y4 + 36 * 6 + 40
    anchors(p, y5, [
        "busy loop 与三个子步骤：vllm/v1/engine/core.py:1465-1476、:1491-1520、:1522-1539",
        "两个 IO 线程的创建与「为什么」的注释：core.py:1107-1134；"
        "收包循环 :1748-1849；发包循环 :1851-1909",
        "worker 监控：vllm/v1/executor/multiproc_executor.py:300-333；"
        "回调注册 core.py:1042-1044；EXECUTOR_FAILED 处理 :1622-1623",
        "LWD 线程：vllm/v1/lwd_control/lwd_comm/lwd_control_plane.py:282-287",
    ], h=96)

    y6 = y5 + 128
    concl(p, y6,
          "<b>一句话记住这一页：</b>EngineCore = 一个不能停的忙循环（主线程）+ "
          "两个只负责收发的 IO 线程 + 一个盯 worker 死活的监控线程；"
          "<b>它自己不碰权重，也不分词</b>，只做调度与下发。")

    legend(p, y6 + 92)
