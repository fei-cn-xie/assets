"""⑦ 进程生命周期：谁在看着谁，挂了怎么被发现。"""
from common import head, concl, anchors, legend
from dg import Doc, table

TITLE = "⑦ 进程生命周期与死亡检测：四条独立路径"
SUB = ("一句话：进程死了**不靠超时轮询**去发现，而是靠四条「谁盯着谁」的独立机制；"
       "看清这四条，就知道为什么有时报 EngineDeadError、有时却只是「请求永远不返回」。")
X = 36


def build(doc: Doc) -> None:
    p = doc.page("⑦ 死亡检测", pw=1560, ph=1420)
    head(p, TITLE, SUB)

    # --- 四条路径 ---------------------------------------------------------
    y = 146
    p.text(X, y - 32, 1000, 22,
           "<b>四条路径：每条由一个「盯着谁」的角色负责</b>", "text_b")

    rows = [
        ["① 协议帧", "EngineCore 自己", "捕获到异常 → 往输出队列塞一帧 ENGINE_CORE_DEAD → 前端读到即抛 EngineDeadError",
         "有报错、能立刻知道", "core.py:1426-1432, 1692-1699"],
        ["② 进程句柄", "前端 MPClientEngineMonitor 线程", "阻塞等 EngineCore 进程句柄（sentinel）→ 返回即认为死了，置 engine_dead",
         "被 OOM/kill -9 时这是唯一信号", "core_client.py:774-801"],
        ["③ worker 回调", "EngineCore 的 MultiprocWorkerMonitor 线程",
         "worker 死了 → 回调 → 往 input_queue 塞 EXECUTOR_FAILED → 主线程 RuntimeError",
         "从 worker 死到引擎死有延迟", "multiproc_executor.py:300-333；core.py:1622-1623"],
        ["④ 前端看门狗", "frontend 的 watchdog_task（协程）", "每 5 秒查「引擎已报错且不再运行」→ 让 uvicorn 退出",
         "保证进程不僵尸化", "launcher.py:106-110, 199-209"],
    ]
    table(p, X, y, [("路径", 150), ("谁在盯", 300), ("怎么发现", 560),
                    ("特点", 240), ("锚点", 200)], rows, row_h=52)

    # --- 从死到收场的顺序 -------------------------------------------------
    y2 = y + 52 * 5 + 44
    p.text(X, y2 - 32, 1000, 22,
           "<b>一个 EngineCore 崩溃后，事情按这个顺序发生</b>", "text_b")
    seq = [
        ("EngineCore 主线程抛异常", "neutral"),
        ("发 ENGINE_CORE_DEAD 帧（最多等 output 线程 5 秒）", "cloudp"),
        ("进程退出", "neutral"),
        ("前端读到 DEAD 帧 → engine_dead=True", "cloudd"),
        ("在飞的请求全部抛 EngineDeadError", "fail"),
        ("watchdog 让 uvicorn 退出 → 整个实例结束", "fail"),
    ]
    cx = X
    for i, (txt, role) in enumerate(seq):
        w = 214
        p.box(cx, y2, w, 74, txt, "box_center", role)
        if i:
            p.edge(src_pt=(cx - 12, y2 + 37), dst_pt=(cx - 2, y2 + 37),
                   st="arrow", role="http")
        cx += w + 16

    # --- 超时表 -----------------------------------------------------------
    y3 = y2 + 116
    p.text(X, y3 - 32, 1000, 22,
           "<b>超时表：哪些地方会「等」，等多久</b>", "text_b")
    trows = [
        ["握手（等 READY 帧）", "5 分钟", "起不来就整体失败"],
        ["引擎就绪（VLLM_ENGINE_READY_TIMEOUT_S）", "600 s（本仓 env 改成 1800）",
         "等前端收到 READY 的上限"],
        ["单次 execute_model（VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS）", "300 s",
         "RPC 的 deadline：跑太久也算失败"],
        ["worker 关闭（VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS）", "5 s", "shutdown 时等子进程退出"],
        ["迭代超时（VLLM_ENGINE_ITERATION_TIMEOUT_S）", "定义是 60，但本分支没人读",
         "⚠ 名义存在、实际未生效"],
        ["请求级超时", "无", "前端 await 输出是无限期——卡住只能靠 py-spy"],
        ["LWD 边端链路（每条）", "605 s", "4 条串行，最坏 ~40 分钟"],
        ["LWD 请求 TTL（D 侧 sweep）", "request_ttl_s", "超时直接 fail-closed"],
    ]
    table(p, X, y3, [("等什么", 340), ("等多久", 400), ("说明", 660)], trows,
          row_h=32)

    # --- 正常退出 ---------------------------------------------------------
    y4 = y3 + 32 * 8 + 36
    p.box(X, y4, 1424, 132,
          "<b>正常退出：没有 pkill，靠三层收尾</b><br>"
          "① <b>信号</b>：入口脚本用 exec 把 shell 换成 python，Ctrl-C 直接打在进程组上，"
          "vLLM 自己处理。<br>"
          "② <b>顺序</b>：API server 子进程先关 → EngineCore 收 shutdown 命令"
          "（按 timeout 决定「直接 abort」还是「把在跑的请求排空」）→ worker 收到退出消息。<br>"
          "③ <b>兜底三段式</b>：先给所有进程 SIGTERM → 在共享的 deadline 内 join → "
          "还活着的直接 <b>kill_process_tree</b>（连子进程一起杀）。"
          "每个 manager 还挂了 weakref.finalize，异常退出也会尝试关。",
          "box", "neutral")

    y5 = y4 + 152
    anchors(p, y5, [
        "发 DEAD 帧并 join：vllm/v1/engine/core.py:1426-1432、:1692-1699、:1886-1891",
        "进程句柄监控：vllm/v1/engine/core_client.py:774-801；EngineDeadError :532-535",
        "worker 监控回调：vllm/v1/executor/multiproc_executor.py:300-333；"
        "EXECUTOR_FAILED core.py:1042-1044, 1622-1623",
        "前端看门狗与收尾：vllm/entrypoints/launchers/launcher.py:106-110, 138-209；"
        "vllm/v1/engine/utils.py:144-148（shutdown 顺序）",
    ], h=96)

    y6 = y5 + 128
    concl(p, y6,
          "<b>一句话记住这一页：</b>「引擎死了」有四条互不依赖的发现路径"
          "（协议帧 / 进程句柄 / worker 回调 / 前端看门狗）；"
          "而「引擎卡住但不死」没有任何路径能发现——那才是最难查的一类故障。")

    legend(p, y6 + 92)
