"""① 全景：一个 vllm serve 到底起了几个进程、进程里又有什么线程。"""
from common import head, concl, anchors, legend
from dg import Doc

TITLE = "① 全景：一个 `vllm serve` 里，进程和线程是怎么分层的"
SUB = ("一句话：<b>进程边界</b>由「谁能碰 NPU」决定，"
       "<b>线程边界</b>由「谁不能阻塞主循环」决定。"
       "看懂这两条，后面所有页都是在填细节。")


def build(doc: Doc) -> None:
    p = doc.page("① 进程线程全景", pw=1660, ph=1320)
    head(p, TITLE, SUB, w=1600)
    X = 36

    # --- 左列：三层进程 ---------------------------------------------------
    p.text(X, 138, 760, 22, "<b>进程层：父子关系（谁 spawn 了谁）</b>", "text_b")

    l1 = p.box(X, 170, 760, 96,
               "<b>主进程 = API server / frontend</b>"
               "　<font style=\"font-size:9px\" color=\"#666666\">"
               "vllm serve 敲下去后第一个活下来的进程</font><br>"
               "它对外提供 HTTP（OpenAI 兼容接口）和 SSE 流式输出；"
               "<b>分词/反分词都在这个进程里</b>，不在引擎里。<br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "过程：uvicorn 起服 → 构造 AsyncLLM → 构造时就把 EngineCore 拉起来</font>",
               "box", "neutral")
    l2 = p.box(X + 60, 296, 700, 104,
               "<b>EngineCore 进程 ×（每个 DP 分片一个）</b><br>"
               "它才是「引擎」：跑调度器 + 主忙循环，是<b>唯一允许碰 NPU 的 EngineCore 侧进程</b>。<br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "进程名 EngineCore / EngineCore_DP0…　·　与前端之间是 ZMQ 通道</font>",
               "box", "cloudp")
    l3 = p.box(X + 120, 430, 640, 104,
               "<b>worker 进程 ×（TP×PP 个）……或者根本没有</b><br>"
               "真正跑模型前向的进程。<b>TP×PP×DP==1 时不起新进程</b>"
               "（uni 后端，worker 与 EngineCore 同进程）。<br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "进程名 VllmWorker-N　·　与 EngineCore 之间是消息队列 RPC</font>",
               "box", "cloudd")
    l4 = p.box(X + 120, 564, 640, 76,
               "<b>（可选）外部进程</b>：DP 协调进程 / LWD 的 ucx_daemon / PD proxy<br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "它们不参与前向，但会影响「进程总数」和「端口表」——见 ② 和 ⑨</font>",
               "box", "data")
    p.edge(l1, l2, "构造 AsyncLLM 时 spawn", st="arrow", role="http")
    p.edge(l2, l3, "executor 决定要不要另起", st="arrow", role="http")
    p.edge(l3, l4, "", st="arrow_dash_open", role="data")

    # --- 右列：线程 -------------------------------------------------------
    p.text(X + 820, 138, 760, 22,
           "<b>线程层：每个进程内部都在忙什么</b>", "text_b")

    def thr(y, title, lines, role):
        p.box(X + 820, y, 764, 20 + 16 * len(lines),
              f"<b>{title}</b><br>"
              + "<br>".join(
                  f"<font style=\"font-size:9px\" color=\"#333333\">{ln}</font>"
                  for ln in lines),
              "box", role)

    thr(170, "frontend：1 条 asyncio 事件循环承所有 HTTP", [
        "主线程跑 uvicorn + FastAPI：每个请求都是这条 loop 上的一个协程",
        "output_handler（一个 Task）：唯一「输出泵」，取引擎输出 → 反分词 → 推给请求",
        "EngineCoreOutputQueueTask（一个 Task）：从 PULL socket 收引擎输出",
        "MPClientEngineMonitor（真线程）：阻塞等 EngineCore 进程句柄，死了立刻知道",
        "renderer 线程池（真线程）：把阻塞的分词/模板渲染甩出去，别卡 loop",
    ], "neutral")
    thr(330, "EngineCore：1 个忙循环 + 2 个 IO 线程 + 1 个监控", [
        "主线程 busy loop：取命令 → 跑一步 → 收集输出（唯一碰 NPU 的线程）",
        "input 线程：收前端发来的 ADD/ABORT（ZMQ DEALER）",
        "output 线程：把引擎输出推回前端（ZMQ PUSH）",
        "MultiprocWorkerMonitor（真线程）：监听 worker 进程句柄",
        "为什么拆 IO 线程：让「ZMQ 收发/序列化」和「NPU 前向」重叠，互不等待",
    ], "cloudp")
    thr(500, "worker：一条阻塞的 RPC 循环", [
        "主线程 worker_busy_loop：逐条执行 EngineCore 下发的 RPC（含 execute_model）",
        "ModelRunner 自己没有线程——它就是这条主线程里被调用的对象",
        "torch 线程池被压到 1：避免多进程抢 CPU 自旋",
        "KV connector 的后台线程（如 Mooncake 的发送/接收线程 + 32 线程池）",
        "通信库（HCCL/NCCL）自己的 C++ 线程：Python 侧看不到",
    ], "cloudd")

    # --- 底部：两条边界 ---------------------------------------------------
    y = 700
    p.box(X, y, 1600, 96,
          "<b>两条边界，记住就不会混</b><br>"
          "① <b>进程边界 = 谁能碰 NPU</b>：worker 进程才是真正调模型的；"
          "EngineCore 是调度 + 下发；frontend 只管 HTTP 和分词。"
          "所以「模型加载」发生在 worker（或 uni 时的 EngineCore），前端永远看不到权重。<br>"
          "② <b>线程边界 = 谁不能阻塞</b>：前端主线程是 asyncio loop（一个阻塞调用就能卡住全站），"
          "EngineCore 主线程是忙循环（一步都不能停），所以阻塞活儿全被赶到别的线程/别的进程。",
          "box", "neutral")

    y2 = y + 126
    anchors(p, y2, [
        "进程树：vllm/v1/engine/utils.py:238-244（Process 名 EngineCore/EngineCore_DPn）、"
        "vllm/config/parallel.py:982-1027（world_size 与 uni/mp 判定）",
        "前端线程：vllm/entrypoints/launchers/launcher.py:100-111（uvicorn loop）、"
        "vllm/v1/engine/async_llm.py:841-908（output_handler）、"
        "vllm/v1/engine/core_client.py:774-801（进程监控线程）",
        "EngineCore 线程：vllm/v1/engine/core.py:1107-1134（两个 IO 线程及注释）、"
        ":1465-1476（busy loop）",
        "worker：vllm/v1/executor/multiproc_executor.py:1088-1118（worker_busy_loop）",
    ], h=104)

    y3 = y2 + 136
    concl(p, y3,
          "<b>一句话记住这一页：</b>一个实例通常长这样——"
          "1 个前端进程（1 条 loop + 若干 Task + 1 个监控线程）→ N 个 EngineCore 进程"
          "（各 1 忙循环 + 2 IO 线程）→ M 个 worker 进程（各 1 条 RPC 循环 + 若干后台线程）；"
          "<b>其中 M 可以为 0（uni 后端，边端就是这样）</b>。")

    legend(p, y3 + 92, extra_fill=("数据面/旁路进程", "data"))
