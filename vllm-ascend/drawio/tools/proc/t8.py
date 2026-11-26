"""⑧ worker 进程里还有什么线程（除了那条 RPC 循环）。"""
from common import head, concl, anchors, legend
from dg import Doc, table

TITLE = "⑧ worker 进程内部：一条阻塞循环 + 三种后台线程"
SUB = ("一句话：worker 主线程只做一件事——阻塞等 RPC、执行、回复；"
       "其它活儿（拷输出、盯父进程、搬 KV、通信库内部）都在后台线程。")
X = 36


def build(doc: Doc) -> None:
    p = doc.page("⑧ worker 线程", pw=1560, ph=1380)
    head(p, TITLE, SUB)

    # --- 主线程 -----------------------------------------------------------
    y = 148
    p.text(X, y - 32, 900, 22,
           "<b>主线程：worker_busy_loop（模型前向就发生在这里）</b>", "text_b")
    m = p.box(X, y, 700, 138,
              "<b>worker_busy_loop</b><br>"
              "<font style=\"font-size:10px\">"
              "阻塞在「从广播队列取 RPC」→ 取到就同步执行 `worker.&lt;method&gt;(...)` "
              "→ 把结果放回响应队列 → 继续等。<br>"
              "`execute_model` 也只是其中一种 RPC。</font><br><br>"
              "<font style=\"font-size:9px;color:#c0392b\">"
              "常见误解：以为 ModelRunner 有自己的线程池——<b>它没有</b>，"
              "它就是这条主线程里被调用的对象。</font>",
              "box", "cloudd")
    r = p.box(X + 760, y, 664, 138,
              "<b>为什么这里可以「阻塞等」</b><br>"
              "<font style=\"font-size:10px\">"
              "因为 worker 不需要同时做别的事：一次只有一个「执行计划」要跑，"
              "跑完回结果，等下一个。<br><br>"
              "对比前端：那里一条 loop 要伺候成千上万个连接，所以绝不能阻塞——"
              "<b>同一个「不能阻塞」的道理，在两处的解法完全相反。</b></font>",
              "box", "neutral")
    p.edge(m, r, "", st="arrow", role="http")

    # --- 后台线程 ---------------------------------------------------------
    y2 = y + 178
    p.text(X, y2 - 32, 1000, 22,
           "<b>后台线程：分三类（哪类出现取决于启动参数与连接器）</b>", "text_b")
    t1 = p.box(X, y2, 460, 168,
               "<b>① 框架自带</b><br><font style=\"font-size:10px\">"
               "· <b>WorkerAsyncOutputCopy</b>（仅 --async-scheduling）："
               "把「拷贝 worker 输出」拆出去，不和主循环抢<br>"
               "· <b>DeathPipeMonitor</b>：盯着父进程的死亡管道，父死就通知收尾<br>"
               "· <b>torch 线程池压到 1</b>：避免多进程互相抢 CPU（这是主动做的，不是限制）"
               "</font>", "box", "neutral")
    t2 = p.box(X + 482, y2, 460, 168,
               "<b>② KV 连接器自带</b><br><font style=\"font-size:10px\">"
               "以本仓 Mooncake 为例：<br>"
               "· KVCacheSendingThread（P 侧「挂号窗口」）<br>"
               "· KVCacheRecvingThread（D 侧搬运工）<br>"
               "· 它内部的 32 线程池（按对端分桶）<br>"
               "<b>这些线程在 register_kv_caches 之后启动并等 ready</b></font>",
               "box", "data")
    t3 = p.box(X + 964, y2, 460, 168,
               "<b>③ 通信库内部（HCCL/NCCL）</b><br><font style=\"font-size:10px\">"
               "集合通信、RDMA 收发、超时重试都在 C++ 层，"
               "Python 里看不到线程名。<br><br>"
               "这类线程出问题时的表现：<br>"
               "主线程卡在 allreduce / 进程 hang 住不退出——"
               "<b>日志里通常只有一行超时</b></font>", "box", "fail")

    # --- 线程表 -----------------------------------------------------------
    y3 = y2 + 208
    p.text(X, y3 - 32, 1000, 22, "<b>一张表：worker 进程里可能有哪些线程</b>",
           "text_b")
    rows = [
        ["MainThread", "worker_busy_loop：等 RPC → 执行 → 回结果", "总是有"],
        ["WorkerAsyncOutputCopy", "输出拷贝与主循环解耦", "仅 --async-scheduling"],
        ["DeathPipeMonitor", "盯父进程死亡管道", "总是有"],
        ["KVCacheSendingThread / RecvingThread", "KV 收发（Mooncake 连接器）", "用该连接器时"],
        ["32 线程池（连接器内部）", "按对端分桶的并行搬运", "用该连接器时"],
        ["lwd-ucx-send-* / 预收线程", "LWD host 数据面收发", "LWD 部署时"],
        ["通信库线程（C++）", "集合通信 / RDMA", "总是有，Python 不可见"],
    ]
    table(p, X, y3, [("线程", 420), ("干什么", 560), ("什么时候有", 360)], rows,
          row_h=32)

    y4 = y3 + 32 * 8 + 34
    anchors(p, y4, [
        "worker 主循环：vllm/v1/executor/multiproc_executor.py:1088-1118（RPC 循环）、"
        ":341-350（EngineCore 侧发 RPC）",
        "输出拷贝线程：同文件 :683-692、:1060-1086（仅 async-scheduling）",
        "父进程死亡监控：同文件 :828-851；torch 线程数 :1164-1184",
        "KV connector 线程：vllm-ascend/.../kv_p2p/mooncake_connector.py:256-270、"
        ":420-495、:2705-2762",
    ], h=96)

    y5 = y4 + 128
    concl(p, y5,
          "<b>一句话记住这一页：</b>worker = 一条「等 RPC → 执行 → 回话」的主线程"
          "+ 三类后台线程（框架自带 / 连接器 / 通信库）；"
          "所以 worker 的线程数会随「是否开 async-scheduling、用哪个 KV 连接器」而变，"
          "**同一个命令在不同配置下进程内线程数并不一样**。")

    legend(p, y5 + 92, extra_fill=("数据面 / 连接器", "data"))
