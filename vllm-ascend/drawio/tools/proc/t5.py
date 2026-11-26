"""⑤ 三条 ZMQ 通道：握手、下行、上行。"""
from common import head, concl, anchors, legend
from dg import Doc, table

TITLE = "⑤ 前端与引擎之间的三条 ZMQ 通道（谁 bind、谁 connect）"
SUB = ("一句话：前端**统一 bind**、引擎**统一 connect**——所以前端可以先起、引擎后起也能接上。"
       "三条通道各管一件事：换地址（握手）、发命令（下行）、收结果（上行）。")
X = 36


def build(doc: Doc) -> None:
    p = doc.page("⑤ 三条 ZMQ 通道", pw=1560, ph=1380)
    head(p, TITLE, SUB)

    # --- 三条通道示意图 ---------------------------------------------------
    y = 150
    p.text(X, y - 32, 1000, 22,
           "<b>第一步：把三条通道画成三条箭头（注意 socket 类型与方向）</b>",
           "text_b")

    fe = p.box(X + 40, y, 380, 240,
               "<b>前端进程（API server）</b><br>"
               "<font style=\"font-size:10px\">"
               "① handshake：<b>ROUTER bind</b><br>"
               "② input：<b>ROUTER bind</b><br>"
               "③ output：<b>PULL bind</b></font><br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "三条全是 bind —— 前端是「服务端」角色</font>",
               "box", "neutral")
    ec = p.box(X + 980, y, 400, 240,
               "<b>EngineCore 进程</b><br>"
               "<font style=\"font-size:10px\">"
               "① handshake：<b>DEALER connect</b><br>"
               "② input：<b>DEALER connect</b><br>"
               "③ output：<b>PUSH connect</b></font><br>"
               "<font style=\"font-size:9px\" color=\"#666666\">"
               "identity = dp_rank，所以前端能定向到某个 DP 分片</font>",
               "box", "cloudp")

    for i, (label, role, dy) in enumerate([("① 握手：我是哪个 DP、我的地址是什么", "ctrl", 24),
                                           ("② 下行：ADD / ABORT / UTILITY", "ctrl", 100),
                                           ("③ 上行：EngineCoreOutputs", "data", 176)]):
        if i < 2:
            p.edge(src_pt=(X + 420, y + dy + 20), dst_pt=(X + 980, y + dy + 20),
                   label=label, st="arrow", role=role, label_pos=0)
        else:
            p.edge(src_pt=(X + 980, y + dy + 20), dst_pt=(X + 420, y + dy + 20),
                   label=label, st="arrow", role=role, label_pos=0)

    # --- 通道表 -----------------------------------------------------------
    y2 = y + 280
    p.text(X, y2 - 32, 1000, 22, "<b>第二步：三条通道的细节对照</b>", "text_b")
    rows = [
        ["① handshake", "ROUTER ↔ DEALER", "前端 bind / 引擎 connect",
         "引擎：HELLO → 前端：地址表 → 引擎：READY", "只有启动期用，5 分钟超时"],
        ["② input", "ROUTER ↔ DEALER", "前端 bind / 引擎 connect",
         "ADD（新请求）/ ABORT / UTILITY（查询）/ WAKEUP", "首帧是命令类型字节，后面是 msgpack 负载"],
        ["③ output", "PUSH ↔ PULL", "前端 bind / 引擎 connect",
         "EngineCoreOutputs（token、统计、utility 回包）", "单向「只推不收」，无需请求-响应配对"],
        ["旁路：DP 协调", "XPUB ↔ XSUB", "协调进程 XPUB / 引擎 XSUB connect",
         "各 DP 的队列长度与首波信号", "DP&gt;1 才有；用来做负载均衡"],
        ["旁路：关闭", "PAIR", "前端 connect / 引擎侧同步收", "空字节 b\"\"",
         "同步客户端用来叫醒引擎退出"],
    ]
    table(p, X, y2, [("通道", 170), ("socket 类型", 190), ("谁 bind / 谁 connect", 250),
                     ("承载什么", 400), ("备注", 310)], rows, row_h=40)

    # --- 为什么这么设计 ---------------------------------------------------
    y3 = y2 + 40 * 6 + 40
    p.box(X, y3, 1424, 118,
          "<b>为什么是这套设计（三个理由，都很实在）</b><br>"
          "① <b>前端 bind、引擎 connect</b>：前端先起来就能开服、挂住端口；"
          "引擎后起也能接上，还支持「引擎崩了重启再连」。<br>"
          "② <b>下行用 ROUTER→DEALER</b>：DEALER 的 identity 就是 dp_rank，"
          "于是多个 API server 子进程可以共享同一批 EngineCore，"
          "按 identity 精确路由到某个 DP 分片。<br>"
          "③ <b>上行用 PUSH→PULL</b>：引擎只管推，前端只管收，不需要请求-响应配对；"
          "配上「每次 step 的输出打成一个包」，天然就是批量传输。",
          "box", "neutral")

    # --- 背压与超时 -------------------------------------------------------
    y4 = y3 + 152
    p.box(X, y4, 1424, 132,
          "<b>两个反直觉的点（踩过才知道）</b><br>"
          "① <b>ZMQ 层不背压</b>：SNDHWM/RCVHWM 被设成 <b>0（无上限）</b>，"
          "所以「发送队列长」不会自动限流；真正的限流在前端准入"
          "（max_num_queued_reqs / max_num_queued_tokens，超了直接 503）。<br>"
          "② <b>没有请求级超时</b>：前端取输出是无限期 await。"
          "所以 EngineCore 卡住（不死也不发）时，请求会一直挂着——"
          "这类问题只能靠 py-spy 看线程栈，日志里没有报错。",
          "box", "fail")

    y5 = y4 + 166
    anchors(p, y5, [
        "握手：前端 bind vllm/v1/engine/utils.py:1261-1277；引擎 connect + HELLO "
        "vllm/v1/engine/core.py:1220-1227；READY :1234-1246",
        "下行：前端 ROUTER vllm/v1/engine/core_client.py:636-642；引擎收包 core.py:1805-1849",
        "上行：前端 PULL core_client.py:643-645；引擎 PUSH core.py:1869-1874",
        "HWM=0 与缓冲区设置：vllm/utils/network_utils.py:324-342；"
        "前端准入 async_llm.py:319-346",
    ], h=96)

    y6 = y5 + 128
    concl(p, y6,
          "<b>一句话记住这一页：</b>三条通道 = 换地址（握手，只用一次）+ 发命令（下行，"
          "ROUTER→DEALER，按 dp_rank 定向）+ 收结果（上行，PUSH→PULL，只推不收）；"
          "全由前端 bind、引擎 connect，所以顺序上「先起前端、后起引擎」也行。")

    legend(p, y6 + 92, extra_line=("虚线箭头 = 返回/异步", "ctrl"))
