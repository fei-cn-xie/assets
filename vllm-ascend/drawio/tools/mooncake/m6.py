"""⑥ MooncakeConnector 里都有谁：一张"分层结构图"。

锚点全部来自 mc-connector 的走读结论（vllm-ascend 分支 fix/lwd-multidp-dense-pd）。
对外注册名 MooncakeConnectorV1，实现文件 kv_p2p/mooncake_connector.py。
"""
from dg import Doc, legend_bar

TITLE = "⑥ MooncakeConnector 里都有谁（一张图看清分层）"
SUB = ("一句话：vLLM 只在固定时刻调几个方法（上层），connector 里分「调度侧」和"
       "「worker 侧」，真正的搬运发生在后台线程 + 传输引擎（底层）。")
X = 36


def box(p, x, y, w, h, label, role, st="box"):
    return p.box(x, y, w, h, label, st, role)


def build(doc: Doc) -> None:
    p = doc.page("⑥ connector 结构", pw=1560, ph=1260)
    p.title_block(TITLE, SUB, x=X, y=20, w=1500)

    # --- 第 1 层：vLLM 核心 -------------------------------------------------
    y1 = 148
    p.text(X, y1 - 32, 800, 22,
           "<b>第 1 层 · vLLM 核心只认「钩子」</b>", "text_b")
    core_s = box(p, X, y1, 690, 92,
                 "<b>调度器 scheduler（决定谁跑、跑多少 token）</b><br>"
                 "它会在固定时刻调用 connector 的<b>调度侧方法</b>："
                 "问「能省多少」、登记块号、打包工单。<br>"
                 "<font style=\"font-size:9px\" color=\"#666666\">"
                 "对应方法名见下一页（⑤ 钩子表）</font>",
                 "neutral")
    core_w = box(p, X + 730, y1, 694, 92,
                 "<b>worker（真正跑模型的进程）</b><br>"
                 "它会在固定时刻调用 connector 的<b>worker 侧方法</b>："
                 "注册显存、下发工单、取「这一轮谁完成了」。<br>"
                 "<font style=\"font-size:9px\" color=\"#666666\">"
                 "worker 侧才有 NPU 句柄，所以搬运动作只能发生在这一侧</font>",
                 "neutral")

    # --- 第 2 层：connector 门面 -------------------------------------------
    y2 = y1 + 140
    p.text(X, y2 - 32, 900, 22,
           "<b>第 2 层 · MooncakeConnector（门面，按 role 分派）</b>", "text_b")
    facade = box(p, X, y2, 1424, 70,
                 "<b>MooncakeConnector</b>"
                 "　<font style=\"font-size:9px\" color=\"#666666\">"
                 "mooncake_connector.py:1608 · 对外注册名 MooncakeConnectorV1</font><br>"
                 "它自己不干活：把调用转给 scheduler 侧或 worker 侧的两套实现。",
                 "neutral")

    # --- 第 3 层：两侧 ------------------------------------------------------
    y3 = y2 + 116
    p.text(X, y3 - 32, 900, 22,
           "<b>第 3 层 · 两侧各管一段（这是读代码时最容易迷路的地方）</b>",
           "text_b")
    s_side = box(p, X, y3, 690, 214,
                 "<b>调度侧 MooncakeConnectorScheduler</b><br><br>"
                 "① <b>get_num_new_matched_tokens</b>：我能省掉多少重复计算？"
                 "→ 返回 (数量, True)<br>"
                 "② <b>update_state_after_alloc</b>：登记「我这些本地块要去远端拉」<br>"
                 "③ <b>build_connector_meta</b>：把这些登记打包成工单<br>"
                 "④ <b>request_finished</b>（P 侧）：算交接清单 + <b>钉住块</b>"
                 "（delay_free_blocks=True）<br><br>"
                 "<font style=\"font-size:9px\" color=\"#666666\">"
                 "锚点：:1916 / :1958 / :1987 / :2016</font>",
                 "cloudp")
    w_side = box(p, X + 730, y3, 694, 214,
                 "<b>worker 侧 MooncakeConnectorWorker</b><br><br>"
                 "① <b>register_kv_caches</b>：把 KV 显存注册给传输引擎，"
                 "并起后台线程<br>"
                 "② <b>start_load_kv</b>：D 侧入口，排班后把工单丢进队列<br>"
                 "③ <b>get_finished</b>：回报「谁发完了 / 谁收完了」，"
                 "交给核心去释放块<br><br>"
                 "<font style=\"font-size:9px\" color=\"#666666\">"
                 "锚点：:2627 / :3835 / :2767-2787</font>",
                 "cloudd")

    # --- 第 4 层：后台线程 + 传输引擎 ---------------------------------------
    y4 = y3 + 262
    p.text(X, y4 - 32, 1000, 22,
           "<b>第 4 层 · 后台线程与传输引擎（真正搬东西的人）</b>", "text_b")
    th_send = box(p, X, y4, 466, 150,
                  "<b>KVCacheSendingThread（P 侧的「挂号窗口」）</b><br>"
                  "名字叫 Sending，但它<b>不搬 KV</b>：<br>"
                  "只做两件事：① 回「仓库地址表」；② 收「我收完了」并回 ACK。<br>"
                  "<font style=\"font-size:9px\" color=\"#c0392b\">"
                  "名字误导：这是本套实现最容易被读错的地方</font>",
                  "cloudp")
    th_recv = box(p, X + 500, y4, 460, 150,
                  "<b>KVCacheRecvingThread（D 侧的「搬运工」）</b><br>"
                  "32 线程池；按「对端 host:port」分桶；每桶每轮最多处理 5 个请求"
                  "就换人（防止一个忙 P 独占）。<br>"
                  "所有 KV 数据都由它发起的 <b>RDMA read</b> 拉过来。",
                  "cloudd")
    te = box(p, X + 994, y4, 430, 150,
             "<b>global_te（TransferEngine 单例）</b><br>"
             "全进程一个：<br>"
             "<font style=\"font-size:9px\" color=\"#666666\">"
             "initialize(host, \"P2PHANDSHAKE\", \"ascend\", device)</font><br>"
             "向下直接调网卡/驱动（RDMA），向上只暴露"
             "「注册内存 / 读 / 写」三种动作。<br>"
             "<font style=\"font-size:9px\" color=\"#666666\">"
             "锚点：utils/mooncake_transfer_engine.py:4-43</font>",
             "data")

    # --- 竖线把分层串起来（用固定锚点，避免穿过中间的框）------------------
    p.edge(src_pt=(X + 345, y1 + 92), dst_pt=(X + 345, y2), st="arrow",
           role="http")
    p.edge(src_pt=(X + 1077, y1 + 92), dst_pt=(X + 1077, y2), st="arrow",
           role="http")
    p.edge(src_pt=(X + 345, y2 + 70), dst_pt=(X + 345, y3), st="arrow",
           role="http")
    p.edge(src_pt=(X + 1077, y2 + 70), dst_pt=(X + 1077, y3), st="arrow",
           role="http")
    p.edge(s_side, th_send, "", st="arrow", role="http")
    p.edge(w_side, th_recv, "", st="arrow", role="http")
    p.edge(th_recv, te, "", st="arrow", role="data")

    # --- 底部：容易踩的三个点 ----------------------------------------------
    y5 = y4 + 204
    p.box(X, y5, 1424, 104,
          "<b>读这套代码时最容易踩的三点</b><br>"
          "① <b>箭头方向 ≠ 数据方向</b>：P 侧线程只回元数据和 ACK，"
          "KV 数据是 D 主动<b>读</b>过去的（pull），P 从不 push。<br>"
          "② <b>不是逐层的</b>：<font style=\"font-size:9px\" color=\"#666666\">"
          "wait_for_layer_load / save_kv_layer / wait_for_save 三个钩子实现就是 pass"
          "（:1691-1703）</font>；逐层版在另一个文件 mooncake_layerwise_connector.py。<br>"
          "③ <b>没有 pin API</b>：「钉住块」是通过 request_finished 返回 "
          "delay_free_blocks=True 让核心延迟释放，不是 connector 自己加锁。",
          "box", "neutral")

    legend_bar(p, y5 + 130)
