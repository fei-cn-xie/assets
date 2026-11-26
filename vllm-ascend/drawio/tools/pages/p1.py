"""P1 · 部署形态与进程/线程全景。"""
from dg import Doc, box_for, legend_bar, measure_h, table


def build(doc: Doc) -> None:
    p = doc.page("01 部署与进程全景", pw=2100, ph=1420)
    p.text(36, 16, 1500, 26, "① · 部署形态与进程/线程全景（feat/pd-sep-lwd-v0.30.0）",
           "title")
    p.text(36, 44, 1600, 20,
           "拉起脚本：assets/c_home/bash/pds/30rc_1124/sidecar/{edge,cloud_p,cloud_d,start_proxy}.sh"
           " ｜ 拓扑：pd_host_topology.yaml（edges=1, clouds=2, data_plane_comm=host, deployment=sidecar）",
           "subtitle")

    # ---------------- 客户端 ----------------
    box_for(p, 40, 210, 130, "<b>客户端</b><br><font style='font-size:9px'>curl / SDK</font>",
            st="box_center", role="client", h=60)

    # ---------------- 边端主机 ----------------
    EDGE = [
        ("进程① API Server（vllm serve :8510）",
         "vllm/entrypoints/openai/api_server.py",
         "uvicorn + OpenAIServingChat + AsyncLLM 前端；唯一对用户开放的入口"),
        ("进程② EngineCore（LwdEdgeEngineCore）",
         "vllm/v1/lwd_control/control_edge_scheduler/lwd_edge_engine.py:95",
         "控制面 ROUTER（ctrl 6100/6101）· LwdPdEdgeRuntime · busy loop"),
        ("进程③ = ② 内的 driver_worker（LwdEdgeModelRunner）",
         "vllm-ascend/vllm_ascend/worker/lwd_edge_model/lwd_edge_worker.py:31",
         "NPU 4；真实 Embedding 计算 + UNEMBED 恢复 token。TP=1 → world_size=1 → "
         "LwdHostUniProcExecutor（vllm/config/vllm.py:1306-1316），worker 与引擎同进程"),
        ("进程④ host transport daemon（sidecar）",
         "vllm/v1/lwd_comm/data_comm/ucx_daemon.py",
         "每机一个；ctl socket /tmp/lwd-host-comm-&lt;uid&gt;.sock；memfd + SPSC ring"),
        ("线程清单",
         "process_input_sockets:1748 · process_output_sockets:1851 · run_busy_loop:1465",
         "前端：uvicorn 循环 + ZMQ 收包；引擎进程内：input socket 线程 · output socket 线程 · "
         "主 busy loop · lwd-pd-results 定时器线程"),
    ]
    GAP = 12
    eh = [measure_h(t, a or "", e or "", 424) for t, a, e in EDGE]
    edge_h = 30 + sum(eh) + GAP * (len(eh) - 1) + 16
    p.box(300, 120, 460, edge_h, "", "lane", "edge", decor=True)
    p.text(314, 126, 430, 22,
           "<b>边端主机 76.76.26.194</b>　NPU 4　TP=1　--enforce-eager", "text_b")
    cy = 156
    for (t, a, e), h in zip(EDGE, eh):
        box_for(p, 318, cy, 424, t, a, e, role="edge", h=h)
        cy += h + GAP

    # ---------------- PD Proxy ----------------
    box_for(p, 840, 250, 190,
            "<b>PD Proxy</b><br><font style='font-size:10px'>:.8102</font>",
            "vllm/v1/lwd_control/pd/proxy.py",
            "单进程，无 NPU / 无 UCX / 无模型。先选 P，拿到 kv_transfer_params，再选 D",
            role="proxy")

    # ---------------- 云端主机 ----------------
    P = [("前端进程：vllm serve :8600 ｜ LWD coord :8100 ｜ ctrl 6100",
          "vllm/v1/engine/utils.py:170-188 start_lwd_cloud_control_server()",
          "coord server 由前端进程在起 EngineCore 之前拉起，PD proxy 打的就是这个端口"),
         ("EngineCore（LwdCloudEngineCore）", "vllm/v1/engine/core.py:1372-1380",
          "控制面 DEALER → 边侧 · W1/W2 到达窗口 · 阶段上下文"),
         ("Worker ×2（TP=2，NPU 0,1）LwdCloudWorker",
          "vllm/v1/lwd_control/control_cloud_scheduler/lwd_cloud_engine.py:1170",
          "UP 接收 → chunk prefill → 内部采样 → 导出 KV（Mooncake producer 14579）"),
         ("host transport daemon（sidecar）· 与本机 D 共用",
          "vllm-ascend/.../lwd_data_plane/host/lwd_host_channels.py:252,340",
          "ctrl/data 端口 6100 / 7200")]
    D = [("前端进程：vllm serve :8601 ｜ LWD coord :8101 ｜ ctrl 6101",
          "vllm/v1/engine/utils.py:170-188",
          "FULL_DECODE_ONLY cudagraph；D 的 coord 端口是 proxy 的第二跳"),
         ("EngineCore（LwdCloudEngineCore）",
          "vllm/v1/lwd_control/pd/cloud_runtime.py:1",
          "G/K 关联 · 最终 usage 形成 · C2e 发布"),
         ("Worker ×2（TP=2，NPU 2,3）LwdCloudWorker",
          "vllm-ascend/vllm_ascend/worker/lwd_cloud_model/lwd_cloud_worker.py",
          "KV 导入（Mooncake consumer 14580）+ 尾输入 UP 接收 + DOWN hidden 发送"),
         ("host transport daemon（sidecar）· 与 P 共用同一进程",
          "vllm/v1/lwd_comm/data_comm/ucx_port.py:142-146",
          "ctrl/data 端口 6101 / 7201")]
    ph_ = [measure_h(t, a, e, 556) for t, a, e in P]
    dh_ = [measure_h(t, a, e, 556) for t, a, e in D]
    p_block = 30 + sum(ph_) + GAP * 3 + 16
    d_block = 30 + sum(dh_) + GAP * 3 + 16
    p_top, d_top = 156, 156 + p_block + 18
    p.box(1100, 120, 620, 26 + p_block + d_block + 34, "", "lane", "grey", decor=True)
    p.text(1114, 126, 590, 22, "<b>云端主机 76.76.26.195</b>　NPU 0,1（P）+ NPU 2,3（D）",
           "text_b")

    def panel(top, height, title, rows, heights, role):
        p.box(1116, top, 588, height, "", "lane", role, decor=True)
        p.text(1130, top + 4, 560, 20, f"<b>{title}</b>", "text_b")
        yy = top + 30
        for (t, a, e), h in zip(rows, heights):
            box_for(p, 1132, yy, 556, t, a, e, role=role, h=h)
            yy += h + GAP

    panel(p_top, p_block, "P 实例 · cloud 0 · stage_role=prefill", P, ph_, "cloudp")
    panel(d_top, d_block, "D 实例 · cloud 1 · stage_role=decode", D, dh_, "cloudd")

    # ---------------- 关键连线（编号与下方矩阵一一对应） ----------------
    p.edge(src_pt=(170, 240), dst_pt=(300, 240), label="① OpenAI HTTP :8510",
           st="arrow", role="data", label_pos=0)
    p.edge(src_pt=(760, 300), dst_pt=(840, 300), label="② 内部 HTTP", st="arrow",
           role="ctrl", label_pos=0)
    p.edge(src_pt=(1030, 280), dst_pt=(1132, p_top + 44), label="③", st="arrow",
           role="ctrl", label_pos=0)
    p.edge(src_pt=(1030, 344), dst_pt=(1132, d_top + 44), label="④", st="arrow",
           role="ctrl", label_pos=0)

    # ---------------- 通道矩阵 ----------------
    ty = max(d_top + d_block, cy) + 90
    p.text(36, ty - 30, 1400, 22,
           "<b>通道矩阵</b>（编号与上图箭头同源；⑤~⑨ 见下表，图中不画连线以免交叉）",
           "text_b")
    cols = [("编号", 56), ("源 → 目的", 300), ("协议 / 承载", 300), ("端口", 130),
            ("载荷与语义", 700)]
    rows = [
        [("①", "edge"), "客户端 → 边端 API Server", "HTTP/1.1，OpenAI 兼容 + SSE",
         "8510", "chat/completions 原文；文本 stop 与 detokenize 只在边侧"],
        [("②", "proxy"), "edge EngineCore → PD Proxy", "内部 HTTP POST + SSE 回读",
         "8102", "控制信封：rid/attempt/tenant/N/M/摘要链；不含原文与 token ID"],
        [("③", "cloudp"), "PD Proxy → P coord server", "HTTP（复用 proxy 原有转发）",
         "8100", "同 body；P 返回顶层 kv_transfer_params（交接扩展）"],
        [("④", "cloudd"), "PD Proxy → D coord server", "HTTP", "8101",
         "原 body + P 的 kv_transfer_params；D 回标准 usage / SSE"],
        [("⑤", "neutral"), "edge EngineCore ↔ P / D EngineCore", "ZMQ ROUTER/DEALER 控制面",
         "6100 / 6101", "BindNotify / RequestNotify / RangeNotify / C2e / 最终 K / 取消"],
        [("⑥", "neutral"), "edge Worker → P Worker", "UCX host 数据面（sidecar 代理）",
         "7200", "UP：prompt 窗口的 Embedding + 位置 + hash 路由等辅助输入"],
        [("⑦", "neutral"), "edge Worker ↔ D Worker", "UCX host 数据面",
         "7201", "UP：真实尾输入窗口；DOWN：hidden（输出恢复用）"],
        [("⑧", "neutral"), "P Worker → D Worker", "Mooncake KV 传输（RDMA/TCP）",
         "14579 / 14580", "KV blocks + conv/recurrent 等已声明支持的模型状态"],
        [("⑨", "neutral"), "同机 Worker ↔ sidecar daemon", "memfd + SPSC ring + unix socket",
         "/tmp/lwd-*.sock", "帧收发与 D2H/H2D 拷贝交接；daemon 每机一个"],
    ]
    bottom = table(p, 36, ty, cols, rows, row_h=40)

    legend_bar(p, bottom + 34)
