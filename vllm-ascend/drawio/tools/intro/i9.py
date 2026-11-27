"""⑨ 需求 → 组件：一页看懂 vLLM 为什么要长成这样。"""
from common import head, concl, anchors, legend, bridge
from dg import Doc, table

TITLE = "⑨ 需求 → 组件：vLLM 的每个部件都是在回答前 8 页的问题"
SUB = ("一句话：<b>没有无缘无故的组件</b>。这一页把「前 8 页的 8 个需求」"
       "与「vLLM 的 8 个部件」一一对上，并标出每个部件在哪套图里细讲。")


def build(doc: Doc) -> None:
    p = doc.page("⑨ 需求到组件", pw=1560, ph=1520)
    head(p, TITLE, SUB)
    X = 36

    y = 152
    p.text(X, y - 34, 1400, 22,
           "<b>第一步：一张映射表（左边是问题，右边是答案）</b>", "text_b")
    rows = [
        ["每一步要算谁、算几个 token？", "Scheduler（调度器）",
         "进程/线程 ⑪⑫"],
        ["KV 放哪、怎么借还、怎么复用？", "KVCacheManager + BlockPool + 块表",
         "生命周期 ⑤"],
        ["权重怎么进来、怎么参与计算？", "ModelLoader + ModelRunner",
         "进程/线程 ⑮⑯ & Transformer ⑤⑫"],
        ["多个请求怎么并行算？", "Executor（uni/mp/ray）+ Worker 进程",
         "进程/线程 ③④⑧"],
        ["前端和引擎怎么通信？", "ZMQ 三通道（握手/下行/上行）",
         "进程/线程 ⑤"],
        ["怎么把 decode 的提交开销压掉？", "CUDA/ACL Graph（图捕获）",
         "图捕获 全套"],
        ["显存不够了怎么办（跨机）？", "KV Connector（Mooncake）+ PD 分离",
         "Mooncake 全套 & 生命周期 ⑥"],
        ["一张卡装不下模型？", "TP/PP/DP/EP 并行",
         "本套 ⑧ & 进程/线程 ②③"],
    ]
    table(p, X, y, [("问题（前 8 页讲过的）", 520), ("vLLM 的答案", 520),
                    ("在哪套图里细讲", 384)], rows, row_h=40)

    # --- 一页全图 ---------------------------------------------------------
    y2 = y + 40 * 9 + 40
    p.text(X, y2 - 34, 1400, 22,
           "<b>第二步：把它们拼成一张图（这就是 vLLM 的骨架）</b>", "text_b")
    # 上：前端
    p.box(X + 380, y2, 660, 56,
          "<b>前端（API server）</b>：收 HTTP / 流式返回 / 分词反分词",
          "box", "neutral")
    # 中：引擎
    p.box(X + 380, y2 + 88, 660, 88,
          "<b>EngineCore 进程</b>：忙循环<br>"
          "<font style=\"font-size:9px\">Scheduler（每步挑一批）→ "
          "KVCacheManager（分块）→ Executor（下发）</font>",
          "box", "cloudp")
    # 右：图
    p.box(X + 1080, y2 + 88, 380, 88,
          "<b>图捕获</b><br>"
          "<font style=\"font-size:9px\">把 forward 录成图，回放时只提交一次</font>",
          "box", "cloudd")
    # 下：worker
    p.box(X + 380, y2 + 208, 660, 88,
          "<b>worker 进程（TP 个）</b>：真正跑 forward<br>"
          "<font style=\"font-size:9px\">权重 + KV 写入 + 前向 + 采样</font>",
          "box", "cloudd")
    # 左下：权重
    p.box(X, y2 + 208, 340, 88,
          "<b>权重文件</b><br>"
          "<font style=\"font-size:9px\">safetensors → 加载 → 显存常量</font>",
          "box", "neutral")
    # 右下：另一个实例
    p.box(X + 1080, y2 + 208, 380, 88,
          "<b>另一个实例 / 另一端</b><br>"
          "<font style=\"font-size:9px\">KV 通过 Mooncake 搬过去（PD 分离）</font>",
          "box", "data")

    p.edge(src_pt=(X + 710, y2 + 56), dst_pt=(X + 710, y2 + 88), st="arrow",
           role="http", label="ZMQ", label_pos=0)
    p.edge(src_pt=(X + 1040, y2 + 132), dst_pt=(X + 1080, y2 + 132), st="arrow",
           role="http")
    p.edge(src_pt=(X + 710, y2 + 176), dst_pt=(X + 710, y2 + 208), st="arrow",
           role="http", label="消息队列 RPC", label_pos=0)
    p.edge(src_pt=(X + 340, y2 + 252), dst_pt=(X + 380, y2 + 252), st="arrow",
           role="http")
    p.edge(src_pt=(X + 1040, y2 + 252), dst_pt=(X + 1080, y2 + 252), st="arrow",
           role="data")

    y3 = y2 + 320
    bridge(p, y3, "本页交接：图到这里就闭合了——<b>每个组件都对应前面一个具体问题</b>。"
                  "下一步（⑩）给出整套材料的学习顺序与自测题。")

    y4 = y3 + 76
    anchors(p, y4, [
        "前端与三通道：vllm/entrypoints/launchers/launcher.py:100-111；"
        "vllm/v1/engine/core_client.py:636-645",
        "EngineCore 忙循环与调度：vllm/v1/engine/core.py:1465-1476、"
        "vllm/v1/core/sched/scheduler.py:568",
        "Executor 与 worker：vllm/v1/executor/multiproc_executor.py:190-206、:1088-1118",
        "KV 管理：vllm/v1/core/kv_cache_manager.py:402、vllm/v1/core/block_pool.py:143",
        "图捕获：vllm/compilation/cuda_graph.py:315-321、:362",
        "KV 搬运：vllm-ascend/.../kv_p2p/mooncake_connector.py:1042（pull）",
    ], h=112)

    y5 = y4 + 146
    concl(p, y5,
          "<b>一句话记住这一页：</b>vLLM = 一个「每步重新组批」的调度器 + "
          "一套「按块借还」的 KV 管理 + 一个「把 forward 录成图」的执行器 + "
          "一条「能把 KV 搬到别的机器上」的通道；其余都是为这四件事服务的工程细节。")

    legend(p, y5 + 92, extra_fill=("跨机数据面（KV 搬运）", "data"))
