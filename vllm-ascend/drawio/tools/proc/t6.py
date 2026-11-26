"""⑥ 启动时序：从 bash 到「全部 ready」，谁在等谁。"""
from common import legend
from dg import Doc
from seq import Seq

TITLE = "⑥ 启动时序：从 bash 到全部 ready，谁在等谁"
SUB = ("四段：① 预检与算进程数　② 拉子进程　③ 引擎握手与 worker 起来　④ 就绪与开服。"
       "紫线＝控制/启动消息，黄线＝数据面。注意：<b>云端不等对端，边端才逐条等</b>。")


def build(doc: Doc) -> None:
    p = doc.page("⑥ 启动时序", pw=2280, ph=1560)
    s = Seq(p,
            lanes=[("sh", "bash 入口脚本<br>"
                          "<font style='font-size:9px'>env.sh + cloud_p.sh</font>", "neutral"),
                   ("la", "主进程 launcher<br>"
                          "<font style='font-size:9px'>算进程数 / 拉子进程</font>", "cloudp"),
                   ("api", "API server<br>"
                           "<font style='font-size:9px'>uvicorn（DP&gt;1 时是子进程）</font>", "neutral"),
                   ("ec", "EngineCore<br>"
                          "<font style='font-size:9px'>调度 + 忙循环</font>", "cloudp"),
                   ("wk", "worker<br>"
                          "<font style='font-size:9px'>VllmWorker-N（uni 时不存在）</font>", "cloudd")],
            lane_w=430,
            title=TITLE,
            subtitle=SUB)

    s.phase("① 预检与算进程数（失败就直接退，不是等）")
    s.step("sh", "先查运行库：缺 libucp.so.0 直接 exit 3", "env.sh:91-110",
           "预检放在最前，是为了避免「起了一半才报错」", "neutral")
    s.step("sh", "exec 接管：把 shell 替换成 python 进程", "env.sh:112-141",
           "所以 Ctrl-C 打在进程组上，vLLM 自己收尾（脚本里没有 pkill）", "neutral")
    s.step("la", "算出这次要起几个进程", "vllm/entrypoints/cli/serve.py:118-135",
           "api_server_count 默认 = DP；EngineCore = DP；worker = TP×PP（world_size=1 时 0）",
           "cloudp")

    s.phase("② 拉子进程（顺序有讲究：先绑 listener，再拉引擎）")
    s.step("la", "先在本进程里绑 coord HTTP listener（线程，不是进程）",
           "lwd_cloud_control.py:639-659",
           "云端会打印 cloud control HTTP ready on 0.0.0.0 ports=[8100, 8101]",
           "cloudp")
    s.step("api", "DP&gt;1：fork 出 N 个 API server 子进程，SO_REUSEPORT 抢同一端口",
           "vllm/entrypoints/cli/serve.py:156-161, 309",
           "DP=1 时没有这一步：主进程自己就是 API server", "neutral")
    s.step("la", "spawn EngineCore_DP0 … DPn-1（每分片一个进程）",
           "vllm/v1/engine/utils.py:1279-1291", None, "cloudp")
    s.msg("la", "ec", "handshake：ROUTER bind 好，等你连", role="ctrl")
    s.msg("ec", "la", "HELLO（我是 dp_rank=n）", mode="return", role="ctrl")
    s.msg("la", "ec", "地址表（input/output/DP 协调各自在哪）", mode="sync",
          role="ctrl", extra="EngineHandshakeMetadata")

    s.phase("③ 引擎握手完成 → 起 worker → 加载模型 → 分配 KV")
    s.msg("ec", "la", "READY（握手完成）", mode="return", role="ctrl",
          extra="超时 5 分钟；失败则实例起不来")
    s.steps([("ec", "mp 后端：spawn VllmWorker-0…N", "multiproc_executor.py:190-206",
              "uni 后端：这一步不存在，worker 就在本进程内", "cloudd", "box", True, None),
             ("wk", "init_device → 加载权重 → 注册 KV 显存",
              "gpu_worker.py · model_runner_v1.py:4684",
              "权重只加载到 worker（uni 时是 EngineCore）", "cloudd", "box", True, None)])
    s.note(("ec", "wk"), "worker 回 RPC 后，EngineCore 才认为「模型可用」；"
                         "此期间前端已经在听端口，但请求会排队", "neutral")

    s.phase("④ 就绪与开服（云端 vs 边端差别最大的一段）")
    s.step("api", "uvicorn 开始接请求（此时集群可能还没全好）", "launcher.py:100-111",
           "所以「服务能连上」≠「整条 PD 链路可用」", "neutral")
    s.step("la", "云端：报「listener ready，异步等 edge」就结束",
           "lwd_host_channels.py:364-374",
           "云端不阻塞，所以启动很快；日志里是 waiting for N edge link(s) asynchronously",
           "cloudp")
    s.step("la", "边端：逐条等链路 ready，605 秒/条 × 4 条",
           "host/ucx_io.py:63；lwd_host_channels.py:391-419",
           "最坏 ~40 分钟；失败日志 [LWD][handshake] host link not ready", "fail")

    legend(p, s._layout_rows(dry=True) + 44,
           extra_line=("紫=控制/启动消息；黄=数据面", "ctrl"))
