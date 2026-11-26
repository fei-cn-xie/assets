"""⑦ 一次 KV 交接的完整时序（pull 模型：D 主动去 P 拿）。

这一页是全套图的核心：把 mc-connector 给出的 14 个节点压成一条四段式时序。
读者看完应当能回答："谁在什么时候调谁、数据往哪个方向流、什么时候算完成"。
"""
from dg import Doc, legend_bar
from seq import Seq

TITLE = "⑦ 一次 KV 交接的完整时序：P 钉住 → D 主动拉 → 回执解钉"
SUB = ("四段：① P 算完并把块钉住　② D 拿到交接单先登记再派活　"
       "③ D 主动去 P 读（数据流），P 只回地址表　④ 回执与释放。"
       "读法：紫线＝控制面消息，黄线＝数据面（真正的 KV 搬运）。")


def build(doc: Doc) -> None:
    p = doc.page("⑦ PD 交接时序", pw=2200, ph=1400)
    s = Seq(p,
            lanes=[("pc", "P 机 · EngineCore<br>"
                          "<font style='font-size:9px'>调度器 + 引擎</font>", "neutral"),
                   ("pcn", "P 机 · connector<br>"
                           "<font style='font-size:9px'>挂号窗口（只回元数据）</font>", "cloudp"),
                   ("dcn", "D 机 · connector<br>"
                           "<font style='font-size:9px'>搬运工（32 线程池）</font>", "cloudd"),
                   ("dc", "D 机 · EngineCore<br>"
                          "<font style='font-size:9px'>调度器 + 引擎</font>", "neutral")],
            lane_w=440,
            title=TITLE,
            subtitle=SUB)

    s.phase("① P 侧：prompt 算完 → 交接清单 + 把块「钉住」")
    s.step("pc", "prompt 前向算完，KV 已经写进本机显存块",
           "本文件的 hook 不参与前向，只在前向之后接管", None, "neutral")
    s.msg("pc", "pcn", "request_finished(request, block_ids)", role="ctrl")
    s.step("pcn", "算出「真正含 prompt KV 的块」，返回延迟释放标记",
           "mooncake_connector.py:2016,2051-2061",
           "delay_free_blocks = 块数>0 → 打日志 "
           "Delaying free of %d blocks；这就是「钉住」，没有 pin API", "cloudp")
    s.msg("pcn", "pc", "kv_transfer_params（块号 / IP 端口 / PCP·DCP 规模）",
          mode="return", role="ctrl")
    s.note(("pc", "dc"), "这张「交接单」由 vLLM 核心随请求带去 D，connector 自己不做跨进程投递"
                         "（mooncake_connector.py:2063-2078）")

    s.phase("② D 侧：拿到交接单 → 登记 → 派活")
    s.step("dc", "先问自己：这批 KV 能省掉多少重复计算？",
           "mooncake_connector.py:1916-1956",
           "返回 (count, True)：True 表示「异步拉，别阻塞调度」", "neutral")
    s.step("dc", "update_state_after_alloc：登记要拉的本地块号",
           "mooncake_connector.py:1958-1985",
           "同时把 do_remote_prefill 置 False，保证只触发一次", "neutral")
    s.step("dc", "build_connector_meta：打包成给 worker 的工单",
           "mooncake_connector.py:1987-2014", None, "neutral")
    s.msg("dc", "dcn", "start_load_kv(metadata)", role="ctrl")
    s.step("dcn", "排班：切分片 → 挑要问的 P 端口 → 按 peer 分桶",
           "mooncake_connector.py:3835-3936, 3034, 3522",
           "32 线程池；每个 peer 每轮最多 5 个请求就换人（防独占）", "cloudd")

    s.phase("③ 真正搬：D 主动去 P 读（P 从不 push）")
    s.msg("dcn", "pcn", "GET_META_MSG：把你的「仓库地址表」给我", role="ctrl")
    s.msg("pcn", "dcn", "MooncakeAgentMetadata（每层基址 / 块长 / 块步长 / te 端口）",
          mode="return", role="ctrl",
          extra="mooncake_connector.py:1466-1500")
    s.step("dcn", "算搬运单：src / dst / length 三个等长列表",
           "mooncake_connector.py:877-1030",
           "三层循环（组 → 层 → 缓存），连续块会被合并成一段", "cloudd")
    s.msg("dcn", "pcn", "RDMA read：把 KV 从 P 的显存读到 D 的显存",
          role="data", dashed=True,
          extra="mooncake_connector.py:1042-1049 · batch_transfer_sync_read")
    s.step("dcn", "拼装：P 的 TP 比 D 大时，把分片 cat 回完整 head 维",
           "mooncake_connector.py:1083-1165", None, "cloudd")

    s.phase("④ 收尾：回执、解钉、失败兜底")
    s.msg("dcn", "pcn", "DONE_RECVING：我收完了", role="ctrl",
          extra="无论成功失败都发：不发会让对端内存泄漏")
    s.msg("pcn", "dcn", "ACK", mode="return", role="ctrl")
    s.step("pcn", "按端口累计回执，达到 D 声明的片数才算这一单完成",
           "mooncake_connector.py:373-397", None, "cloudp")
    s.step("pc", "get_finished() → done_sending → 真正释放那些块",
           "mooncake_connector.py:2767-2787, 213-224",
           "至此「钉住」解除，P 的显存才回来", "neutral")
    s.note(("pcn", "dcn"),
           "兜底：480 秒还没人来拿 → 强制释放并打 error（"
           "VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT，vllm/envs.py:260）", "fail")

    legend_bar(p, s._layout_rows(dry=True) + 44,
               extra_line=("虚线 = 数据流向与调用方向相反（D 读 P）", "data"))
