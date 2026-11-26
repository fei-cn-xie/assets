"""⑭ 一个请求在这个拓扑里的完整旅程（dp=2 × tp=4 的 PD 分离）。"""
from common import legend
from dg import Doc
from seq import Seq

TITLE = "⑭ 一个请求的完整旅程：从 edge 的 loop 到 8 张卡的 worker"
SUB = ("六段：① edge 收单发号　② Proxy 挑 P/D 的 DP　③ P 注册并派活　"
       "④ P 的 4 卡协同算 prefill　⑤ 交接单与 KV 一对一拉取　⑥ D 算 tail 并吐 token。"
       "紫线＝控制面/HTTP，黄线＝数据面。")


def build(doc: Doc) -> None:
    p = doc.page("⑭ 请求旅程 dp2tp4", pw=2400, ph=1900)
    s = Seq(p,
            lanes=[("cl", "客户端 + edge 前端<br>"
                          "<font style='font-size:9px'>asyncio loop / SSE</font>", "neutral"),
                   ("ee", "edge EngineCore<br>"
                          "<font style='font-size:9px'>PD runtime（1 卡）</font>", "cloudp"),
                   ("px", "PD Proxy<br>"
                          "<font style='font-size:9px'>uvicorn，挑 DP</font>", "proxy"),
                   ("pe", "P 实例：coord → EngineCore<br>"
                          "<font style='font-size:9px'>2 个 DP rank 各一个引擎</font>", "cloudp"),
                   ("pw", "P 的 4 个 TP worker<br>"
                          "<font style='font-size:9px'>卡 0-3 或 4-7</font>", "cloudd"),
                   ("de", "D 实例：coord → EngineCore → 4 worker<br>"
                          "<font style='font-size:9px'>拉 KV、算 tail、出 token</font>", "cloudd")],
            lane_w=380,
            title=TITLE,
            subtitle=SUB)

    s.phase("① edge 收单：分配 attempt，自己不路由")
    s.step("cl", "收 /v1/chat/completions，给这次尝试发一个 attempt 号",
           "pd/edge_runtime.py:97-99", "PD 场景必需：后续每一步都带 attempt 去重", "neutral")
    s.step("ee", "见 pd.enabled 就直接交给 PD runtime——不做 DP 打分",
           "lwd_edge_engine.py:363-371",
           "edge 只有 1 卡，它不算整模型，只做信封与汇聚", "cloudp")
    s.msg("ee", "px", "信封（不含原始 prompt）+ 流式 POST", role="http")

    s.phase("② Proxy：挑一个 P 的 DP、一个 D 的 DP")
    s.step("px", "按 (tenant,edge,rid,attempt) 去重，round-robin 选 target_dp",
           "pd/proxy.py:131-174（选择 :159-162）",
           "两个独立计数器：P 一个、D 一个 → (dp_P, dp_D) 是自由组合", "proxy")
    s.msg("px", "pe", "POST 到 P 的 coord 端口（带 target_dp、stage_role=prefill）",
          role="http", extra="max_tokens=1，剥掉 kv 参数")

    s.phase("③ P 注册：校验这个 DP 归我管 → 回 BIND")
    s.step("pe", "coord 线程校验 target_dp 属于本进程 → 投进该 DP 的命令队列",
           "lwd_cloud_control.py:615-634；cloud_http.py:41-85",
           "每个 DP 一个 coord_port，所以 dp=2 就要 2 个端口", "cloudp")
    s.step("pe", "engine 被唤醒取命令 → register（校验 dp_idx / TTL / handoff 计划）",
           "lwd_cloud_engine.py:865-891；cloud_runtime.py:138-273",
           "它把请求按 max_tokens=1 注册成「只算 prompt」的一次执行", "cloudp")
    s.msg("pe", "ee", "BIND：这个请求归我，绑定信息给你", mode="return", role="ctrl")
    s.step("ee", "记路由 → 让 P 开始算（claim / ADD 进 P 的 input_queue）",
           "edge_runtime.py:193-280；cloud_runtime.py:413-456", None, "cloudp")

    s.phase("④ P 算 prefill：一份计划广播给 4 张卡")
    s.step("pe", "调度器产出 SchedulerOutput（每请求几个 token + 用哪些块）",
           "core.py:589-619；scheduler.py:1471-1493",
           "注意：SchedulerOutput 里没有 slot_mapping", "cloudp")
    s.msg("pe", "pw", "广播同一份 SchedulerOutput（共享内存队列）", role="ctrl",
          extra="multiproc_executor.py:421")
    s.step("pw", "4 个 worker 各算自己 1/4：QKV 列并行 → attention → o_proj 行并行",
           "linear.py:1047-1053, 1775-1776",
           "行并行末尾要 all_reduce —— 这就是 4 卡的锁步点", "cloudd")
    s.step("pw", "只有 TP rank0 回采样结果；lm_head 出 logits 时做 all_gather",
           "multiproc_executor.py:542-556；logits_processor.py:124-133",
           "其他 3 个 rank 仍然全程参与集合通信，只是不回话", "cloudd")

    s.phase("⑤ 交接单 + KV 一对一拉取（P/D 的 TP 相等 → 恒等映射）")
    s.step("pe", "请求结束：连接器导出 remote_engine/host/port + 块号；引擎打 lwd_handoff",
           "mooncake_connector.py:2016-2070；cloud_runtime.py:697-717",
           "p_end / i_end / r_pos 都在交接单里（R = N-1）", "cloudp")
    s.msg("pe", "px", "P 的 HTTP 返回 kv_transfer_params（含交接单）", role="http")
    s.step("px", "拿交接单去注册 D 的某个 DP（此处才确定 dp_D）",
           "proxy.py:201-217；cloud_runtime.py:310-343", None, "proxy")
    s.msg("px", "de", "POST 到 D 的 coord（带 target_dp 与交接单）", role="http")
    s.step("de", "D 的调度器把请求置 WAITING_FOR_REMOTE_KVS，后台线程去拉",
           "pd/cloud_scheduler.py:106-149",
           "每个 TP worker 拉自己那 1/4 头（P worker r ⇄ D worker r）", "cloudd")
    s.msg("pw", "de", "块级 pull：4 条一对一通道各自搬 1/4 的 KV", role="data",
          dashed=True, extra="mooncake_connector.py:4028-4029, 4078-4087")
    s.step("de", "到齐断言 num_computed_tokens == r_pos，然后算 tail 并开始逐字生成",
           "pd/cloud_scheduler.py:127-130, 151-218", None, "cloudd")

    s.phase("⑥ 结果回 edge → 客户端")
    s.msg("de", "px", "D 一边算一边把结果事件推回", role="data", dashed=True)
    s.msg("px", "ee", "Proxy 回放事件给 edge", mode="async", role="http")
    s.step("ee", "edge 做 UNEMBED(lm_head) + tick() 推进状态机",
           "lwd_edge_down_receiver.py:40；lwd_edge_worker.py:157-166；"
           "edge_runtime.py:524-535", None, "cloudp")
    s.step("cl", "前端 output_handler 反分词 → SSE 逐 token 吐给客户端",
           "async_llm.py:841-908",
           "注意：分词/反分词一直在 edge 前端进程里，不在沿线的任何引擎里", "neutral")

    legend(p, s._layout_rows(dry=True) + 44,
           extra_line=("虚线 = 数据面/异步回推", "data"))
