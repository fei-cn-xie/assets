"""P6 · PD 分离下的 KV 交接时序。"""
from dg import Doc, legend_bar
from seq import Seq


def build(doc: Doc) -> None:
    p = doc.page("06 PD·KV 交接时序", pw=2400, ph=1900)
    s = Seq(
        p,
        lanes=[("px", "PD Proxy<br><font style='font-size:9px'>:8102</font>", "proxy"),
               ("p", "云 P<br><font style='font-size:9px'>prefill / producer</font>", "cloudp"),
               ("mk", "Mooncake 通道<br><font style='font-size:9px'>kv_port 14579/14580</font>",
                "data"),
               ("d", "云 D<br><font style='font-size:9px'>decode / consumer</font>", "cloudd"),
               ("e", "边侧 EngineCore", "edge")],
        lane_w=360, title="⑥ · PD 分离 · KV 交接时序（P 导出 → Proxy 转交 → D 导入 → 解 pin）",
        subtitle="时间自上而下。第 ③ 段的 KV 导入与第 ④ 段的「尾输入 UP」是并行的两条路，"
                   "D 必须两者都就绪才允许 forward。左三条要点：pin/引用、块级 pull、KV_READY 判定。")

    s.phase("① P 侧：跑完 prompt 就导出，但块先不释放")
    s.step("p", "P 收到 HTTP 登记 → Bind → RequestNotify 认领 → 分块 prefill",
           "pd/cloud_runtime.py:138-255 · lwd_cloud_scheduler.py:115-143",
           "输入是边端 UP 上来的 embedding（不是 token id）", role="cloudp")
    s.step("p", "末次 forward 后内部采样 a（max_tokens=1 → FINISHED_LENGTH_CAPPED）",
           "cloud_runtime.py:340 · vllm/v1/core/sched/utils.py:114",
           "a 不进用户输出、不进 D 历史、不占用户预算", role="cloudp")
    s.step("p", "`request_finished` → kv_transfer_params + `lwd_handoff`",
           "mooncake_connector.py:2014-2076 · pd/cloud_runtime.py:679-699",
           "关键字段：p_end / i_end / r_pos / hit_tokens / export_id", role="cloudp")
    s.step("p", "**pin 住块**：delay_free_blocks=True",
           "mooncake_connector.py:2056-2059 · scheduler.py:2648-2650",
           "调度器跳过 _free_blocks，块进 delayed_free_requests 账本", role="neutral")
    s.msg("p", "e", "ZMQ：LwdPdStageDoneNotify(export_id, ok)", mode="sync", role="ctrl")
    s.msg("p", "px", "HTTP 200 JSON：顶层 kv_transfer_params", mode="sync", role="data")

    s.phase("② Proxy：只做 HTTP 转交，不看内容")
    s.step("px", "选 P → 拿到 exported → 组 D 的 body（整块塞回 kv_transfer_params）",
           "pd/proxy.py:167-211",
           "P 侧剥掉、D 侧塞入；proxy 不参与 ZMQ、不进 NPU world", role="proxy")
    s.msg("px", "d", "POST :8101/v1/chat/completions（stream=true）", mode="sync", role="data")

    s.phase("③ D 侧：导入 KV（与 ④ 尾输入并行）")
    s.step("d", "`_validate_handoff`：version / rid / attempt / fingerprint / n_tokens / plan",
           "pd/cloud_runtime.py:292-325",
           "校验不过直接拒绝，不建执行 Request", role="cloudd")
    s.step("d", "`get_num_new_matched_tokens` → (i_end, True) → `_reqs_need_recv`",
           "mooncake_connector.py:1914-1983",
           "do_remote_prefill=True：告诉调度器「这段前缀由外部供给」", role="cloudd")
    s.step("d", "`allocate_slots(delay_cache_blocks=True)` → 状态 "
                "`WAITING_FOR_REMOTE_KVS`",
           "vllm/v1/core/sched/scheduler.py:1273-1303（:1276）",
           "本轮 forward token 数 = 0，请求不参与计费式调度", role="cloudd")
    s.msg("d", "mk", "start_load_kv → kv_recv_thread.add_request → 发起块级 pull", mode="data",
          role="data")
    s.msg("mk", "p", "RDMA / memcpy 读 KV blocks（+ conv/recurrent 状态组）", mode="data",
          role="data")
    s.step("mk", "读失败/部分 rank 失败 → 不置 KV_READY，汇总后回收引用",
           "pd/cloud_scheduler.py:118-130",
           "LWD 保持报错策略，不返回 `stop_reason=recomputed` 触发文本拼接重算", role="fail")
    s.step("d", "`get_finished` → promote → **KV_READY：num_computed_tokens == r_pos**",
           "scheduler.py:2983-3041 · pd/cloud_scheduler.py:121-130",
           "r_pos = N−1：D 只重算最后一行（普通 attention）；状态模型 N&gt;1 则 p_end=i_end=N−1",
           role="cloudd")

    s.phase("④ 并行：真实尾输入（1 帧 1 行）")
    s.msg("e", "d", "ZMQ：LwdRangeNotify（seqno, 1 行 tail）", mode="sync", role="ctrl")
    s.msg("e", "d", "UP：tail 帧（1 行 Embedding + 辅助输入）", mode="data", role="data")
    s.step("d", "硬校验「恰好最后一行」→ `_lwd_buffer_received_tails` → LwdTailBuffer.put",
           "lwd_cloud_worker.py:448-464 · :647-674 · lwd_tail_buffer.py:42",
           "D 只保留自己真正要算的那一行，绝不保留整段 UP 视图", role="cloudd")
    s.step("d", "**联合就绪**：kv_ready ∧ output_quota&gt;0 ∧ tail_ready ∧ ¬tail_inflight",
           "pd/cloud_scheduler.py:106-149",
           "四条件缺一不可；满足后扣配额、标 tail_inflight、安排 forward", role="cloudd")

    s.phase("⑤ 生成、回执与解 pin")
    s.step("d", "tail forward → 采样 → DOWN hidden（每步 1 帧）+ C2e 元数据",
           "lwd_cloud_model_runner.py:1351-1399 · lwd_cloud_worker.py:676-750",
           "rank 编码：只传 token 在 logits 里的名次，边端本地复现（lwd_token_rank.py:13）",
           role="cloudd")
    s.msg("d", "mk", "DONE_RECVING_MSG（本导出已读完）", mode="return", role="data")
    s.msg("mk", "p", "转发完成回执", mode="return", role="data")
    s.step("p", "`finished_sending` → `_free_blocks`（此时块才真正回到空闲队列）",
           "scheduler.py:3086-3089 · mooncake_connector.py:213-224",
           "超时兜底 `Force freed expired request`（:232-253）；D 掉线时会挂 pin 到超时",
           role="fail")
    s.step("d", "结果闭环：G（D 记账）与 K（边 detokenizer 记账）→ usage",
           "pd/results.py:94-135 · output_processor.py:505",
           "usage.completion_tokens = K；K≤G；complete ⇒ K==G", role="cloudd")

    s.gap(6)
    s.note(("p", "d"),
           "<b>为什么 P 要 pin</b>：KV 在 P 的块池里，D 的读是异步网络操作。"
           "若 P 先释放，块可能已被别的请求复写。所以 P 用 delay_free_blocks 把块从「可分配」"
           "降为「只读引用」，直到 D 显式回执才真正释放。")

    # ---- 统一图例（全文件同义）----
    legend_bar(p, s._layout_rows(dry=True) + 44)
