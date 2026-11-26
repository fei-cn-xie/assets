"""P7 · 单请求端到端生命周期。"""
from dg import Doc, legend_bar
from seq import Seq


def build(doc: Doc) -> None:
    p = doc.page("07 单请求端到端", pw=2600, ph=2800)
    s = Seq(
        p,
        lanes=[("cl", "客户端", "client"),
               ("fe", "边前端<br><font style='font-size:9px'>API server</font>", "edge"),
               ("ee", "边引擎<br><font style='font-size:9px'>Engine+Sched+PdRuntime</font>", "edge"),
               ("px", "PD Proxy<br><font style='font-size:9px'>:8102</font>", "proxy"),
               ("cp", "云 P<br><font style='font-size:9px'>prefill</font>", "cloudp"),
               ("cd", "云 D<br><font style='font-size:9px'>decode</font>", "cloudd"),
               ("dp", "数据面 UP/DOWN<br><font style='font-size:9px'>UCX + Mooncake</font>", "data")],
        lane_w=330, title="⑦ · 一个请求的端到端生命周期（客户端 → 边 → Proxy → P → D → 边 → 客户端）",
        subtitle="八段。实线=HTTP/数据面，虚线=ZMQ 控制面。注意两件事：① 边侧一次请求只发一次"
                   "控制面信封给 Proxy，之后靠 ZMQ 与两朵云对话；② 文本 stop / detokenize / K 计数只在边侧。")

    s.phase("① 接入：tokenize 与身份生成（都在边侧，原文不出边）")
    s.msg("cl", "fe", "POST /v1/chat/completions（messages / max_tokens / stream）", mode="sync",
          role="data")
    s.step("fe", "模板渲染 + tokenize + 建 `EngineCoreRequest`",
           "vllm/entrypoints/openai/serving_chat.py · vllm/v1/engine/input_processor.py:288,429",
           "rid = `<外部id>-<8hex>`；lwd_pd_attempt_id = uuid4 注入 sampling_params.extra_args",
           role="edge")
    s.msg("fe", "ee", "ZMQ input：add_request", mode="sync", role="ctrl")
    s.step("ee", "`LwdEdgeEngineCore.add_request` → `LwdPdEdgeRuntime.add`",
           "lwd_edge_engine.py:322-332 · pd/edge_runtime.py:90",
           "PD 下 add_request 直接交给 PD runtime，不走 prefix probe 那条老路", role="edge")

    s.phase("② 信封与选路：HTTP 一路，控制面一路，并行推进")
    s.step("ee", "`build_envelope`：N/M + 采样参数 + 摘要链 + tenant + ttl",
           "pd/envelope.py:79-130 · lwd_prefix.py:354-421",
           "信封里**没有原文、没有 token id**；只有长度、摘要链与语义参数", role="edge")
    s.msg("ee", "px", "POST :8102（body.messages[0].content = 信封 JSON；Accept: SSE）",
          mode="sync", role="data")
    s.step("px", "`register`：解析信封 → 幂等键 (tenant,edge,rid,attempt) → 选 P",
           "pd/proxy.py:115-157,269-279", "命中缓存直接重放上次结果，不重跑阶段", role="proxy")
    s.msg("px", "cp", "POST :8100（stage_role=prefill, stream=false, max_tokens=1）",
          mode="sync", role="data")

    s.phase("③ P 登记 → Bind → 认领：请求在云端此刻才真正存在")
    s.step("cp", "coord 收到 → 命令队列 → 引擎线程 `register` → 发 Bind",
           "lwd_cloud_control.py:615-634 · pd/cloud_runtime.py:138-255,367-393",
           "登记只建阶段上下文，**还不建执行 Request**", role="cloudp")
    s.msg("cp", "ee", "ZMQ：`LwdPdBindNotify`(stage=prefill, hit/p_end/i_end/r_pos, 窗口额度)",
          mode="sync", role="ctrl")
    s.step("ee", "Bind(P)：num_computed=hit → `lwd_admit_request` → 发 RequestNotify",
           "pd/edge_runtime.py:236-260 · lwd_edge_scheduler.py:840-853",
           "边侧不选云：路由来自 Bind，而不是本地 picker", role="edge")
    s.msg("ee", "cp", "ZMQ：`LwdRequestNotify`（12 项采样 + block_hashes + mrope + spans）",
          mode="sync", role="ctrl")
    s.step("cp", "`claim`：校验 → 写参数 → `input_queue(ADD)`（内部采样、抑制用户输出）",
           "pd/cloud_runtime.py:395-438 · lwd_cloud_engine.py:416-434",
           "sampling 被改成 max_tokens=1/min_tokens=1/ignore_eos；输出数=1 只为触发导出",
           role="cloudp")

    s.phase("④ UP 分块 prefill：W1（到达）与 W2（原生窗口）双窗口")
    s.step("ee", "`_lwd_schedule_step`：按 upload_limit 选 prefill 批 → RangeNotify",
           "lwd_edge_scheduler.py:339-392,684-771,879-919",
           "窗口大小被云端给的接收信用夹住（max_inflight_windows=4）", role="edge")
    s.msg("ee", "cp", "ZMQ：`LwdRangeNotify`(seqno, offset/num_tokens/attempt/binding)",
          mode="sync", role="ctrl")
    s.msg("ee", "dp", "UP：`[embeds | hash-MoE | engram]` 单帧", mode="data", role="data")
    s.msg("dp", "cp", "D2H 落 pinned → cumulative arrival", mode="data", role="data")
    s.step("cp", "W1 由 arrival 推进；W2 限制本次 native prefill 窗口 → forward",
           "lwd_cloud_engine.py:963-1031 · lwd_cloud_scheduler.py:262-359",
           "`scheduled <= window` 是硬断言；跨连接用软配额轮转防 HOL 阻塞", role="cloudp")

    s.phase("⑤ P 导出 → Proxy 转交")
    s.step("cp", "末次 forward → 内部采样 → `request_finished` → 导出并 pin 块",
           "mooncake_connector.py:2014-2076",
           "P 的槽位结束，但导出状态与块引用继续由 P 管理", role="cloudp")
    s.msg("cp", "ee", "ZMQ：`LwdPdStageDoneNotify`(export_id)", mode="sync", role="ctrl")
    s.msg("cp", "px", "HTTP 200：顶层 `kv_transfer_params`（含 lwd_handoff）", mode="return",
          role="data")
    s.msg("px", "cd", "POST :8101（stream=true）+ 整块塞回 kv_transfer_params", mode="sync",
          role="data")

    s.phase("⑥ D 绑定 → KV 导入 ∥ 尾输入")
    s.step("cd", "`_validate_handoff` → 登记 → 发 Bind(decode)",
           "pd/cloud_runtime.py:162-255,292-325", "校验不过直接拒，不建 Request", role="cloudd")
    s.msg("cd", "ee", "ZMQ：`LwdPdBindNotify`(stage=decode, r_pos, source_p_binding)",
          mode="sync", role="ctrl")
    s.step("ee", "Bind(D)：迁移 account → 第二 RequestNotify → 输出配额",
           "pd/edge_runtime.py:261-303", "`LwdPdQuotaNotify(kind=output)` 给 D 发输出行信用",
           role="edge")
    s.msg("cd", "dp", "Mooncake：块级 pull（非 layerwise）", mode="data", role="data")
    s.step("cd", "`WAITING_FOR_REMOTE_KVS` → KV_READY（num_computed == r_pos）",
           "scheduler.py:1273-1303 · pd/cloud_scheduler.py:121-130",
           "此时 D 只有 1 行 tail 要自己算", role="cloudd")
    s.msg("ee", "dp", "tail 帧（1 行）+ RangeNotify", mode="data", role="data")
    s.step("cd", "联合就绪：kv_ready ∧ quota ∧ tail_ready → tail forward",
           "pd/cloud_scheduler.py:106-149", role="cloudd")

    s.phase("⑦ 有效生成 → DOWN → 边侧恢复 token")
    s.step("cd", "`sample_tokens`：`_lwd_collect_down_payload` → rank 编码 + valid 掩码",
           "lwd_cloud_model_runner.py:1352-1399,1408-1613 · lwd_token_rank.py:13",
           "只传「token 在 logits 里的名次」，边端用本地 logits 复现同一个 token", role="cloudd")
    s.msg("cd", "dp", "DOWN：hidden 帧", mode="data", role="data")
    s.msg("cd", "ee", "ZMQ：`LwdC2eNotify`(seqno / valid_ranges / ranks / final_g / finish)",
          mode="sync", role="ctrl")
    s.step("ee", "`_lwd_schedule_unembed` → authorize_frame → UNEMBED 批",
           "lwd_edge_scheduler.py:457-470,1221-1247 · pd/edge_runtime.py:337-383",
           "校验身份/几何/seqno 连续，防重放与乱序", role="edge")
    s.msg("dp", "ee", "hidden 张量到边 → compute_logits → select_token_batch", mode="data",
          role="data")
    s.step("ee", "`update_from_output` → `_lwd_deliver_unembed` → 追加 token / finish",
           "lwd_edge_scheduler.py:472-512,539-666",
           "先吃 lwd_up_send_watermarks（释放 pending batch），再交付有效行", role="edge")
    s.msg("ee", "fe", "EngineCoreOutputs", mode="sync", role="ctrl")

    s.phase("⑧ 输出与计量闭环（K 在边、G 在 D）")
    s.step("fe", "detokenize → 文本 stop 判定 → SSE 给客户端",
           "output_processor.py:726-777 · serving.py:450",
           "stop 与计数按 token ID 记，不按可见文本/SSE 帧数", role="edge")
    s.msg("fe", "cl", "SSE 增量内容", mode="return", role="data")
    s.step("fe", "冻结 K = detokenizer 输出 token 数 → `lwd_pd_finalize`",
           "output_processor.py:487-509 · lwd_frontend_result.py:105-137",
           "K 是「前端实际接纳的连续前缀」，正常完成时 K==G", role="edge")
    s.msg("ee", "cd", "ZMQ：`LwdPdResultCommit`(K, reason)（每 1s 重发直到回执）",
          mode="sync", role="ctrl")
    s.step("cd", "ledger.commit(K) → 合并 G/K → 形成 usage（N, K, N+K）",
           "pd/results.py:111-135 · pd/cloud_runtime.py:790-823",
           "校验 K≤G；complete ⇒ K==G；冲突即 fail", role="cloudd")
    s.msg("cd", "ee", "ZMQ：`LwdPdResultReceipt`（+ HTTP 侧 SSE data:[DONE]）", mode="return",
          role="ctrl")
    s.step("ee", "三方一致性校验（HTTP 结果 vs 回执）→ `runtime.release(rid)`",
           "pd/edge_runtime.py:492-522,462-467",
           "边侧不重新算 usage，只校验 D 给的结果与 HTTP 路一致", role="edge")
    s.msg("px", "ee", "SSE 收尾 → 边 `tick()` 取 done 队列", mode="return", role="data")

    # ---- 统一图例（全文件同义）----
    legend_bar(p, s._layout_rows(dry=True) + 44)
