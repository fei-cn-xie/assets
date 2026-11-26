"""P8 · 控制面协议与四类对象的状态机。"""
from dg import Doc, box_for, legend_bar, table


def _sm(page, x, y, title, subtitle, states, back_note, w=880):
    """Vertical state chain with labelled transitions."""
    page.text(x, y, w, 22, f"<b>{title}</b>", "text_b")
    page.text(x, y + 22, w, 18, subtitle, "subtitle")
    cy = y + 48
    ids = []
    ys = []
    for i, (name, anchor, role) in enumerate(states):
        cid, hh = box_for(page, x + 40, cy, w - 260, name, anchor or "", "",
                          role=role)
        ids.append(cid)
        ys.append(cy)
        cy += hh
        if i != len(states) - 1:
            cy += 34
    for i in range(len(ids) - 1):
        page.edge(ids[i], ids[i + 1], st="arrow", role="ctrl")
    mid = len(ids) // 2
    term = page.box(x + w - 195, ys[mid], 175, 44,
                    '<font style="font-size:10px">terminal=True</font>', "box_center",
                    "warn")
    page.edge(ids[mid], term, st="arrow_dash_open", role="warn", label="失败/取消/超时")
    page.text(x + w - 195, ys[mid] + 50, 190, 96, back_note, "subtitle")
    return cy


def build(doc: Doc) -> None:
    p = doc.page("08 控制面协议与状态机", pw=2000, ph=2300)
    p.text(36, 16, 1700, 26, "⑧ · 控制面协议（ZMQ）与四类对象的状态机", "title")
    p.text(36, 44, 1800, 20,
           "云每 DP bind 一个 ROUTER 到 ctrl_port；边每云 DP connect 一个 DEALER。"
           "帧 = [kind, body]，kind 只有 LINK(0x00 握手) / PAYLOAD(0x01 业务)。"
           "业务版本 LWD_MESSAGE_VERSION=7，PD 协议 LWD_PD_VERSION=lwd-pd-v2。", "subtitle")

    p.text(36, 88, 1500, 22,
           "<b>消息清单</b>（11 类走在本 PD 路径上；LwdPrefixReadyNotify / LwdRefillNotify 在 PD 下不适用）",
           "text_b")
    cols = [("消息", 250), ("方向", 170), ("关键字段", 620), ("请求时间轴位置", 420),
            ("定义", 380)]
    rows = [
        ["LwdLinkHello / Accept / Reject", "边→云 / 云→边",
         "link_version, message_version, instance_id, dp_rank, dp_size, session_id",
         "启动期（边阻塞等 605s/条 IO）", "lwd_link_protocol.py:62-90"],
        ["LwdCloudParamsNotify", "云→边",
         "max_num_batched_tokens, max_num_seqs, num_speculative_tokens, prefix_caching, "
         "hash_block_size, pd_version, stage_role", "握手后第一条；边启动等它",
         "lwd_messages.py:241-263"],
        ["LwdPdBindNotify", "云→边",
         "request_id, attempt_id, stage_role, binding_id, hit_tokens, p_end, i_end, r_pos, "
         "max_inflight_windows, recv_windows, source_p_binding", "P 登记后 / D 登记后各一条",
         "lwd_messages.py:315-335"],
        ["LwdRequestNotify", "边→云",
         "num_prompt_tokens, 采样 12 项, eos, block_hashes, prefix_nonce, mrope, image spans, "
         "attempt_id, binding_id", "收到 Bind 之后立刻；P/D 各一条", "lwd_messages.py:62-129"],
        ["LwdRangeNotify", "边→云",
         "seqno, items[request_id, offset, num_tokens, has_mrope, window_id]",
         "每个 EMBED 批一条（D 的 tail 窗口也是它）", "lwd_messages.py:30-59"],
        ["LwdPdQuotaNotify", "双向", "kind(\"output\"/\"recv\"), watermark",
         "边→云=输出行信用；云→边=输入窗口回收", "lwd_messages.py:346-351"],
        ["LwdC2eNotify", "云→边",
         "down_seqno, valid_ranges, row_counts, top_id_ths, num_accepted_tokens, "
         "final_g, finish_reasons, range_digests", "云每步 post_step；终结行 elements=0",
         "lwd_messages.py:285-312"],
        ["LwdPdStageDoneNotify", "云P→边", "export_id, ok, error", "P 导出完成或失败",
         "lwd_messages.py:337-343"],
        ["LwdPdResultCommit", "边→云D", "accepted_tokens(K), frontend_reason, verified_tokens",
         "前端 detokenize/stop 定案后（每 1s 重发）", "lwd_messages.py:354-360"],
        ["LwdPdResultReceipt", "云D→边", "result{usage…}", "D 合并 G/K 之后",
         "lwd_messages.py:363-367"],
        ["LwdAbortNotify", "边→云", "request_id, prefix_nonce, attempt_id, binding_id",
         "取消 / 断连 / 超时（幂等，凭证是终态回执）", "lwd_messages.py:213-220"],
    ]
    table(p, 36, 116, cols, rows, row_h=44, header_role="neutral")

    p.text(36, 700, 1700, 22,
           "<b>四类对象的生命周期</b>（字段名取自代码；任何状态叠加「错误 / 取消 / 超时」都收敛到 terminal）",
           "text_b")

    _sm(p, 36, 736, "① P 阶段记录 · LwdPdCloudStage(stage_role=prefill)",
        "pd/cloud_runtime.py:52-77 · register():162 · bind():367 · claim():395 · _terminal():540",
        [("无记录", "", "neutral"),
         ("已登记（阶段上下文，尚无执行 Request）", "register() · cloud_runtime.py:162", "cloudp"),
         ("已绑定：发 LwdPdBindNotify(binding_id)", "bind() · :367", "cloudp"),
         ("已认领 claimed=True → ADD 执行 Request",
          "claim() · :395（≤60s，pd.notify_claim_timeout_s）", "cloudp"),
         ("已导出：export + published，块被 pin", "on_outputs() · :600 → kv_transfer_params",
          "cloudp"),
         ("终态 terminal=True + pending_terminal", "_terminal() · :540", "fail")],
        "p_bind 30s / notify_claim 60s / 执行 1800s 到期 → _terminal(ERROR|ABORT)")

    _sm(p, 1010, 736, "② D 阶段记录 · LwdPdCloudStage(stage_role=decode)",
        "pd/cloud_runtime.py:52-77 · _validate_handoff():292 · cloud_scheduler.py:106-149",
        [("无记录", "", "neutral"),
         ("已登记（校验 handoff：version/rid/attempt/fingerprint/plan）",
          "_validate_handoff() · cloud_runtime.py:292", "cloudd"),
         ("已绑定：发 Bind(decode, r_pos, source_p_binding)", "bind() · :367", "cloudd"),
         ("已认领 → ADD；请求侧 WAITING_FOR_REMOTE_KVS", "claim() · :395 · scheduler.py:1276",
          "cloudd"),
         ("KV_READY：num_computed_tokens == r_pos → 联合就绪", "pd/cloud_scheduler.py:121-130",
          "cloudd"),
         ("G 固定：pending_terminal=(reason, G) → 发 C2e 终结行", "_terminal() · :540", "cloudd"),
         ("已发布 published：K 到齐 → usage + Receipt", "publish_result() · :809", "cloudd")],
        "KV 导入失败 / 抢占 / 边界不符 → fail-fast（不重算）")

    end3 = _sm(p, 36, 1460, "③ 边侧请求记录 · LwdPdEdgeState",
        "pd/edge_state.py:35-55 · bind():57 · freeze():118 · edge_runtime.py:410-439",
        [("added：信封已建，state 进 states[rid]", "LwdPdEdgeRuntime.add() · :90", "edge"),
         ("P 已绑定（bindings[\"prefill\"]，num_computed=hit）", "bind() · edge_state.py:57",
          "edge"),
         ("D 已绑定（bindings[\"decode\"]，tail_pending=True，建 output 流）",
          "bind() · edge_state.py:81-99", "edge"),
         ("执行完成 execution_done", "update_from_output 收尾", "edge"),
         ("前端冻结 K：frontend_commit=(K, reason, cancel)", "freeze() · edge_state.py:118",
          "edge"),
         ("回执 receipt + http_result 三方一致", "_result() · edge_runtime.py:492", "ok"),
         ("released：tick 过期清理", "release() · :462", "neutral")],
        "K 超过 verified_tokens / 冲突结果 → fail")

    end4 = _sm(p, 1010, 1460, "④ 物理帧记录 · frames[connection]",
        "pd/edge_runtime.py:54 · authorize_frame():337 · verify_row():385",
        [("DOWN 帧未到", "", "neutral"),
         ("C2e 到达：校验 binding 属 decode、seqno 连续无重放", "authorize_frame() · :337",
          "ctrl"),
         ("帧记录入 frames[conn] = (down_seqno, 编码后整帧)", "authorize_frame() · :337-383",
          "ctrl"),
         ("UNEMBED 消费：hidden 张量 → compute_logits → select_token_batch",
          "lwd_edge_model_runner.py:343-438", "edge"),
         ("行校验 verify_row：rank 复现与摘要一致", "verify_row() · :385", "edge"),
         ("出帧：更新 down_seqno，允许下一帧", "_lwd_deliver_unembed() · :539", "cloudd")],
        "seqno 跳号 / 重复 / 行几何不符 → 整请求 fail")

    legend_bar(p, max(end3, end4) + 40,
               extra_fill=("terminal = 终态", "fail"))
