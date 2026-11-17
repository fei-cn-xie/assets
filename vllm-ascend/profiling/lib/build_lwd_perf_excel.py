from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import mean

from openpyxl import Workbook, load_workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

DATA = Path(os.environ["PD_EDGE_JSON"])
OUTPUT = Path(os.environ["PD_EDGE_BOOK"])
REQ = os.environ["PD_REQUEST_ID"]

BLUE = "1F4E78"
LIGHT_BLUE = "D9EAF7"
PALE_BLUE = "EAF3F8"
ORANGE = "F4B183"
PALE_ORANGE = "FCE4D6"
GREEN = "E2F0D9"
GRAY = "E7E6E6"
WHITE = "FFFFFF"
THIN = Side(style="thin", color="D9E1F2")


def pct(values, q):
    values = sorted(v for v in values if v is not None and v >= 0)
    if not values:
        return None
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def stats(values):
    vals = [v for v in values if v is not None and v >= 0]
    if not vals:
        return {}
    return {
        "count": len(vals),
        "mean": mean(vals),
        "p50": pct(vals, 0.5),
        "p90": pct(vals, 0.9),
        "p95": pct(vals, 0.95),
        "p99": pct(vals, 0.99),
        "max": max(vals),
        "sum": sum(vals),
    }


def base_sheet(wb, title, subtitle=None):
    ws = wb.create_sheet(title)
    ws.sheet_view.showGridLines = False
    ws.sheet_view.zoomScale = 85
    ws["A1"] = title
    ws["A1"].font = Font(name="Microsoft YaHei", size=16, bold=True, color=BLUE)
    if subtitle:
        ws["A2"] = subtitle
        ws["A2"].font = Font(name="Microsoft YaHei", size=10, color="666666")
        ws["A2"].alignment = Alignment(wrap_text=True)
    return ws


def write_table(ws, row, headers, rows, widths=None, mono_headers=()):
    for col, header in enumerate(headers, 1):
        cell = ws.cell(row, col, header)
        cell.font = Font(name="Microsoft YaHei", size=10, bold=True, color=WHITE)
        cell.fill = PatternFill("solid", fgColor=BLUE)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(bottom=THIN)
    for r_idx, values in enumerate(rows, row + 1):
        for c_idx, value in enumerate(values, 1):
            cell = ws.cell(r_idx, c_idx, value)
            cell.font = Font(name="Microsoft YaHei", size=9)
            cell.alignment = Alignment(vertical="center", wrap_text=c_idx in {2, 3, len(headers)})
            cell.border = Border(bottom=THIN)
            if headers[c_idx - 1] in mono_headers:
                cell.number_format = "0.000000"
            elif isinstance(value, float):
                cell.number_format = "0.00"
        if (r_idx - row) % 2 == 0:
            for c_idx in range(1, len(headers) + 1):
                ws.cell(r_idx, c_idx).fill = PatternFill("solid", fgColor=PALE_BLUE)
    if rows:
        ws.auto_filter.ref = f"A{row}:{get_column_letter(len(headers))}{row + len(rows)}"
    if widths:
        for idx, width in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(idx)].width = width
    else:
        for idx, header in enumerate(headers, 1):
            ws.column_dimensions[get_column_letter(idx)].width = min(max(len(str(header)) + 4, 12), 24)
    ws.freeze_panes = ws.cell(row + 1, 1)
    return row + len(rows) + 1


def section(ws, row, title, end_col=8):
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=end_col)
    cell = ws.cell(row, 1, title)
    cell.font = Font(name="Microsoft YaHei", size=11, bold=True, color=BLUE)
    cell.fill = PatternFill("solid", fgColor=LIGHT_BLUE)
    cell.alignment = Alignment(vertical="center")
    ws.row_dimensions[row].height = 22
    return row + 1


def add_kv_rows(ws, row, entries, note=""):
    for key, value, *rest in entries:
        ws.cell(row, 1, key)
        ws.cell(row, 2, value)
        ws.cell(row, 1).font = Font(name="Microsoft YaHei", size=10, bold=True)
        ws.cell(row, 2).font = Font(name="Microsoft YaHei", size=10)
        if isinstance(value, float):
            ws.cell(row, 2).number_format = "0.00"
        if rest:
            ws.cell(row, 3, rest[0])
            ws.cell(row, 3).font = Font(name="Microsoft YaHei", size=9, color="666666")
            ws.cell(row, 3).alignment = Alignment(wrap_text=True)
        row += 1
    if note:
        ws.cell(row, 1, note)
        ws.cell(row, 1).font = Font(name="Microsoft YaHei", size=9, italic=True, color="777777")
        row += 1
    return row


def main():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    assert data["request"] == REQ
    counts = data["counts"]
    assert counts["down_send"] == counts["down_recv"] == counts["unembed"] == counts["cloud_steps"]
    assert counts["accepted_sum"] > 0
    steps = data["decode_steps"]
    assert len(steps) == counts["down_send"]
    assert [s["seq"] for s in steps] == sorted(s["seq"] for s in steps)
    output_tokens = (
        data.get("request_done", {}).get("completion_tokens")
        if data.get("request_done")
        else None
    ) or counts["accepted_sum"]
    assert sum(s["accepted"] for s in steps) == output_tokens
    positive_steps = sum(1 for s in steps if s["accepted"] > 0)
    first_down_seq = steps[0]["seq"]
    meta = data.get("request_meta", {})
    decode_meta = data.get("decode_meta", {})
    stages = {s["stage"]: s for s in data["edge_stages"]}
    central = data["central_stages"]
    by_role_stage = {(c["role"], c["stage"]): c for c in central}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    wb.remove(wb.active)
    wb.properties.title = "边云 PD 请求分阶段耗时分析"
    wb.properties.subject = REQ
    wb.properties.creator = "DeepSeek Harness"

    # -------------------- 请求总览 --------------------
    ws = base_sheet(
        wb,
        "请求总览",
        "数据仅来自原始日志；elapsed/total 采用日志值，mono 不跨主机相减。",
    )
    ws.column_dimensions["A"].width = 34
    ws.column_dimensions["B"].width = 24
    ws.column_dimensions["C"].width = 72
    row = section(ws, 4, "1. 请求基本信息", 3)
    p_npu = max(e["npu_ms"] for e in data["up"] if e["kind"] == "model_forward" and e["file_role"] == "prefill")
    d_first_npu = steps[0]["forward_max_npu_ms"]
    decode_ms = stages["decode_generation"]["elapsed_ms"]
    finalize_ms = stages["edge_finalize"]["elapsed_ms"]
    done_mono = (
        data.get("request_done", {}).get("mono")
        if data.get("request_done")
        else None
    ) or stages["edge_finalize"]["mono"]
    e2e_ms = (done_mono - stages["request_accepted"]["mono"]) * 1000
    tpot = decode_ms / (output_tokens - 1) if output_tokens > 1 else None
    steady_forward_stats = stats([s["forward_max_npu_ms"] for s in steps[1:]])
    p_forward_events = [e for e in data["up"] if e["kind"] == "model_forward" and e["file_role"] == "prefill"]
    p_forward_detail = "，".join(f"{e.get('worker', 'rank')}={e['npu_ms']:.2f}" for e in p_forward_events)
    worker_exec_values = [e["total_ms"] for e in data["up"] if e["kind"] == "worker_exec" and e["total_ms"] is not None]
    worker_exec_max = max(worker_exec_values) if worker_exec_values else None
    transfer_values = [t["took_ms"] for t in data.get("transfers", [])]
    transfer_max = max(transfer_values) if transfer_values else None
    import_ready = data.get("import_ready") or {}
    info = [
        ("请求 ID", REQ, "只统计该请求，日志中其他请求已排除"),
        ("Prompt tokens N", meta.get("n", "-"), "请求原始 prompt 长度"),
        ("Prefix cache 命中 H", meta.get("h", "-"), "命中后减少实际上传/重算量"),
        ("实际上传 tokens", (meta.get("n") - meta.get("h")) if meta.get("n") is not None and meta.get("h") is not None else "-", "按请求元数据 N-H 计算"),
        ("输出 tokens G/K", output_tokens, "最终 accepted/delivered/usage"),
        ("P 绑定", meta.get("binding", "-"), "P export ID=e0-<request_id>"),
        ("D 绑定", decode_meta.get("binding", "-"), f"R={decode_meta.get('r', '-')}，D 尾窗 UP"),
        ("DOWN 批次", f"{len(steps)}（{steps[0]['seq']}–{steps[-1]['seq']}）", f"{positive_steps} 个有效生成批次 + 空行"),
        ("平均有效 tokens/批", output_tokens / positive_steps if positive_steps else None, "按有效生成批次计算"),
    ]
    row = add_kv_rows(ws, row, info) + 1

    row = section(ws, row, "2. 核心端到端指标", 3)
    metrics = [
        ("边云 TTFT（ms）", stages["d_to_edge_first_token"]["total_ms"], "request_accepted → 边侧恢复首 token"),
        ("P 绑定（ms）", stages["edge_to_p_bind"]["elapsed_ms"], "登记请求 → P 绑定"),
        ("P 上传（ms）", stages["p_upload"]["elapsed_ms"], "P 绑定 → 边侧输入全部派发"),
        ("P 计算/导出（ms）", stages["p_compute_export"]["elapsed_ms"], "最后上传 → P 完成通知"),
        ("P 阶段往返（ms）", stages["p_upload"]["elapsed_ms"] + stages["p_compute_export"]["elapsed_ms"], "p_upload + p_compute_export"),
        ("P→D 绑定（ms）", stages["p_to_d_bind"]["elapsed_ms"], "P 完成 → D 绑定"),
        ("D 尾窗派发（ms）", stages["d_tail_dispatch"]["elapsed_ms"], "D 绑定 → 尾窗全部派发"),
        ("D→边侧首 token（ms）", stages["d_to_edge_first_token"]["elapsed_ms"], "含 D 导入、首步 Decode、DOWN、UNEMBED"),
        ("持续 Decode（ms）", decode_ms, "首 token → D finished"),
        ("最终收尾（ms）", finalize_ms, "D finished → 边侧最终 usage"),
        ("请求总耗时（ms）", e2e_ms, "同一边侧 monotonic：request done - request accepted"),
        ("平均 TPOT（ms/token）", tpot, f"decode_generation / ({output_tokens}-1)"),
        ("Decode 输出吞吐（tokens/s）", output_tokens / (decode_ms / 1000), "仅按持续 Decode 区间计算"),
        ("全请求输出吞吐（tokens/s）", output_tokens / (e2e_ms / 1000), "按请求总耗时计算"),
    ]
    row = add_kv_rows(ws, row, metrics) + 1

    row = section(ws, row, "3. P/D 实例与关键 NPU 数据", 3)
    instance_metrics = [
        ("P accepted→first_scheduled（ms）", by_role_stage[("kv_producer", "first_scheduled")]["elapsed_ms"], "P 调度准入"),
        ("P first_scheduled→first_token（ms）", by_role_stage[("kv_producer", "first_token")]["elapsed_ms"], "P Prefill 主阶段"),
        ("P NPU forward 最大值（ms）", p_npu, f"{p_forward_detail}；批 tokens={p_forward_events[0]['tokens'] if p_forward_events else '-'}"),
        ("P worker exec 最大值（ms）", worker_exec_max, "worker_exec_done 最大值"),
        ("P kv_send_done（ms）", by_role_stage[("kv_producer", "kv_send_done")]["elapsed_ms"], "P 首 token 后观察到 KV 发送完成"),
        ("D kv_wait_start→kv_recv_done（ms）", by_role_stage[("kv_consumer", "kv_recv_done")]["elapsed_ms"], "D 远端 KV 等待"),
        ("Mooncake KV transfer（ms）", transfer_max, "来自实际 KV backend transfer 日志"),
        ("D import admit_to_ready（s）", import_ready.get("total_ms", 0) / 1000 if import_ready else None, "日志原值；与 kv_recv_done 口径不同"),
        ("D kv_recv_done→first_scheduled（ms）", by_role_stage[("kv_consumer", "first_scheduled")]["elapsed_ms"], "KV 就绪后的调度等待"),
        ("D first_scheduled→first_token（ms）", by_role_stage[("kv_consumer", "first_token")]["elapsed_ms"] - by_role_stage[("kv_consumer", "first_scheduled")]["elapsed_ms"], "D 首步调度/前向区间"),
        ("D accepted→first_token（ms）", by_role_stage[("kv_consumer", "first_token")]["total_ms"], "D 实例本地累计"),
        ("D 首步 NPU forward 最大值（ms）", d_first_npu, f"TP0={steps[0]['forward_tp0_npu_ms']:.2f}，TP1={steps[0]['forward_tp1_npu_ms']:.2f}；tokens=1"),
        ("首个 DOWN sidecar 发送（ms）", steps[0]["send"]["total_ms"], f"seqno={first_down_seq}"),
        ("首个 UNEMBED（ms）", steps[0]["unembed"]["total_ms"], f"seqno={first_down_seq}"),
    ]
    row = add_kv_rows(ws, row, instance_metrics) + 1

    row = section(ws, row, "4. 主要观察结论", 3)
    down_stats = data["stats"]
    conclusions = [
        ("TTFT 主要来源", "P 计算/导出", f"{stages['p_compute_export']['elapsed_ms']:.2f}ms，占边云 TTFT 的 {stages['p_compute_export']['elapsed_ms'] / stages['d_to_edge_first_token']['total_ms'] * 100:.2f}%"),
        ("首 token 第二来源", "D→边侧首 token", f"{stages['d_to_edge_first_token']['elapsed_ms']:.2f}ms，占边云 TTFT 的 {stages['d_to_edge_first_token']['elapsed_ms'] / stages['d_to_edge_first_token']['total_ms'] * 100:.2f}%"),
        ("稳态 Decode NPU", f"约 {steady_forward_stats['mean']:.2f}ms/步", f"{positive_steps} 个有效 Decode 步，TP0/TP1 取每步最大值"),
        ("Decode 归一化", f"约 {tpot:.2f}ms/token" if tpot is not None else "不可计算", f"MTP 平均约 {output_tokens / positive_steps:.2f} tokens/有效步，不能把 batch NPU 时间当成每 token 时间"),
        ("DOWN sidecar", f"平均 {down_stats['down_send_total_ms']['mean']:.2f}ms，P95={down_stats['down_send_total_ms']['p95']:.2f}ms", f"存在明显 D2H/network 尖峰，最大 total={down_stats['down_send_total_ms']['max']:.2f}ms"),
        ("边侧 UNEMBED", f"平均 {down_stats['unembed_total_ms']['mean']:.2f}ms，P95={down_stats['unembed_total_ms']['p95']:.2f}ms", "wait_tensor 和周期性 lm_head 尖峰需要单独分析"),
        ("Meta harvest", f"平均 {down_stats['meta_total_ms']['mean']:.2f}ms，P95={down_stats['meta_total_ms']['p95']:.2f}ms", "输出线程存在周期性等待"),
    ]
    row = add_kv_rows(ws, row, conclusions)

    # -------------------- 边云主阶段 --------------------
    ws = base_sheet(wb, "边云主阶段", "按边侧 request 状态机的实际日志顺序排列；mono 为边侧原始秒值。")
    headers = ["顺序", "阶段", "作用", "mono(s)", "elapsed_ms", "total_ms", "占 TTFT", "证据位置"]
    stage_explain = {
        "request_accepted": "边侧登记 PD 请求",
        "edge_to_p_bind": "选择并绑定 P 实例",
        "p_upload": "边侧输入窗口全部派发",
        "p_compute_export": "P 尾部计算、KV 导出及完成通知",
        "p_to_d_bind": "完成 D 实例绑定",
        "d_tail_dispatch": "D 尾窗全部派发",
        "d_to_edge_first_token": "D 导入、首步 Decode、DOWN、UNEMBED 到首 token",
        "decode_generation": "首 token 到 D 生成结束",
        "edge_finalize": "K 回报、D usage 形成和 HTTP 收尾",
    }
    rows = []
    for idx, stage in enumerate(data["edge_stages"], 1):
        total = stage["total_ms"]
        rows.append([
            idx,
            stage["stage"],
            stage_explain[stage["stage"]],
            stage["mono"],
            stage["elapsed_ms"],
            total,
            (total / stages["d_to_edge_first_token"]["total_ms"] if total is not None else None),
            f"edge.log:{stage['line']}",
        ])
    rows.append([
        len(rows) + 1,
        "request done",
        "请求清理完成，G/K/delivered=" + str(output_tokens),
        done_mono,
        e2e_ms - stages["edge_finalize"]["total_ms"],
        e2e_ms,
        e2e_ms / stages["d_to_edge_first_token"]["total_ms"],
        f"edge.log:{data.get('request_done', {}).get('line', stages['edge_finalize']['line'])}",
    ])
    write_table(ws, 4, headers, rows, [8, 28, 52, 18, 14, 14, 12, 18], {"mono(s)"})
    for r in range(5, 5 + len(rows)):
        ws.cell(r, 7).number_format = "0.00%"

    # -------------------- P/D 实例 --------------------
    ws = base_sheet(wb, "P与D实例", "集中式 PD 日志按 ts_wall 排序；mono 仅保留各实例原值，不跨进程相减。")
    headers = ["实例", "角色", "阶段", "ts_wall(s)", "mono(s)", "elapsed_ms", "total_ms", "证据位置"]
    role_name = {"kv_producer": "P", "kv_consumer": "D"}
    rows = []
    for event in sorted(central, key=lambda e: e["ts_wall"]):
        rows.append([
            role_name[event["role"]],
            event["role"],
            event["stage"],
            event["ts_wall"],
            event["mono"],
            event["elapsed_ms"],
            event["total_ms"],
            f"{event['file_role']}:{event['line']}",
        ])
    write_table(ws, 4, headers, rows, [10, 18, 24, 20, 18, 14, 14, 20], {"ts_wall(s)", "mono(s)"})

    # -------------------- UP 与首 token --------------------
    ws = base_sheet(wb, "UP与首token", f"UP seq={sorted({e['seq'] for e in data['up']})}；DOWN seq={first_down_seq} 为首个输出。")
    headers = [
        "UP/DOWN", "seq", "组件", "角色", "mono(s)", "total_ms", "queue_ms", "clone_ms",
        "d2h_ms", "serialize_ms", "network_ms", "rx_wait_ms", "h2d_ms", "npu_ms", "补充说明", "证据位置",
    ]
    rows = []
    up = sorted(data["up"], key=lambda e: (e["seq"], e["file_role"], e["line"]))
    for e in up:
        kind_label = {
            "embed": "边侧 EMBED",
            "sidecar": "sidecar 数据面",
            "frame": "frame_host_ready",
            "model_forward": "NPU forward",
            "worker_exec": "Worker 执行",
            "gate_ready": "UP gate 放行",
        }[e["kind"]]
        rows.append([
            "UP",
            e["seq"],
            kind_label,
            f"{e.get('role', e.get('worker', e['file_role']))}/{e['op'] if e['kind'] == 'sidecar' else ''}",
            e["mono"],
            e.get("total_ms"),
            e.get("queue_ms"),
            e.get("clone_ms"),
            e.get("d2h_ms"),
            e.get("serialize_ms"),
            e.get("network_ms"),
            e.get("rx_wait_ms"),
            e.get("h2d_ms"),
            e.get("npu_ms"),
            f"tokens={e.get('tokens', '')}; forward={e.get('forward_ms', '')}; submit={e.get('submit_send_ms', '')}",
            f"{e['file_role']}:{e['line']}",
        ])
    first = steps[0]
    for label, e in (("cloud_send", first["send"]), ("edge_recv", first["recv"]), ("edge_unembed", first["unembed"])):
        rows.append([
            "DOWN",
            first_down_seq,
            label,
            "cloud→edge" if label == "cloud_send" else "edge",
            e["mono"],
            e["total_ms"],
            e.get("queue_ms"),
            e.get("clone_ms"),
            e.get("d2h_ms"),
            e.get("serialize_ms"),
            e.get("network_ms"),
            e.get("rx_wait_ms"),
            e.get("h2d_ms"),
            None,
            f"wait_tensor={e.get('wait_tensor', '')}; lm_head={e.get('lm_head', '')}; select={e.get('select', '')}",
            f"{e['file_role']}:{e['line']}",
        ])
    write_table(ws, 4, headers, rows, [10, 8, 20, 20, 18, 12, 12, 12, 12, 14, 14, 14, 12, 12, 40, 20], {"mono(s)"})

    # -------------------- Decode 逐步 --------------------
    ws = base_sheet(wb, "Decode逐步", "每行一个 DOWN batch；D 与 edge 的 mono 分列保留，不能跨实例直接相减。")
    headers = [
        "seq", "accepted", "累计accepted", "D_forward_TP0_ms", "D_forward_TP1_ms", "D_forward_max_ms",
        "D_sample_ms", "D_step_send_ms", "D_step_total_ms", "D_collect_max_ms", "D_meta_ms",
        "send_queue_ms", "send_clone_ms", "send_d2h_ms", "send_serialize_ms", "send_network_ms", "send_total_ms",
        "recv_queue_ms", "recv_rx_wait_ms", "recv_h2d_ms", "recv_total_ms",
        "edge_post_recv_ms", "edge_wait_tensor_ms", "edge_lm_head_ms", "edge_select_ms", "edge_unembed_total_ms",
        "edge_interval_ms", "interval/token_ms", "D_forward_mono", "D_step_mono", "send_mono", "recv_mono", "edge_unembed_mono", "证据行",
    ]
    rows = []
    cumulative = 0
    for s in steps:
        cumulative += s["accepted"]
        send, recv, un = s["send"], s["recv"], s["unembed"]
        rows.append([
            s["seq"], s["accepted"], cumulative,
            s["forward_tp0_npu_ms"], s["forward_tp1_npu_ms"], s["forward_max_npu_ms"],
            s["cloud_step_sample_ms"], s["cloud_step_send_ms"], s["cloud_step_total_ms"], s["collect_max_ms"], s["meta_total_ms"],
            send["queue_ms"], send["clone_ms"], send["d2h_ms"], send["serialize_ms"], send["network_ms"], send["total_ms"],
            recv["queue_ms"], recv["rx_wait_ms"], recv["h2d_ms"], recv["total_ms"],
            un["post_recv"], un["wait_tensor"], un["lm_head"], un["select"], un["total_ms"],
            s["unembed_interval_ms"], s["interval_per_accepted_token_ms"],
            s["forward_mono_max"], s["cloud_step_mono"], send["mono"], recv["mono"], un["mono"],
            f"D:{send['line']}; edge:{un['line']}",
        ])
    write_table(ws, 4, headers, rows, [8, 10, 14] + [12] * 25 + [18] * 5 + [24], {
        "D_forward_mono", "D_step_mono", "send_mono", "recv_mono", "edge_unembed_mono",
    })
    for idx, header in enumerate(headers, 1):
        if header.endswith("_ms") and idx >= 4:
            ws.conditional_formatting.add(
                f"{get_column_letter(idx)}5:{get_column_letter(idx)}{4 + len(rows)}",
                ColorScaleRule(
                    start_type="min", start_color="E2F0D9",
                    mid_type="percentile", mid_value=80, mid_color="FFEB9C",
                    end_type="max", end_color="F8696B",
                ),
            )
    ws.freeze_panes = "D5"

    # -------------------- 统计与异常 --------------------
    ws = base_sheet(wb, "统计与异常", "Decode 有效步的均值、分位数与尖峰；sum 仅表示各指标独立求和，不能直接累加为 E2E。")
    metric_map = {
        "decode_forward_npu_ms": "D NPU forward（tokens>1）",
        "down_send_total_ms": "DOWN sidecar send total",
        "down_send_d2h_ms": "DOWN sidecar send D2H",
        "down_send_network_ms": "DOWN sidecar send network",
        "down_recv_total_ms": "DOWN sidecar recv total",
        "down_recv_rx_wait_ms": "DOWN sidecar recv rx_wait",
        "down_recv_h2d_ms": "DOWN sidecar recv H2D",
        "unembed_total_ms": "边侧 UNEMBED total",
        "unembed_wait_tensor_ms": "边侧 wait_tensor",
        "unembed_lm_head_ms": "边侧 lm_head",
        "unembed_select_ms": "边侧 select",
        "cloud_step_total_ms": "D cloud-step total",
        "cloud_step_sample_ms": "D cloud-step sample",
        "collect_total_ms": "D collect total（双 rank 日志）",
        "meta_total_ms": "D meta-harvest total",
        "accepted_per_step": "accepted tokens/有效步",
    }
    headers = ["指标", "count", "mean_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "max_ms", "sum_ms", "说明"]
    rows = []
    for key, label in metric_map.items():
        s = data["stats"][key]
        rows.append([
            label, s["count"], s["mean"], s["p50"], s["p90"], s["p95"], s["p99"], s["max"], s["sum"],
            "批次/日志原值；非纯 kernel 或纯网络口径",
        ])
    rows.append([
        "D NPU forward（每步双 rank 最大）",
        steady_forward_stats["count"], steady_forward_stats["mean"], steady_forward_stats["p50"],
        steady_forward_stats["p90"], steady_forward_stats["p95"], steady_forward_stats["p99"],
        steady_forward_stats["max"], steady_forward_stats["sum"],
        "先对每个 Decode 步取 TP0/TP1 最大值，再计算统计",
    ])
    interval_stats = stats([s["unembed_interval_ms"] for s in steps])
    rows.append([
        "边侧 UNEMBED 完成间隔", interval_stats["count"], interval_stats["mean"], interval_stats["p50"],
        interval_stats["p90"], interval_stats["p95"], interval_stats["p99"], interval_stats["max"], interval_stats["sum"],
        "同一边侧 mono 差值；包含调度/等待，不是纯 token kernel 间隔",
    ])
    per_token_stats = stats([s["interval_per_accepted_token_ms"] for s in steps])
    rows.append([
        "归一化 interval/accepted token", per_token_stats["count"], per_token_stats["mean"], per_token_stats["p50"],
        per_token_stats["p90"], per_token_stats["p95"], per_token_stats["p99"], per_token_stats["max"], per_token_stats["sum"],
        "accepted>0 的批次；最后空行不参与",
    ])
    write_table(ws, 4, headers, rows, [34, 10, 12, 12, 12, 12, 12, 12, 14, 48])

    start = 4 + len(rows) + 3
    row = section(ws, start, "重点尖峰（每类前 5 条）", 10)
    headers = ["类别", "seq", "峰值_ms", "关联指标", "mono", "证据位置"]
    outlier_rows = []
    for category, entries in (
        ("send D2H", data["top"]["send_d2h"][:5]),
        ("send network", data["top"]["send_network"][:5]),
        ("UNEMBED total", data["top"]["unembed"][:5]),
        ("recv total", data["top"]["recv"][:5]),
        ("meta-harvest", data["top"]["meta"][:5]),
    ):
        for e in entries:
            value = e.get("d2h_ms") if category == "send D2H" else e.get("network_ms") if category == "send network" else e["total_ms"]
            related = f"total={e['total_ms']:.2f}ms"
            if e.get("network_ms") is not None:
                related += f"; network={e['network_ms']:.2f}ms"
            if "wait_tensor" in e:
                related += f"; wait_tensor={e['wait_tensor']:.2f}ms"
            outlier_rows.append([category, e.get("seq"), value, related, e["mono"], f"{e['file_role']}:{e['line']}"])
    write_table(ws, row, headers, outlier_rows, [22, 10, 14, 42, 18, 20], {"mono"})

    chart = LineChart()
    chart.title = "Decode 关键阶段耗时趋势"
    chart.y_axis.title = "ms"
    chart.x_axis.title = "DOWN seq"
    chart.height = 15
    chart.width = 28
    decode_ws = wb["Decode逐步"]
    header_index = {cell.value: cell.column for cell in decode_ws[4]}
    for name in ("D_forward_max_ms", "send_total_ms", "recv_total_ms", "edge_unembed_total_ms"):
        series = Reference(decode_ws, min_col=header_index[name], min_row=4, max_row=4 + len(steps))
        chart.add_data(series, titles_from_data=True)
    categories = Reference(decode_ws, min_col=1, min_row=5, max_row=4 + len(steps))
    chart.set_categories(categories)
    chart.legend.position = "b"
    ws.add_chart(chart, "K4")

    # -------------------- 说明与日志映射 --------------------
    ws = base_sheet(wb, "说明与日志映射", "数据口径、日志文件和证据定位。")
    ws.column_dimensions["A"].width = 28
    ws.column_dimensions["B"].width = 105
    notes = [
        ("请求范围", f"仅 {REQ}；其他请求和后台 empty step 不纳入。"),
        ("edge.log", "边侧 API、PD 状态机、EMBED/UNEMBED、sidecar recv、最终 request done。"),
        ("cloud_p.log", "P accepted/first_scheduled/first_token/kv_send_done、UP recv、P forward、P export。"),
        ("cloud_d.log", "D accepted/KV wait/KV recv/first_scheduled/first_token/finished、Decode forward、DOWN send。"),
        ("sidecar_p.log", "UP seq=2 的 frame_host_ready；独立 proxy 文件为空。"),
        ("sidecar_d.log", "UP seq=1 的 frame_host_ready；其余 DOWN 使用 worker sidecar 明细。"),
        ("mono", "保留 time.monotonic() 原秒值。不同主机/进程不得直接相减；P/D 跨实例排序使用 ts_wall。"),
        ("elapsed/total", "优先采用日志打印值；表格派生值仅在同一单调钟或明确公式下计算。"),
        ("TP0/TP1", "Decode forward 和 collect 每步取两个 rank 的最大值；原始两列同时保留。"),
        ("MTP", f"{positive_steps} 个有效 Decode 步生成 {output_tokens} tokens，平均约 {output_tokens / positive_steps:.4f} tokens/步；空行不计入。"),
        ("sidecar", "queue/clone/d2h/serialize/network/rx_wait/h2d 为嵌套子阶段，不能与 outer stage 全量相加。"),
        ("NPU 同步", "model-forward 与 embedding 的 npu_ms 是批级同步墙钟，不是每请求纯 kernel 时间。"),
        ("slot_wait=-1", "日志中未形成对应等待区间，按原值保留，不改写成 0。"),
        ("usage", f"若 APIServer 出现 final usage unavailable warning，将使用本地计数；本请求最终 completion={output_tokens}。"),
        ("输出文件", OUTPUT.name),
    ]
    row = 4
    for key, value in notes:
        ws.cell(row, 1, key)
        ws.cell(row, 2, value)
        ws.cell(row, 1).font = Font(name="Microsoft YaHei", size=10, bold=True)
        ws.cell(row, 2).font = Font(name="Microsoft YaHei", size=10)
        ws.cell(row, 2).alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[row].height = 32
        row += 1

    # Consistent workbook presentation.
    for sheet in wb.worksheets:
        sheet.sheet_properties.pageSetUpPr.fitToPage = True
        sheet.page_setup.fitToWidth = 1
        sheet.page_setup.fitToHeight = 0
        sheet.sheet_view.zoomScale = 85

    wb.save(OUTPUT)

    # Reopen and validate required content and row counts.
    checked = load_workbook(OUTPUT, data_only=False)
    required = {"请求总览", "边云主阶段", "P与D实例", "UP与首token", "Decode逐步", "统计与异常", "说明与日志映射"}
    assert required == set(checked.sheetnames)
    assert checked["Decode逐步"].max_row == len(steps) + 4
    seq_values = [checked["Decode逐步"].cell(r, 1).value for r in range(5, len(steps) + 5)]
    accepted_values = [checked["Decode逐步"].cell(r, 2).value for r in range(5, len(steps) + 5)]
    assert seq_values == [s["seq"] for s in steps]
    assert sum(accepted_values) == output_tokens
    assert checked["边云主阶段"]["E12"].value is not None
    assert OUTPUT.stat().st_size > 10_000
    print(f"saved={OUTPUT.resolve()}")
    print(f"size={OUTPUT.stat().st_size}; sheets={checked.sheetnames}; decode_rows={len(seq_values)}; accepted={sum(accepted_values)}")


if __name__ == "__main__":
    main()
