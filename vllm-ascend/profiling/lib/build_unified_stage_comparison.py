from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from lib.build_lwd_perf_excel import base_sheet, write_table

CENTRAL = Path(os.environ["PD_CENTRAL_JSON"])
EDGE = Path(os.environ["PD_EDGE_JSON"])
BOOK = Path(os.environ["PD_COMPARE_BOOK"])


def mean(values):
    vals = [v for v in values if v is not None and v >= 0]
    return sum(vals) / len(vals) if vals else None


def main():
    c = json.loads(CENTRAL.read_text(encoding="utf-8"))
    e = json.loads(EDGE.read_text(encoding="utf-8"))
    es = {s["stage"]: s for s in e["edge_stages"]}
    cs = {(s["kv_role"], s["stage"]): s for s in c["stages"]}
    steps = e["decode_steps"]
    first = steps[0]
    derived = c["derived"]
    central_output_tokens = derived.get("d_first_to_finished_output_tokens")
    central_tpot = (
        derived["d_decode_wall_ms"] / (central_output_tokens - 1)
        if central_output_tokens and central_output_tokens > 1
        else None
    )
    output_tokens = (
        e.get("request_done", {}).get("completion_tokens")
        if e.get("request_done")
        else None
    ) or e["counts"]["accepted_sum"]
    request_done = e.get("request_done") or {}
    e2e_ms = (
        (request_done["mono"] - es["request_accepted"]["mono"]) * 1000
        if request_done.get("mono") is not None
        else None
    )

    def up_event(kind, seqno=None, op=None, channel=None):
        candidates = [
            event for event in e["up"]
            if event["kind"] == kind
            and (seqno is None or event.get("seq") == seqno)
            and (op is None or event.get("op") == op)
            and (channel is None or event.get("channel") == channel)
        ]
        return candidates[0] if candidates else {}

    def cm(role, stage, key="mono"):
        return cs[(role, stage)][key]

    rows = []
    seq = 0

    def add(stage, central_ms, edge_ms, central_mono, edge_mono, code, parent=None):
        nonlocal seq
        seq += 1
        parent_seq = None
        if parent is not None:
            parent_seq = next(r[0] for r in rows if r[1] == parent)
        rows.append([
            seq,
            stage,
            "-" if central_ms is None else central_ms,
            "-" if edge_ms is None else edge_ms,
            "-" if central_mono is None else central_mono,
            "-" if edge_mono is None else edge_mono,
            code,
            "-" if parent is None else parent,
            "-" if parent_seq is None else parent_seq,
        ])

    add("请求接入至首token", derived["global_ttft_ms"], es["d_to_edge_first_token"]["total_ms"], cm("kv_consumer", "first_token"), es["d_to_edge_first_token"]["mono"], "集中式: Scheduler.update_from_output；边云: lwd_pd_deliver_tokens")
    add("Prefill阶段（P accepted→first token）", cm("kv_producer", "first_token", "total_ms"), next(s["total_ms"] for s in e["central_stages"] if s["role"] == "kv_producer" and s["stage"] == "first_token"), cm("kv_producer", "first_token"), next(s["mono"] for s in e["central_stages"] if s["role"] == "kv_producer" and s["stage"] == "first_token"), "集中式: scheduler.py:1489；边云: lwd_edge_scheduler.py:2195", "请求接入至首token")
    add("API预处理", None, (e.get("api_response") or {}).get("total_ms"), None, (e.get("api_response") or {}).get("mono"), "边云: api_router.py:65，Streaming response cost；无单独预处理日志时填-。", "请求接入至首token")
    add("请求登记/accepted", cm("kv_producer", "accepted", "elapsed_ms"), es["request_accepted"]["total_ms"], cm("kv_producer", "accepted"), es["request_accepted"]["mono"], "集中式: scheduler.py:1892；边云: lwd_edge_scheduler.py:1724", "请求接入至首token")
    add("P绑定/排队", cm("kv_producer", "first_scheduled", "elapsed_ms"), es["edge_to_p_bind"]["elapsed_ms"], cm("kv_producer", "first_scheduled"), es["edge_to_p_bind"]["mono"], "集中式: scheduler.py:1000；边云: lwd_edge_scheduler.py:2102", "Prefill阶段（P accepted→first token）")
    embedding = up_event("embed", 2)
    add("边侧Embedding", None, embedding.get("forward_ms"), None, embedding.get("mono"), "边云特有: lwd_edge_worker.py:316-329", "Prefill阶段（P accepted→first token）")
    add("P上传调度", None, es["p_upload"]["elapsed_ms"], None, es["p_upload"]["mono"], "边云特有: lwd_edge_scheduler.py:3708", "Prefill阶段（P accepted→first token）")
    up_send = up_event("sidecar", 2, "send", "lwd_up")
    add("UP sidecar发送", None, up_send.get("total_ms"), None, up_send.get("mono"), "边云特有: sidecar_channel.py:305", "Prefill阶段（P accepted→first token）")
    up_recv = up_event("sidecar", 2, "recv", "lwd_up")
    add("UP sidecar接收", None, up_recv.get("total_ms"), None, up_recv.get("mono"), "边云特有: sidecar_channel.py:329", "Prefill阶段（P accepted→first token）")
    up_gate = up_event("gate_ready", 2)
    add("UP gate/调度准入", None, up_gate.get("total_ms"), None, up_gate.get("mono"), "边云特有: lwd_cloud_mixed_scheduler.py:847", "Prefill阶段（P accepted→first token）")
    p_forward = max(
        [event for event in e["up"] if event["kind"] == "model_forward" and event["file_role"] == "prefill"],
        key=lambda event: event["npu_ms"],
        default={},
    )
    add("P前向计算", derived["p_forward_max_ms"], p_forward.get("npu_ms"), next((f["mono"] for f in c["forwards"] if f["role"] == "prefill" and f["npu_ms"] == derived["p_forward_max_ms"]), None), p_forward.get("mono"), "model_runner_v1.py:2333-2338/2890-2926", "Prefill阶段（P accepted→first token）")
    add("P导出/完成通知", None, es["p_compute_export"]["elapsed_ms"], None, es["p_compute_export"]["mono"], "边云特有: lwd_edge_scheduler.py:2195", "Prefill阶段（P accepted→first token）")

    add("P→D交接/首token", (cm("kv_consumer", "first_token", "ts_wall") - cm("kv_producer", "first_token", "ts_wall")) * 1000, (next(s["ts_wall"] for s in e["central_stages"] if s["role"] == "kv_consumer" and s["stage"] == "first_token") - next(s["ts_wall"] for s in e["central_stages"] if s["role"] == "kv_producer" and s["stage"] == "first_token")) * 1000, cm("kv_consumer", "first_token"), next(s["mono"] for s in e["central_stages"] if s["role"] == "kv_consumer" and s["stage"] == "first_token"), "集中式/边云: P.first_token→D.first_token(ts_wall)", "请求接入至首token")
    add("KV send done", cm("kv_producer", "kv_send_done", "elapsed_ms"), next(s["elapsed_ms"] for s in e["central_stages"] if s["role"] == "kv_producer" and s["stage"] == "kv_send_done"), cm("kv_producer", "kv_send_done"), next(s["mono"] for s in e["central_stages"] if s["role"] == "kv_producer" and s["stage"] == "kv_send_done"), "集中式/边云: scheduler.py:2322", "P→D交接/首token")
    add("KV wait/recv done", cm("kv_consumer", "kv_recv_done", "elapsed_ms"), next(s["elapsed_ms"] for s in e["central_stages"] if s["role"] == "kv_consumer" and s["stage"] == "kv_recv_done"), cm("kv_consumer", "kv_recv_done"), next(s["mono"] for s in e["central_stages"] if s["role"] == "kv_consumer" and s["stage"] == "kv_recv_done"), "集中式/边云: scheduler.py:2313", "P→D交接/首token")
    add("D绑定", None, es["p_to_d_bind"]["elapsed_ms"], None, es["p_to_d_bind"]["mono"], "边云特有: lwd_edge_scheduler.py:2139", "P→D交接/首token")
    add("D尾窗派发", None, es["d_tail_dispatch"]["elapsed_ms"], None, es["d_tail_dispatch"]["mono"], "边云特有: lwd_edge_scheduler.py:2421", "P→D交接/首token")
    add("D KV import admit_to_ready", None, (e.get("import_ready") or {}).get("total_ms"), None, (e.get("import_ready") or {}).get("mono"), "边云: lwd_cloud_mixed_scheduler.py:348", "P→D交接/首token")

    add("Decode首token（D accepted→first token）", cm("kv_consumer", "first_token", "total_ms"), next(s["total_ms"] for s in e["central_stages"] if s["role"] == "kv_consumer" and s["stage"] == "first_token"), cm("kv_consumer", "first_token"), next(s["mono"] for s in e["central_stages"] if s["role"] == "kv_consumer" and s["stage"] == "first_token"), "集中式/边云: scheduler.py:1489", "P→D交接/首token")
    add("D首次调度", cm("kv_consumer", "first_scheduled", "elapsed_ms"), next(s["elapsed_ms"] for s in e["central_stages"] if s["role"] == "kv_consumer" and s["stage"] == "first_scheduled"), cm("kv_consumer", "first_scheduled"), next(s["mono"] for s in e["central_stages"] if s["role"] == "kv_consumer" and s["stage"] == "first_scheduled"), "集中式/边云: scheduler.py:1000", "Decode首token（D accepted→first token）")
    add("D首步NPU forward", derived["d_first_forward_max_ms"], first["forward_max_npu_ms"], max(f["mono"] for f in c["forwards"] if f["role"] == "decode" and f["tokens"] == 1), first["forward_mono_max"], "model_runner_v1.py:2333-2338/2890-2926", "Decode首token（D accepted→first token）")
    add("首次DOWN发送", None, first["send"]["total_ms"], None, first["send"]["mono"], "边云特有: sidecar_channel.py:305，SidecarDataRuntime._step", "Decode首token（D accepted→first token）")
    add("首次DOWN接收", None, first["recv"]["total_ms"], None, first["recv"]["mono"], "边云特有: sidecar_channel.py:329，SidecarDataRuntime._step", "Decode首token（D accepted→first token）")
    add("首次边侧UNEMBED", None, first["unembed"]["total_ms"], None, first["unembed"]["mono"], "边云特有: lwd_edge_worker.py:514-605", "Decode首token（D accepted→first token）")
    add("首token交付", cm("kv_consumer", "first_token", "elapsed_ms"), es["d_to_edge_first_token"]["elapsed_ms"], cm("kv_consumer", "first_token"), es["d_to_edge_first_token"]["mono"], "集中式: kv_recv_done→first_token；边云: d_tail_dispatch→d_to_edge_first_token", "Decode首token（D accepted→first token）")

    add("持续Decode", derived["d_decode_wall_ms"], es["decode_generation"]["elapsed_ms"], cm("kv_consumer", "finished"), es["decode_generation"]["mono"], "集中式: scheduler.py:1563；边云: lwd_edge_scheduler.py:2562", "Decode首token（D accepted→first token）")
    add("D稳态NPU forward（均值）", derived["d_steady_forward_max_per_step"]["mean"], mean([s["forward_max_npu_ms"] for s in steps[1:]]), None, None, "model_runner_v1.py:2340；均按每步双rank最大", "持续Decode")
    add("D采样与收集（均值）", None, mean([s["cloud_step_sample_ms"] for s in steps]), None, None, "边云特有: lwd_cloud_worker.py:777-780", "持续Decode")
    add("collect.total（均值）", None, mean([max(s["collect_max_ms"] for s in steps)]), None, None, "边云特有: lwd_cloud_model_runner.py:1035-1256", "持续Decode")
    add("meta-harvest（均值）", None, mean([s["meta_total_ms"] for s in steps]), None, None, "边云特有: multiproc_executor.py:1232-1251", "持续Decode")
    add("DOWN sidecar发送（均值）", None, mean([s["send"]["total_ms"] for s in steps]), None, None, "边云特有: sidecar_channel.py:305", "持续Decode")
    for field, label, code in (
        ("queue_ms", "queue/activate", "sidecar_channel.py:492-550"),
        ("clone_ms", "snapshot/clone", "sidecar_channel.py:220-224,373-380"),
        ("d2h_ms", "D2H", "sidecar_channel.py:379-383"),
        ("serialize_ms", "serialize", "sidecar_channel.py:384-385"),
        ("network_ms", "network/ACK", "sidecar_channel.py:261-282,245-259"),
        ("slot_wait_ms", "slot_wait", "sidecar_channel.py:228-233,375-378,398-401"),
    ):
        value = mean([s["send"][field] for s in steps])
        add(f"  ↳ send.{label}（均值）", None, value, None, None, f"边云特有: {code}", "DOWN sidecar发送（均值）")
    add("DOWN sidecar接收（均值）", None, mean([s["recv"]["total_ms"] for s in steps]), None, None, "边云特有: sidecar_channel.py:329", "持续Decode")
    for field, label, code in (
        ("queue_ms", "queue/activate", "sidecar_channel.py:492-550,365-370"),
        ("rx_wait_ms", "rx_wait", "sidecar_channel.py:391-398,245-259"),
        ("h2d_ms", "H2D", "sidecar_channel.py:403-410"),
        ("slot_wait_ms", "slot_wait", "sidecar_channel.py:228-233,398-401"),
    ):
        value = mean([s["recv"][field] for s in steps])
        add(f"  ↳ recv.{label}（均值）", None, value, None, None, f"边云特有: {code}", "DOWN sidecar接收（均值）")
    add("边侧UNEMBED（均值）", None, mean([s["unembed"]["total_ms"] for s in steps]), None, None, "边云特有: lwd_edge_worker.py:514-605", "持续Decode")
    for field, label, code in (
        ("post_recv", "post_recv", "lwd_edge_worker.py:514-525"),
        ("wait_tensor", "wait_tensor", "lwd_edge_worker.py:526-533，非纯网络耗时"),
        ("lm_head", "lm_head", "lwd_edge_worker.py:534-575"),
        ("select", "select", "lwd_edge_worker.py:595-605"),
    ):
        value = mean([s["unembed"][field] for s in steps])
        add(f"  ↳ unembed.{label}（均值）", None, value, None, None, f"边云特有: {code}", "边侧UNEMBED（均值）")
    add("输出间隔/TPOT", central_tpot, es["decode_generation"]["elapsed_ms"] / (output_tokens - 1) if output_tokens > 1 else None, None, None, f"集中式: 有completion_tokens时计算；边云: decode_generation/({output_tokens}-1)", "持续Decode")

    add("收尾", None, es["edge_finalize"]["elapsed_ms"], None, es["edge_finalize"]["mono"], "边云特有: lwd_edge_scheduler.py:1912", "持续Decode")
    add("D finished", cm("kv_consumer", "finished", "elapsed_ms"), es["decode_generation"]["elapsed_ms"], cm("kv_consumer", "finished"), es["decode_generation"]["mono"], "集中式/边云: scheduler.py:1563/lwd_edge_scheduler.py:2562", "收尾")
    add("最终usage", None, es["edge_finalize"]["elapsed_ms"], None, es["edge_finalize"]["mono"], "边云特有: lwd_edge_scheduler.py:1912；K回报与usage形成", "收尾")
    add("请求完成", cm("kv_consumer", "finished", "total_ms"), e2e_ms, cm("kv_consumer", "finished"), request_done.get("mono"), "集中式: D accepted→finished；边云: request done", "收尾")

    for row in rows:
        for index in (2, 3):
            value = row[index]
            if isinstance(value, (int, float)) and value < 0:
                raise ValueError(f"Negative stage duration at row {row[0]} ({row[1]}): {value}")

    wb = load_workbook(BOOK)
    if "阶段对比总表" in wb.sheetnames:
        del wb["阶段对比总表"]
    ws = base_sheet(
        wb,
        "阶段对比总表",
        "字段：序号/阶段/耗时/ms/mono/代码位置/父阶段。顶层阶段用于总览，子阶段用于定位；异步重叠阶段不得相加。",
    )
    headers = ["序号", "阶段", "耗时-集中式(ms)", "耗时-边云PD(ms)", "time.monotonic()-集中式", "time.monotonic()-边云PD", "代码位置", "父阶段", "父阶段序号"]
    write_table(ws, 4, headers, rows, [8, 36, 20, 20, 24, 24, 78, 30, 14], {"time.monotonic()-集中式", "time.monotonic()-边云PD"})
    for r in range(5, 5 + len(rows)):
        ws.cell(r, 2).alignment = Alignment(wrap_text=True, vertical="center")
        ws.cell(r, 7).alignment = Alignment(wrap_text=True, vertical="center")
        if str(ws.cell(r, 2).value).startswith("  ↳"):
            for col in range(1, 10):
                ws.cell(r, col).fill = PatternFill("solid", fgColor="FFF2CC")
                ws.cell(r, col).font = Font(name="Microsoft YaHei", size=9, italic=True)
    ws.freeze_panes = "C5"
    wb.save(BOOK)

    checked = load_workbook(BOOK, data_only=False)
    ws = checked["阶段对比总表"]
    assert ws.max_row == len(rows) + 4
    assert ws.cell(5, 1).value == 1
    assert ws.cell(5, 2).value == "请求接入至首token"
    assert ws.cell(5, 3).value == derived["global_ttft_ms"]
    assert ws.cell(5, 4).value == es["d_to_edge_first_token"]["total_ms"]
    assert ws.cell(6, 2).value == "Prefill阶段（P accepted→first token）"
    assert any(ws.cell(r, 3).value == "-" for r in range(5, ws.max_row + 1))
    print(f"saved={BOOK.resolve()}; rows={len(rows)}")


if __name__ == "__main__":
    main()
