from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openpyxl import load_workbook

from lib.build_lwd_perf_excel import add_kv_rows, base_sheet, section, stats, write_table

CENTRAL = Path(os.environ["PD_CENTRAL_JSON"])
EDGE = Path(os.environ["PD_EDGE_JSON"])
OUT_DIR = Path(os.environ.get("PD_OUTPUT_DIR", "."))
CENTRAL_BOOK = Path(os.environ["PD_CENTRAL_BOOK"])
COMPARE_BOOK = Path(os.environ["PD_COMPARE_BOOK"])


def code(text):
    m = re.search(r"\[([A-Za-z0-9_]+\.py):(\d+)\]", text)
    return f"{m.group(1)}:{m.group(2)}" if m else ""


def by_role_stage(data, role, stage):
    return next(s for s in data["stages"] if s["kv_role"] == role and s["stage"] == stage)


def pair_value(pair, worker, field="npu_ms"):
    return next(f[field] for f in pair if f["worker"] == worker)


def build_central_steps(data):
    forwards = sorted(
        [f for f in data["forwards"] if f["role"] == "decode" and f["mono"] <= by_role_stage(data, "kv_consumer", "finished")["mono"]],
        key=lambda f: f["mono"],
    )
    first = [f for f in forwards if f["tokens"] == 1]
    steady = [f for f in forwards if f["tokens"] and f["tokens"] > 1]
    steps = []
    if first:
        steps.append({
            "index": 0,
            "tp0": pair_value(first, "TP0"),
            "tp1": pair_value(first, "TP1"),
            "max": max(f["npu_ms"] for f in first),
            "mono": max(f["mono"] for f in first),
            "interval": None,
            "tokens": 1,
            "evidence": ",".join(str(f["line"]) for f in first),
        })
    previous_mono = steps[0]["mono"] if steps else None
    for idx in range(0, len(steady), 2):
        pair = steady[idx : idx + 2]
        if len(pair) != 2:
            continue
        mono = max(f["mono"] for f in pair)
        steps.append({
            "index": len(steps),
            "tp0": pair_value(pair, "TP0"),
            "tp1": pair_value(pair, "TP1"),
            "max": max(f["npu_ms"] for f in pair),
            "mono": mono,
            "interval": (mono - previous_mono) * 1000 if previous_mono is not None else None,
            "tokens": pair[0]["tokens"],
            "evidence": ",".join(str(f["line"]) for f in pair),
        })
        previous_mono = mono
    return steps


def build_central_book(data):
    steps = build_central_steps(data)
    assert len(steps) == central["counts"]["d_steady_step_pairs"] + 1
    derived = data["derived"]
    wb = base_sheet(
        __import__("openpyxl").Workbook(),
        "集中式总览",
        "集中式 PD 单请求：P/D request_id 不同，按公共业务前缀与 Mooncake KV transfer 关联。",
    )
    # Replace the temporary workbook created above with the real workbook in the caller.
    return wb, steps


def main():
    central = json.loads(CENTRAL.read_text(encoding="utf-8"))
    edge = (
        json.loads(EDGE.read_text(encoding="utf-8"))
        if os.environ.get("PD_WRITE_COMPARISON", "1") == "1"
        else {}
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    derived = central["derived"]
    steps = build_central_steps(central)
    assert len(steps) == central["counts"]["d_steady_step_pairs"] + 1
    assert central["counts"]["d_steady_step_pairs"] == derived["d_steady_forward_max_per_step"]["count"]
    central_output_tokens = derived.get("d_first_to_finished_output_tokens")
    central_tpot = (
        derived["d_decode_wall_ms"] / (central_output_tokens - 1)
        if central_output_tokens and central_output_tokens > 1
        else None
    )
    p_forward_tokens = max(
        (f["tokens"] for f in central["forwards"] if f["role"] == "prefill" and f["tokens"] is not None),
        default=None,
    )

    from openpyxl import Workbook

    wb = Workbook()
    wb.remove(wb.active)
    ws = base_sheet(
        wb,
        "集中式总览",
        "集中式 PD 单请求：P/D request_id 不同，按公共业务前缀与 Mooncake KV transfer 关联。",
    )
    ws.column_dimensions["A"].width = 36
    ws.column_dimensions["B"].width = 24
    ws.column_dimensions["C"].width = 76
    row = section(ws, 4, "1. 请求与可比性", 3)
    info = [
        ("公共业务前缀", central["base_request"], "P/D request_id 后缀不同，不能直接按完整 ID join"),
        ("P request_id", "…-95c9180c", central["request_ids"][0]),
        ("D request_id", "…-be90ebbf", central["request_ids"][1]),
        ("P forward tokens", p_forward_tokens, "实际模型前向 token 数"),
        ("输出 token 数", central_output_tokens if central_output_tokens is not None else "未显式记录", "仅在有明确 completion_tokens 时计算 TPOT"),
        ("D 稳态 Decode 步", central["counts"]["d_steady_step_pairs"], "每步双 rank 日志，first token 另计 1 步"),
    ]
    row = add_kv_rows(ws, row, info) + 1

    row = section(ws, row, "2. 集中式核心指标", 3)
    p_first = by_role_stage(central, "kv_producer", "first_token")
    p_scheduled = by_role_stage(central, "kv_producer", "first_scheduled")
    p_send = by_role_stage(central, "kv_producer", "kv_send_done")
    d_first = by_role_stage(central, "kv_consumer", "first_token")
    d_recv = by_role_stage(central, "kv_consumer", "kv_recv_done")
    d_scheduled = by_role_stage(central, "kv_consumer", "first_scheduled")
    d_finished = by_role_stage(central, "kv_consumer", "finished")
    metrics = [
        ("全局近似 TTFT（ms）", derived["global_ttft_ms"], "D.first_token.ts_wall - P.accepted.ts_wall"),
        ("P 排队（ms）", p_scheduled["elapsed_ms"], "P accepted → P first_scheduled"),
        ("P Prefill（ms）", p_first["elapsed_ms"], "P first_scheduled → P first_token"),
        ("P accepted→first_token（ms）", p_first["total_ms"], "P 实例本地累计"),
        ("P NPU forward 最大值（ms）", derived["p_forward_max_ms"], f"{p_forward_tokens} tokens，两个 rank 的最大值"),
        ("P kv_send_done（ms）", p_send["elapsed_ms"], "P first_token → 观察到 KV 发送完成"),
        ("D 远端 KV 等待（ms）", d_recv["elapsed_ms"], "D kv_wait_start → kv_recv_done"),
        ("Mooncake KV transfer（ms）", max(t["took_ms"] for t in central["transfers"]), "两个 device 为 11.98/12.01ms"),
        ("D KV 就绪→first_scheduled（ms）", d_scheduled["elapsed_ms"], "D 调度等待"),
        ("D accepted→first_token（ms）", d_first["total_ms"], "D 实例本地累计"),
        ("D 首步 NPU forward 最大值（ms）", derived["d_first_forward_max_ms"], "首步两个 rank 的最大值"),
        ("D 持续 Decode（ms）", derived["d_decode_wall_ms"], "D first_token → finished"),
        ("D accepted→finished（ms）", d_finished["total_ms"], "D 实例本地累计"),
        ("D 稳态 NPU forward（ms/步）", derived["d_steady_forward_max_per_step"]["mean"], f"{central['counts']['d_steady_step_pairs']} 步，每步双 rank 最大值"),
        ("平均 TPOT", central_tpot if central_tpot is not None else "不可计算", "无明确 completion_tokens 时不计算"),
    ]
    row = add_kv_rows(ws, row, metrics) + 1

    row = section(ws, row, "3. 代码位置与注意事项", 3)
    notes = [
        ("P/D 阶段埋点", "scheduler.py:1848", "P accepted/scheduled/token/send，D accepted/wait/recv/scheduled/token/finished"),
        ("NPU forward", "model_runner_v1.py:2340", "批级同步墙钟，不是每请求独占纯 kernel"),
        ("KV transfer", "mooncake_connector.py:980", "实际 backend transfer 日志"),
        ("输出数缺失", "未计算 TPOT", "不能与边云 24.22ms/token 直接做差"),
        ("proxy", "仅有 HTTP 200 记录", "不包含逐阶段 proxy 时间"),
    ]
    add_kv_rows(ws, row, notes)

    ws = base_sheet(wb, "P与D阶段", "按 ts_wall 排序；保留各进程 mono 原值，不跨进程相减。")
    rows = []
    for event in central["stages"]:
        rows.append([
            "P" if event["kv_role"] == "kv_producer" else "D",
            event["stage"], event["ts_wall"], event["mono"], event["elapsed_ms"], event["total_ms"],
            event["code"], event["line"], event["request_id"],
        ])
    write_table(
        ws, 4,
        ["实例", "阶段", "ts_wall(s)", "mono(s)", "elapsed_ms", "total_ms", "代码位置", "日志行号", "request_id"],
        rows,
        [10, 24, 20, 18, 14, 14, 24, 12, 60],
        {"ts_wall(s)", "mono(s)"},
    )

    ws = base_sheet(wb, "D逐步NPU", f"首步 tokens=1 单独列出；其余 {central['counts']['d_steady_step_pairs']} 步为 tokens=4 的 forward，不代表实际 accepted 数。")
    rows = []
    for step in steps:
        rows.append([
            step["index"], step["tokens"], step["tp0"], step["tp1"], step["max"], step["interval"], step["mono"],
            "model_runner_v1.py:2340", step["evidence"],
        ])
    write_table(
        ws, 4,
        ["step", "forward_tokens", "TP0_npu_ms", "TP1_npu_ms", "max_npu_ms", "forward_interval_ms", "mono_max", "代码位置", "日志行号"],
        rows,
        [10, 16, 14, 14, 14, 20, 18, 26, 18],
        {"mono_max"},
    )

    ws = base_sheet(wb, "统计与代码", "统计只针对集中式日志中可确认的数据。")
    stat = derived["d_steady_forward_max_per_step"]
    rows = [[
        "D 稳态 forward max/步", stat["count"], stat["mean"], stat["p50"], stat["p90"], stat["p95"], stat["p99"], stat["max"], stat["sum"],
        "先取每步 TP0/TP1 最大值",
    ]]
    write_table(
        ws, 4,
        ["指标", "count", "mean_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "max_ms", "sum_ms", "说明"],
        rows,
        [30, 10, 12, 12, 12, 12, 12, 12, 14, 42],
    )
    row = section(ws, 8, "代码位置", 4)
    code_rows = [
        ("P/D 调度阶段", "vllm/v1/core/sched/scheduler.py:1848", "central_pd latency 日志"),
        ("NPU forward", "vllm-ascend/vllm_ascend/worker/model_runner_v1.py:2340", "同步 forward npu_ms"),
        ("Mooncake transfer", "vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py:980", "KV backend transfer"),
        ("P KV block 延迟释放", "vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py:1958", "Delaying free of 21 blocks"),
    ]
    write_table(ws, row, ["组件", "代码位置", "说明"], code_rows, [24, 90, 42])
    wb.save(CENTRAL_BOOK)

    if os.environ.get("PD_WRITE_COMPARISON", "1") == "0":
        central_checked = load_workbook(CENTRAL_BOOK, data_only=False)
        assert central_checked["D逐步NPU"].max_row == len(steps) + 4
        print(f"saved={CENTRAL_BOOK.resolve()}")
        return

    # -------------------- 对比工作簿 --------------------
    edge_stages = {s["stage"]: s for s in edge["edge_stages"]}
    edge_central = {(s["role"], s["stage"]): s for s in edge["central_stages"]}
    edge_steps = edge["decode_steps"]
    edge_npu_stats = stats([s["forward_max_npu_ms"] for s in edge_steps[1:]])
    edge_output_tokens = (
        edge.get("request_done", {}).get("completion_tokens")
        if edge.get("request_done")
        else None
    ) or edge["counts"]["accepted_sum"]
    edge_tpot = (
        edge_stages["decode_generation"]["elapsed_ms"] / (edge_output_tokens - 1)
        if edge_output_tokens > 1
        else None
    )
    edge_prefill_tokens = (
        edge.get("request_meta", {}).get("n") - edge.get("request_meta", {}).get("h")
        if edge.get("request_meta", {}).get("n") is not None
        and edge.get("request_meta", {}).get("h") is not None
        else None
    )
    wb = Workbook()
    wb.remove(wb.active)
    ws = base_sheet(
        wb,
        "对比总览",
        "集中式 1117-1055 vs 边云 1117-1020；单请求对比，输出数不同或缺失的项目不做归一化结论。",
    )
    ws.column_dimensions["A"].width = 34
    ws.column_dimensions["B"].width = 22
    ws.column_dimensions["C"].width = 22
    ws.column_dimensions["D"].width = 22
    ws.column_dimensions["E"].width = 20
    ws.column_dimensions["F"].width = 68
    row = section(ws, 4, "1. 样本可比性", 6)
    comparisons = [
        ("实际 Prefill tokens", p_forward_tokens, edge_prefill_tokens, (edge_prefill_tokens - p_forward_tokens) if edge_prefill_tokens is not None and p_forward_tokens is not None else None, "按实际前向/上传 token 数", "Prefill 计算量对比"),
        ("输出 tokens", central_output_tokens if central_output_tokens is not None else "未显式记录", edge_output_tokens, "按实际完成数", "有明确输出数时才计算 TPOT", "缺失一侧时不做 TPOT 差值"),
        ("TTFT（ms）", derived["global_ttft_ms"], edge_stages["d_to_edge_first_token"]["total_ms"], edge_stages["d_to_edge_first_token"]["total_ms"] - derived["global_ttft_ms"], "集中式更快", "集中式=D.first_token-P.accepted；边云=edge request_accepted→首 token"),
        ("Decode wall（ms）", derived["d_decode_wall_ms"], edge_stages["decode_generation"]["elapsed_ms"], edge_stages["decode_generation"]["elapsed_ms"] - derived["d_decode_wall_ms"], "仅列原值", "输出 token 数未对齐时不据此判断吞吐优劣"),
        ("稳态 NPU forward（ms/步）", derived["d_steady_forward_max_per_step"]["mean"], edge_npu_stats["mean"], edge_npu_stats["mean"] - derived["d_steady_forward_max_per_step"]["mean"], "基本接近", f"集中式{central['counts']['d_steady_step_pairs']}步，边云{len(edge_steps)-1}步；均按双rank最大"),
        ("平均 TPOT（ms/token）", central_tpot if central_tpot is not None else "不可计算", edge_tpot if edge_tpot is not None else "不可计算", (edge_tpot - central_tpot) if edge_tpot is not None and central_tpot is not None else "不可比", "按实际 completion_tokens", "缺失一侧时不做 TPOT 差值"),
    ]
    write_table(ws, row, ["指标", "集中式", "边云式", "边云-集中式", "结论", "口径说明"], comparisons, [32, 20, 20, 20, 20, 70])

    row = section(ws, 4 + len(comparisons) + 3, "2. 主要结论", 6)
    conclusions = [
        ("TTFT 差值", f"边云高 {edge_stages['d_to_edge_first_token']['total_ms'] - derived['global_ttft_ms']:.2f}ms", "集中式 828.24ms，边云 1038.77ms；本次样本集中式更低"),
        ("Prefill NPU 差异", f"边云-集中式={max(e['npu_ms'] for e in edge['up'] if e['kind'] == 'model_forward' and e['file_role'] == 'prefill') - derived['p_forward_max_ms']:.2f}ms", "模型前向本身接近时，差异主要在其他阶段"),
        ("D 首步 NPU 差异", f"边云-集中式={edge_steps[0]['forward_max_npu_ms'] - derived['d_first_forward_max_ms']:.2f}ms", "比较首步双 rank 最大 forward"),
        ("Decode 总时长", "不直接下结论", f"输出 token 数未完全对齐；仅列 {derived['d_decode_wall_ms']:.2f} vs {edge_stages['decode_generation']['elapsed_ms']:.2f}ms"),
        ("边云特有开销", "sidecar/UNEMBED/边云绑定与尾窗", "边云 DOWN send P95=42.38ms，UNEMBED P95=32.96ms"),
        ("实验限制", "单请求、不同时间段", "结果用于定位阶段差异，不等同于严格多轮 benchmark"),
    ]
    row = add_kv_rows(ws, row, conclusions)

    ws = base_sheet(wb, "共同阶段对比", "仅比较语义相近阶段；差值=边云-集中式，正值表示边云更慢。")
    common = [
        ("P accepted→first_token", p_first["total_ms"], edge_central[("kv_producer", "first_token")]["total_ms"], "P 实例本地累计", "scheduler.py:1848", "scheduler.py:1848"),
        ("P first_scheduled→first_token", p_first["elapsed_ms"], edge_central[("kv_producer", "first_token")]["elapsed_ms"], "P Prefill 主阶段", "scheduler.py:1848", "scheduler.py:1848"),
        ("P NPU forward max", derived["p_forward_max_ms"], max(e["npu_ms"] for e in edge["up"] if e["kind"] == "model_forward" and e["file_role"] == "prefill"), f"tokens={p_forward_tokens}，双rank最大", "model_runner_v1.py:2340", "model_runner_v1.py:2340"),
        ("P first_token→kv_send_done", p_send["elapsed_ms"], edge_central[("kv_producer", "kv_send_done")]["elapsed_ms"], "发送完成观察，不是纯网络", "scheduler.py:1848", "scheduler.py:1848"),
        ("D kv_wait_start→kv_recv_done", d_recv["elapsed_ms"], edge_central[("kv_consumer", "kv_recv_done")]["elapsed_ms"], "远端 KV 等待", "scheduler.py:1848", "scheduler.py:1848"),
        ("Mooncake KV transfer", max(t["took_ms"] for t in central["transfers"]), max((t["took_ms"] for t in edge.get("transfers", [])), default=None), "两个实例 backend transfer", "mooncake_connector.py:980", "mooncake_connector.py:980"),
        ("D kv_recv_done→first_scheduled", d_scheduled["elapsed_ms"], edge_central[("kv_consumer", "first_scheduled")]["elapsed_ms"], "D 调度等待", "scheduler.py:1848", "scheduler.py:1848"),
        ("D accepted→first_token", d_first["total_ms"], edge_central[("kv_consumer", "first_token")]["total_ms"], "D 实例本地累计", "scheduler.py:1848", "scheduler.py:1848"),
        ("D 首步 NPU forward max", derived["d_first_forward_max_ms"], edge_steps[0]["forward_max_npu_ms"], "首步 tokens=1，双rank最大", "model_runner_v1.py:2340", "model_runner_v1.py:2340"),
        ("D 稳态 forward max/步", derived["d_steady_forward_max_per_step"]["mean"], edge_npu_stats["mean"], "按每步双rank最大", "model_runner_v1.py:2340", "model_runner_v1.py:2340"),
        ("D first_token→finished", derived["d_decode_wall_ms"], edge_stages["decode_generation"]["elapsed_ms"], "输出数未对齐，只比较原始区间", "scheduler.py:1848", "lwd_edge_scheduler.py:2562"),
    ]
    rows = []
    for label, cval, eval_, note, ccode, ecode in common:
        delta = eval_ - cval if eval_ is not None and cval is not None else None
        rows.append([label, cval, eval_, delta, delta / cval if cval else None, note, ccode, ecode])
    write_table(
        ws, 4,
        ["阶段", "集中式_ms", "边云式_ms", "差值_ms", "差值比例", "阶段口径", "集中式代码位置", "边云代码位置"],
        rows,
        [34, 16, 16, 16, 14, 44, 28, 32],
    )
    for r in range(5, 5 + len(rows)):
        ws.cell(r, 5).number_format = "0.00%"

    ws = base_sheet(wb, "架构特有阶段", "边云特有绑定、UP/DOWN、尾窗与 UNEMBED，不与集中式 KV 阶段强行一一对应。")
    special = [
        ("P 绑定", None, edge_stages["edge_to_p_bind"]["elapsed_ms"], "边云 probe、控制面和 P 绑定", "lwd_edge_scheduler.py:2102"),
        ("P 上传", None, edge_stages["p_upload"]["elapsed_ms"], "边侧输入窗口派发", "lwd_edge_scheduler.py:3708"),
        ("P 计算/导出", None, edge_stages["p_compute_export"]["elapsed_ms"], "最后上传到 P 完成通知", "lwd_edge_scheduler.py:2195"),
        ("D 绑定", None, edge_stages["p_to_d_bind"]["elapsed_ms"], "P 完成到 D 绑定", "lwd_edge_scheduler.py:2139"),
        ("D 尾窗派发", None, edge_stages["d_tail_dispatch"]["elapsed_ms"], "D 绑定到尾窗派发", "lwd_edge_scheduler.py:2421"),
        ("D→边侧首 token", None, edge_stages["d_to_edge_first_token"]["elapsed_ms"], "D 导入、首步、DOWN、UNEMBED", "lwd_edge_scheduler.py:2534"),
        ("DOWN send 平均/P95", None, f"{edge['stats']['down_send_total_ms']['mean']:.2f} / {edge['stats']['down_send_total_ms']['p95']:.2f}", "边云 sidecar 发送", "sidecar_channel.py:305"),
        ("DOWN recv 平均/P95", None, f"{edge['stats']['down_recv_total_ms']['mean']:.2f} / {edge['stats']['down_recv_total_ms']['p95']:.2f}", "边云 sidecar 接收", "sidecar_channel.py:329"),
        ("UNEMBED 平均/P95", None, f"{edge['stats']['unembed_total_ms']['mean']:.2f} / {edge['stats']['unembed_total_ms']['p95']:.2f}", "边侧 LM head/select", "lwd_edge_worker.py:617"),
        ("Meta-harvest 平均/P95", None, f"{edge['stats']['meta_total_ms']['mean']:.2f} / {edge['stats']['meta_total_ms']['p95']:.2f}", "输出线程收割", "multiproc_executor.py:1271"),
        ("边云收尾", None, edge_stages["edge_finalize"]["elapsed_ms"], "K 回报与最终 usage", "lwd_edge_scheduler.py:1912"),
    ]
    write_table(ws, 4, ["边云特有阶段", "集中式", "边云式", "阶段作用", "边云代码位置"], special, [30, 16, 22, 58, 34])

    ws = base_sheet(wb, "Decode统计对比", "两种架构 NPU forward 统计；不把 tokens=4 直接当作 accepted tokens。")
    cstat = derived["d_steady_forward_max_per_step"]
    rows = [
        ["步数", cstat["count"], edge_npu_stats["count"], edge_npu_stats["count"] - cstat["count"], f"集中式{cstat['count']}，边云{edge_npu_stats['count']}"],
        ["mean_ms", cstat["mean"], edge_npu_stats["mean"], edge_npu_stats["mean"] - cstat["mean"], "按每步双rank最大"],
        ["p50_ms", cstat["p50"], edge_npu_stats["p50"], edge_npu_stats["p50"] - cstat["p50"], "中位数"],
        ["p90_ms", cstat["p90"], edge_npu_stats["p90"], edge_npu_stats["p90"] - cstat["p90"], "P90"],
        ["p95_ms", cstat["p95"], edge_npu_stats["p95"], edge_npu_stats["p95"] - cstat["p95"], "P95"],
        ["p99_ms", cstat["p99"], edge_npu_stats["p99"], edge_npu_stats["p99"] - cstat["p99"], "P99"],
        ["max_ms", cstat["max"], edge_npu_stats["max"], edge_npu_stats["max"] - cstat["max"], "最大值"],
        ["Decode wall_ms", derived["d_decode_wall_ms"], edge_stages["decode_generation"]["elapsed_ms"], edge_stages["decode_generation"]["elapsed_ms"] - derived["d_decode_wall_ms"], "输出数未对齐，不评价吞吐"],
        ["TPOT_ms", central_tpot, edge_tpot, (edge_tpot - central_tpot) if edge_tpot is not None and central_tpot is not None else None, "无明确 completion_tokens 的一侧留空"],
    ]
    write_table(ws, 4, ["指标", "集中式", "边云式", "差值", "说明"], rows, [28, 20, 20, 18, 58])

    ws = base_sheet(wb, "代码与证据", "代码行号来自 logger 调用位置；日志行号来自原始文件物理行。")
    evidence = [
        ("集中式 P/D 阶段", "vllm/v1/core/sched/scheduler.py:1848", "cloud_p.log / cloud_d.log"),
        ("边云主阶段", "vllm/v1/lwd_control/control_edge_scheduler/lwd_edge_scheduler.py:1724/2102/2195/2139/2421/2534/2562/3708/1912/2826", "edge.log"),
        ("NPU forward", "vllm-ascend/vllm_ascend/worker/model_runner_v1.py:2340", "cloud_p.log / cloud_d.log"),
        ("边侧 EMBED/UNEMBED", "vllm-ascend/vllm_ascend/worker/lwd_edge_worker.py:358/617", "edge.log"),
        ("Sidecar", "vllm-ascend/vllm_ascend/distributed/lwd_comm/sidecar_channel.py:305/329", "worker sidecar 日志"),
        ("D import", "vllm/v1/lwd_control/control_cloud_scheduler/lwd_cloud_mixed_scheduler.py:348", "cloud_d.log"),
        ("KV transfer", "vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py:980", "cloud_d.log"),
        ("对比限制", "单请求、不同部署时间、集中式输出数缺失", "不做严格吞吐或 TPOT 优劣结论"),
    ]
    write_table(ws, 4, ["内容", "代码位置/说明", "证据文件"], evidence, [28, 105, 34])
    wb.save(COMPARE_BOOK)

    # Validate both workbooks.
    central_checked = load_workbook(CENTRAL_BOOK, data_only=False)
    compare_checked = load_workbook(COMPARE_BOOK, data_only=False)
    assert central_checked["D逐步NPU"].max_row == 411
    assert central_checked["P与D阶段"].max_row == len(central["stages"]) + 4
    assert compare_checked["共同阶段对比"].max_row == 15
    assert compare_checked["对比总览"]["C7"].value is not None
    assert CENTRAL_BOOK.stat().st_size > 30_000
    assert COMPARE_BOOK.stat().st_size > 10_000
    print(f"saved={CENTRAL_BOOK.resolve()}")
    print(f"saved={COMPARE_BOOK.resolve()}")


if __name__ == "__main__":
    main()
