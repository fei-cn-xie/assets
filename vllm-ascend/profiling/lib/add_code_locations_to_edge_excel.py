from __future__ import annotations

import json
import os
import re
from copy import copy
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

BOOK = Path(os.environ["PD_EDGE_BOOK"])
DATA = Path(os.environ["PD_EDGE_JSON"])


def code(text):
    m = re.search(r"\[([A-Za-z0-9_]+\.py):(\d+)\]", text)
    return f"{m.group(1)}:{m.group(2)}" if m else ""


def add_code_column(ws, header, values, header_row=4, width=30):
    col = ws.max_column + 1
    source_col = col - 1
    ws.cell(header_row, col, header)
    for attr in ("font", "fill", "border", "alignment"):
        setattr(ws.cell(header_row, col), attr, copy(getattr(ws.cell(header_row, source_col), attr)))
    for offset, value in enumerate(values, 1):
        cell = ws.cell(header_row + offset, col, value)
        for attr in ("font", "fill", "border", "alignment"):
            setattr(cell, attr, copy(getattr(ws.cell(header_row + offset, source_col), attr)))
    ws.column_dimensions[get_column_letter(col)].width = width
    if ws.auto_filter.ref:
        ws.auto_filter.ref = f"A{header_row}:{get_column_letter(col)}{header_row + len(values)}"


def main():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    wb = load_workbook(BOOK)

    ws = wb["边云主阶段"]
    codes = [code(e["text"]) for e in data["edge_stages"]]
    codes.append(data.get("request_done", {}).get("code", "lwd_edge_scheduler.py:2826"))
    add_code_column(ws, "代码位置", codes)

    ws = wb["P与D实例"]
    central = sorted(data["central_stages"], key=lambda e: e["ts_wall"])
    add_code_column(ws, "代码位置", [code(e["text"]) for e in central])

    ws = wb["UP与首token"]
    codes = [code(e["text"]) for e in data["up"]]
    first = data["decode_steps"][0]
    codes.extend([
        code(first["send"]["text"]),
        code(first["recv"]["text"]),
        code(first["unembed"]["text"]),
    ])
    add_code_column(ws, "代码位置", codes)

    ws = wb["Decode逐步"]
    steps = data["decode_steps"]
    add_code_column(ws, "D代码位置", [code(s["send"]["text"]) for s in steps], width=28)
    add_code_column(ws, "Edge代码位置", [code(s["unembed"]["text"]) for s in steps], width=28)

    ws = wb["统计与异常"]
    entries = []
    for category, events in (
        ("send D2H", data["top"]["send_d2h"][:5]),
        ("send network", data["top"]["send_network"][:5]),
        ("UNEMBED total", data["top"]["unembed"][:5]),
        ("recv total", data["top"]["recv"][:5]),
        ("meta-harvest", data["top"]["meta"][:5]),
    ):
        entries.extend(events)
    outlier_header_row = 4 + len(data["stats"]) + 3 + 1 + 2
    # Locate the outlier table by its header instead of relying on layout arithmetic.
    header_row = next(
        row for row in range(1, ws.max_row + 1) if ws.cell(row, 1).value == "类别"
    )
    add_code_column(ws, "代码位置", [code(e["text"]) for e in entries], header_row=header_row)

    ws = wb["请求总览"]
    row = ws.max_row + 2
    ws.cell(row, 1, "关键代码位置")
    ws.cell(row, 1).font = copy(ws.cell(4, 1).font)
    ws.cell(row, 1).fill = copy(ws.cell(4, 1).fill)
    code_rows = [
        ("边云各阶段", "vllm/v1/lwd_control/control_edge_scheduler/lwd_edge_scheduler.py:1724/2102/2195/2139/2421/2534/2562/3708/1912/2826"),
        ("P/D 调度与 KV", "vllm/v1/core/sched/scheduler.py:1848"),
        ("NPU forward", "vllm-ascend/vllm_ascend/worker/model_runner_v1.py:2340"),
        ("边侧 EMBED/UNEMBED", "vllm-ascend/vllm_ascend/worker/lwd_edge_worker.py:358/617"),
        ("Sidecar", "vllm-ascend/vllm_ascend/distributed/lwd_comm/sidecar_channel.py:305/329"),
        ("D import", "vllm/v1/lwd_control/control_cloud_scheduler/lwd_cloud_mixed_scheduler.py:348"),
    ]
    for key, value in code_rows:
        row += 1
        ws.cell(row, 1, key)
        ws.cell(row, 2, value)
        ws.cell(row, 1).font = copy(ws.cell(5, 1).font)
        ws.cell(row, 2).font = copy(ws.cell(5, 2).font)
        ws.cell(row, 2).alignment = copy(ws.cell(5, 3).alignment)
    ws.column_dimensions["C"].width = 105

    ws = wb["说明与日志映射"]
    row = ws.max_row + 1
    ws.cell(row, 1, "代码位置")
    ws.cell(row, 2, "文件.py:行号 来自原始日志的 logger 调用位置；证据行 是日志文件中的物理行号，两者用途不同。")
    for col in (1, 2):
        ws.cell(row, col).font = copy(ws.cell(row - 1, col).font)
        ws.cell(row, col).alignment = copy(ws.cell(row - 1, col).alignment)

    wb.save(BOOK)

    checked = load_workbook(BOOK, data_only=False)
    assert checked["边云主阶段"].cell(4, 9).value == "代码位置"
    assert checked["Decode逐步"].cell(4, 35).value == "D代码位置"
    assert checked["Decode逐步"].cell(4, 36).value == "Edge代码位置"
    assert checked["Decode逐步"]["AI5"].value == "sidecar_channel.py:305"
    assert checked["Decode逐步"]["AJ5"].value == "lwd_edge_worker.py:617"
    print(f"updated={BOOK.resolve()}")


if __name__ == "__main__":
    main()
