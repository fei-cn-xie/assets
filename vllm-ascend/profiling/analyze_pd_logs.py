#!/usr/bin/env python3
"""One-command PD log analysis for edge-cloud and centralized deployments."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

LIB_DIR = Path(__file__).resolve().parent / "lib"


def log_files(directory: Path):
    if not directory.is_dir():
        raise ValueError(f"日志目录不存在: {directory}")
    files = sorted(directory.rglob("*.log"))
    if not files:
        raise ValueError(f"目录中没有 .log 文件: {directory}")
    return files


def detect_edge_request(directory: Path) -> str:
    request_ids = set()
    for path in log_files(directory):
        with path.open(encoding="utf-8", errors="replace") as file:
            for line in file:
                if "stage=request_accepted" not in line:
                    continue
                match = re.search(r"req=(\S+)", line)
                if match:
                    request_ids.add(match.group(1))
    if len(request_ids) != 1:
        raise ValueError(
            f"边云请求 ID 数量应为 1，实际为 {len(request_ids)}。"
            "请用 --request-id 指定，或缩小日志时间范围。"
        )
    return request_ids.pop()


def detect_central_base(directory: Path) -> str:
    request_ids = set()
    for path in log_files(directory):
        with path.open(encoding="utf-8", errors="replace") as file:
            for line in file:
                if "[central_pd][latency]" not in line or "stage=accepted" not in line:
                    continue
                if "role=kv_producer" not in line:
                    continue
                match = re.search(r"req=(\S+)", line)
                if match:
                    request_ids.add(match.group(1))
    bases = {re.sub(r"-[0-9a-f]{8}$", "", request_id) for request_id in request_ids}
    if len(bases) != 1:
        raise ValueError(
            f"集中式请求前缀数量应为 1，实际为 {len(bases)}。"
            "请用 --request-id 指定，或缩小日志时间范围。"
        )
    return bases.pop()


def run_script(name: str, env: dict[str, str], verbose: bool) -> None:
    script = LIB_DIR / name
    if verbose:
        print(f"run: {sys.executable} {script}")
    subprocess.run([sys.executable, str(script)], env=env, check=True)


def base_env(output_dir: Path, temp_dir: Path, verbose: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(LIB_DIR.parent)
    env["PD_OUTPUT_DIR"] = str(output_dir)
    env["PD_VERBOSE"] = "1" if verbose else "0"
    env["PD_REQUEST_ID"] = "central-analysis"
    env["PD_EDGE_JSON"] = str(temp_dir / "edge.json")
    env["PD_CENTRAL_JSON"] = str(temp_dir / "central.json")
    return env


def output_paths(output_dir: Path):
    return {
        "edge": output_dir / "edge_pd_analysis.xlsx",
        "central": output_dir / "central_pd_analysis.xlsx",
        "compare": output_dir / "pd_comparison.xlsx",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    edge_parser = subparsers.add_parser("edge", help="分析边云 PD 日志")
    edge_parser.add_argument("--log-dir", required=True)
    edge_parser.add_argument("--request-id")
    edge_parser.add_argument("--output-dir", default=".")
    edge_parser.add_argument("--verbose", action="store_true")

    central_parser = subparsers.add_parser("central", help="分析集中式 PD 日志")
    central_parser.add_argument("--log-dir", required=True)
    central_parser.add_argument("--request-id")
    central_parser.add_argument("--output-dir", default=".")
    central_parser.add_argument("--verbose", action="store_true")

    compare_parser = subparsers.add_parser("compare", help="对比边云与集中式 PD 日志")
    compare_parser.add_argument("--edge-dir", required=True)
    compare_parser.add_argument("--central-dir", required=True)
    compare_parser.add_argument("--edge-request-id")
    compare_parser.add_argument("--central-request-id")
    compare_parser.add_argument("--output-dir", default=".")
    compare_parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir)

    with tempfile.TemporaryDirectory(prefix="pd-log-analysis-") as temp:
        temp_dir = Path(temp)
        env = base_env(output_dir, temp_dir, args.verbose)
        env["PD_EDGE_BOOK"] = str(paths["edge"])
        env["PD_CENTRAL_BOOK"] = str(paths["central"])
        env["PD_COMPARE_BOOK"] = str(paths["compare"])

        if args.mode == "edge":
            log_dir = Path(args.log_dir).expanduser().resolve()
            env["PD_EDGE_LOG_DIR"] = str(log_dir)
            env["PD_REQUEST_ID"] = args.request_id or detect_edge_request(log_dir)
            env["PD_WRITE_COMPARISON"] = "0"
            run_script("lwd_perf_analyze.py", env, args.verbose)
            run_script("build_lwd_perf_excel.py", env, args.verbose)
            run_script("add_code_locations_to_edge_excel.py", env, args.verbose)
            env["PD_CODE_MAP_BOOKS"] = str(paths["edge"])
            run_script("add_decode_code_map.py", env, args.verbose)
            print(f"saved: {paths['edge']}")

        elif args.mode == "central":
            log_dir = Path(args.log_dir).expanduser().resolve()
            env["PD_CENTRAL_LOG_DIR"] = str(log_dir)
            request_id = args.request_id or detect_central_base(log_dir)
            env["PD_CENTRAL_REQUEST_BASE"] = re.sub(r"-[0-9a-f]{8}$", "", request_id)
            env["PD_WRITE_COMPARISON"] = "0"
            run_script("lwd_central_analyze.py", env, args.verbose)
            run_script("build_central_and_comparison_excel.py", env, args.verbose)
            env["PD_CODE_MAP_BOOKS"] = str(paths["central"])
            run_script("add_decode_code_map.py", env, args.verbose)
            print(f"saved: {paths['central']}")

        else:
            edge_dir = Path(args.edge_dir).expanduser().resolve()
            central_dir = Path(args.central_dir).expanduser().resolve()
            env["PD_EDGE_LOG_DIR"] = str(edge_dir)
            env["PD_CENTRAL_LOG_DIR"] = str(central_dir)
            env["PD_REQUEST_ID"] = args.edge_request_id or detect_edge_request(edge_dir)
            env["PD_CENTRAL_REQUEST_BASE"] = args.central_request_id or detect_central_base(central_dir)
            env["PD_WRITE_COMPARISON"] = "1"
            run_script("lwd_perf_analyze.py", env, args.verbose)
            run_script("lwd_central_analyze.py", env, args.verbose)
            run_script("build_lwd_perf_excel.py", env, args.verbose)
            run_script("build_central_and_comparison_excel.py", env, args.verbose)
            run_script("add_code_locations_to_edge_excel.py", env, args.verbose)
            run_script("build_unified_stage_comparison.py", env, args.verbose)
            env["PD_CODE_MAP_BOOKS"] = os.pathsep.join(
                str(paths[key]) for key in ("central", "compare", "edge")
            )
            run_script("add_decode_code_map.py", env, args.verbose)
            print(f"saved: {paths['edge']}")
            print(f"saved: {paths['central']}")
            print(f"saved: {paths['compare']}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"分析失败: {exc}")
