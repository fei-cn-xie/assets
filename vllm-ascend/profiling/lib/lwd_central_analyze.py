from __future__ import annotations

import json
import os
import re
from pathlib import Path
from statistics import mean

ROOT = Path(os.environ["PD_CENTRAL_LOG_DIR"])
BASE = os.environ.get("PD_CENTRAL_REQUEST_BASE")
if not BASE:
    raise ValueError("PD_CENTRAL_REQUEST_BASE is required; use --request-id or let the CLI detect it")


def find_one(pattern: str) -> Path:
    matches = sorted(ROOT.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"日志文件匹配 {pattern} 应有 1 个，实际为 {len(matches)} 个")
    return matches[0]


P = find_one("*_cloud_p.log")
D = find_one("*_cloud_d.log")
PROXY = next(iter(ROOT.glob("*_proxy.log")), None)


def num(text, key, cast=float):
    m = re.search(rf"\b{key}=(-?[\d.]+)", text)
    return cast(m.group(1)) if m else None


def mono(text):
    return num(text, "mono")


def code_location(text):
    m = re.search(r"\[([A-Za-z0-9_]+\.py):(\d+)\]", text)
    return f"{m.group(1)}:{m.group(2)}" if m else ""


def stage(text, role, lineno):
    return {
        "role": role,
        "request_id": re.search(r"req=(\S+)", text).group(1),
        "kv_role": re.search(r"role=(\S+)", text).group(1),
        "stage": re.search(r"stage=(\S+)", text).group(1),
        "elapsed_ms": num(text, "elapsed_ms"),
        "total_ms": num(text, "total_ms"),
        "prompt_tokens": num(text, "prompt_tokens", int),
        "completion_tokens": num(text, "completion_tokens", int),
        "new_token_count": num(text, "new_token_count", int),
        "ts_wall": num(text, "ts_wall"),
        "mono": mono(text),
        "code": code_location(text),
        "line": lineno,
        "text": text,
    }


def forward(text, role, lineno):
    return {
        "role": role,
        "mono": mono(text),
        "npu_ms": num(text, "npu_ms"),
        "tokens": num(text, "tokens", int),
        "reqs": num(text, "reqs", int),
        "worker": "TP0" if "Worker_TP0" in text else "TP1",
        "code": code_location(text),
        "line": lineno,
        "text": text,
    }


def stats(values):
    vals = sorted(v for v in values if v is not None and v >= 0)
    if not vals:
        return {}
    def pct(q):
        pos = (len(vals) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(vals) - 1)
        return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)
    return {
        "count": len(vals),
        "mean": mean(vals),
        "p50": pct(0.5),
        "p90": pct(0.9),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": max(vals),
        "sum": sum(vals),
    }


def main():
    stages = []
    forwards = []
    transfers = []
    for role, path in (("prefill", P), ("decode", D)):
        with path.open(encoding="utf-8") as f:
            for lineno, text in enumerate(f, 1):
                if "[central_pd][latency]" in text and BASE in text:
                    stages.append(stage(text, role, lineno))
                elif "model-forward" in text:
                    forwards.append(forward(text, role, lineno))
                elif "KV cache transfer for request" in text and BASE in text:
                    transfers.append({
                        "role": role,
                        "took_ms": float(re.search(r"took ([\d.]+) ms", text).group(1)),
                        "code": code_location(text),
                        "line": lineno,
                        "text": text,
                    })
    by_role_stage = {(s["kv_role"], s["stage"]): s for s in stages}
    d_finished = by_role_stage[("kv_consumer", "finished")]
    d_forwards = [f for f in forwards if f["role"] == "decode" and f["mono"] <= d_finished["mono"]]
    p_forwards = [f for f in forwards if f["role"] == "prefill"]
    steady = [f for f in d_forwards if f["tokens"] and f["tokens"] > 1]
    per_step_max = []
    for idx in range(0, len(steady), 2):
        pair = steady[idx : idx + 2]
        if len(pair) == 2:
            per_step_max.append(max(f["npu_ms"] for f in pair))
    p_accept = by_role_stage[("kv_producer", "accepted")]
    d_first = by_role_stage[("kv_consumer", "first_token")]
    result = {
        "base_request": BASE,
        "request_ids": sorted({s["request_id"] for s in stages}),
        "stages": sorted(stages, key=lambda s: s["ts_wall"]),
        "transfers": transfers,
        "forwards": p_forwards + d_forwards,
        "counts": {
            "p_forward_logs": len(p_forwards),
            "d_forward_logs_through_finished": len(d_forwards),
            "d_steady_forward_logs": len(steady),
            "d_steady_step_pairs": len(per_step_max),
            "central_stage_count": len(stages),
        },
        "derived": {
            "global_ttft_ms": (d_first["ts_wall"] - p_accept["ts_wall"]) * 1000,
            "d_decode_wall_ms": d_finished["elapsed_ms"],
            "d_first_forward_max_ms": max(f["npu_ms"] for f in d_forwards if f["tokens"] == 1),
            "p_forward_max_ms": max(f["npu_ms"] for f in p_forwards),
            "d_steady_forward_max_per_step": stats(per_step_max),
            "d_steady_forward_all_rank_logs": stats([f["npu_ms"] for f in steady]),
            "output_tokens_logged": d_finished.get("completion_tokens") is not None,
            "d_first_to_finished_output_tokens": d_finished.get("completion_tokens"),
        },
        "code_locations": sorted({f["code"] for f in forwards if f["code"]}),
    }
    out = Path(os.environ["PD_CENTRAL_JSON"])
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if os.environ.get("PD_VERBOSE") == "1":
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
