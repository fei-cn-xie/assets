from __future__ import annotations

import json
import os
import re
from pathlib import Path
from statistics import mean

ROOT = Path(os.environ["PD_EDGE_LOG_DIR"])
REQ = os.environ.get("PD_REQUEST_ID")
if not REQ:
    raise ValueError("PD_REQUEST_ID is required; use --request-id or let the CLI detect it")
def find_one(pattern: str) -> Path:
    matches = sorted(ROOT.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"日志文件匹配 {pattern} 应有 1 个，实际为 {len(matches)} 个")
    return matches[0]


EDGE = find_one("*_edge.log")
P = find_one("*_cloud_p.log")
D = find_one("*_cloud_d.log")
SP = next(iter(ROOT.glob("*_sidecar_p.log")), None)
SD = next(iter(ROOT.glob("*_sidecar_d.log")), None)

FILES = {"edge": EDGE, "prefill": P, "decode": D, "sidecar_p": SP, "sidecar_d": SD}


def lines(path: Path):
    if path is None:
        return
    with path.open(encoding="utf-8") as f:
        for lineno, text in enumerate(f, 1):
            yield lineno, text.rstrip("\n")


def mono(text: str):
    m = re.search(r"\bmono=(-?[\d.]+)", text)
    return float(m.group(1)) if m else None


def code_loc(text: str):
    m = re.search(r"\[([A-Za-z0-9_]+\.py):(\d+)\]", text)
    return f"{m.group(1)}:{m.group(2)}" if m else ""


def seq(text: str, key="seq"):
    m = re.search(rf"\b{key}=(\d+)", text)
    return int(m.group(1)) if m else None


def field(text: str, key: str, cast=float):
    m = re.search(rf"\b{key}=(-?[\d.]+)", text)
    return cast(m.group(1)) if m else None


def sidecar(text: str, file_role: str, lineno: int):
    return {
        "kind": "sidecar",
        "file_role": file_role,
        "line": lineno,
        "role": re.search(r"\brole=(\w+)", text).group(1),
        "op": re.search(r"\bop=(\w+)", text).group(1),
        "channel": re.search(r"\bchannel=(\w+)", text).group(1),
        "seq": seq(text),
        "mono": mono(text),
        "bytes": field(text, "bytes", int),
        "queue_ms": field(text, "queue_ms"),
        "clone_ms": field(text, "clone_ms"),
        "d2h_ms": field(text, "d2h_ms"),
        "serialize_ms": field(text, "serialize_ms"),
        "network_ms": field(text, "network_ms"),
        "slot_wait_ms": field(text, "slot_wait_ms"),
        "rx_wait_ms": field(text, "rx_wait_ms"),
        "h2d_ms": field(text, "h2d_ms"),
        "total_ms": field(text, "total_ms"),
        "text": text,
    }


def model_forward(text: str, file_role: str, lineno: int):
    return {
        "kind": "model_forward",
        "file_role": file_role,
        "line": lineno,
        "seq": seq(text, "seqno"),
        "mono": mono(text),
        "npu_ms": field(text, "npu_ms"),
        "tokens": field(text, "tokens", int),
        "reqs": field(text, "reqs", int),
        "worker": "TP0" if "Worker_TP0" in text else "TP1",
        "text": text,
    }


def unembed(text: str, lineno: int):
    return {
        "kind": "unembed",
        "file_role": "edge",
        "line": lineno,
        "seq": seq(text, "seqno"),
        "mono": mono(text),
        "post_recv": field(text, "post_recv"),
        "wait_tensor": field(text, "wait_tensor"),
        "lm_head": field(text, "lm_head"),
        "select": field(text, "select"),
        "total_ms": field(text, "total"),
        "text": text,
    }


def cloud_step(text: str, lineno: int):
    return {
        "kind": "cloud_step",
        "file_role": "decode",
        "line": lineno,
        "mono": mono(text),
        "sample_ms": field(text, "sample"),
        "send_ms": field(text, "send"),
        "total_ms": field(text, "total"),
        "text": text,
    }


def collect(text: str, lineno: int):
    return {
        "kind": "collect",
        "file_role": "decode",
        "line": lineno,
        "mono": mono(text),
        "rank_ms": field(text, "rank"),
        "pinned_ms": field(text, "pinned"),
        "total_ms": field(text, "total"),
        "worker": "TP0" if "Worker_TP0" in text else "TP1",
        "text": text,
    }


def meta(text: str, lineno: int):
    return {
        "kind": "meta",
        "file_role": "decode",
        "line": lineno,
        "mono": mono(text),
        "total_ms": field(text, "dur"),
        "text": text,
    }


def percentile(values, q):
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def stats(values):
    vals = [v for v in values if v is not None and v >= 0]
    if not vals:
        return {}
    return {
        "count": len(vals),
        "mean": mean(vals),
        "p50": percentile(vals, 0.5),
        "p90": percentile(vals, 0.9),
        "p95": percentile(vals, 0.95),
        "p99": percentile(vals, 0.99),
        "max": max(vals),
        "sum": sum(vals),
    }


def main():
    events = []
    accepted = {}
    edge_stage = []
    central = []
    transfers = []
    request_meta = {}
    decode_meta = {}
    request_done = None
    import_ready = None
    api_response = None
    for role, path in FILES.items():
        for lineno, text in lines(path):
            if "registered req=" in text and REQ in text and "role=prefill" in text:
                request_meta = {
                    "n": field(text, "N", int),
                    "m": field(text, "M", int),
                    "h": field(text, "H", int),
                    "p_end": field(text, "p_end", int),
                    "r": field(text, "R", int),
                    "binding": re.search(r"binding=(\S+)", text).group(1),
                }
                continue
            if "registered req=" in text and REQ in text and "role=decode" in text:
                decode_meta = {
                    "binding": re.search(r"binding=(\S+)", text).group(1),
                    "r": field(text, "R", int),
                }
                continue
            if "Streaming response for chat completion" in text:
                cost_match = re.search(r"cost\s*=\s*([\d.]+)", text)
                now_match = re.search(r"now\s*=\s*([\d.]+)", text)
                api_response = {
                    "total_ms": float(cost_match.group(1)) * 1000 if cost_match else None,
                    "mono": float(now_match.group(1)) if now_match else None,
                    "line": lineno,
                    "text": text,
                }
                continue
            if REQ not in text and not any(
                marker in text
                for marker in (
                    "model-forward",
                    "event=sidecar",
                    "cloud-step",
                    "[perf] collect",
                    "meta-harvest",
                    "[perf] embed",
                    "[perf] unembed",
                    "[edge-worker] UNEMBED",
                    "frame_host_ready",
                    "worker_exec_done",
                    "gate_ready",
                    "D import ready",
                    "KV cache transfer for request",
                    "request done",
                )
            ):
                continue
            if "[Lwd][pd][perf] request done" in text and REQ in text:
                request_done = {
                    "mono": mono(text),
                    "completion_tokens": field(text, "G", int),
                    "accepted_k": field(text, "K", int),
                    "delivered": field(text, "delivered", int),
                    "code": code_loc(text),
                    "line": lineno,
                    "text": text,
                }
            elif "D import ready" in text and REQ in text:
                import_ready = {
                    "total_ms": (field(text, "admit_to_ready") or 0) * 1000,
                    "mono": mono(text),
                    "code": code_loc(text),
                    "line": lineno,
                    "text": text,
                }
            elif "KV cache transfer for request" in text and REQ in text:
                transfers.append({
                    "took_ms": float(re.search(r"took ([\d.]+) ms", text).group(1)),
                    "code": code_loc(text),
                    "line": lineno,
                    "text": text,
                })
            elif "worker_exec_done" in text:
                events.append({
                    "kind": "worker_exec",
                    "file_role": role,
                    "line": lineno,
                    "seq": seq(text),
                    "mono": mono(text),
                    "total_ms": field(text, "total_ms"),
                    "text": text,
                })
            elif "gate_ready" in text:
                events.append({
                    "kind": "gate_ready",
                    "file_role": role,
                    "line": lineno,
                    "seq": seq(text),
                    "mono": mono(text),
                    "total_ms": field(text, "gate_wait_ms"),
                    "text": text,
                })
            elif "[Lwd][pd][latency]" in text and REQ in text:
                m = re.search(r"stage=(\S+)", text)
                row = {
                    "kind": "edge_stage",
                    "file_role": role,
                    "line": lineno,
                    "stage": m.group(1),
                    "mono": mono(text),
                    "elapsed_ms": field(text, "elapsed_ms"),
                    "total_ms": field(text, "total_ms"),
                    "text": text,
                }
                edge_stage.append(row)
            elif "[central_pd][latency]" in text and REQ in text:
                central.append({
                    "kind": "central",
                    "file_role": role,
                    "line": lineno,
                    "role": re.search(r"role=(\S+)", text).group(1),
                    "stage": re.search(r"stage=(\S+)", text).group(1),
                    "mono": mono(text),
                    "ts_wall": field(text, "ts_wall"),
                    "elapsed_ms": field(text, "elapsed_ms"),
                    "total_ms": field(text, "total_ms"),
                    "text": text,
                })
            elif "event=sidecar" in text:
                events.append(sidecar(text, role, lineno))
            elif "model-forward" in text:
                events.append(model_forward(text, role, lineno))
            elif "[perf] unembed" in text:
                events.append(unembed(text, lineno))
            elif "[perf] cloud-step" in text:
                events.append(cloud_step(text, lineno))
            elif "[perf] collect" in text:
                events.append(collect(text, lineno))
            elif "meta-harvest" in text:
                events.append(meta(text, lineno))
            elif "[perf] embed" in text:
                events.append({
                    "kind": "embed",
                    "file_role": role,
                    "line": lineno,
                    "seq": seq(text, "seqno"),
                    "mono": mono(text),
                    "forward_ms": field(text, "forward"),
                    "submit_send_ms": field(text, "submit_send"),
                    "npu_ms": field(text, "npu_ms"),
                    "total_ms": field(text, "total"),
                    "tokens": field(text, "tokens", int),
                    "text": text,
                })
            elif "frame_host_ready" in text:
                events.append({
                    "kind": "frame",
                    "file_role": role,
                    "line": lineno,
                    "seq": seq(text),
                    "mono": mono(text),
                    "grant_ms": field(text, "grant_ms"),
                    "receive_ms": field(text, "receive_ms"),
                    "pool_write_ms": field(text, "pool_write_ms"),
                    "total_ms": field(text, "total_ms"),
                    "text": text,
                })
            elif "[edge-worker] UNEMBED" in text:
                s = seq(text, "seqno")
                if s is None:
                    continue
                accepted[s] = {
                    "accepted": field(text, "accepted", int),
                    "num_elements": field(text, "num_elements", int),
                    "line": lineno,
                    "text": text,
                }

    down_send = sorted(
        [e for e in events if e["kind"] == "sidecar" and e["op"] == "send" and e["channel"] == "lwd_down" and e["file_role"] == "decode"],
        key=lambda e: e["seq"],
    )
    down_recv = sorted(
        [e for e in events if e["kind"] == "sidecar" and e["op"] == "recv" and e["channel"] == "lwd_down" and e["file_role"] == "edge"],
        key=lambda e: e["seq"],
    )
    unembeds = sorted(
        [e for e in events if e["kind"] == "unembed"], key=lambda e: e["seq"]
    )
    forwards_d = [e for e in events if e["kind"] == "model_forward" and e["file_role"] == "decode"]
    forwards_p = [e for e in events if e["kind"] == "model_forward" and e["file_role"] == "prefill"]
    steps = [e for e in events if e["kind"] == "cloud_step"]
    collects = [e for e in events if e["kind"] == "collect"]
    metas = [e for e in events if e["kind"] == "meta"]

    seqs = [e["seq"] for e in down_send]
    if not seqs:
        raise ValueError("未找到目标请求的 DOWN sidecar 数据，请检查 request_id 和日志范围")
    accepted = {s: v for s, v in accepted.items() if s in set(seqs)}
    required = {
        "down_recv": len(down_recv),
        "unembed": len(unembeds),
        "cloud_step": len(steps),
        "collect": len(collects),
        "meta": len(metas),
        "model_forward": len(forwards_d),
    }
    minimum = {
        "down_recv": len(seqs),
        "unembed": len(seqs),
        "cloud_step": len(seqs),
        "collect": len(seqs) * 2,
        "meta": len(seqs),
        "model_forward": len(seqs) * 2,
    }
    insufficient = [f"{name}={actual} < {minimum[name]}" for name, actual in required.items() if actual < minimum[name]]
    if insufficient:
        raise ValueError("Decode 日志不完整，无法按 seqno 对齐：" + ", ".join(insufficient))
    steady_forwards = [e["npu_ms"] for e in forwards_d if e["tokens"] is not None and e["tokens"] > 1]
    decode_steps = []
    for idx, seqno in enumerate(seqs):
        forward_pair = forwards_d[idx * 2 : idx * 2 + 2]
        collect_pair = collects[idx * 2 : idx * 2 + 2]
        step = {
            "seq": seqno,
            **accepted.get(seqno, {}),
            "forward_tp0_npu_ms": forward_pair[0]["npu_ms"],
            "forward_tp1_npu_ms": forward_pair[1]["npu_ms"],
            "forward_max_npu_ms": max(e["npu_ms"] for e in forward_pair),
            "forward_mono_max": max(e["mono"] for e in forward_pair),
            "cloud_step_mono": steps[idx]["mono"],
            "cloud_step_sample_ms": steps[idx]["sample_ms"],
            "cloud_step_send_ms": steps[idx]["send_ms"],
            "cloud_step_total_ms": steps[idx]["total_ms"],
            "collect_max_ms": max(e["total_ms"] for e in collect_pair),
            "meta_mono": metas[idx]["mono"],
            "meta_total_ms": metas[idx]["total_ms"],
            "send": down_send[idx],
            "recv": down_recv[idx],
            "unembed": unembeds[idx],
        }
        if decode_steps:
            previous = decode_steps[-1]["unembed"]["mono"]
            interval = (step["unembed"]["mono"] - previous) * 1000.0
            step["unembed_interval_ms"] = interval
            accepted_count = step.get("accepted", 0)
            step["interval_per_accepted_token_ms"] = (
                interval / accepted_count if accepted_count > 0 else None
            )
        else:
            step["unembed_interval_ms"] = None
            step["interval_per_accepted_token_ms"] = None
        decode_steps.append(step)

    result = {
        "request": REQ,
        "request_meta": request_meta,
        "decode_meta": decode_meta,
        "request_done": request_done,
        "import_ready": import_ready,
        "api_response": api_response,
        "transfers": transfers,
        "edge_stages": edge_stage,
        "central_stages": central,
        "counts": {
            "down_send": len(down_send),
            "down_recv": len(down_recv),
            "unembed": len(unembeds),
            "cloud_steps": len(steps),
            "decode_forward_logs": len(forwards_d),
            "decode_forward_tokens_gt1": len(steady_forwards),
            "collect_logs": len(collects),
            "meta_logs": len(metas),
            "down_seq_min": min(seqs),
            "down_seq_max": max(seqs),
            "accepted_sum": sum(v["accepted"] for v in accepted.values()),
            "accepted_steps": len(accepted),
        },
        "stats": {
            "decode_forward_npu_ms": stats(steady_forwards),
            "down_send_total_ms": stats([e["total_ms"] for e in down_send]),
            "down_send_d2h_ms": stats([e["d2h_ms"] for e in down_send]),
            "down_send_network_ms": stats([e["network_ms"] for e in down_send]),
            "down_recv_total_ms": stats([e["total_ms"] for e in down_recv]),
            "down_recv_rx_wait_ms": stats([e["rx_wait_ms"] for e in down_recv]),
            "down_recv_h2d_ms": stats([e["h2d_ms"] for e in down_recv]),
            "unembed_total_ms": stats([e["total_ms"] for e in unembeds]),
            "unembed_wait_tensor_ms": stats([e["wait_tensor"] for e in unembeds]),
            "unembed_lm_head_ms": stats([e["lm_head"] for e in unembeds]),
            "unembed_select_ms": stats([e["select"] for e in unembeds]),
            "cloud_step_total_ms": stats([e["total_ms"] for e in steps]),
            "cloud_step_sample_ms": stats([e["sample_ms"] for e in steps]),
            "collect_total_ms": stats([e["total_ms"] for e in collects]),
            "meta_total_ms": stats([e["total_ms"] for e in metas]),
            "accepted_per_step": stats([v["accepted"] for v in accepted.values() if v["accepted"] > 0]),
        },
        "up": [
            e for e in events
            if e["kind"] in {"sidecar", "embed", "frame", "model_forward", "worker_exec", "gate_ready"}
            and e.get("seq") in {1, 2}
        ],
        "decode_steps": decode_steps,
        "top": {
            "send_d2h": sorted(down_send, key=lambda e: e["d2h_ms"], reverse=True)[:10],
            "send_network": sorted(down_send, key=lambda e: e["network_ms"], reverse=True)[:10],
            "unembed": sorted(unembeds, key=lambda e: e["total_ms"], reverse=True)[:10],
            "recv": sorted(down_recv, key=lambda e: e["total_ms"], reverse=True)[:10],
            "meta": sorted(metas, key=lambda e: e["total_ms"], reverse=True)[:10],
        },
    }
    out = Path(os.environ["PD_EDGE_JSON"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if os.environ.get("PD_VERBOSE") == "1":
        print(json.dumps({"counts": result["counts"], "stats": result["stats"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
