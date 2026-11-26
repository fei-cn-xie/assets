"""Stdlib-only checker for pd_host_topology.yaml and pd_proxy.json here.

Run it after changing addresses, ports or buffer sizes:

    python3 check_configs.py                     # this directory's files
    python3 check_configs.py other.yaml other.json

It reimplements the restricted YAML subset these files use plus the rules of
vllm/config/lwd/topology.py and vllm/v1/lwd_control/pd/proxy.py on
feat/pd-sep-lwd-v0.30.0; every rule cites the source line it mirrors. It is a
fast pre-flight, not a replacement for the real thing: the authoritative check
is vLLM starting up and printing its [LWD][config] digest.
"""

from __future__ import annotations

import ipaddress
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- YAML subset


class YamlError(ValueError):
    pass


def _scalar(text: str):
    text = text.strip()
    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        return text[1:-1]
    if text.startswith("'") and text.endswith("'") and len(text) >= 2:
        return text[1:-1]
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "~"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _parse_flow_map(text: str) -> dict:
    if not (text.startswith("{") and text.endswith("}")):
        raise YamlError(f"bad flow mapping: {text!r}")
    out: dict = {}
    body = text[1:-1].strip()
    if not body:
        return out
    for part in body.split(","):
        key, sep, value = part.partition(":")
        if not sep:
            raise YamlError(f"bad flow entry: {part!r}")
        key = key.strip()
        if key in out:
            raise YamlError(f"duplicate flow key: {key!r}")
        out[key] = _scalar(value)
    return out


def _tokenize(text: str) -> list[tuple[int, str]]:
    lines = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        stripped = raw.strip()
        if stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if "\t" in raw[:indent]:
            raise YamlError("tab indentation")
        lines.append((indent, stripped))
    return lines


def _parse_node(lines, i, indent, end):
    if lines[i][1].startswith("- "):
        return _parse_seq(lines, i, indent, end)
    return _parse_map(lines, i, indent, end)


def _parse_map(lines, i, indent, end):
    out: dict = {}
    while i < end:
        ind, content = lines[i]
        if ind < indent or content.startswith("- "):
            break
        if ind > indent:
            raise YamlError(f"unexpected indent {ind} > {indent}: {content!r}")
        key, sep, rest = content.partition(":")
        if not sep:
            raise YamlError(f"not a mapping entry: {content!r}")
        key = key.strip()
        if key in out:
            raise YamlError(f"duplicate YAML key: {key!r}")  # _UniqueKeyLoader
        rest = rest.strip()
        if rest:
            out[key] = _parse_flow_map(rest) if rest.startswith("{") else _scalar(rest)
            i += 1
            continue
        i += 1
        if i < end and lines[i][0] > indent:
            out[key], i = _parse_node(lines, i, lines[i][0], end)
        else:
            out[key] = None
    return out, i


def _parse_seq(lines, i, indent, end):
    out: list = []
    while i < end:
        ind, content = lines[i]
        if ind != indent or not content.startswith("- "):
            break
        item = content[2:].strip()
        i += 1
        j = i
        while j < end and lines[j][0] > indent:
            j += 1
        if item.startswith("{"):
            out.append(_parse_flow_map(item))
        elif ":" in item:
            sub = [(indent + 2, item)] + lines[i:j]
            value, consumed = _parse_map(sub, 0, indent + 2, len(sub))
            if consumed != len(sub):
                raise YamlError(f"trailing content in sequence item: {item!r}")
            out.append(value)
        else:
            out.append(_scalar(item))
        i = j
    return out, i


def load_yaml(path: Path):
    lines = _tokenize(path.read_text(encoding="utf-8"))
    if not lines:
        raise YamlError("empty file")
    value, consumed = _parse_map(lines, 0, lines[0][0], len(lines))
    if consumed != len(lines):
        raise YamlError(f"unparsed trailing lines from {lines[consumed]}")
    return value


# ------------------------------------------------------------ rule helpers


class Report:
    def __init__(self, label: str) -> None:
        self.label = label
        self.errors: list[str] = []
        self.notes: list[str] = []

    def check(self, ok: bool, message: str) -> bool:
        if not ok:
            self.errors.append(message)
        return ok

    def done(self) -> int:
        status = "PASS" if not self.errors else "FAIL"
        print(f"[{status}] {self.label}")
        for note in self.notes:
            print(f"        note: {note}")
        for err in self.errors:
            print(f"        ERROR: {err}")
        return 0 if not self.errors else 1


def _keys(node: dict, allowed: set[str], where: str, report: Report) -> None:
    """extra='forbid' on every topology dataclass."""
    extra = set(node) - allowed
    report.check(not extra, f"{where}: unknown keys {sorted(extra)} (extra=forbid)")


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_contiguous(values: list[int]) -> bool:
    return values == list(range(len(values)))


def _unicast(value, where: str, report: Report) -> None:
    if not report.check(isinstance(value, str), f"{where}: addr must be a string"):
        return
    try:
        addr = ipaddress.ip_address(value)
    except ValueError:
        report.check(False, f"{where}: addr {value!r} is not a valid IP")
        return
    report.check(
        not addr.is_unspecified and not addr.is_multicast,
        f"{where}: addr must be unicast (topology.py:34-38)",
    )


DEPLOYMENT_KEYS = {
    "mode", "scene", "edges_num", "clouds_num", "hccl_world_size",
    "data_plane_comm", "data_plane_host", "pd",
}
HOST_KEYS = {"up_buffer_size", "down_buffer_size", "deployment"}
PD_KEYS = {
    "enabled", "max_window_tokens", "max_inflight_windows", "d_output_quota_rows",
    "p_bind_timeout_s", "d_bind_timeout_s", "usage_wait_timeout_s", "request_ttl_s",
    "tenant_id", "notify_claim_timeout_s", "export_hold_timeout_s",
    "result_hold_timeout_s",
}
FEATURE_KEYS = {"enable_early_recv", "scramble"}
SCRAMBLE_KEYS = {"enable", "device_id", "custom_factor"}
INSTANCE_KEYS = {"id", "dp", "coordination", "stage_role"}
DP_KEYS = {"dp_idx", "addr", "ranks", "ctrl_port", "data_port", "coord_port", "coord_url"}
COORD_KEYS = {
    "enabled", "pd_proxy_url", "placement_only", "tenant_key_file", "consumer_id",
    "listen_host", "probe_poll_budget", "connect_timeout", "probe_timeout",
    "enforce_mm_abi_match",
}
TOP_KEYS = {
    "deployment", "feature_ctrl", "edges", "clouds", "instance_links", "qos",
    "control_plane", "digest",
}


def validate_topology(path: Path, label: str) -> int:
    report = Report(f"{label} ({path})")
    raw = load_yaml(path)

    _keys(raw, TOP_KEYS, "top", report)
    for required in ("deployment", "feature_ctrl", "edges", "clouds"):
        report.check(required in raw, f"top: missing required block {required!r}")

    dep = raw.get("deployment") or {}
    _keys(dep, DEPLOYMENT_KEYS, "deployment", report)
    report.check(dep.get("mode") == 0, "deployment.mode must be 0 (ge=0, le=0)")
    scene = dep.get("scene")
    report.check(
        scene in {"single_instance", "edge_share", "cloud_share", "lwd_cluster"},
        f"deployment.scene invalid: {scene!r}",
    )
    report.check(dep.get("data_plane_comm") == "host", "deployment.data_plane_comm must be host")
    report.check(
        "hccl_world_size" not in dep,
        "deployment.hccl_world_size must be omitted for host comm (topology.py:474-480)",
    )
    host = dep.get("data_plane_host")
    if report.check(isinstance(host, dict), "deployment.data_plane_host is required for host comm"):
        _keys(host, HOST_KEYS, "deployment.data_plane_host", report)
        report.check(
            _is_int(host.get("up_buffer_size")) and host["up_buffer_size"] >= 1,
            "deployment.data_plane_host.up_buffer_size must be explicit for PD (topology.py:435-441)",
        )
        down = host.get("down_buffer_size", 32)
        report.check(_is_int(down) and down >= 1, "down_buffer_size must be an int >= 1")
        report.check(
            host.get("deployment", "in_process") in {"in_process", "sidecar"},
            "data_plane_host.deployment must be in_process|sidecar",
        )

    pd = dep.get("pd") or {}
    _keys(pd, PD_KEYS, "deployment.pd", report)
    pd_enabled = pd.get("enabled", False)
    report.check(pd_enabled is True, "deployment.pd.enabled must be true")

    feat = raw.get("feature_ctrl")
    if report.check(isinstance(feat, dict), "feature_ctrl must be a mapping"):
        _keys(feat, FEATURE_KEYS, "feature_ctrl", report)
        scramble = feat.get("scramble")
        if isinstance(scramble, dict):
            _keys(scramble, SCRAMBLE_KEYS, "feature_ctrl.scramble", report)

    edges = raw.get("edges") or []
    clouds = raw.get("clouds") or []
    report.check(
        dep.get("edges_num") == len(edges) and dep.get("clouds_num") == len(clouds),
        "deployment.edges_num/clouds_num must match edges/clouds length (topology.py:387-393)",
    )
    if scene == "lwd_cluster":
        report.check(len(clouds) >= 2, "lwd_cluster requires at least two clouds (topology.py:633-634)")
    report.check(
        scene != "edge_share",
        "PD requires scene != edge_share (topology.py:442-443)",
    )

    def instance(inst, kind: str, idx: int):
        where = f"{kind}[{idx}]"
        _keys(inst, INSTANCE_KEYS, where, report)
        dp_list = inst.get("dp") or []
        report.check(bool(dp_list), f"{where}.dp must not be empty")
        dps = []
        for k, dp in enumerate(dp_list):
            dw = f"{where}.dp[{k}]"
            _keys(dp, DP_KEYS, dw, report)
            _unicast(dp.get("addr"), dw, report)
            report.check(
                "ranks" not in dp,
                f"{dw}: ranks must be omitted for host comm (topology.py:467-473)",
            )
            for port_key in ("ctrl_port", "data_port", "coord_port"):
                if port_key in dp:
                    port = dp[port_key]
                    report.check(
                        _is_int(port) and 1 <= port <= 65535,
                        f"{dw}.{port_key} must be 1..65535",
                    )
            dps.append(dp)
        indices = [dp.get("dp_idx") for dp in dps]
        report.check(
            all(_is_int(i) for i in indices) and _is_contiguous(indices),
            f"{where}.dp_idx must be unique and contiguous from 0 (topology.py:274-280)",
        )
        coord = inst.get("coordination")
        if coord is not None:
            _keys(coord, COORD_KEYS, f"{where}.coordination", report)
        return inst, dps

    edge_instances = [instance(e, "edges", i) for i, e in enumerate(edges)]
    cloud_instances = [instance(c, "clouds", i) for i, c in enumerate(clouds)]

    for kind, instances in (("edges", edges), ("clouds", clouds)):
        ids = [i.get("id") for i in instances]
        report.check(
            all(_is_int(i) for i in ids) and _is_contiguous(ids),
            f"{kind}.id must be unique and contiguous from 0 (topology.py:375-383)",
        )

    for i, (inst, dps) in enumerate(edge_instances):
        for dp in dps:
            report.check(
                "ctrl_port" not in dp and dp.get("ctrl_port") is None,
                f"edges[{i}]: ctrl_port must be omitted on edges (topology.py:401-407)",
            )
            report.check(
                dp.get("data_port") is None,
                f"edges[{i}]: data_port is cloud-only (topology.py:481-485)",
            )
    for i, (inst, dps) in enumerate(cloud_instances):
        for dp in dps:
            report.check(
                _is_int(dp.get("ctrl_port")),
                f"clouds[{i}]: ctrl_port is required on clouds (topology.py:401-407)",
            )
            report.check(
                _is_int(dp.get("data_port")),
                f"clouds[{i}]: data_port is required on every cloud DP (topology.py:487-494)",
            )

    # port namespaces are per address (topology.py:581-623)
    all_cloud_dps = [dp for _, dps in cloud_instances for dp in dps]
    for key in ("ctrl_port", "data_port", "coord_port"):
        pairs = [(dp.get("addr"), dp[key]) for dp in all_cloud_dps if dp.get(key) is not None]
        report.check(
            len(pairs) == len(set(pairs)),
            f"cloud DPs on the same address must use distinct {key} (topology.py:586-609)",
        )
    listeners = {
        (dp.get("addr"), dp[key]) for dp in all_cloud_dps for key in ("ctrl_port", "data_port")
    }
    for dp in all_cloud_dps:
        if dp.get("coord_port") is not None:
            report.check(
                (dp.get("addr"), dp["coord_port"]) not in listeners,
                f"coord_port {dp.get('coord_port')} conflicts with a ctrl/data port on {dp.get('addr')}",
            )

    # PD-specific rules (topology.py:426-458)
    report.check(
        not any(e.get("stage_role") for e in edges),
        "edges must not carry stage_role (topology.py:427-428)",
    )
    roles = {c.get("stage_role") for c in clouds}
    report.check(
        roles == {"prefill", "decode"},
        f"PD requires exactly one prefill and one decode cloud, got {sorted(map(str, roles))}",
    )
    for kind, instances in (("edges", edges), ("clouds", clouds)):
        for i, inst in enumerate(instances):
            coord = inst.get("coordination") or {}
            report.check(
                coord.get("enabled") is True,
                f"PD requires coordination.enabled on every instance ({kind}[{i}])",
            )
    for i, inst in enumerate(edges):
        coord = inst.get("coordination") or {}
        report.check(
            bool(coord.get("tenant_key_file")),
            f"PD requires edges[{i}].coordination.tenant_key_file (topology.py:449-453)",
        )
        report.check(
            coord.get("placement_only") in (None, False),
            f"PD rejects placement_only on edges[{i}] (topology.py:449-453)",
        )
        url = coord.get("pd_proxy_url") or ""
        parts = urlsplit(url)
        report.check(
            parts.scheme in ("http", "https") and bool(parts.hostname),
            f"PD requires an HTTP(S) pd_proxy_url on edges[{i}] (topology.py:454-456)",
        )
    edge_consumers = {(e.get("coordination") or {}).get("consumer_id") for e in edges}
    report.check(
        bool(pd.get("tenant_id")) or any(edge_consumers),
        "PD requires deployment.pd.tenant_id or an edge consumer_id (topology.py:457-458)",
    )
    for i, (inst, dps) in enumerate(cloud_instances):
        coord = inst.get("coordination") or {}
        if coord.get("enabled"):
            for dp in dps:
                report.check(
                    _is_int(dp.get("coord_port")),
                    f"clouds[{i}]: enabled coordination requires coord_port on every DP "
                    "(topology.py:500-509)",
                )

    links = raw.get("instance_links")
    expected = {(e, c) for e in range(len(edges)) for c in range(len(clouds))}
    if links is None:
        report.notes.append("instance_links omitted -> full mesh derived (topology.py:341-350)")
    else:
        got = set()
        for k, link in enumerate(links):
            _keys(link, {"edge", "cloud"}, f"instance_links[{k}]", report)
            got.add((link.get("edge"), link.get("cloud")))
        report.check(len(got) == len(links), "instance_links must not contain duplicates")
        report.check(
            got == expected,
            f"scene requires full-mesh instance_links, missing {sorted(expected - got)}",
        )

    # host memory budget (vllm_ascend .../lwd_host_channels.py:201-216, 262-266)
    up = host.get("up_buffer_size", 0) if isinstance(host, dict) else 0
    down = host.get("down_buffer_size", 32) if isinstance(host, dict) else 32
    # The EDGE owns one connection per edge -> cloud DP link
    # (``local_conns = [conn for conn in connections if role == "edge" ...]``),
    # and links = 2 * connections, so the estimate scales with the number of
    # cloud DPs -- not with the number of cloud instances.
    links = 2 * len(all_cloud_dps)
    need = (links // 2) * 4 * (up + down)
    report.check(
        need <= 12 * 1024,
        f"host memory budget exceeded: {need} MiB > 12288 MiB",
    )
    report.notes.append(
        f"host memory estimate: {need} MiB/rank on the edge "
        f"({len(all_cloud_dps)} cloud DP link(s), budget 12288)"
    )

    return report.done()


PROXY_TARGET_KEYS = {"cloud_id", "target_dp", "url", "headers"}
PROXY_KEYS = {
    "prefill",
    "decode",
    "max_attempts",
    "max_response_bytes",
    "request_timeout_s",
    # Added upstream when the proxy gained a shared httpx pool
    # (proxy.py:66-67); both have defaults, but the example config spells them out.
    "max_connections",
    "max_keepalive_connections",
}


def validate_proxy(path: Path, label: str) -> int:
    report = Report(f"{label} ({path})")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    _keys(cfg, PROXY_KEYS, "proxy", report)

    def targets(key: str) -> list[dict]:
        entries = cfg.get(key)
        if not report.check(isinstance(entries, list) and bool(entries), f"{key} must be a non-empty list"):
            return []
        seen = set()
        out = []
        for i, entry in enumerate(entries):
            where = f"{key}[{i}]"
            _keys(entry, PROXY_TARGET_KEYS, where, report)
            cid, tdp = entry.get("cloud_id"), entry.get("target_dp")
            report.check(
                _is_int(cid) and cid >= 0, f"{where}.cloud_id must be a non-negative int"
            )
            report.check(
                _is_int(tdp) and tdp >= 0, f"{where}.target_dp must be a non-negative int"
            )
            parts = urlsplit(entry.get("url") or "")
            report.check(
                parts.scheme in ("http", "https") and bool(parts.hostname),
                f"{where}.url must be an HTTP(S) endpoint",
            )
            report.check(
                not (parts.username or parts.password or parts.fragment),
                f"{where}.url must not carry credentials or a fragment (proxy.py:40-48)",
            )
            report.check((cid, tdp) not in seen, f"{where}: duplicate (cloud_id,target_dp)")
            seen.add((cid, tdp))
            out.append(entry)
        return out

    prefill = targets("prefill")
    decode = targets("decode")
    overlap = {t.get("cloud_id") for t in prefill} & {t.get("cloud_id") for t in decode}
    report.check(
        not overlap,
        f"P and D must be independent cloud instances, overlapping cloud_id {sorted(overlap)} "
        "(proxy.py:70-71)",
    )
    report.check(
        _is_int(cfg.get("max_attempts", 4096)) and cfg.get("max_attempts", 4096) >= 1,
        "max_attempts must be an int >= 1",
    )
    report.check(
        _is_int(cfg.get("max_response_bytes", 1048576)) and cfg.get("max_response_bytes", 1048576) >= 1024,
        "max_response_bytes must be an int >= 1024",
    )
    timeout = cfg.get("request_timeout_s", 600.0)
    report.check(
        isinstance(timeout, (int, float)) and not isinstance(timeout, bool) and timeout > 0,
        "request_timeout_s must be > 0",
    )
    return report.done()


LAUNCHERS = ("edge.sh", "cloud_p.sh", "cloud_d.sh")
# Knobs the edge and both clouds must agree on: the first one is hashed into the
# PD ABI fingerprint, the other two bound what the pipeline as a whole can admit.
PARITY_FLAGS = ("--max-model-len", "--max-num-seqs", "--max-num-batched-tokens")
LAUNCH_VALUE_RE = r'(?:"\$\{LWD_[A-Z_]+:-(\d+)\}"|(\d+))'


def _launcher_flag(path: Path, flag: str) -> int | None:
    pattern = rf"{re.escape(flag)}\s+{LAUNCH_VALUE_RE}"
    match = re.search(pattern, path.read_text(encoding="utf-8"))
    if match is None:
        return None
    return int(match.group(1) or match.group(2))


def validate_launch_parity(where: Path) -> int:
    """All three launchers must agree on the pipeline-wide knobs.

    model_config_hash feeds the PD input-processor ABI fingerprint
    (lwd_mm_identity.compute_processor_fingerprint) and max_model_len is not in
    ModelConfig.compute_hash's ignored_factors, so drift there rejects every
    request with "PD input processor ABI does not match this cloud".
    """
    report = Report(f"launch parity ({where})")
    missing = [name for name in LAUNCHERS if not (where / name).is_file()]
    report.check(not missing, f"missing launcher(s): {missing}")
    if missing:
        return report.done()
    for flag in PARITY_FLAGS:
        values = {name: _launcher_flag(where / name, flag) for name in LAUNCHERS}
        report.check(
            None not in values.values(),
            f"{flag}: readable in all three launchers (got {values})",
        )
        reason = (
            "hashed into the PD ABI fingerprint"
            if flag == "--max-model-len"
            else "bounds what the whole pipeline can admit"
        )
        report.check(
            len({v for v in values.values() if v is not None}) <= 1,
            f"edge and both clouds must pass the same {flag} (got {values}); {reason}",
        )
    return report.done()


def main(argv: list[str]) -> int:
    # This directory holds exactly one deployment: one topology + one proxy file.
    topology = Path(argv[1]) if len(argv) > 1 else ROOT / "pd_host_topology.yaml"
    proxy = Path(argv[2]) if len(argv) > 2 else ROOT / "pd_proxy.json"
    failures = 0
    failures += validate_topology(topology, "topology")
    failures += validate_proxy(proxy, "proxy")
    failures += validate_launch_parity(ROOT)
    print()
    print("RESULT:", "ALL CHECKS PASSED" if not failures else f"{failures} FILE(S) FAILED")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
