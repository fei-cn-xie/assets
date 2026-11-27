"""Static pre-flight for the centralized DP=2 x TP=4 Mooncake PD launchers.

Stdlib only, no imports from vLLM: it parses the launchers plus env.sh and checks
the invariants that otherwise fail deep inside vLLM on the real host:

    python3 check_configs.py
    python3 check_configs.py --dir other_package

What it asserts
---------------
* both sides run DP=2 x TP=4 and take those sizes from env.sh (no literals)
* the two NPU sets are disjoint, sized DP * TP, and together cover 0-15
* every listening port is distinct, including the KV handshake ranges
  (``kv_port + dp_rank * tp`` .. ``+ DP*TP-1``)
* the KV connector extra config carries dp_size/tp_size matching each side
* the knobs pinned for the performance comparison agree between prefill and
  decode, and with the edge-cloud sibling package (same values on both sides is
  the whole point of the comparison)
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

LAUNCHERS = ("prefill.sh", "decode.sh", "start_proxy.sh")
# The knobs this package exists to keep identical to ../lwd_dp2_tp4_17_231.
SHARED_KNOBS = (
    ("PD_GPU_MEM_UTIL", "the comparison fixes utilisation at 0.95 on both sides"),
    ("PD_MAX_MODEL_LEN", "max_model_len is hashed into the KV connector layout"),
    ("PD_MAX_NUM_SEQS", "bounds what the whole pipeline can admit"),
    ("PD_MAX_NUM_BATCHED_TOKENS", "sizes the chunk window on both sides"),
    ("PD_ENABLE_MTP", "draft on/off changes the throughput being compared"),
)
SIBLING = "lwd_dp2_tp4_17_231"
SIBLING_KNOBS = {
    "PD_GPU_MEM_UTIL": "LWD_GPU_MEM_UTIL",
    "PD_MAX_MODEL_LEN": "LWD_MAX_MODEL_LEN",
    "PD_MAX_NUM_SEQS": "LWD_MAX_NUM_SEQS",
    "PD_MAX_NUM_BATCHED_TOKENS": "LWD_MAX_NUM_BATCHED_TOKENS",
    "PD_ENABLE_MTP": "LWD_ENABLE_MTP",
}
ENV_RE = re.compile(r'^export\s+([A-Z0-9_]+)="?\$\{\1:-([^}"]*)\}"?', re.MULTILINE)


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


def parse_env(root: Path) -> dict[str, str]:
    text = (root / "env.sh").read_text(encoding="utf-8")
    return {name: value for name, value in ENV_RE.findall(text)}


def script_source(path: Path) -> str:
    """Launcher text with comment lines stripped.

    Comments quote example flags (``--data-parallel-size 2``) that must not be
    mistaken for the real argument list.
    """
    return "\n".join(
        line for line in path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )


def expand_vars(text: str, env: dict[str, str]) -> str:
    """Only double-quoted references occur in these launchers; expand those."""
    return re.sub(
        r'"\$(?:\{)?([A-Z0-9_]+)(?:\})?"',
        lambda m: env.get(m.group(1), m.group(1)),
        text,
    )


def parse_script(path: Path, env: dict[str, str]) -> dict[str, str]:
    """flag -> value, with shell quoting and $VAR expansion applied."""
    source = expand_vars(script_source(path), env).replace("\\\n", " ")
    args: dict[str, str] = {}
    pos = 0
    tokens = shlex.split(source, comments=False)
    while pos < len(tokens):
        token = tokens[pos]
        nxt = tokens[pos + 1] if pos + 1 < len(tokens) else ""
        # A following token that is still an unexpanded shell reference (e.g. the
        # array expansion ${PD_MTP_ARGS[@]+...}) is not this flag's value. JSON
        # values start with "{" and must still be accepted.
        has_value = nxt and not nxt.startswith("--") and not nxt.startswith("$")
        if token.startswith("--") and has_value:
            args.setdefault(token, nxt)
            pos += 2
        elif token.startswith("--"):
            args.setdefault(token, "")
            pos += 1
        else:
            pos += 1
    return args


def parse_kv_config(path: Path, env: dict[str, str]) -> dict | None:
    raw = parse_script(path, env).get("--kv-transfer-config")
    return json.loads(raw) if raw else None


def visible_devices(path: Path, env: dict[str, str]) -> list[int]:
    """Cards the launcher hands to the engine.

    The launchers set ``ASCEND_RT_VISIBLE_DEVICES="$PD_PREFILL_DEVICES"``, so the
    reference has to be expanded through env.sh before the digits can be read.
    """
    source = expand_vars(script_source(path), env)
    match = re.search(r'ASCEND_RT_VISIBLE_DEVICES="?([0-9,]+)', source)
    if not match:
        return []
    return [int(x) for x in match.group(1).split(",") if x.isdigit()]


def kv_port_range(base: int, dp: int, tp: int) -> set[int]:
    """One handshake port per (dp, tp) pair: base + dp_rank * tp + device."""
    return set(range(base, base + dp * tp))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument(
        "--parity-dir",
        default=None,
        help=f"edge-cloud sibling package to compare shared knobs with (default ../{SIBLING})",
    )
    args_ns = parser.parse_args(argv[1:])
    root = Path(args_ns.dir).resolve()
    sibling = (
        Path(args_ns.parity_dir).resolve()
        if args_ns.parity_dir
        else (root.parent / SIBLING)
    )

    report = Report(f"centralized PD DP=2 x TP=4 ({root.name})")
    missing = [n for n in ("env.sh", *LAUNCHERS) if not (root / n).is_file()]
    report.check(not missing, f"missing file(s): {missing}")
    if missing:
        return report.done()

    # --- every text file must be LF-only -------------------------------------
    crlf = sorted(
        p.name
        for p in root.iterdir()
        if p.is_file()
        and p.suffix in (".sh", ".py", ".md", ".yaml", ".json")
        and b"\r\n" in p.read_bytes()
    )
    report.check(not crlf, f"CRLF line endings (breaks the host copy): {crlf}")

    env = parse_env(root)
    prefill = parse_script(root / "prefill.sh", env)
    decode = parse_script(root / "decode.sh", env)

    dp = int(env.get("PD_DP_SIZE", "0") or 0)
    tp = int(env.get("PD_TP_SIZE", "0") or 0)
    report.check(dp == 2, f"PD_DP_SIZE must be 2 (got {dp})")
    report.check(tp == 4, f"PD_TP_SIZE must be 4 (got {tp})")

    # --- both engines are DP=2 x TP=4 and take the sizes from env.sh ----------
    for name, script_args in (("prefill", prefill), ("decode", decode)):
        report.check(script_args.get("--data-parallel-size") == env.get("PD_DP_SIZE"),
                     f"{name}: --data-parallel-size resolves to {env.get('PD_DP_SIZE')}")
        report.check(script_args.get("--tensor-parallel-size") == env.get("PD_TP_SIZE"),
                     f"{name}: --tensor-parallel-size resolves to {env.get('PD_TP_SIZE')}")
        report.check("--data-parallel-address" in script_args,
                     f"{name}: DP=2 needs an explicit --data-parallel-address")

    # --- prefill and decode must share the engine flags ----------------------
    allowed = {
        "--port",
        "--master-port",
        "--data-parallel-rpc-port",
        "--kv-transfer-config",
        "--compilation-config",
    }
    for flag in sorted(set(prefill) | set(decode)):
        if flag in allowed:
            continue
        report.check(
            prefill.get(flag) == decode.get(flag),
            f"{flag} differs between prefill and decode: "
            f"{prefill.get(flag)!r} vs {decode.get(flag)!r}",
        )

    # --- NPU sets ------------------------------------------------------------
    devices = {
        "prefill": visible_devices(root / "prefill.sh", env),
        "decode": visible_devices(root / "decode.sh", env),
    }
    for name, cards in devices.items():
        report.check(bool(cards), f"{name}: ASCEND_RT_VISIBLE_DEVICES missing")
        report.check(len(cards) == dp * tp,
                     f"{name}: needs DP*TP = {dp * tp} cards, got {len(cards)}")
    report.check(not set(devices["prefill"]) & set(devices["decode"]),
                 f"NPU sets overlap: {devices}")
    union = set(devices["prefill"]) | set(devices["decode"])
    report.check(union == set(range(16)),
                 f"expected cards 0-15 split between the two engines, got {sorted(union)}")
    report.check(
        set(devices["prefill"]) == set(range(8)) and set(devices["decode"]) == set(range(8, 16)),
        f"expected prefill on 0-7 and decode on 8-15, got {devices}",
    )

    # --- KV connector roles and layout ---------------------------------------
    layouts = {}
    for name in ("prefill", "decode"):
        script = f"{name}.sh"
        try:
            cfg = parse_kv_config(root / script, env)
        except json.JSONDecodeError as exc:
            report.check(False, f"{name}: --kv-transfer-config is not valid JSON ({exc})")
            continue
        if not report.check(cfg is not None, f"{name}: --kv-transfer-config not found"):
            continue
        layouts[name] = cfg
        want_role = {"prefill": "kv_producer", "decode": "kv_consumer"}[name]
        report.check(cfg.get("kv_role") == want_role, f"{name}: kv_role must be {want_role}")
        extra = cfg.get("kv_connector_extra_config", {})
        for side in ("prefill", "decode"):
            report.check(extra.get(side) == {"dp_size": dp, "tp_size": tp},
                         f"{name}: extra_config.{side} must be dp_size={dp}/tp_size={tp}")
    if len(layouts) == 2:
        report.check(layouts["prefill"].get("engine_id") != layouts["decode"].get("engine_id"),
                     "prefill and decode must use different kv engine_id values")

    # --- ports ---------------------------------------------------------------
    ints = {k: int(v) for k, v in env.items() if v.isdigit()}
    pk = kv_port_range(ints["PD_PREFILL_KV_PORT"], dp, tp)
    dk = kv_port_range(ints["PD_DECODE_KV_PORT"], dp, tp)
    report.check(not pk & dk, f"KV handshake ports overlap: {sorted(pk)} vs {sorted(dk)}")
    singles = {
        "proxy": ints["PD_PROXY_PORT"],
        "prefill api": ints["PD_PREFILL_PORT"],
        "decode api": ints["PD_DECODE_PORT"],
        "prefill dp rpc": ints["PD_PREFILL_DP_RPC"],
        "decode dp rpc": ints["PD_DECODE_DP_RPC"],
        "prefill master": ints["PD_PREFILL_MASTER_PORT"],
        "decode master": ints["PD_DECODE_MASTER_PORT"],
    }
    values = list(singles.values())
    report.check(len(values) == len(set(values)), f"listening ports collide: {singles}")
    clash = sorted(set(values) & (pk | dk))
    report.check(not clash, f"KV port range overlaps a listening port: {clash}")
    report.notes.append(f"KV handshake ports: prefill {sorted(pk)}, decode {sorted(dk)}")

    # --- shared knobs, in-package -------------------------------------------
    for key, reason in SHARED_KNOBS:
        report.check(key in env, f"{key} missing from env.sh")
        for script in ("prefill.sh", "decode.sh"):
            report.check(
                "$" + key in (root / script).read_text(encoding="utf-8")
                or key in ("PD_ENABLE_MTP", "PD_GPU_MEM_UTIL"),
                f"{key} must be referenced by {script} ({reason})",
            )
    report.check(env.get("PD_GPU_MEM_UTIL") == "0.95",
                 f"PD_GPU_MEM_UTIL must be 0.95 for the comparison (got {env.get('PD_GPU_MEM_UTIL')})")
    report.check(env.get("PD_ENABLE_MTP") == "1",
                 "this comparison runs with MTP on: set PD_ENABLE_MTP=1")
    report.notes.append(
        f"shared knobs: util={env.get('PD_GPU_MEM_UTIL')} "
        f"max_model_len={env.get('PD_MAX_MODEL_LEN')} "
        f"max_num_seqs={env.get('PD_MAX_NUM_SEQS')} "
        f"max_num_batched_tokens={env.get('PD_MAX_NUM_BATCHED_TOKENS')} "
        f"mtp={env.get('PD_ENABLE_MTP')}"
    )

    # --- shared knobs, against the edge-cloud sibling ------------------------
    if (sibling / "env.sh").is_file():
        sib_env = parse_env(sibling)
        for key, sib_key in SIBLING_KNOBS.items():
            sib_value = sib_env.get(sib_key)
            report.check(
                sib_value == env.get(key),
                f"{key}={env.get(key)!r} but {sibling.name}:{sib_key}={sib_value!r} "
                "— both deployments must run the same knobs",
            )
        report.notes.append(f"parity with {sibling.name}: {'ok' if not report.errors else 'see errors'}")
    else:
        report.notes.append(
            f"parity check skipped: {sibling} not present next to this package"
        )

    # TP=4 leaves 4 cards worth of room per replica, so 128 sequences no longer
    # hit the TP=1 Mamba-block wall; keep the reminder rather than a hard assert.
    if int(env.get("PD_MAX_NUM_SEQS", "0") or 0) > 64:
        report.notes.append(
            "max_num_seqs=128 needs 'Available Mamba cache blocks >= 128' in the "
            "engine log; lower PD_MAX_NUM_SEQS if the pool is smaller"
        )
    return report.done()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
