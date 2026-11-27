"""Static pre-flight for the centralized DP=2 Mooncake PD launchers.

Stdlib only, no imports from vLLM: it parses the three launchers plus env.sh and
checks the invariants that otherwise fail deep inside vLLM on the real host:

    python3 check_configs.py
    python3 check_configs.py --dir other_package

What it asserts
---------------
* both sides run DP=2 x TP=1 and reference the shared env.sh knobs (no literals)
* the two NPU sets are disjoint and sized DP * TP
* every listening port is distinct, including the KV handshake ranges
  (``kv_port + dp_rank * tp * pp`` .. ``+ DP*TP-1``)
* the KV connector extra config carries dp_size/tp_size matching each side
* the knobs that must agree between prefill and decode do agree
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

LAUNCHERS = ("prefill.sh", "decode.sh", "start_proxy.sh")
SHARED_KNOBS = (
    ("PD_MAX_MODEL_LEN", "max_model_len is hashed into the KV connector layout"),
    ("PD_MAX_NUM_SEQS", "bounds what the whole pipeline can admit"),
    ("PD_MAX_NUM_BATCHED_TOKENS", "sizes the chunk window on both sides"),
)
ENV_RE = re.compile(r'^export\s+([A-Z0-9_]+)="?\$\{\1:-([^}"]*)\}"?', re.MULTILINE)
ARG_RE = re.compile(r"--([a-z0-9-]+)\s+(\"[^\"]*\"|'[^']*'|\S+)")


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


def resolve(token: str, env: dict[str, str]) -> str:
    match = re.fullmatch(r"\$(?:\{)?([A-Z0-9_]+)(?:\})?", token)
    if match:
        return env.get(match.group(1), f"<undefined {match.group(1)}>")
    return token


def kv_ports(base: int, dp: int, tp: int) -> set[int]:
    return {base + rank * tp for rank in range(dp)}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=str(Path(__file__).resolve().parent))
    root = Path(parser.parse_args(argv[1:]).dir).resolve()

    report = Report(f"centralized PD DP=2 ({root})")
    missing = [n for n in ("env.sh", *LAUNCHERS) if not (root / n).is_file()]
    report.check(not missing, f"missing file(s): {missing}")
    if missing:
        return report.done()

    # --- every text file must be LF-only -------------------------------------
    # A CRLF line leaves a "\r" at the end of the last word, so bash on the host
    # fails in ways that point nowhere. Git stores LF here (.gitattributes), but
    # edits made through Windows tooling can leak CRLF into the working copy that
    # gets copied to the host, so check the bytes instead of trusting git.
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
    report.check(tp == 1, f"PD_TP_SIZE must be 1 (got {tp})")

    # --- both engines are DP=2 x TP=1 and take the sizes from env.sh ----------
    for name, args in (("prefill", prefill), ("decode", decode)):
        report.check(args.get("--data-parallel-size") == env.get("PD_DP_SIZE"),
                     f"{name}: --data-parallel-size resolves to {env.get('PD_DP_SIZE')}")
        report.check(args.get("--tensor-parallel-size") == env.get("PD_TP_SIZE"),
                     f"{name}: --tensor-parallel-size resolves to {env.get('PD_TP_SIZE')}")
        report.check("--data-parallel-address" in args,
                     f"{name}: DP=2 needs an explicit --data-parallel-address")

    # --- prefill and decode must share the engine flags ----------------------
    # The deployments differ only in the ports, the KV role/config and the graph
    # mode; everything else stays identical so a regression is attributable.
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
        same = prefill.get(flag) == decode.get(flag)
        report.check(same, f"{flag} differs between prefill and decode: "
                           f"{prefill.get(flag)!r} vs {decode.get(flag)!r}")

    # --- NPU sets -------------------------------------------------------------
    devices: dict[str, list[int]] = {}
    for name, script in (("prefill", "prefill.sh"), ("decode", "decode.sh")):
        text = (root / script).read_text(encoding="utf-8")
        match = re.search(r"ASCEND_RT_VISIBLE_DEVICES=([0-9,]+)", text)
        report.check(match is not None, f"{name}: ASCEND_RT_VISIBLE_DEVICES missing")
        devices[name] = [int(x) for x in match.group(1).split(",")] if match else []
        report.check(len(devices[name]) == dp * tp,
                     f"{name}: needs DP*TP = {dp * tp} cards, got {devices[name]}")
    report.check(not set(devices["prefill"]) & set(devices["decode"]),
                 f"NPU sets overlap: {devices}")
    report.check(len(set(devices["prefill"]) | set(devices["decode"])) == 4,
                 f"expected 4 distinct cards total, got {devices}")

    # --- KV connector roles and layout ---------------------------------------
    layouts = {}
    for name, args in (("prefill", prefill), ("decode", decode)):
        script = "prefill.sh" if name == "prefill" else "decode.sh"
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
    pk = kv_ports(ints["PD_PREFILL_KV_PORT"], dp, tp)
    dk = kv_ports(ints["PD_DECODE_KV_PORT"], dp, tp)
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
    report.check(ints["PD_PREFILL_DP_RPC"] != 29550 and ints["PD_DECODE_DP_RPC"] != 29550,
                 "set both DP rpc ports explicitly: 29550 is the ParallelConfig default "
                 "and prefill/decode share this host")
    report.notes.append(f"KV ports: prefill {sorted(pk)}, decode {sorted(dk)}")

    # --- shared knobs --------------------------------------------------------
    for key, reason in SHARED_KNOBS:
        report.check(key in env, f"{key} missing from env.sh")
        report.check("$" + key in (root / "prefill.sh").read_text(encoding="utf-8")
                     and "$" + key in (root / "decode.sh").read_text(encoding="utf-8"),
                     f"{key} must be referenced by both launchers ({reason})")
    report.notes.append(
        f"shared knobs: max_model_len={env.get('PD_MAX_MODEL_LEN')} "
        f"max_num_seqs={env.get('PD_MAX_NUM_SEQS')} "
        f"max_num_batched_tokens={env.get('PD_MAX_NUM_BATCHED_TOKENS')} "
        f"gpu_mem={env.get('PD_GPU_MEM_UTIL')} mtp={env.get('PD_ENABLE_MTP')}"
    )
    # TP=1 puts the whole model on one card, so the Mamba block count caps the
    # batch: 32 is the value that starts, 64 was rejected on the real host.
    report.check(int(env.get("PD_MAX_NUM_SEQS", "0") or 0) <= 32,
                 "with TP=1 and MTP on, max_num_seqs above 32 exceeded the Mamba "
                 "block count on the real host")
    # Same switch as the working edge-cloud DP=2 package, so this is a warning and
    # not a failure: if the centralized engine runs out of KV, dropping the draft
    # is the first knob to try (measured: 0.30 GiB with MTP vs ~6.8 GiB without).
    if tp == 1 and env.get("PD_ENABLE_MTP") == "1":
        report.notes.append(
            "MTP on: centralized TP=1 measured only ~0.30 GiB of KV on this host; "
            "if the engine refuses to start, rerun with PD_ENABLE_MTP=0 (~6.8 GiB)"
        )
    return report.done()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
