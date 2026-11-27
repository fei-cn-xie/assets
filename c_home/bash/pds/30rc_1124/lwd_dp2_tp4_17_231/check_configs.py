"""Static pre-flight for the edge/cloud DP=2 x TP=4 LWD launchers.

Stdlib only, no PyYAML and no imports from vLLM: the authoritative schema check
is LwdTopology on the host. What this script does is catch the mistakes that
would otherwise surface only after minutes of model loading:

    python3 check_configs.py
    python3 check_configs.py --dir other_package

What it asserts
---------------
* the topology declares 1 edge + 2 clouds, one prefill and one decode, 2 DP each
  on the cloud host, with unique ctrl/data/coord ports and a full edge-cloud mesh
* the edge runs TP=1 / DP=1 (LWD forces it) and points at the proxy this package
  starts
* both cloud engines run DP=2 x TP=4, own disjoint 8-card sets covering 0-15, and
  declare dp_size/tp_size matching inside the KV extra config
* every listening port (proxy, edge API, cloud APIs, masters, DP rpc, KV ranges,
  topology ctrl/data/coord ports) is distinct
* the knobs pinned for the performance comparison are referenced by all three
  launchers and agree with the centralized sibling package
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

LAUNCHERS = ("edge.sh", "cloud_p.sh", "cloud_d.sh", "start_proxy.sh")
SHARED_KNOBS = (
    ("LWD_GPU_MEM_UTIL", "the comparison fixes utilisation at 0.95 on both sides"),
    ("LWD_MAX_MODEL_LEN", "max_model_len is hashed into the KV connector layout"),
    ("LWD_MAX_NUM_SEQS", "bounds what the whole pipeline can admit"),
    ("LWD_MAX_NUM_BATCHED_TOKENS", "sizes the chunk window on all three engines"),
    ("LWD_ENABLE_MTP", "draft on/off changes the throughput being compared"),
)
SIBLING = "mooncake_pd2_tp4_231"
SIBLING_KNOBS = {
    "LWD_GPU_MEM_UTIL": "PD_GPU_MEM_UTIL",
    "LWD_MAX_MODEL_LEN": "PD_MAX_MODEL_LEN",
    "LWD_MAX_NUM_SEQS": "PD_MAX_NUM_SEQS",
    "LWD_MAX_NUM_BATCHED_TOKENS": "PD_MAX_NUM_BATCHED_TOKENS",
    "LWD_ENABLE_MTP": "PD_ENABLE_MTP",
}
# The three engine launchers hand these to every role.
KNOB_SCRIPTS = ("edge.sh", "cloud_p.sh", "cloud_d.sh")
ENV_RE = re.compile(r'^export\s+([A-Z0-9_]+)="?\$\{\1:-([^}"]*)\}"?', re.MULTILINE)
BUFFER_BUDGET_MIB = 12288


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

    Comments quote example flags that must not be mistaken for the real argv.
    """
    return "\n".join(
        line for line in path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )


def expand_vars(text: str, env: dict[str, str]) -> str:
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
        # A following unexpanded array expansion (${LWD_MTP_ARGS[@]+...}) is not
        # this flag's value; JSON values start with "{" and stay acceptable.
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


def visible_devices(path: Path, env: dict[str, str]) -> list[int]:
    source = expand_vars(script_source(path), env)
    match = re.search(r'ASCEND_RT_VISIBLE_DEVICES="?([0-9,]+)', source)
    if not match:
        return []
    return [int(x) for x in match.group(1).split(",") if x.isdigit()]


def kv_port_range(base: int, dp: int, tp: int) -> set[int]:
    return set(range(base, base + dp * tp))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument(
        "--parity-dir",
        default=None,
        help=f"centralized sibling package to compare shared knobs with (default ../{SIBLING})",
    )
    ns = parser.parse_args(argv[1:])
    root = Path(ns.dir).resolve()
    sibling = Path(ns.parity_dir).resolve() if ns.parity_dir else (root.parent / SIBLING)

    report = Report(f"edge/cloud PD DP=2 x TP=4 ({root.name})")
    missing = [n for n in ("env.sh", "pd_host_topology.yaml", *LAUNCHERS)
               if not (root / n).is_file()]
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
    edge = parse_script(root / "edge.sh", env)
    cloud_p = parse_script(root / "cloud_p.sh", env)
    cloud_d = parse_script(root / "cloud_d.sh", env)
    proxy = parse_script(root / "start_proxy.sh", env)

    topology_path = root / "pd_host_topology.yaml"
    topology_text = topology_path.read_text(encoding="utf-8")
    dp_size = int(env.get("LWD_DP_SIZE", "0") or 0)
    tp_size = int(env.get("LWD_TP_SIZE", "0") or 0)
    report.check(dp_size == 2, f"LWD_DP_SIZE must be 2 (got {dp_size})")
    report.check(tp_size == 4, f"LWD_TP_SIZE must be 4 (got {tp_size})")

    # --- topology shape ------------------------------------------------------
    def scalar(name: str) -> str | None:
        match = re.search(rf"^\s*{name}:\s*(\S+)\s*$", topology_text, re.MULTILINE)
        return match.group(1).strip("\"'") if match else None

    for name, want in (
        ("edges_num", "1"),
        ("clouds_num", "2"),
        ("data_plane_comm", "host"),
        ("deployment", "sidecar"),
        ("enabled", "true"),
    ):
        got = scalar(name)
        # "enabled" appears twice (pd + coordination); any true is fine there.
        report.check(got == want, f"topology: {name} should be {want}, found {got}")

    reports = re.findall(r"stage_role:\s*(\w+)", topology_text)
    report.check(sorted(reports) == ["decode", "prefill"],
                 f"topology: expected one prefill and one decode cloud, found {reports}")

    edge_addrs = re.findall(r"^\s*addr:\s*(\S+)\s*$", topology_text, re.MULTILINE)
    cloud_host = env.get("LWD_CLOUD_HOST", "")
    edge_host = env.get("LWD_EDGE_HOST", "")
    report.check(edge_host in edge_addrs, f"topology: edge addr {edge_host} missing")
    report.check(
        edge_addrs.count(cloud_host) == dp_size * 2,
        f"topology: expected {dp_size * 2} cloud DP addrs for {cloud_host}, "
        f"found {edge_addrs.count(cloud_host)}",
    )

    proxy_url = re.search(r"pd_proxy_url:\s*(\S+)", topology_text)
    want_url = f"http://{cloud_host}:{env.get('LWD_PROXY_PORT')}/v1/chat/completions"
    report.check(
        proxy_url is not None and proxy_url.group(1) == want_url,
        f"topology: pd_proxy_url should be {want_url}, found "
        f"{proxy_url.group(1) if proxy_url else None}",
    )

    ctrl = [int(x) for x in re.findall(r"ctrl_port:\s*(\d+)", topology_text)]
    data = [int(x) for x in re.findall(r"data_port:\s*(\d+)", topology_text)]
    coord = [int(x) for x in re.findall(r"coord_port:\s*(\d+)", topology_text)]
    for name, values in (("ctrl_port", ctrl), ("data_port", data), ("coord_port", coord)):
        report.check(len(values) == dp_size * 2,
                     f"topology: expected {dp_size * 2} {name} entries, found {len(values)}")
        report.check(len(values) == len(set(values)),
                     f"topology: duplicate {name}: {values}")

    up = int(scalar("up_buffer_size") or 0)
    down = int(scalar("down_buffer_size") or 0)
    links = dp_size * 2
    pool_mib = links * 4 * (up + down)
    report.check(
        pool_mib <= BUFFER_BUDGET_MIB,
        f"topology: host data-plane pools need {pool_mib} MiB for {links} links, "
        f"over the {BUFFER_BUDGET_MIB} MiB budget",
    )
    report.notes.append(f"host data-plane budget: {pool_mib} MiB over {links} links")

    links_found = re.findall(r"\{edge:\s*(\d+),\s*cloud:\s*(\d+)\}", topology_text)
    report.check(sorted(links_found) == [("0", "0"), ("0", "1")],
                 f"topology: instance_links must be a full mesh, found {links_found}")

    # --- engines -------------------------------------------------------------
    report.check(edge.get("--tensor-parallel-size") == "1",
                 "edge: LWD forces TP=1")
    report.check("--data-parallel-size" not in edge,
                 "edge: must not pass --data-parallel-size (one process serves every DP)")
    report.check("--enforce-eager" in edge, "edge: keep --enforce-eager")
    report.check('"role":"edge"' in edge.get("--additional-config", ""),
                 "edge: --additional-config must carry role=edge")

    for name, script_args, role, instance in (
        ("cloud_p", cloud_p, "cloud", "0"),
        ("cloud_d", cloud_d, "cloud", "1"),
    ):
        report.check(script_args.get("--tensor-parallel-size") == env.get("LWD_TP_SIZE"),
                     f"{name}: --tensor-parallel-size must resolve to {env.get('LWD_TP_SIZE')}")
        report.check(script_args.get("--data-parallel-size") == env.get("LWD_DP_SIZE"),
                     f"{name}: --data-parallel-size must resolve to {env.get('LWD_DP_SIZE')}")
        report.check("--data-parallel-address" in script_args,
                     f"{name}: DP=2 needs an explicit --data-parallel-address")
        config = script_args.get("--additional-config", "")
        report.check(f'"role":"{role}"' in config, f"{name}: --additional-config must carry role={role}")
        report.check(f'"instance_id":{instance}' in config,
                     f"{name}: --additional-config must carry instance_id={instance}")

    # --- KV layout -----------------------------------------------------------
    layouts = {}
    for name, script_args, want_role in (
        ("cloud_p", cloud_p, "kv_producer"),
        ("cloud_d", cloud_d, "kv_consumer"),
    ):
        raw = script_args.get("--kv-transfer-config")
        if not report.check(raw is not None, f"{name}: --kv-transfer-config not found"):
            continue
        try:
            cfg = json.loads(raw)
        except json.JSONDecodeError as exc:
            report.check(False, f"{name}: --kv-transfer-config is not valid JSON ({exc})")
            continue
        layouts[name] = cfg
        report.check(cfg.get("kv_role") == want_role, f"{name}: kv_role must be {want_role}")
        extra = cfg.get("kv_connector_extra_config", {})
        for side in ("prefill", "decode"):
            report.check(
                extra.get(side) == {"dp_size": dp_size, "tp_size": tp_size},
                f"{name}: extra_config.{side} must be dp_size={dp_size}/tp_size={tp_size}",
            )
        report.check("engine_id" not in cfg,
                     f"{name}: the LWD topology assigns engine ids; do not pass engine_id")

    # --- prefill / decode flag parity ---------------------------------------
    allowed = {
        "--port",
        "--master-port",
        "--data-parallel-rpc-port",
        "--kv-transfer-config",
        "--compilation-config",
        "--additional-config",  # differs by instance_id
    }
    for flag in sorted(set(cloud_p) | set(cloud_d)):
        if flag in allowed:
            continue
        report.check(
            cloud_p.get(flag) == cloud_d.get(flag),
            f"{flag} differs between the two clouds: "
            f"{cloud_p.get(flag)!r} vs {cloud_d.get(flag)!r}",
        )

    # --- devices -------------------------------------------------------------
    devices = {
        "prefill": visible_devices(root / "cloud_p.sh", env),
        "decode": visible_devices(root / "cloud_d.sh", env),
    }
    for name, cards in devices.items():
        report.check(len(cards) == dp_size * tp_size,
                     f"cloud {name}: needs DP*TP = {dp_size * tp_size} cards, got {len(cards)}")
    report.check(not set(devices["prefill"]) & set(devices["decode"]),
                 f"NPU sets overlap: {devices}")
    report.check(set(devices["prefill"]) | set(devices["decode"]) == set(range(16)),
                 f"expected cards 0-15 split between the clouds, got {devices}")
    # The edge picks its card inside lwd_host_env (env.sh), so the launcher has no
    # literal device list to read; fall back to the knob that drives it.
    edge_cards = visible_devices(root / "edge.sh", env) or [
        int(x) for x in env.get("LWD_EDGE_DEVICES", "").split(",") if x.isdigit()
    ]
    report.check(len(edge_cards) == 1, f"edge: expected exactly one card, got {edge_cards}")
    # The edge sits on its own host, so sharing card indices with the clouds is
    # expected (edge NPU 0 on 17, cloud NPUs 0-15 on 231). What must hold is that
    # the two hosts really are different.
    report.check(
        env.get("LWD_EDGE_HOST") != env.get("LWD_CLOUD_HOST"),
        "the edge and the clouds must run on different hosts, otherwise their "
        "card ranges overlap",
    )
    report.notes.append(
        f"cards: edge {edge_cards} on {env.get('LWD_EDGE_HOST')}, "
        f"cloud prefill {devices['prefill']} / decode {devices['decode']} "
        f"on {env.get('LWD_CLOUD_HOST')}"
    )

    # --- ports ---------------------------------------------------------------
    ints = {k: int(v) for k, v in env.items() if v.isdigit()}
    pk = kv_port_range(ints["LWD_PREFILL_KV_PORT"], dp_size, tp_size)
    dk = kv_port_range(ints["LWD_DECODE_KV_PORT"], dp_size, tp_size)
    report.check(not pk & dk, f"KV handshake ranges overlap: {sorted(pk)} vs {sorted(dk)}")
    cloud_singles = {
        "proxy": ints["LWD_PROXY_PORT"],
        "prefill api": ints["LWD_PREFILL_PORT"],
        "decode api": ints["LWD_DECODE_PORT"],
        "prefill dp rpc": ints["LWD_PREFILL_DP_RPC"],
        "decode dp rpc": ints["LWD_DECODE_DP_RPC"],
        "prefill master": ints["LWD_PREFILL_MASTER_PORT"],
        "decode master": ints["LWD_DECODE_MASTER_PORT"],
    }
    cloud_values = list(cloud_singles.values())
    report.check(len(cloud_values) == len(set(cloud_values)),
                 f"cloud listening ports collide: {cloud_singles}")
    report.check(not set(cloud_values) & (pk | dk),
                 f"KV range overlaps a cloud listening port: "
                 f"{sorted(set(cloud_values) & (pk | dk))}")
    # The edge lives on its own host, but its ports must not collide with the
    # topology's cloud-side ports either (they would if both ran on one host).
    edge_port = ints["LWD_EDGE_PORT"]
    topology_ports = set(ctrl) | set(data) | set(coord)
    report.check(not topology_ports & (set(cloud_values) | pk | dk),
                 f"topology ports collide with engine ports: "
                 f"{sorted(topology_ports & (set(cloud_values) | pk | dk))}")
    report.check(edge_port not in topology_ports,
                 f"edge API {edge_port} collides with a topology port")
    report.notes.append(f"KV handshake ports: prefill {sorted(pk)}, decode {sorted(dk)}")

    # --- knobs ---------------------------------------------------------------
    for key, reason in SHARED_KNOBS:
        report.check(key in env, f"{key} missing from env.sh")
        for script in KNOB_SCRIPTS:
            text = (root / script).read_text(encoding="utf-8")
            # LWD_ENABLE_MTP is consumed by lwd_mtp_args() in env.sh, so calling
            # that helper is the evidence the launcher honours the knob.
            referenced = ("$" + key in text) or (
                key == "LWD_ENABLE_MTP" and "lwd_mtp_args" in text
            )
            report.check(referenced, f"{key} must be referenced by {script} ({reason})")
    report.check(env.get("LWD_GPU_MEM_UTIL") == "0.95",
                 f"LWD_GPU_MEM_UTIL must be 0.95 for the comparison (got {env.get('LWD_GPU_MEM_UTIL')})")
    report.check(env.get("LWD_ENABLE_MTP") == "1",
                 "this comparison runs with MTP on: set LWD_ENABLE_MTP=1")
    report.notes.append(
        f"shared knobs: util={env.get('LWD_GPU_MEM_UTIL')} "
        f"max_model_len={env.get('LWD_MAX_MODEL_LEN')} "
        f"max_num_seqs={env.get('LWD_MAX_NUM_SEQS')} "
        f"max_num_batched_tokens={env.get('LWD_MAX_NUM_BATCHED_TOKENS')} "
        f"mtp={env.get('LWD_ENABLE_MTP')}"
    )

    # --- parity with the centralized sibling --------------------------------
    if (sibling / "env.sh").is_file():
        sib_env = parse_env(sibling)
        for key, sib_key in SIBLING_KNOBS.items():
            report.check(
                sib_env.get(sib_key) == env.get(key),
                f"{key}={env.get(key)!r} but {sibling.name}:{sib_key}="
                f"{sib_env.get(sib_key)!r} — both deployments must run the same knobs",
            )
        report.notes.append(f"parity with {sibling.name}: checked")
    else:
        report.notes.append(f"parity check skipped: {sibling} not present next to this package")

    # --- UCX runtime ---------------------------------------------------------
    # UCX is installed outside the vllm-workspace checkout on both hosts; deriving
    # the path from CODEBASE_DIR used to point at a tree that does not exist.
    report.check(
        env.get("LWD_UCX_LIB_DIR") == "/home/extra/ucx/lib",
        f"LWD_UCX_LIB_DIR must default to /home/extra/ucx/lib on both hosts "
        f"(got {env.get('LWD_UCX_LIB_DIR')!r})",
    )

    # --- UCX transports ------------------------------------------------------
    env_sh = (root / "env.sh").read_text(encoding="utf-8")
    report.check(
        "rocep196s0f0:1,enp196s0f0" in env_sh,
        "the 231 default must be rocep196s0f0:1,enp196s0f0 — the RoCE device that "
        "pairs with NIC enp196s0f0 (host 231 IB devices: rocep162s0f0/f1, "
        "rocep196s0f0/f1)",
    )
    report.check(
        "LWD_UCX_TCP_ONLY" in env_sh,
        "keep the LWD_UCX_TCP_ONLY escape hatch for hosts without RoCE",
    )
    if env.get("LWD_UCX_NET_DEVICES_231"):
        report.notes.append("UCX: 231 will use explicit "
                            f"{env.get('LWD_UCX_NET_DEVICES_231')}")
    else:
        report.notes.append("UCX: 231 will use rocep196s0f0:1,enp196s0f0 (RoCE)")
    return report.done()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
