#!/usr/bin/env bash
# Client-facing proxy for the edge/cloud deployment — 76.76.26.231:8102.
#
# IMPORTANT: this is NOT the standard Mooncake PD proxy used by the centralized
# package next door. LWD routes by cloud instance and DP rank, so the proxy is the
# LWD control-plane one:
#
#   python -m vllm.v1.lwd_control.pd.proxy --config pd_proxy.json
#
# Its config (pd_proxy.json next to this script, CLI: vllm/v1/lwd_control/pd/
# proxy.py:351-353 --config/--host/--port) lists one target per cloud DP, pointing
# at that DP's coord_port from pd_host_topology.yaml:
#   prefill dp0 -> 8100, dp1 -> 8101, decode dp0 -> 8110, dp1 -> 8111
#
# The edge reaches this proxy through the topology's
# edges[0].coordination.pd_proxy_url, so the port here must match that URL.
#
# Start it after both cloud engines are up and before the edge. No NPU, no UCX and
# no model are needed here.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

CONFIG="$SCRIPT_DIR/pd_proxy.json"
if [[ ! -f $CONFIG ]]; then
  printf '[lwd][preflight] ERROR: %s not found (per-DP target list for the LWD proxy)\n' "$CONFIG" >&2
  exit 5
fi

lwd_preflight "$LWD_PROXY_PORT"

lwd_run "$LWD_PYTHON" -m vllm.v1.lwd_control.pd.proxy \
  --config "$CONFIG" \
  --host 0.0.0.0 \
  --port "$LWD_PROXY_PORT"
