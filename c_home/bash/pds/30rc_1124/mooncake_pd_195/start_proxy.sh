#!/usr/bin/env bash
# Client-facing PD proxy on 76.76.26.195:9060 (no edge, no LWD).
#
# It performs the two-hop hand-off that a standard PD deployment needs:
#   client -> proxy:9060 -> P:9160 (max_tokens=1, do_remote_decode)
#                       -> D:9161 (with P's kv_transfer_params) -> client
#
# Uses the small stdlib proxy in this directory instead of
# examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py, because that
# example never sets kv_transfer_params on the prefill hop and never forwards
# them to decode, so MooncakeConnectorV1 would transfer nothing.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

pd_preflight "$PD_PROXY_PORT"

pd_run "$PD_PYTHON" "$SCRIPT_DIR/pd_proxy.py" \
  --host 0.0.0.0 \
  --port "$PD_PROXY_PORT" \
  --prefill "http://$PD_HOST:$PD_PREFILL_PORT" \
  --decode "http://$PD_HOST:$PD_DECODE_PORT"
