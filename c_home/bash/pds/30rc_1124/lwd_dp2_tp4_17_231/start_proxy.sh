#!/usr/bin/env bash
# Client-facing proxy for the edge/cloud deployment — 76.76.26.231:8102.
#
# The LWD topology points the edge at this URL (edges[0].coordination.pd_proxy_url),
# and the proxy itself is the plain PD proxy shipped with vllm-ascend:
#   examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py
# It does not know about LWD: the edge talks to it exactly like a client, and the
# proxy performs the usual prefill -> decode hand-off.
#
# Start it after both cloud engines are up and before the edge.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

PROXY="$CODEBASE_DIR/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py"
if [[ ! -f $PROXY ]]; then
  printf '[lwd][preflight] ERROR: %s not found.\n' "$PROXY" >&2
  printf '[lwd][preflight]        expected inside the vllm-ascend checkout at %s\n' "$CODEBASE_DIR/vllm-ascend" >&2
  [[ $LWD_DRY_RUN == 1 ]] || exit 5
fi

lwd_preflight "$LWD_PROXY_PORT"

lwd_run "$LWD_PYTHON" "$PROXY" \
  --host 0.0.0.0 \
  --port "$LWD_PROXY_PORT" \
  --prefiller-hosts "$LWD_CLOUD_HOST" \
  --prefiller-ports "$LWD_PREFILL_PORT" \
  --decoder-hosts "$LWD_CLOUD_HOST" \
  --decoder-ports "$LWD_DECODE_PORT"
