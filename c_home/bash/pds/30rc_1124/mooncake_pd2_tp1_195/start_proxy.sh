#!/usr/bin/env bash
# Client-facing PD proxy on 76.76.26.195:9060 (no edge, no LWD).
#
# Uses the proxy shipped with vllm-ascend, which performs the two-hop hand-off a
# standard PD deployment needs:
#   examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py
#     - sets kv_transfer_params={"do_remote_decode": true, ...} on the prefill hop
#     - forwards the prefill response's kv_transfer_params to the decode hop
#
# With DP=2 on each side the proxy still needs only one endpoint per side: each
# `vllm serve` fronts its DP replicas through its own internal load balancer
# (api_server_count defaults to data_parallel_size). Pass extra host/port pairs
# only if you later split the replicas across processes or hosts:
#   pd_preflight ... ; pd_run "$PD_PYTHON" "$PROXY" \
#     --prefiller-hosts $H1 $H2 --prefiller-ports $P1 $P2 ...
#
# No Python glue lives in this repository: the script is only a launcher.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

PROXY="$PD_ASCEND_DIR/load_balance_proxy_server_example.py"
if [[ ! -f $PROXY ]]; then
  printf '[pd][preflight] ERROR: %s not found.\n' "$PROXY" >&2
  printf '[pd][preflight]        expected inside the vllm-ascend checkout at %s\n' "$CODEBASE_DIR/vllm-ascend" >&2
  printf '[pd][preflight]        check the branch: examples/disaggregated_prefill_v1/ must exist.\n' >&2
  [[ $PD_DRY_RUN == 1 ]] || exit 5
fi

pd_preflight "$PD_PROXY_PORT"

pd_run "$PD_PYTHON" "$PROXY" \
  --host 0.0.0.0 \
  --port "$PD_PROXY_PORT" \
  --prefiller-hosts "$PD_HOST" \
  --prefiller-ports "$PD_PREFILL_PORT" \
  --decoder-hosts "$PD_HOST" \
  --decoder-ports "$PD_DECODE_PORT"
