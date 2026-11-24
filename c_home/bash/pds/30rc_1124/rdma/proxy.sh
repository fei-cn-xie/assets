#!/usr/bin/env bash
# PD proxy on cloud B. Start after cloud_p.sh and cloud_d.sh.
# v0.30.0 replaced the old "proxy.py --prefill-url ... --decode-url ..." form with
# a JSON target list (cloud_id + target_dp + coord url) so the proxy can pick the
# actual cloud DP and hand a target_dp to the engine.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"
# INFO by default: the proxy logs the P/D it picked, which is how you tell whether
# a request actually fanned out to both stages.
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

lwd_run "$LWD_PYTHON" -m vllm.v1.lwd_control.pd.proxy \
  --config "$LWD_SCRIPT_DIR/pd_proxy.json" \
  --host 0.0.0.0 \
  --port 8500
