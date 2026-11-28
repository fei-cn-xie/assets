#!/usr/bin/env bash
# PD proxy on the cloud host (76.76.26.195:8102).
# Copied from the curl-verified reference v30_rc/start_proxy.sh; the target list
# is read from pd_proxy.json next to this script instead of /vllm-workspace.
#
# The proxy needs no NPU, no UCX and no model: it only wires the edge to the P
# and D coordination endpoints listed in pd_proxy.json.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

# The LWD proxy takes no --log-level (its CLI is --config/--host/--port), so the
# only way to quieten it like the engines is the environment.
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-ERROR}"
export VLLM_LWD_LOG_LEVEL="${VLLM_LWD_LOG_LEVEL:-ERROR}"

lwd_run "$LWD_PYTHON" -m vllm.v1.lwd_control.pd.proxy \
  --config "$SCRIPT_DIR/pd_proxy.json" \
  --host 0.0.0.0 \
  --port 8102
