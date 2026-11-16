#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

lwd_run "$LWD_PYTHON" "$CODEBASE_DIR/vllm-ascend/examples/edge_cloud_pd_sidecar/proxy.py" \
  --host 0.0.0.0 \
  --port 8500 \
  --prefill-url http://76.76.26.189:8660/v1/chat/completions \
  --decode-url http://76.76.26.189:8661/v1/chat/completions
