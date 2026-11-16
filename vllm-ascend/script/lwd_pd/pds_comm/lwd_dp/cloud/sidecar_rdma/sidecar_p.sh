#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

lwd_ucx_env cloud
lwd_prepare_ipc cloud-p

lwd_run "$LWD_PYTHON" "$CODEBASE_DIR/vllm/tools/lwd_sidecar.py" \
  --directory "$LWD_DIRECTORY" \
  --instance cloud-p \
  --log-level INFO \
  --stats-interval 10
