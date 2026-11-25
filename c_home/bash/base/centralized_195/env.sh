#!/usr/bin/env bash
# Shared environment for the centralized (non-PD) Qwen3.6-27B service.
#
# Derived from c_home/bash/pds/30rc_1124/sidecar/env.sh. Everything the
# curl-verified v30_rc reference proved on this hardware is kept verbatim; the
# PD/LWD-only pieces are dropped, because a single engine has no cross-host
# data plane:
#   dropped: UCX_* tuning, LWD_TOPOLOGY / LWD_TOPOLOGY_JSON, LWD_UCX_LIB_DIR,
#            lwd_require_ucx_runtime (libucp preflight), VLLM_LWD_LOG_LEVEL
#   kept:    PYTHONPATH, NPU allocator/HCCL/OMP knobs, spawn worker method,
#            jemalloc preload, log level, engine-ready budget
#
# Sourced, never executed. Overrides may be exported beforehand:
#   CODEBASE_DIR, BASE_PYTHON, VLLM_LOGGING_LEVEL, VLLM_ENGINE_READY_TIMEOUT_S
export CODEBASE_DIR="${CODEBASE_DIR:-/vllm-workspace}"
export BASE_PYTHON="${BASE_PYTHON:-python3}"
# v30_rc invokes the module entry point rather than the `vllm` script, so the
# interpreter is explicit and independent of PATH.
export PYTHONPATH="$CODEBASE_DIR/vllm-ascend:$CODEBASE_DIR/vllm${PYTHONPATH:+:$PYTHONPATH}"

BASE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export BASE_SCRIPT_DIR

BASE_DRY_RUN=0
if [[ $# == 1 && $1 == --dry-run ]]; then
  BASE_DRY_RUN=1
elif [[ $# != 0 ]]; then
  printf 'Usage: bash %s [--dry-run]\n' "$0" >&2
  exit 2
fi

base_runtime_env() {
  # Single node, so the torch/HCCL bootstrap socket groups stay on loopback.
  # This is the same choice the PD launchers made on this host (they ran TP=2
  # per instance here); only the process world gets bigger (TP=4).
  export GLOO_SOCKET_IFNAME=lo
  export TP_SOCKET_IFNAME=lo
  export HCCL_SOCKET_IFNAME=lo
  export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
  export HCCL_OP_EXPANSION_MODE="AIV"
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export HCCL_BUFFSIZE=1024
  export OMP_PROC_BIND=false
  export OMP_NUM_THREADS=1
  export TASK_QUEUE_ENABLE=1
  export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}"
  export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
  # A 27B weight load plus graph capture takes minutes; v30_rc raised this to
  # 1800s and the same budget applies here.
  export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"
}

# Two checks that turn a confusing runtime failure into a clear message.
#   * NPU 0-3 are exactly what the PD engines on this host use (cloud_p owns
#     0,1 and cloud_d owns 2,3), so a still-running PD pair must be stopped.
#   * the service port must be free.
base_preflight() {
  local port="${1:-9060}"
  local pd_ports=()
  local candidate
  if command -v ss >/dev/null 2>&1; then
    for candidate in 8600 8601; do
      if ss -lnt 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${candidate}\$"; then
        pd_ports+=("$candidate")
      fi
    done
    if ((${#pd_ports[@]})); then
      printf '[base][preflight] WARNING: PD engine port(s) %s still listening on this host.\n' "${pd_ports[*]}" >&2
      printf '[base][preflight]          Those engines own NPU 0-3, which this script also uses.\n' >&2
      printf '[base][preflight]          Stop cloud_p.sh / cloud_d.sh first.\n' >&2
    fi
    if ss -lnt 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${port}\$"; then
      printf '[base][preflight] ERROR: port %s is already in use on this host.\n' "$port" >&2
      [[ $BASE_DRY_RUN == 1 ]] || exit 4
    fi
  fi
}

base_run() {
  if [[ $BASE_DRY_RUN == 1 ]]; then
    "$BASE_PYTHON" - "$@" <<'PY'
import json
import os
import sys

names = (
    "CODEBASE_DIR", "PYTHONPATH", "VLLM_HOST_IP", "MASTER_ADDR",
    "ASCEND_RT_VISIBLE_DEVICES",
    "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME", "HCCL_SOCKET_IFNAME",
    "PYTORCH_NPU_ALLOC_CONF", "HCCL_OP_EXPANSION_MODE", "HCCL_BUFFSIZE",
    "VLLM_WORKER_MULTIPROC_METHOD", "OMP_PROC_BIND", "OMP_NUM_THREADS",
    "TASK_QUEUE_ENABLE", "VLLM_LOGGING_LEVEL", "VLLM_ENGINE_READY_TIMEOUT_S",
    "LD_PRELOAD", "LD_LIBRARY_PATH",
)
print(json.dumps({
    "command": sys.argv[1:],
    "environment": {name: os.environ.get(name) for name in names},
}, ensure_ascii=False, indent=2))
PY
  else
    exec "$@"
  fi
}
