#!/usr/bin/env bash
# Shared environment for the centralized Mooncake PD deployment (P + D + proxy,
# all on 76.76.26.195, no LWD control plane).
#
# The NPU/vLLM runtime knobs are the ones the curl-verified v30_rc reference
# proved on this hardware. The LWD-only pieces of
# c_home/bash/pds/30rc_1124/sidecar/env.sh are dropped, because this deployment
# does not use the LWD control plane or its host data plane:
#   dropped: UCX_*, LWD_TOPOLOGY*, LWD_UCX_LIB_DIR, libucp preflight,
#            VLLM_LWD_LOG_LEVEL
#   kept:    PYTHONPATH, allocator/HCCL/OMP knobs, spawn worker method,
#            jemalloc preload, log level, engine-ready budget
#
# Sourced, never executed. Overrides may be exported beforehand:
#   CODEBASE_DIR, PD_PYTHON, VLLM_LOGGING_LEVEL, VLLM_ENGINE_READY_TIMEOUT_S
export CODEBASE_DIR="${CODEBASE_DIR:-/vllm-workspace}"
export PD_PYTHON="${PD_PYTHON:-python3}"
export PYTHONPATH="$CODEBASE_DIR/vllm-ascend:$CODEBASE_DIR/vllm${PYTHONPATH:+:$PYTHONPATH}"

PD_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PD_SCRIPT_DIR

PD_DRY_RUN=0
if [[ $# == 1 && $1 == --dry-run ]]; then
  PD_DRY_RUN=1
elif [[ $# != 0 ]]; then
  printf 'Usage: bash %s [--dry-run]\n' "$0" >&2
  exit 2
fi

# Ports of this deployment (single source of truth for the launchers/README).
export PD_PROXY_PORT=9060
export PD_PREFILL_PORT=9160
export PD_DECODE_PORT=9161
export PD_PREFILL_KV_PORT=14589
export PD_DECODE_KV_PORT=14590
export PD_PREFILL_MASTER_PORT=29535
export PD_DECODE_MASTER_PORT=29536
export PD_HOST=76.76.26.195
export PD_MODEL=/home/weight/Qwen3.6-27B/
# The PD proxy is the one shipped with vllm-ascend (no local Python glue):
# examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py
export PD_ASCEND_DIR="${PD_ASCEND_DIR:-$CODEBASE_DIR/vllm-ascend/examples/disaggregated_prefill_v1}"

pd_runtime_env() {
  # Single node: the torch/HCCL bootstrap socket groups stay on loopback.
  export GLOO_SOCKET_IFNAME=lo
  export TP_SOCKET_IFNAME=lo
  export HCCL_SOCKET_IFNAME=lo
  export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
  export HCCL_OP_EXPANSION_MODE="AIV"
  # Values from the vllm-ascend Mooncake deployment guide: the prefill side can
  # sit on a long prefill before the decode side hears from it.
  export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-204}"
  export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-120}"
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export HCCL_BUFFSIZE=1024
  export OMP_PROC_BIND=false
  export OMP_NUM_THREADS=1
  export TASK_QUEUE_ENABLE=1
  export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}"
  export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
  # 27B weights plus graph capture take minutes.
  export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"
}

# Warn when a port of this deployment is already taken (a stale engine, or a
# PD/LWD deployment still running), which would otherwise fail deep inside vLLM.
pd_preflight() {
  local port
  local busy=()
  command -v ss >/dev/null 2>&1 || return 0
  for port in "$@"; do
    if ss -lnt 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${port}\$"; then
      busy+=("$port")
    fi
  done
  if ((${#busy[@]})); then
    printf '[pd][preflight] ERROR: port(s) %s already in use on this host.\n' "${busy[*]}" >&2
    printf '[pd][preflight]        stop the previous deployment first (prefill.sh/decode.sh, start_proxy.sh).\n' >&2
    [[ $PD_DRY_RUN == 1 ]] || exit 4
  fi
}

pd_run() {
  if [[ $PD_DRY_RUN == 1 ]]; then
    "$PD_PYTHON" - "$@" <<'PY'
import json
import os
import sys

names = (
    "CODEBASE_DIR", "PYTHONPATH", "VLLM_HOST_IP", "MASTER_ADDR",
    "ASCEND_RT_VISIBLE_DEVICES",
    "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME", "HCCL_SOCKET_IFNAME",
    "PYTORCH_NPU_ALLOC_CONF", "HCCL_OP_EXPANSION_MODE", "HCCL_BUFFSIZE",
    "HCCL_EXEC_TIMEOUT", "HCCL_CONNECT_TIMEOUT",
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
