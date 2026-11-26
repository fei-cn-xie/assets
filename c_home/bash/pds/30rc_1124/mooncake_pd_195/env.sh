#!/usr/bin/env bash
# Shared environment for the centralized Mooncake PD deployment (P + D + proxy,
# all on 76.76.26.195, no LWD control plane).
#
# Every value here is taken from the confirmed-working reference
# private_home/vllm_script/pds/centralized/{prefill.sh,decode.sh}:
#   * the real NIC for GLOO/TP/HCCL (plus HCCL_IF_IP) instead of loopback
#   * LD_LIBRARY_PATH pointing at the ascend-toolkit mooncake site-packages
#   * jemalloc preload, AIV expansion mode, expandable segments, OMP/queue knobs
# Deliberately absent (also absent upstream): MASTER_ADDR / VLLM_HOST_IP,
# --master-addr/--master-port, --additional-config, explicit HCCL timeouts.
#
# Sourced, never executed. Overrides may be exported beforehand:
#   CODEBASE_DIR, PD_PYTHON, PD_NIC_NAME, PD_HOST, PD_MODEL,
#   PD_MOONCAKE_LIB, PD_ASCEND_DIR, VLLM_LOGGING_LEVEL
export CODEBASE_DIR="${CODEBASE_DIR:-/vllm-workspace}"
export PD_PYTHON="${PD_PYTHON:-python3}"
export PYTHONPATH="$CODEBASE_DIR/vllm-ascend:$CODEBASE_DIR/vllm${PYTHONPATH:+:$PYTHONPATH}"

PD_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PD_SCRIPT_DIR

# Mooncake's python bindings live in the ascend-toolkit site-packages; without
# this the engines start fine but the first KV transfer fails.
export PD_MOONCAKE_LIB="${PD_MOONCAKE_LIB:-/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake}"

export PD_NIC_NAME="${PD_NIC_NAME:-enp189s0f0}"
export PD_HOST="${PD_HOST:-76.76.26.195}"
export PD_MODEL="${PD_MODEL:-/home/weight/Qwen3.6-27B/}"
# The PD proxy is the one shipped with vllm-ascend (no local Python glue).
export PD_ASCEND_DIR="${PD_ASCEND_DIR:-$CODEBASE_DIR/vllm-ascend/examples/disaggregated_prefill_v1}"

# Ports of this deployment (single source of truth for the launchers/README).
export PD_PROXY_PORT=9060
export PD_PREFILL_PORT=9160
export PD_DECODE_PORT=9161
# KV handshake ports, same values as the confirmed-working reference.
export PD_PREFILL_KV_PORT=30000
export PD_DECODE_KV_PORT=30100

PD_DRY_RUN=0
if [[ $# == 1 && $1 == --dry-run ]]; then
  PD_DRY_RUN=1
elif [[ $# != 0 ]]; then
  printf 'Usage: bash %s [--dry-run]\n' "$0" >&2
  exit 2
fi

pd_runtime_env() {
  export LD_LIBRARY_PATH="$PD_MOONCAKE_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export HCCL_IF_IP="$PD_HOST"
  export GLOO_SOCKET_IFNAME="$PD_NIC_NAME"
  export TP_SOCKET_IFNAME="$PD_NIC_NAME"
  export HCCL_SOCKET_IFNAME="$PD_NIC_NAME"
  export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}"
  export HCCL_OP_EXPANSION_MODE="AIV"
  export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
  export OMP_PROC_BIND=false
  export OMP_NUM_THREADS=1
  export TASK_QUEUE_ENABLE=1
  export HCCL_BUFFSIZE=1024
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  # The reference runs at ERROR; INFO keeps the failure visible when a bench
  # request dies, which is what you want while bringing the deployment up.
  export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
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
    # Clear LD_PRELOAD for the dry-run helper: on a non-Ascend host (or any host
    # without the Linux jemalloc) the preload above would break the spawn.
    LD_PRELOAD= "$PD_PYTHON" - "$@" <<'PY'
import json
import os
import sys

names = (
    "CODEBASE_DIR", "PYTHONPATH", "ASCEND_RT_VISIBLE_DEVICES",
    "HCCL_IF_IP", "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME", "HCCL_SOCKET_IFNAME",
    "PYTORCH_NPU_ALLOC_CONF", "HCCL_OP_EXPANSION_MODE", "HCCL_BUFFSIZE",
    "VLLM_WORKER_MULTIPROC_METHOD", "OMP_PROC_BIND", "OMP_NUM_THREADS",
    "TASK_QUEUE_ENABLE", "VLLM_LOGGING_LEVEL", "LD_PRELOAD", "LD_LIBRARY_PATH",
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
