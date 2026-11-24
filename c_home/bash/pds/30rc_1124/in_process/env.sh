#!/usr/bin/env bash
# Shared environment for the 194/195 edge/cloud PD deployment on
# feat/pd-sep-lwd-v0.30.0.
#
# Every runtime value below is copied verbatim from the curl-verified reference
# private_home/vllm_script/pds_comm_1117/v30_rc (branch
# debug/pd-empty-response-trace): the UCX tuning, LD_LIBRARY_PATH, the server
# entry point, the master ports and the logging switches all come from there.
# Only the topology schema had to be translated (see pd_host_topology.yaml).
#
# Sourced, never executed. Overrides may be exported beforehand:
#   CODEBASE_DIR, LWD_PYTHON, LWD_TOPOLOGY, LWD_UCX_LIB_DIR,
#   VLLM_LOGGING_LEVEL, VLLM_LWD_LOG_LEVEL
export CODEBASE_DIR="${CODEBASE_DIR:-/vllm-workspace}"
export LWD_PYTHON="${LWD_PYTHON:-python3}"
# v30_rc invokes the module entry point rather than the `vllm` script, so the
# interpreter is explicit and independent of PATH.
# The model directory is written literally in each launcher, exactly as in
# v30_rc: the edge host (194) uses /weight/Qwen3.6-27B/, the cloud host (195)
# uses /home/weight/Qwen3.6-27B/. No override knob is added here.
export PYTHONPATH="$CODEBASE_DIR/vllm-ascend:$CODEBASE_DIR/vllm${PYTHONPATH:+:$PYTHONPATH}"

LWD_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LWD_SCRIPT_DIR
# One topology file for all three roles; it must be identical on both hosts.
# Topology name and location follow v30_rc (pd_host_topology.yaml), but the path
# is resolved from this directory so the tree stays relocatable.
export LWD_TOPOLOGY="${LWD_TOPOLOGY:-$LWD_SCRIPT_DIR/pd_host_topology.yaml}"
LWD_TOPOLOGY_JSON="$("$LWD_PYTHON" -c 'import json, os; print(json.dumps(os.environ["LWD_TOPOLOGY"]))')"
export LWD_TOPOLOGY_JSON

LWD_DRY_RUN=0
if [[ $# == 1 && $1 == --dry-run ]]; then
  LWD_DRY_RUN=1
elif [[ $# != 0 ]]; then
  printf 'Usage: bash %s [--dry-run]\n' "$0" >&2
  exit 2
fi

# Runtime environment common to the three engine processes, copied from v30_rc.
# Roles only differ in VLLM_HOST_IP / MASTER_ADDR / ASCEND_RT_VISIBLE_DEVICES.
lwd_runtime_env() {
  # v30_rc binds the torch/HCCL socket groups to loopback: each host runs its
  # own single-node world, the cross-host path is the LWD host plane.
  export GLOO_SOCKET_IFNAME=lo
  export TP_SOCKET_IFNAME=lo
  export HCCL_SOCKET_IFNAME=lo
  export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
  export HCCL_OP_EXPANSION_MODE="AIV"
  export TORCH_DISTRIBUTED_DETAIL=DEBUG
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export HCCL_BUFFSIZE=1024
  export OMP_PROC_BIND=false
  export OMP_NUM_THREADS=1
  export TASK_QUEUE_ENABLE=1
  export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}"
  export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
  export VLLM_LWD_LOG_LEVEL="${VLLM_LWD_LOG_LEVEL:-INFO}"
  # The cloud engines need minutes for weights + graph capture before they
  # answer the edge; v30_rc raises the engine-ready budget to 1800s.
  export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"

  # ---- UCX: the exact tuning validated by v30_rc on this hardware ----------
  # rc_verbs first (RoCE), tcp for the handshake/bootstrap, self for intra-host.
  unset UCX_IB_NUM_PATHS UCX_MAX_RNDV_LANES
  export UCX_RNDV_THRESH="${UCX_RNDV_THRESH:-512}"
  export UCX_PROTO_ENABLE="${UCX_PROTO_ENABLE:-n}"
  export UCX_RNDV_SCHEME="${UCX_RNDV_SCHEME:-put_zcopy}"
  export UCX_TLS="${UCX_TLS:-rc_verbs,tcp,self}"
  # rocep67s0f0 = IB device (gid index 5 below), enp67s0f0np0 = its netdev.
  export UCX_NET_DEVICES="${UCX_NET_DEVICES:-rocep67s0f0:1,enp67s0f0np0}"
  export UCX_IB_GID_INDEX="${UCX_IB_GID_INDEX:-5}"
  export UCX_IB_MTU="${UCX_IB_MTU:-1024}"

  # libucp lives in the workspace UCX tree on these hosts (v30_rc uses lib/ only).
  export LWD_UCX_LIB_DIR="${LWD_UCX_LIB_DIR:-$CODEBASE_DIR/ucx/lib}"
  if [[ -d $LWD_UCX_LIB_DIR ]]; then
    export LD_LIBRARY_PATH="$LWD_UCX_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
}

# Fail fast when libucp is absent. Mirrors ucx_ucp.py, which loads libucp.so.0
# then libucp.so; without either, the link builder raises inside a background
# thread and the engine only dies at the 605s handshake timeout with a
# misleading "[LWD][handshake] host link not ready" message.
lwd_require_ucx_runtime() {
  if "$LWD_PYTHON" -c 'import ctypes, sys
for name in ("libucp.so.0", "libucp.so"):
    try:
        ctypes.CDLL(name)
    except OSError:
        continue
    sys.exit(0)
sys.exit(1)' >/dev/null 2>&1; then
    return 0
  fi
  local hint="put libucp on the loader path: LWD_UCX_LIB_DIR=<dir with libucp.so.0> (v30_rc uses $CODEBASE_DIR/ucx/lib); verify with: ls $CODEBASE_DIR/ucx/lib/libucp.so*"
  if [[ $LWD_DRY_RUN == 1 ]]; then
    printf '[LWD][preflight] WARNING: libucp.so.0 is not loadable here; a real launch would exit now.\n[LWD][preflight] hint: %s\n' "$hint" >&2
    return 0
  fi
  printf '[LWD][preflight] ERROR: libucp.so.0 not found on this host.\n' >&2
  printf '[LWD][preflight] hint: %s\n' "$hint" >&2
  exit 3
}

lwd_run() {
  if [[ $LWD_DRY_RUN == 1 ]]; then
    "$LWD_PYTHON" - "$@" <<'PY'
import json
import os
import sys

names = (
    "CODEBASE_DIR", "PYTHONPATH", "LWD_TOPOLOGY", "LWD_TOPOLOGY_JSON",
    "VLLM_HOST_IP", "MASTER_ADDR", "ASCEND_RT_VISIBLE_DEVICES",
    "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME", "HCCL_SOCKET_IFNAME",
    "PYTORCH_NPU_ALLOC_CONF", "HCCL_OP_EXPANSION_MODE", "HCCL_BUFFSIZE",
    "VLLM_WORKER_MULTIPROC_METHOD", "TORCH_DISTRIBUTED_DETAIL",
    "OMP_PROC_BIND", "OMP_NUM_THREADS", "TASK_QUEUE_ENABLE",
    "VLLM_LOGGING_LEVEL", "VLLM_LWD_LOG_LEVEL", "VLLM_ENGINE_READY_TIMEOUT_S",
    "UCX_TLS", "UCX_NET_DEVICES", "UCX_IB_GID_INDEX", "UCX_IB_MTU",
    "UCX_RNDV_THRESH", "UCX_RNDV_SCHEME", "UCX_PROTO_ENABLE",
    "UCX_IB_NUM_PATHS", "UCX_MAX_RNDV_LANES",
    "LWD_UCX_LIB_DIR", "LD_LIBRARY_PATH", "LD_PRELOAD",
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
