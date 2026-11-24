#!/usr/bin/env bash
# Shared by the four launch scripts of the v0.30.0 edge/cloud PD deployment.
# Transport variant: TCP (the host data plane is UCX; only the UCX transport
# layer is pinned to TCP here).
#
# Sourced, never executed. Deployment overrides may be exported beforehand:
#   CODEBASE_DIR, LWD_PYTHON, LWD_TOPOLOGY, UCX_PREFIX, UCX_TLS, UCX_NET_DEVICES,
#   LWD_CLOUD_UCX_NET_DEVICES, LWD_EDGE_UCX_NET_DEVICES, VLLM_LOGGING_LEVEL,
#   VLLM_LWD_LOG_LEVEL
export CODEBASE_DIR="${CODEBASE_DIR:-/vllm-workspace}"
export LWD_PYTHON="${LWD_PYTHON:-python}"
export PYTHONPATH="$CODEBASE_DIR/vllm-ascend:$CODEBASE_DIR/vllm${PYTHONPATH:+:$PYTHONPATH}"

LWD_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LWD_SCRIPT_DIR
# The topology YAML is v0.30.0's single source of truth and must be identical on
# both hosts. It is resolved from this directory so the tree is relocatable.
export LWD_TOPOLOGY="${LWD_TOPOLOGY:-$LWD_SCRIPT_DIR/lwd_topology.yaml}"
# Encode the path as a JSON string before embedding it in --additional-config.
LWD_TOPOLOGY_JSON="$("$LWD_PYTHON" -c 'import json, os; print(json.dumps(os.environ["LWD_TOPOLOGY"]))')"
export LWD_TOPOLOGY_JSON

# LWD also has its own log stream, off by default (VLLM_LWD_LOG_LEVEL=DISABLE).
# VLLM_LOGGING_LEVEL defaults to INFO in these scripts; add the [lwd] stream with
# VLLM_LWD_LOG_LEVEL=INFO. The [LWD][config] line carries the topology digest,
# which must match on both hosts.

LWD_DRY_RUN=0
if [[ $# == 1 && $1 == --dry-run ]]; then
  LWD_DRY_RUN=1
elif [[ $# != 0 ]]; then
  printf 'Usage: bash %s [--dry-run]\n' "$0" >&2
  exit 2
fi

# v0.30.0 always uses UCX for data_plane_comm=host (ctypes over libucp), so
# libucp is required even in the TCP variant. Only UCX_TLS / UCX_NET_DEVICES
# decide whether frames leave on TCP or on RoCE: no sidecar, no directory file,
# no ucx_port_base, no ipc socket.
#
# This is NEW relative to dev/pd-sep-comm: the old TCP deployment used its own
# sockets, so a host without UCX worked. Here every engine host needs the UCX
# runtime - the edge included. lwd_require_ucx_runtime() enforces it up front
# instead of letting the link builder retry for 605s.
lwd_resolve_ucx_prefix() {
  # An explicitly set UCX_PREFIX wins, empty included ("keep the system path").
  if [[ -v UCX_PREFIX ]]; then
    return 0
  fi
  # Otherwise use the workspace tree only when it actually exists, so we never
  # prepend a bogus lib dir; empty means "keep the system search path".
  if [[ -d /vllm-workspace/ucx/lib || -d /vllm-workspace/ucx/lib64 ]]; then
    UCX_PREFIX=/vllm-workspace/ucx
  else
    UCX_PREFIX=
  fi
  export UCX_PREFIX
}

lwd_transport_env() {
  local devices
  case "$1" in
    cloud) devices="${LWD_CLOUD_UCX_NET_DEVICES:-${UCX_NET_DEVICES:-enp189s0f0}}" ;;
    edge) devices="${LWD_EDGE_UCX_NET_DEVICES:-${UCX_NET_DEVICES:-enp189s0f0}}" ;;
    *) return 2 ;;
  esac
  # tcp,self: no rc_verbs -> everything goes over the business NIC's TCP stack.
  export UCX_TLS="${UCX_TLS:-tcp,self}"
  export UCX_NET_DEVICES="$devices"
  lwd_resolve_ucx_prefix
  if [[ -n $UCX_PREFIX ]]; then
    export PATH="$UCX_PREFIX/bin:$PATH"
    export LD_LIBRARY_PATH="$UCX_PREFIX/lib:$UCX_PREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
}

# Fail fast when the UCX runtime is absent. Mirrors ucx_ucp.py, which tries
# libucp.so.0 then libucp.so; without either, the link builder raises inside a
# background thread and the engine only dies at the 605s handshake timeout with
# a misleading "[LWD][handshake] host link not ready" message.
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
  local hint='install the UCX runtime on this host (libucp), or point UCX_PREFIX at an existing tree: UCX_PREFIX=/path/to/ucx; verify with: ldconfig -p | grep libucp'
  if [[ $LWD_DRY_RUN == 1 ]]; then
    printf '[LWD][preflight] WARNING: libucp.so.0 is not loadable here; a real launch would exit now.\n[LWD][preflight] hint: %s\n' "$hint" >&2
    return 0
  fi
  printf '[LWD][preflight] ERROR: libucp.so.0 not found on this host (UCX_PREFIX=%s).\n' "${UCX_PREFIX:-<system path>}" >&2
  printf '[LWD][preflight] hint: %s\n' "$hint" >&2
  printf '[LWD][preflight] note: v0.30.0 uses UCX for BOTH transport variants; the edge needs it too.\n' >&2
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
    "VLLM_HOST_IP", "VLLM_LOGGING_LEVEL", "VLLM_LWD_LOG_LEVEL",
    "ASCEND_RT_VISIBLE_DEVICES", "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME",
    "HCCL_SOCKET_IFNAME", "UCX_PREFIX", "UCX_NET_DEVICES", "UCX_TLS",
    "LD_LIBRARY_PATH", "LD_PRELOAD", "VLLM_ASCEND_ENABLE_FLASHCOMM1",
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
