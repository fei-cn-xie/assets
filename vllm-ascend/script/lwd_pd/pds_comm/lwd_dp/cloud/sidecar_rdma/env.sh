#!/usr/bin/env bash
# Shared by the six launch scripts; deployment overrides may be exported first.
export CODEBASE_DIR="${CODEBASE_DIR:-/vllm-workspace}"
export LWD_PYTHON="${LWD_PYTHON:-python}"
export PYTHONPATH="$CODEBASE_DIR/vllm-ascend:$CODEBASE_DIR/vllm${PYTHONPATH:+:$PYTHONPATH}"

LWD_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LWD_DIRECTORY="${LWD_DIRECTORY:-$LWD_SCRIPT_DIR/lwd_sidecar_directory.json}"
# Encode the path as a JSON string before embedding it in --additional-config.
LWD_DIRECTORY_JSON="$("$LWD_PYTHON" -c 'import json, os; print(json.dumps(os.environ["LWD_DIRECTORY"]))')"
export LWD_DIRECTORY_JSON

LWD_DRY_RUN=0
if [[ $# == 1 && $1 == --dry-run ]]; then
  LWD_DRY_RUN=1
elif [[ $# != 0 ]]; then
  printf 'Usage: bash %s [--dry-run]\n' "$0" >&2
  exit 2
fi

lwd_ucx_env() {
  local devices
  case "$1" in
    cloud) devices="${LWD_CLOUD_UCX_NET_DEVICES:-${UCX_NET_DEVICES:-rocep67s0f0:1,enp189s0f0}}" ;;
    edge) devices="${LWD_EDGE_UCX_NET_DEVICES:-${UCX_NET_DEVICES:-rocep67s0f0:1,enp189s0f0}}" ;;
    *) return 2 ;;
  esac
  export UCX_NET_DEVICES="$devices"
  export UCX_TLS="${UCX_TLS:-rc_verbs,tcp,self}"
  # An explicitly empty prefix keeps the existing library search path.
  export UCX_PREFIX="${UCX_PREFIX-/vllm-workspace/ucx}"
  if [[ -n $UCX_PREFIX ]]; then
    export PATH="$UCX_PREFIX/bin:$PATH"
    export LD_LIBRARY_PATH="$UCX_PREFIX/lib:$UCX_PREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
}

lwd_prepare_ipc() {
  if [[ $LWD_DRY_RUN == 1 ]]; then
    return
  fi
  "$LWD_PYTHON" - "$LWD_DIRECTORY" "$1" <<'PY'
import json
import sys
from pathlib import Path

with open(sys.argv[1], encoding="utf-8") as file:
    directory = json.load(file)
instance = next(i for i in directory["instances"] if i["instance_id"] == sys.argv[2])
endpoint = instance["local_endpoint"]
if not endpoint.startswith("ipc:///"):
    raise ValueError("local_endpoint must be an absolute IPC path")
Path(endpoint.removeprefix("ipc://")).parent.mkdir(parents=True, exist_ok=True)
PY
}

lwd_run() {
  if [[ $LWD_DRY_RUN == 1 ]]; then
    "$LWD_PYTHON" - "$@" <<'PY'
import json
import os
import sys

names = (
    "CODEBASE_DIR", "PYTHONPATH", "LWD_DIRECTORY", "VLLM_HOST_IP",
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
