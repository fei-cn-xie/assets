#!/usr/bin/env bash
# Shared environment for the edge/cloud (LWD) PD deployment used as the
# baseline against ../mooncake_pd2_tp4_231:
#
#   edge   76.76.26.17  NPU 0     TP=1 DP=1     -> API 8510
#   cloud 0 prefill 76.76.26.231  NPU 0-7   DP=2 TP=4 -> API 8600, kv_producer 14570
#   cloud 1 decode  76.76.26.231  NPU 8-15  DP=2 TP=4 -> API 8601, kv_consumer 14580
#   proxy  76.76.26.231:8102 (CPU only)
#
# 17 is an a2 host with data NIC enp189s0f0 (same family as 194/195); 231 is an
# a3 host with enp196s0f0. Both run feat/pd-sep-lwd-v0.30.0 plus the
# fix/lwd-multidp-dense-pd branches (dense models with DP>1 need them).
#
# Sourced, never executed. Overrides may be exported beforehand:
#   CODEBASE_DIR, LWD_PYTHON, LWD_TOPOLOGY, LWD_UCX_LIB_DIR, VLLM_LOGGING_LEVEL,
#   LWD_GPU_MEM_UTIL, LWD_MAX_MODEL_LEN, LWD_MAX_NUM_SEQS,
#   LWD_MAX_NUM_BATCHED_TOKENS, LWD_ENABLE_MTP, plus the per-host UCX knobs below.
#
# The performance comparison this package exists for is only fair if both
# deployments use the same knobs: 95% memory utilisation, a 65536 context, 128
# sequences, an 8192 batch window and MTP on. check_configs.py asserts that
# against ../mooncake_pd2_tp4_231.

export CODEBASE_DIR="${CODEBASE_DIR:-/vllm-workspace}"
export LWD_PYTHON="${LWD_PYTHON:-python3}"
# v30_rc invokes the module entry point rather than the `vllm` script, so the
# interpreter is explicit and independent of PATH.
export PYTHONPATH="$CODEBASE_DIR/vllm-ascend:$CODEBASE_DIR/vllm${PYTHONPATH:+:$PYTHONPATH}"

LWD_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LWD_SCRIPT_DIR
# One topology file for all three roles; it must be identical on both hosts
# (LwdTopology hashes it, every process logs "[LWD][config] ... digest=").
export LWD_TOPOLOGY="${LWD_TOPOLOGY:-$LWD_SCRIPT_DIR/pd_host_topology.yaml}"
LWD_TOPOLOGY_JSON="$("$LWD_PYTHON" -c 'import json, os; print(json.dumps(os.environ["LWD_TOPOLOGY"]))')"
export LWD_TOPOLOGY_JSON

# --- hosts and their data NICs ---------------------------------------------
export LWD_EDGE_HOST="${LWD_EDGE_HOST:-76.76.26.17}"
export LWD_EDGE_NIC="${LWD_EDGE_NIC:-enp189s0f0}"
export LWD_CLOUD_HOST="${LWD_CLOUD_HOST:-76.76.26.231}"
export LWD_CLOUD_NIC="${LWD_CLOUD_NIC:-enp196s0f0}"
export LWD_MODEL="${LWD_MODEL:-/home/weight/Qwen3.6-27B/}"
# UCX tree, identical on both hosts and deliberately NOT derived from
# CODEBASE_DIR: it lives at /home/extra/ucx, outside the vllm-workspace checkout.
# lwd_require_ucx_runtime() prepends $LWD_UCX_LIB_DIR/lib to LD_LIBRARY_PATH and
# refuses to start when libucp is missing there.
export LWD_UCX_LIB_DIR="${LWD_UCX_LIB_DIR:-/home/extra/ucx/lib}"

# --- parallelism -----------------------------------------------------------
export LWD_DP_SIZE="${LWD_DP_SIZE:-2}"
export LWD_TP_SIZE="${LWD_TP_SIZE:-4}"
export LWD_PREFILL_DEVICES="${LWD_PREFILL_DEVICES:-0,1,2,3,4,5,6,7}"
export LWD_DECODE_DEVICES="${LWD_DECODE_DEVICES:-8,9,10,11,12,13,14,15}"
export LWD_EDGE_DEVICES="${LWD_EDGE_DEVICES:-0}"

# --- the knobs held identical across both deployments ----------------------
export LWD_GPU_MEM_UTIL="${LWD_GPU_MEM_UTIL:-0.95}"
export LWD_MAX_MODEL_LEN="${LWD_MAX_MODEL_LEN:-65536}"
export LWD_MAX_NUM_SEQS="${LWD_MAX_NUM_SEQS:-128}"
export LWD_MAX_NUM_BATCHED_TOKENS="${LWD_MAX_NUM_BATCHED_TOKENS:-8192}"
export LWD_ENABLE_MTP="${LWD_ENABLE_MTP:-1}"

# --- ports -----------------------------------------------------------------
# Kept clear of ../mooncake_pd2_tp4_231 (9060/9160/9161/29545/29546/29610/29611
# and KV 30000-30007 / 30100-30107) so both deployments can live on these hosts.
export LWD_PROXY_PORT="${LWD_PROXY_PORT:-8102}"
export LWD_EDGE_PORT="${LWD_EDGE_PORT:-8510}"
export LWD_PREFILL_PORT="${LWD_PREFILL_PORT:-8600}"
export LWD_DECODE_PORT="${LWD_DECODE_PORT:-8601}"
export LWD_PREFILL_MASTER_PORT="${LWD_PREFILL_MASTER_PORT:-29525}"
export LWD_DECODE_MASTER_PORT="${LWD_DECODE_MASTER_PORT:-29526}"
export LWD_PREFILL_DP_RPC="${LWD_PREFILL_DP_RPC:-29620}"
export LWD_DECODE_DP_RPC="${LWD_DECODE_DP_RPC:-29621}"
# Producer binds one handshake port per (dp, tp) pair: 14570-14577; the consumer
# only connects, its base just has to stay clear.
export LWD_PREFILL_KV_PORT="${LWD_PREFILL_KV_PORT:-14570}"
export LWD_DECODE_KV_PORT="${LWD_DECODE_KV_PORT:-14580}"

LWD_DRY_RUN=0
if [[ $# == 1 && $1 == --dry-run ]]; then
  LWD_DRY_RUN=1
elif [[ $# != 0 ]]; then
  printf 'Usage: bash %s [--dry-run]\n' "$0" >&2
  exit 2
fi

# Runtime environment common to the three engine processes, copied from the
# curl-verified v30_rc reference used by ../dp2_tp1_194_195.
lwd_runtime_env() {
  export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
  export HCCL_OP_EXPANSION_MODE="AIV"
  export TORCH_DISTRIBUTED_DETAIL=DEBUG
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export HCCL_BUFFSIZE=1024
  export OMP_PROC_BIND=false
  export OMP_NUM_THREADS=1
  export TASK_QUEUE_ENABLE=1
  # Only preload jemalloc where it exists: an unconditional LD_PRELOAD floods
  # every child with "cannot be preloaded: ignored" ERROR lines.
  if [[ -f /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ]]; then
    export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}"
  fi
  export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
  export VLLM_LWD_LOG_LEVEL="${VLLM_LWD_LOG_LEVEL:-INFO}"
  # The cloud engines need minutes for weights + graph capture before they
  # answer the edge; v30_rc raises the engine-ready budget to 1800s.
  export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"
}

# Per-host addressing: role is "edge" (17) or "cloud" (231). Also selects the
# UCX transports, because the two hosts are different card generations.
lwd_host_env() {
  local role=$1
  case $role in
    edge)
      export VLLM_HOST_IP="$LWD_EDGE_HOST"
      export MASTER_ADDR="$LWD_EDGE_HOST"
      export HCCL_IF_IP="$LWD_EDGE_HOST"
      export GLOO_SOCKET_IFNAME="$LWD_EDGE_NIC"
      export TP_SOCKET_IFNAME="$LWD_EDGE_NIC"
      export HCCL_SOCKET_IFNAME="$LWD_EDGE_NIC"
      export ASCEND_RT_VISIBLE_DEVICES="$LWD_EDGE_DEVICES"
      # a2 host: the RoCE pair validated on 194/195 (IB device + its netdev).
      export UCX_NET_DEVICES="${LWD_UCX_NET_DEVICES_17:-rocep67s0f0:1,enp67s0f0np0}"
      ;;
    cloud)
      export VLLM_HOST_IP="$LWD_CLOUD_HOST"
      export MASTER_ADDR="$LWD_CLOUD_HOST"
      export HCCL_IF_IP="$LWD_CLOUD_HOST"
      export GLOO_SOCKET_IFNAME="$LWD_CLOUD_NIC"
      export TP_SOCKET_IFNAME="$LWD_CLOUD_NIC"
      export HCCL_SOCKET_IFNAME="$LWD_CLOUD_NIC"
      # 231 is an a3 host: its RoCE device names are NOT the a2 ones, so they
      # must be supplied rather than guessed. Until they are, fall back to plain
      # tcp over the data NIC (works, but slower — do not compare bandwidth).
      if [[ -n ${LWD_UCX_NET_DEVICES_231:-} ]]; then
        export UCX_NET_DEVICES="$LWD_UCX_NET_DEVICES_231"
      else
        export UCX_NET_DEVICES="$LWD_CLOUD_NIC"
        export LWD_UCX_TCP_FALLBACK=1
        printf '[lwd][warn] LWD_UCX_NET_DEVICES_231 is unset: falling back to UCX_TLS=tcp\n' >&2
        printf '[lwd][warn]   export LWD_UCX_NET_DEVICES_231=rocepXXs0f0:1,enpXXs0f0np0\n' >&2
        printf '[lwd][warn]   (find it with: ibdev2netdev ; ls /sys/class/infiniband)\n' >&2
      fi
      ;;
    *)
      printf '[lwd][error] lwd_host_env: role must be edge or cloud, got %s\n' "$role" >&2
      return 2
      ;;
  esac

  # UCX tuning validated by v30_rc on the a2 hosts.
  unset UCX_IB_NUM_PATHS UCX_MAX_RNDV_LANES
  export UCX_RNDV_THRESH="${UCX_RNDV_THRESH:-512}"
  export UCX_PROTO_ENABLE="${UCX_PROTO_ENABLE:-n}"
  export UCX_RNDV_SCHEME="${UCX_RNDV_SCHEME:-put_zcopy}"
  if [[ ${LWD_UCX_TCP_FALLBACK:-0} == 1 ]]; then
    export UCX_TLS="${UCX_TLS:-tcp,self}"
  else
    export UCX_TLS="${UCX_TLS:-rc_verbs,tcp,self}"
    export UCX_IB_GID_INDEX="${UCX_IB_GID_INDEX:-5}"
    export UCX_IB_MTU="${UCX_IB_MTU:-1024}"
  fi
}

# Fail fast when libucp is absent. Mirrors ucx_ucp.py, which loads libucp.so.0
# then libucp.so; without either the link builder raises inside a background
# thread and the engine only dies at the 605s handshake timeout with a
# misleading "[LWD][handshake] host link not ready" message.
lwd_require_ucx_runtime() {
  if [[ -d $LWD_UCX_LIB_DIR ]]; then
    export LD_LIBRARY_PATH="$LWD_UCX_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  if [[ $LWD_DRY_RUN == 1 ]]; then
    return 0
  fi
  if ! ls "$LWD_UCX_LIB_DIR"/libucp.so* >/dev/null 2>&1; then
    printf '[lwd][preflight] ERROR: libucp not found under %s\n' "$LWD_UCX_LIB_DIR" >&2
    printf '[lwd][preflight]        LWD_UCX_LIB_DIR points at the UCX tree on this host.\n' >&2
    printf '[lwd][preflight]        without it the host data plane fails only after the 605s handshake.\n' >&2
    exit 5
  fi
}

# MTP draft list; empty when LWD_ENABLE_MTP=0 so the same launcher can serve as
# the no-draft baseline.
lwd_mtp_args() {
  LWD_MTP_ARGS=()
  if [[ $LWD_ENABLE_MTP == 1 ]]; then
    LWD_MTP_ARGS+=(--speculative-config '{"method":"mtp","num_speculative_tokens":3,"enforce_eager":true}')
  fi
}

# Refuse to start when a port this role needs is taken, naming the owner.
lwd_preflight() {
  local port
  for port in "$@"; do
    if command -v ss >/dev/null 2>&1 && [[ -n $(ss -ltnH "sport = :${port}" 2>/dev/null) ]]; then
      printf '[lwd][preflight] ERROR: port %s is already in use:\n' "$port" >&2
      ss -ltnp "sport = :${port}" >&2 || true
      exit 1
    fi
  done
}

# Run the server, or print the resolved argv in --dry-run mode.
lwd_run() {
  if [[ $LWD_DRY_RUN == 1 ]]; then
    "$LWD_PYTHON" - "$@" <<'PY'
import json, os, sys
argv = sys.argv[1:]
cmd = argv[argv.index("serve") + 1:] if "serve" in argv else argv
print(json.dumps({
    "command": cmd,
    "model": argv[argv.index("serve") + 1] if "serve" in argv else None,
    "environment": {
        k: v for k, v in os.environ.items()
        if k.startswith(("LWD_", "VLLM_", "HCCL_", "GLOO_", "TP_", "ASCEND_", "UCX_"))
    },
}, ensure_ascii=False, default=str))
PY
    return 0
  fi
  exec "$@"
}
