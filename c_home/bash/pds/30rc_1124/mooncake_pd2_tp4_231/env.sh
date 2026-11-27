#!/usr/bin/env bash
# Shared environment for the centralized Mooncake PD deployment
# (prefill 2 DP x 4 TP on NPU 0-7, decode 2 DP x 4 TP on NPU 8-15) on
# 76.76.26.231, an a3 card host whose data NIC is enp196s0f0.
#
# Sourced, never executed. Everything a run may want to change lives here; the
# launchers only name the pieces. Overrides may be exported beforehand:
#   PD_PYTHON, PD_MODEL, PD_HOST, PD_NIC_NAME, PD_GPU_MEM_UTIL,
#   PD_MAX_MODEL_LEN, PD_MAX_NUM_SEQS, PD_MAX_NUM_BATCHED_TOKENS,
#   PD_ENABLE_MTP, PD_SKIP_MM_PROFILING, PD_API_SERVER_COUNT
#
# The performance comparison this package exists for is only fair if both
# deployments use the same knobs: 95% memory utilisation, a 65536 context,
# 128 sequences and an 8192 batch window. Those four are asserted by
# check_configs.py against the edge-cloud sibling package
# ../lwd_dp2_tp4_17_231 via PD_PARITY_PACKAGE.

export PD_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PD_PYTHON="${PD_PYTHON:-python3}"
export PD_MODEL="${PD_MODEL:-/home/weight/Qwen3.6-27B/}"

# Upstream checkout the launchers run from; the proxy example comes from
# vllm-ascend rather than being vendored here.
export CODEBASE_DIR="${CODEBASE_DIR:-/vllm-workspace}"
export PD_ASCEND_DIR="${PD_ASCEND_DIR:-$CODEBASE_DIR/vllm-ascend/examples/disaggregated_prefill_v1}"
export PYTHONPATH="$CODEBASE_DIR/vllm-ascend:$CODEBASE_DIR/vllm${PYTHONPATH:+:$PYTHONPATH}"

# This deployment is single host, so the address is also its own peer address.
export PD_HOST="${PD_HOST:-76.76.26.231}"
export PD_NIC_NAME="${PD_NIC_NAME:-enp196s0f0}"

# --- parallelism: 2 data parallel replicas, 4 cards each -------------------
export PD_DP_SIZE="${PD_DP_SIZE:-2}"
export PD_TP_SIZE="${PD_TP_SIZE:-4}"
# NPU 0-7 prefill, NPU 8-15 decode; DP rank i owns the i-th group of 4 cards.
export PD_PREFILL_DEVICES="${PD_PREFILL_DEVICES:-0,1,2,3,4,5,6,7}"
export PD_DECODE_DEVICES="${PD_DECODE_DEVICES:-8,9,10,11,12,13,14,15}"

# --- the four knobs held identical across both deployments ------------------
export PD_GPU_MEM_UTIL="${PD_GPU_MEM_UTIL:-0.95}"
export PD_MAX_MODEL_LEN="${PD_MAX_MODEL_LEN:-65536}"
export PD_MAX_NUM_SEQS="${PD_MAX_NUM_SEQS:-128}"
export PD_MAX_NUM_BATCHED_TOKENS="${PD_MAX_NUM_BATCHED_TOKENS:-8192}"

# --- ports -----------------------------------------------------------------
# A second deployment (the edge-cloud sibling) may run on this host at another
# time, so keep both port sets apart and let the validator prove it.
export PD_PROXY_PORT="${PD_PROXY_PORT:-9060}"
export PD_PREFILL_PORT="${PD_PREFILL_PORT:-9160}"
export PD_DECODE_PORT="${PD_DECODE_PORT:-9161}"
export PD_PREFILL_MASTER_PORT="${PD_PREFILL_MASTER_PORT:-29545}"
export PD_DECODE_MASTER_PORT="${PD_DECODE_MASTER_PORT:-29546}"
export PD_PREFILL_DP_RPC="${PD_PREFILL_DP_RPC:-29610}"
export PD_DECODE_DP_RPC="${PD_DECODE_DP_RPC:-29611}"
# KV handshake base: the producer binds one port per (dp, tp) pair, i.e.
# base + dp_rank * tp + device, so 30000-30007 for prefill. The consumer only
# connects, its base just has to stay clear.
export PD_PREFILL_KV_PORT="${PD_PREFILL_KV_PORT:-30000}"
export PD_DECODE_KV_PORT="${PD_DECODE_KV_PORT:-30100}"

# --- optional engine switches ----------------------------------------------
# MTP drafts. Both deployments of the comparison turn this on so the numbers are
# comparable; with 4-way tensor parallelism the draft is sharded as well, which
# is what makes it affordable here (on a single card it used to cost ~6.5 GiB).
export PD_ENABLE_MTP="${PD_ENABLE_MTP:-1}"
# Skip the multimodal encoder profiling run (saves memory, less accurate for
# image requests). Off here: this deployment has room at TP=4.
export PD_SKIP_MM_PROFILING="${PD_SKIP_MM_PROFILING:-0}"
# Front-end process count. Empty keeps vLLM's default (= data_parallel_size,
# i.e. two front-ends). One front-end means much less absolute throughput, so a
# performance comparison must leave it empty.
export PD_API_SERVER_COUNT="${PD_API_SERVER_COUNT:-}"

PD_DRY_RUN=0
if [[ $# == 1 && $1 == --dry-run ]]; then
  PD_DRY_RUN=1
elif [[ $# != 0 ]]; then
  printf 'Usage: bash %s [--dry-run]\n' "$0" >&2
  exit 2
fi

pd_runtime_env() {
  # vLLM derives every peer address from get_ip(), which honours VLLM_HOST_IP and
  # otherwise auto-detects an interface (on the 195 host that picked a docker
  # bridge). Pin it to the data NIC's address.
  export VLLM_HOST_IP="$PD_HOST"
  export HCCL_IF_IP="$PD_HOST"
  export GLOO_SOCKET_IFNAME="$PD_NIC_NAME"
  export TP_SOCKET_IFNAME="$PD_NIC_NAME"
  export HCCL_SOCKET_IFNAME="$PD_NIC_NAME"
  export MASTER_ADDR="$PD_HOST"
  export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
  export HCCL_OP_EXPANSION_MODE="AIV"
  export HCCL_BUFFSIZE=1024
  export OMP_PROC_BIND=false
  export OMP_NUM_THREADS=1
  export TASK_QUEUE_ENABLE=1
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
  export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"
}

# Optional MTP flag list; empty when PD_ENABLE_MTP=0 so the same launcher can be
# used for a baseline measurement.
pd_mtp_args() {
  PD_MTP_ARGS=()
  if [[ $PD_ENABLE_MTP == 1 ]]; then
    PD_MTP_ARGS+=(--speculative-config '{"method":"mtp","num_speculative_tokens":3,"enforce_eager":true}')
  fi
}

pd_mm_args() {
  PD_MM_ARGS=()
  if [[ $PD_SKIP_MM_PROFILING == 1 ]]; then
    PD_MM_ARGS+=(--skip-mm-profiling)
  fi
}

pd_api_args() {
  PD_API_ARGS=()
  if [[ -n ${PD_API_SERVER_COUNT:-} ]]; then
    PD_API_ARGS+=(--api-server-count "$PD_API_SERVER_COUNT")
  fi
}

# Refuse to start when a port this deployment needs is already taken, and name
# the process holding it. Cheap insurance: a leftover engine from the sibling
# package looks exactly like a hung start otherwise.
pd_preflight() {
  local port
  for port in "$@"; do
    if command -v ss >/dev/null 2>&1 && [[ -n $(ss -ltnH "sport = :${port}" 2>/dev/null) ]]; then
      printf '[pd][preflight] ERROR: port %s is already in use:\n' "$port" >&2
      ss -ltnp "sport = :${port}" >&2 || true
      exit 1
    fi
  done
}

# Run the server, or print the resolved argv in --dry-run mode so the launchers
# can be checked without a card, a model or a network.
pd_run() {
  if [[ $PD_DRY_RUN == 1 ]]; then
    "$PD_PYTHON" - "$@" <<'PY'
import json, sys
argv = sys.argv[1:]
cmd = argv[argv.index("serve") + 1:] if "serve" in argv else argv
print(json.dumps({
    "command": cmd,
    "model": argv[argv.index("serve") + 1] if "serve" in argv else None,
    "environment": {
        k: v for k, v in __import__("os").environ.items()
        if k.startswith(("PD_", "VLLM_", "HCCL_", "GLOO_", "TP_", "ASCEND_"))
    },
}, ensure_ascii=False, default=str))
PY
    return 0
  fi
  exec "$@"
}
