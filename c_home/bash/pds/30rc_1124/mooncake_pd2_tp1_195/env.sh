#!/usr/bin/env bash
# Shared environment for the CENTRALIZED (no edge, no LWD) Mooncake PD
# deployment with DP=2 x TP=1 on both sides, all on 76.76.26.195.
#
# Derived from the confirmed-working reference in ../mooncake_pd_195 (which is
# TP=2 x DP=1); only the parallelism and the DP plumbing differ. The runtime
# environment (NIC, HCCL_IF_IP, mooncake site-packages, jemalloc, AIV expansion,
# expandable segments, OMP/queue knobs, spawn method) is copied verbatim.
#
# Sourced, never executed. Overrides may be exported beforehand:
#   CODEBASE_DIR, PD_PYTHON, PD_NIC_NAME, PD_HOST, PD_MODEL,
#   PD_MOONCAKE_LIB, PD_ASCEND_DIR, VLLM_LOGGING_LEVEL,
#   PD_MAX_MODEL_LEN, PD_MAX_NUM_SEQS, PD_MAX_NUM_BATCHED_TOKENS,
#   PD_GPU_MEM_UTIL, PD_ENABLE_MTP
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
export PD_ASCEND_DIR="${PD_ASCEND_DIR:-$CODEBASE_DIR/vllm-ascend/examples/disaggregated_prefill_v1}"

# ---------------------------------------------------------------- parallelism
# Two independent replicas per side, one card each. Both sides use the same
# shape so the KV connector sees a symmetric layout.
export PD_DP_SIZE="${PD_DP_SIZE:-2}"
export PD_TP_SIZE="${PD_TP_SIZE:-1}"

# ---------------------------------------------------------------------- ports
export PD_PROXY_PORT="${PD_PROXY_PORT:-9060}"
export PD_PREFILL_PORT="${PD_PREFILL_PORT:-9160}"
export PD_DECODE_PORT="${PD_DECODE_PORT:-9161}"
# KV handshake bases. The producer binds base + dp_rank * tp * pp + device_index,
# so this side needs PD_DP_SIZE * PD_TP_SIZE consecutive free ports per engine.
export PD_PREFILL_KV_PORT="${PD_PREFILL_KV_PORT:-30000}"
export PD_DECODE_KV_PORT="${PD_DECODE_KV_PORT:-30100}"
# DP coordinator RPC ports; P and D share the host, so they must differ
# (ParallelConfig default is 29550 for both).
export PD_PREFILL_DP_RPC="${PD_PREFILL_DP_RPC:-29600}"
export PD_DECODE_DP_RPC="${PD_DECODE_DP_RPC:-29601}"
# torch-distributed rendezvous ports. Prefill and decode share this host and the
# edge-cloud deployment already owns 29525/29526, so keep a separate pair.
export PD_PREFILL_MASTER_PORT="${PD_PREFILL_MASTER_PORT:-29535}"
export PD_DECODE_MASTER_PORT="${PD_DECODE_MASTER_PORT:-29536}"

# --------------------------------------------------------------------- memory
# TP=1 means every replica holds the full 64-layer model (50.2 GB ≈ 46.8 GiB) on
# one 64 GB card, so the KV pool is the binding constraint: 0.95 utilization
# covers ~32-63 GDN(Mamba) blocks with MTP on, and each decode sequence needs
# one block, hence max_num_seqs 32. These three values are shared by both sides
# on purpose: max_model_len is hashed into the KV connector's extra config
# layout, and the batch limits bound what the pipeline as a whole can admit.
export PD_MAX_MODEL_LEN="${PD_MAX_MODEL_LEN:-512}"
export PD_MAX_NUM_SEQS="${PD_MAX_NUM_SEQS:-32}"
export PD_MAX_NUM_BATCHED_TOKENS="${PD_MAX_NUM_BATCHED_TOKENS:-8192}"
export PD_GPU_MEM_UTIL="${PD_GPU_MEM_UTIL:-0.98}"
# MTP drafts, same switch and same method string as the working edge-cloud DP=2
# package. Verified before this package existed: with MTP on, a centralized TP=1
# engine on this host reported only 0.30 GiB of KV for a 32768 context that needs
# 3.02 GiB and refused to start, while the edge-cloud engine starts with MTP on.
# The two remaining differences are inherent to LWD: its cloud workers skip the
# visual tower ("role=cloud skips loading tensors of visual") and use their own
# LwdMTPProposer. If this deployment is short on KV, drop the draft with
# PD_ENABLE_MTP=0 and expect ~6.8 GiB instead.
export PD_ENABLE_MTP="${PD_ENABLE_MTP:-1}"
# A centralized engine loads the multimodal model itself, so its profiling run also
# covers the vision encoder. LWD clouds never do: the module ownership table gives
# `visual` to the edge only (vllm/model_executor/models/utils.py:864-875), so the
# cloud neither builds nor loads the tower (lwd_weight_filter.py). Skipping the
# encoder profiling run frees part of that reservation at the cost of profiling
# accuracy for image requests; turn it on when KV is short.
export PD_SKIP_MM_PROFILING="${PD_SKIP_MM_PROFILING:-0}"
# How many front-end (API server) processes serve this engine. Empty keeps vLLM's
# default, which is `data_parallel_size` (2 here) and therefore two front-ends
# that each have to pair with one engine. Set it to 1 to let a single front-end
# load-balance both DP engines; that removes the front-end <-> engine pairing
# from startup, which is where "Did not receive response from front-end" and the
# DP coordinator's "unexpected message while waiting for engines to subscribe"
# come from. Costs a little LB throughput.
export PD_API_SERVER_COUNT="${PD_API_SERVER_COUNT:-}"

PD_DRY_RUN=0
if [[ $# == 1 && $1 == --dry-run ]]; then
  PD_DRY_RUN=1
elif [[ $# != 0 ]]; then
  printf 'Usage: bash %s [--dry-run]\n' "$0" >&2
  exit 2
fi

pd_runtime_env() {
  export LD_LIBRARY_PATH="$PD_MOONCAKE_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  # vLLM derives every peer address from get_ip(), which honours VLLM_HOST_IP and
  # otherwise auto-detects an interface. On this host auto-detection picked a
  # docker bridge (76.1.1.2) and the KV handshake metadata advertised that
  # address instead of 76.76.26.195. The working edge-cloud package sets it in
  # every engine script; set it here once for both engines.
  export VLLM_HOST_IP="$PD_HOST"
  export HCCL_IF_IP="$PD_HOST"
  export GLOO_SOCKET_IFNAME="$PD_NIC_NAME"
  export TP_SOCKET_IFNAME="$PD_NIC_NAME"
  export HCCL_SOCKET_IFNAME="$PD_NIC_NAME"
  # Only preload jemalloc where it exists: these hosts do not ship it, and an
  # unconditional LD_PRELOAD floods every child process with ld.so ERROR lines.
  if [[ -f /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ]]; then
    export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}"
  fi
  export HCCL_OP_EXPANSION_MODE="AIV"
  export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
  export OMP_PROC_BIND=false
  export OMP_NUM_THREADS=1
  export TASK_QUEUE_ENABLE=1
  export HCCL_BUFFSIZE=1024
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
}

# Build the optional MTP flag list. Empty when PD_ENABLE_MTP=0 so an unfixed
# branch fails on the DP problem rather than on the draft config.
pd_mtp_args() {
  PD_MTP_ARGS=()
  if [[ $PD_ENABLE_MTP == 1 ]]; then
    PD_MTP_ARGS+=(--speculative-config '{"method":"mtp","num_speculative_tokens":3,"enforce_eager":true}')
  fi
}

# Warn when a port of this deployment is already taken (a stale engine, or the
# LWD deployment still running), which would otherwise fail deep inside vLLM.
# Optional multimodal profiling switch (see PD_SKIP_MM_PROFILING in the header).
pd_mm_args() {
  PD_MM_ARGS=()
  if [[ $PD_SKIP_MM_PROFILING == 1 ]]; then
    PD_MM_ARGS+=(--skip-mm-profiling)
  fi
}

# Optional front-end process count (see PD_API_SERVER_COUNT in the header).
pd_api_args() {
  PD_API_ARGS=()
  if [[ -n ${PD_API_SERVER_COUNT:-} ]]; then
    PD_API_ARGS+=(--api-server-count "$PD_API_SERVER_COUNT")
  fi
}

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
    LD_PRELOAD= "$PD_PYTHON" - "$@" <<'PY'
import json
import os
import sys

names = (
    "CODEBASE_DIR", "PYTHONPATH", "ASCEND_RT_VISIBLE_DEVICES",
    "VLLM_HOST_IP", "HCCL_IF_IP", "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME", "HCCL_SOCKET_IFNAME",
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
