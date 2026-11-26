#!/usr/bin/env bash
# Prefill engine of the centralized Mooncake PD deployment, DP=2 x TP=1
# (76.76.26.195, NPU 0,1 — one replica per card).
#
# Differences from ../mooncake_pd_195 (the confirmed TP=2 x DP=1 reference):
#   * --tensor-parallel-size 1 --data-parallel-size 2 (+ address / rpc port)
#   * kv_connector_extra_config describes dp 2 / tp 1 on both sides
#   * max-model-len 32768 / max-num-seqs 32 / 0.95: TP=1 puts the full 46.8 GiB
#     of weights on each card, leaving only a few GiB of KV
#   * MTP drafts behind PD_ENABLE_MTP
#
# On a branch without the multi-DP fix this engine fails in one of two ways:
#   * PD_ENABLE_MTP=0 -> "ZMQError: Address already in use" on the KV handshake
#     port (both DP ranks derive the same port from data_parallel_rank)
#   * PD_ENABLE_MTP=1 -> earlier, in the MTP drafter, with
#     "KV transfer ... Expected 1, but got 2"
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"
pd_runtime_env
pd_mtp_args

export ASCEND_RT_VISIBLE_DEVICES=0,1

pd_preflight "$PD_PREFILL_PORT" "$PD_PREFILL_KV_PORT" "$PD_PREFILL_DP_RPC"

pd_run "$PD_PYTHON" -m vllm.entrypoints.cli.main serve "$PD_MODEL" \
  --host 0.0.0.0 \
  --port "$PD_PREFILL_PORT" \
  --tensor-parallel-size "$PD_TP_SIZE" \
  --data-parallel-size "$PD_DP_SIZE" \
  --data-parallel-address "$PD_HOST" \
  --data-parallel-rpc-port "$PD_PREFILL_DP_RPC" \
  --served-model-name qwen3.6 \
  --trust-remote-code \
  --max-num-seqs "$PD_MAX_NUM_SEQS" \
  --max-model-len "$PD_MAX_MODEL_LEN" \
  --max-num-batched-tokens "$PD_MAX_NUM_BATCHED_TOKENS" \
  --enable-prefix-caching \
  --gpu-memory-utilization "$PD_GPU_MEM_UTIL" \
  --async-scheduling \
  --distributed-executor-backend mp \
  --enforce-eager \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_port":"'"$PD_PREFILL_KV_PORT"'","engine_id":"0","kv_connector_extra_config":{"prefill":{"dp_size":'"$PD_DP_SIZE"',"tp_size":'"$PD_TP_SIZE"'},"decode":{"dp_size":'"$PD_DP_SIZE"',"tp_size":'"$PD_TP_SIZE"'}}}' \
  ${PD_MTP_ARGS[@]+"${PD_MTP_ARGS[@]}"}
