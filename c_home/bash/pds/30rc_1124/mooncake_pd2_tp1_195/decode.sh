#!/usr/bin/env bash
# Decode engine of the centralized Mooncake PD deployment, DP=2 x TP=1
# (76.76.26.195, NPU 2,3 — one replica per card).
#
# Mirrors prefill.sh except for the KV role, the FULL_DECODE_ONLY graph mode and
# the DP rpc port. The consumer side only connects out, so its kv_port base just
# has to stay clear of the producer's range.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"
pd_runtime_env
pd_mtp_args

export ASCEND_RT_VISIBLE_DEVICES=2,3

pd_preflight "$PD_DECODE_PORT" "$PD_DECODE_KV_PORT" "$PD_DECODE_DP_RPC"

pd_run "$PD_PYTHON" -m vllm.entrypoints.cli.main serve "$PD_MODEL" \
  --host 0.0.0.0 \
  --port "$PD_DECODE_PORT" \
  --tensor-parallel-size "$PD_TP_SIZE" \
  --data-parallel-size "$PD_DP_SIZE" \
  --data-parallel-address "$PD_HOST" \
  --data-parallel-rpc-port "$PD_DECODE_DP_RPC" \
  --served-model-name qwen3.6 \
  --trust-remote-code \
  --max-num-seqs "$PD_MAX_NUM_SEQS" \
  --max-model-len "$PD_MAX_MODEL_LEN" \
  --max-num-batched-tokens "$PD_MAX_NUM_BATCHED_TOKENS" \
  --async-scheduling \
  --distributed-executor-backend mp \
  --enable-prefix-caching \
  --gpu-memory-utilization "$PD_GPU_MEM_UTIL" \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,12,16,20,24,28,32]}' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_port":"'"$PD_DECODE_KV_PORT"'","engine_id":"1","kv_connector_extra_config":{"prefill":{"dp_size":'"$PD_DP_SIZE"',"tp_size":'"$PD_TP_SIZE"'},"decode":{"dp_size":'"$PD_DP_SIZE"',"tp_size":'"$PD_TP_SIZE"'}}}' \
  ${PD_MTP_ARGS[@]+"${PD_MTP_ARGS[@]}"}
