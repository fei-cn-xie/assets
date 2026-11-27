#!/usr/bin/env bash
# Decode engine of the centralized Mooncake PD deployment:
# 2 DP x 4 TP on NPU 8-15 of 76.76.26.231 (DP rank 0 owns 8-11, rank 1 owns 12-15).
#
# Identical to prefill.sh except the KV role, the master / DP rpc ports and the
# FULL_DECODE_ONLY graph mode.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"
pd_runtime_env
pd_mtp_args
pd_mm_args
pd_api_args

export ASCEND_RT_VISIBLE_DEVICES="$PD_DECODE_DEVICES"

pd_preflight "$PD_DECODE_PORT" "$PD_DECODE_KV_PORT" "$PD_DECODE_DP_RPC" "$PD_DECODE_MASTER_PORT"

pd_run "$PD_PYTHON" -m vllm.entrypoints.cli.main serve "$PD_MODEL" \
  --host 0.0.0.0 \
  --port "$PD_DECODE_PORT" \
  --master-addr "$PD_HOST" --master-port "$PD_DECODE_MASTER_PORT" \
  --pipeline-parallel-size 1 \
  --tensor-parallel-size "$PD_TP_SIZE" \
  --data-parallel-size "$PD_DP_SIZE" \
  --data-parallel-address "$PD_HOST" --data-parallel-rpc-port "$PD_DECODE_DP_RPC" \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_rank":1,"engine_id":"1","kv_port":'"$PD_DECODE_KV_PORT"',"kv_connector_extra_config":{"prefill":{"dp_size":'"$PD_DP_SIZE"',"tp_size":'"$PD_TP_SIZE"'},"decode":{"dp_size":'"$PD_DP_SIZE"',"tp_size":'"$PD_TP_SIZE"'}}}' \
  --trust-remote-code --served-model-name qwen3.6 \
  --max-model-len "$PD_MAX_MODEL_LEN" --max-num-seqs "$PD_MAX_NUM_SEQS" --max-num-batched-tokens "$PD_MAX_NUM_BATCHED_TOKENS" \
  --gpu-memory-utilization "$PD_GPU_MEM_UTIL" \
  --async-scheduling \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  ${PD_API_ARGS[@]+"${PD_API_ARGS[@]}"} \
  ${PD_MM_ARGS[@]+"${PD_MM_ARGS[@]}"} \
  ${PD_MTP_ARGS[@]+"${PD_MTP_ARGS[@]}"}
