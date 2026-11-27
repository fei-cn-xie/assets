#!/usr/bin/env bash
# Prefill engine of the centralized Mooncake PD deployment:
# 2 DP x 4 TP on NPU 0-7 of 76.76.26.231 (DP rank 0 owns 0-3, rank 1 owns 4-7).
#
# Flags mirror ../mooncake_pd2_tp1_195/prefill.sh (the curl-verified DP=2 package)
# so the two packages differ only in parallelism, host and the knobs this
# performance comparison pins: 0.95 utilisation, 65536 context, 128 sequences,
# 8192 batch window.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"
pd_runtime_env
pd_mtp_args
pd_mm_args
pd_api_args

export ASCEND_RT_VISIBLE_DEVICES="$PD_PREFILL_DEVICES"

pd_preflight "$PD_PREFILL_PORT" "$PD_PREFILL_KV_PORT" "$PD_PREFILL_DP_RPC" "$PD_PREFILL_MASTER_PORT"

pd_run "$PD_PYTHON" -m vllm.entrypoints.cli.main serve "$PD_MODEL" \
  --host 0.0.0.0 \
  --port "$PD_PREFILL_PORT" \
  --master-addr "$PD_HOST" --master-port "$PD_PREFILL_MASTER_PORT" \
  --pipeline-parallel-size 1 \
  --tensor-parallel-size "$PD_TP_SIZE" \
  --data-parallel-size "$PD_DP_SIZE" \
  --data-parallel-address "$PD_HOST" --data-parallel-rpc-port "$PD_PREFILL_DP_RPC" \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_rank":0,"engine_id":"0","kv_port":'"$PD_PREFILL_KV_PORT"',"kv_connector_extra_config":{"prefill":{"dp_size":'"$PD_DP_SIZE"',"tp_size":'"$PD_TP_SIZE"'},"decode":{"dp_size":'"$PD_DP_SIZE"',"tp_size":'"$PD_TP_SIZE"'}}}' \
  --trust-remote-code --served-model-name qwen3.6 \
  --max-model-len "$PD_MAX_MODEL_LEN" --max-num-seqs "$PD_MAX_NUM_SEQS" --max-num-batched-tokens "$PD_MAX_NUM_BATCHED_TOKENS" \
  --gpu-memory-utilization "$PD_GPU_MEM_UTIL" \
  --async-scheduling \
  ${PD_API_ARGS[@]+"${PD_API_ARGS[@]}"} \
  ${PD_MM_ARGS[@]+"${PD_MM_ARGS[@]}"} \
  ${PD_MTP_ARGS[@]+"${PD_MTP_ARGS[@]}"}
