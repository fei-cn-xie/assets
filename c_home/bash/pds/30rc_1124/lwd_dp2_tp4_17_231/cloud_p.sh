#!/usr/bin/env bash
# Cloud 0 (prefill) — 76.76.26.231, NPU 0-7 as 2 DP x 4 TP.
#
# DP rank 0 owns NPU 0-3, rank 1 owns NPU 4-7; the producer binds one KV
# handshake port per (dp, tp) pair, i.e. 14570-14577.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"
lwd_runtime_env
lwd_host_env cloud
lwd_mtp_args
lwd_require_ucx_runtime

export ASCEND_RT_VISIBLE_DEVICES="$LWD_PREFILL_DEVICES"

lwd_preflight "$LWD_PREFILL_PORT" "$LWD_PREFILL_KV_PORT" "$LWD_PREFILL_DP_RPC"

lwd_run "$LWD_PYTHON" -m vllm.entrypoints.cli.main serve "$LWD_MODEL" \
  --host 0.0.0.0 \
  --port "$LWD_PREFILL_PORT" \
  --master-addr "$LWD_CLOUD_HOST" --master-port "$LWD_PREFILL_MASTER_PORT" \
  --pipeline-parallel-size 1 \
  --tensor-parallel-size "$LWD_TP_SIZE" \
  --data-parallel-size "$LWD_DP_SIZE" \
  --data-parallel-address "$LWD_CLOUD_HOST" --data-parallel-rpc-port "$LWD_PREFILL_DP_RPC" \
  --additional-config '{"enable_cpu_binding":true,"lwd_config":{"path":'"$(lwd_topology_json)"',"role":"cloud","instance_id":0}}' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_rank":0,"kv_port":'"$LWD_PREFILL_KV_PORT"',"kv_connector_extra_config":{"prefill":{"dp_size":'"$LWD_DP_SIZE"',"tp_size":'"$LWD_TP_SIZE"'},"decode":{"dp_size":'"$LWD_DP_SIZE"',"tp_size":'"$LWD_TP_SIZE"'}}}' \
  --trust-remote-code --served-model-name qwen3.6 \
  --max-model-len "$LWD_MAX_MODEL_LEN" --max-num-seqs "$LWD_MAX_NUM_SEQS" --max-num-batched-tokens "$LWD_MAX_NUM_BATCHED_TOKENS" \
  --gpu-memory-utilization "$LWD_GPU_MEM_UTIL" \
  --async-scheduling \
  ${LWD_MTP_ARGS[@]+"${LWD_MTP_ARGS[@]}"}
