#!/usr/bin/env bash
# Cloud 1 (decode) — 76.76.26.231, NPU 8-15 as 2 DP x 4 TP.
#
# DP rank 0 owns NPU 8-11, rank 1 owns NPU 12-15. The consumer connects out, so
# its kv_port base only has to stay clear of the producer's range.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"
lwd_runtime_env
lwd_host_env cloud
lwd_mtp_args
lwd_require_ucx_runtime

export ASCEND_RT_VISIBLE_DEVICES="$LWD_DECODE_DEVICES"

lwd_preflight "$LWD_DECODE_PORT" "$LWD_DECODE_KV_PORT" "$LWD_DECODE_DP_RPC"

lwd_run "$LWD_PYTHON" -m vllm.entrypoints.cli.main serve "$LWD_MODEL" \
  --host 0.0.0.0 \
  --port "$LWD_DECODE_PORT" \
  --master-addr "$LWD_CLOUD_HOST" --master-port "$LWD_DECODE_MASTER_PORT" \
  --pipeline-parallel-size 1 \
  --tensor-parallel-size "$LWD_TP_SIZE" \
  --data-parallel-size "$LWD_DP_SIZE" \
  --data-parallel-address "$LWD_CLOUD_HOST" --data-parallel-rpc-port "$LWD_DECODE_DP_RPC" \
  --additional-config '{"enable_cpu_binding":true,"lwd_config":{"path":'"$(lwd_topology_json)"',"role":"cloud","instance_id":1}}' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_rank":1,"kv_port":'"$LWD_DECODE_KV_PORT"',"kv_connector_extra_config":{"prefill":{"dp_size":'"$LWD_DP_SIZE"',"tp_size":'"$LWD_TP_SIZE"'},"decode":{"dp_size":'"$LWD_DP_SIZE"',"tp_size":'"$LWD_TP_SIZE"'}}}' \
  --trust-remote-code --served-model-name qwen3.6 \
  --max-model-len "$LWD_MAX_MODEL_LEN" --max-num-seqs "$LWD_MAX_NUM_SEQS" --max-num-batched-tokens "$LWD_MAX_NUM_BATCHED_TOKENS" \
  --gpu-memory-utilization "$LWD_GPU_MEM_UTIL" \
  --async-scheduling \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  ${LWD_MTP_ARGS[@]+"${LWD_MTP_ARGS[@]}"}
