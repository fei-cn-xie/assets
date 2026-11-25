#!/usr/bin/env bash
# Decode engine of the centralized Mooncake PD deployment (76.76.26.195, NPU 3,4).
#
# Same flags as pds/30rc_1124/sidecar/cloud_d.sh minus the LWD layer:
#   * no --additional-config lwd_config
# Kept as-is: graph mode (--compilation-config FULL_DECODE_ONLY), MTP with three
# draft tokens, Mooncake consumer. MTP belongs on this side: decode is the stage
# that generates tokens.
# The --kv-transfer-config field set follows the vllm-ascend guide:
#   vllm-ascend/examples/disaggregated_prefill_v1/mooncake_connector_deployment_guide.md
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"
pd_runtime_env

export VLLM_HOST_IP="$PD_HOST"
export MASTER_ADDR="$PD_HOST"
export ASCEND_RT_VISIBLE_DEVICES=3,4

pd_preflight "$PD_DECODE_PORT" "$PD_DECODE_KV_PORT" "$PD_DECODE_MASTER_PORT"

pd_run "$PD_PYTHON" -m vllm.entrypoints.cli.main serve "$PD_MODEL" \
  --host 0.0.0.0 \
  --port "$PD_DECODE_PORT" \
  --master-addr "$PD_HOST" --master-port "$PD_DECODE_MASTER_PORT" \
  --pipeline-parallel-size 1 \
  --tensor-parallel-size 2 \
  --additional-config '{"enable_cpu_binding":true}' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_buffer_device":"npu","kv_role":"kv_consumer","kv_parallel_size":1,"kv_rank":1,"kv_port":"'"$PD_DECODE_KV_PORT"'","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":2},"decode":{"dp_size":1,"tp_size":2}}}' \
  --trust-remote-code --served-model-name qwen3.6 \
  --max-model-len 65536 --max-num-seqs 128 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --async-scheduling \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"enforce_eager":true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
