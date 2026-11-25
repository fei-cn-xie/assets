#!/usr/bin/env bash
# Prefill engine of the centralized Mooncake PD deployment (76.76.26.195, NPU 0,1).
#
# Same flags as pds/30rc_1124/sidecar/cloud_p.sh minus the LWD layer:
#   * no --additional-config lwd_config (no control plane, no host data plane)
#   * no --speculative-config: this engine only ever answers the proxy's
#     "max_tokens=1" prefill probe, so draft tokens would be pure overhead.
# Kept as-is: --enforce-eager (prefill-only, no decode graphs), Mooncake producer.
# The --kv-transfer-config field set (kv_buffer_device / kv_parallel_size /
# string kv_port / kv_rank) follows the vllm-ascend guide:
#   vllm-ascend/examples/disaggregated_prefill_v1/mooncake_connector_deployment_guide.md
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"
pd_runtime_env

export VLLM_HOST_IP="$PD_HOST"
export MASTER_ADDR="$PD_HOST"
export ASCEND_RT_VISIBLE_DEVICES=0,1

pd_preflight "$PD_PREFILL_PORT" "$PD_PREFILL_KV_PORT" "$PD_PREFILL_MASTER_PORT"

pd_run "$PD_PYTHON" -m vllm.entrypoints.cli.main serve "$PD_MODEL" \
  --host 0.0.0.0 \
  --port "$PD_PREFILL_PORT" \
  --master-addr "$PD_HOST" --master-port "$PD_PREFILL_MASTER_PORT" \
  --pipeline-parallel-size 1 \
  --tensor-parallel-size 2 \
  --additional-config '{"enable_cpu_binding":true}' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_buffer_device":"npu","kv_role":"kv_producer","kv_parallel_size":1,"kv_rank":0,"kv_port":"'"$PD_PREFILL_KV_PORT"'","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":2},"decode":{"dp_size":1,"tp_size":2}}}' \
  --enforce-eager \
  --trust-remote-code --served-model-name qwen3.6 \
  --max-model-len 65536 --max-num-seqs 128 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --async-scheduling
