#!/usr/bin/env bash
# Decode engine of the centralized Mooncake PD deployment (76.76.26.195, NPU 2,3).
#
# Flags mirror the confirmed-working reference
# private_home/vllm_script/pds/centralized/decode.sh, with only these deltas:
#   * --port 9161 instead of 8061 (internal to this deployment)
#   * --kv-port 30100 kept identical to the reference
# Everything else (TP=2, prefix caching, 0.95 memory, async scheduling, mp
# executor, FULL_DECODE_ONLY with the reference's capture-size list, qwen3_5_mtp
# drafts, no --enforce-eager) is byte-for-byte the reference.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"
pd_runtime_env

export ASCEND_RT_VISIBLE_DEVICES=2,3

pd_preflight "$PD_DECODE_PORT" "$PD_DECODE_KV_PORT"

pd_run "$PD_PYTHON" -m vllm.entrypoints.cli.main serve "$PD_MODEL" \
  --host 0.0.0.0 \
  --port "$PD_DECODE_PORT" \
  --tensor-parallel-size 2 \
  --served-model-name qwen3.6 \
  --trust-remote-code \
  --max-num-seqs 128 \
  --max-model-len 65536 \
  --max-num-batched-tokens 8192 \
  --async-scheduling \
  --distributed-executor-backend mp \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.95 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100]}' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_port":"'"$PD_DECODE_KV_PORT"'","engine_id":"1","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":2},"decode":{"dp_size":1,"tp_size":2}}}' \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}'
