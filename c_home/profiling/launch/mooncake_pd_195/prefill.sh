#!/usr/bin/env bash
# Prefill engine of the centralized Mooncake PD deployment (76.76.26.195, NPU 0,1).
#
# Flags mirror the confirmed-working reference
# private_home/vllm_script/pds/centralized/prefill.sh, with only these deltas:
#   * --port 9160 instead of 8060 (internal to this deployment)
#   * --kv-port 30000 kept identical to the reference
# Everything else (TP=2, prefix caching, 0.95 memory, async scheduling, mp
# executor, --enforce-eager, qwen3_5_mtp drafts) is byte-for-byte the reference.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"
pd_runtime_env

export ASCEND_RT_VISIBLE_DEVICES=0,1

pd_preflight "$PD_PREFILL_PORT" "$PD_PREFILL_KV_PORT"

pd_run "$PD_PYTHON" -m vllm.entrypoints.cli.main serve "$PD_MODEL" \
  --host 0.0.0.0 \
  --port "$PD_PREFILL_PORT" \
  --tensor-parallel-size 2 \
  --served-model-name qwen3.6 \
  --trust-remote-code \
  --max-num-seqs 128 \
  --max-model-len 65536 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.95 \
  --async-scheduling \
  --distributed-executor-backend mp \
  --enforce-eager \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_port":"'"$PD_PREFILL_KV_PORT"'","engine_id":"0","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":2},"decode":{"dp_size":1,"tp_size":2}}}' \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}'
