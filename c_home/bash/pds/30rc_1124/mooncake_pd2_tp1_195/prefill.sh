#!/usr/bin/env bash
# Prefill engine of the centralized Mooncake PD deployment, DP=2 x TP=1
# (76.76.26.195, NPU 0,1 — one replica per card).
#
# Engine flags are kept identical to the confirmed-working edge-cloud DP=2
# package (../dp2_tp1_194_195/cloud_p.sh) so a difference between the two
# deployments can only come from LWD itself. Compared to that file this script
# only changes what centralized PD inherently needs:
#   * --port / --master-port / --data-parallel-rpc-port of this deployment
#   * the KV roles, kv_port base and engine_id
#   * no --additional-config (there is no LWD topology here)
#
# Both DP replicas of this engine bind a KV handshake port
# (kv_port + dp_rank); on a branch without the multi-DP fix they collide and the
# engine dies with "ZMQError: Address already in use".
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
  ${PD_MTP_ARGS[@]+"${PD_MTP_ARGS[@]}"}
