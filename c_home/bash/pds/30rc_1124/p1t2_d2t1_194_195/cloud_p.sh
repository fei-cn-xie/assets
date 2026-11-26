#!/usr/bin/env bash
# Prefill cloud (76.76.26.195, NPU 0,1) — cloud instance 0, DP=1 x TP=2.
#
# Identical in shape to pds/30rc_1124/sidecar/cloud_p.sh: a single prefill
# replica over two cards. DP stays 1 on purpose -- the MooncakeConnectorV1
# producer binds one ZMQ ROUTER per rank on
#   kv_port + data_parallel_rank * tp * pp * pcp + device_index
# and the LWD non-MoE path resets data_parallel_rank to 0, so a producer with
# DP>1 collides on the same port. Decode (the consumer) connects out and is free
# to run DP=2 -- see cloud_d.sh.
#
# --kv-transfer-config describes BOTH sides (dp1/tp2 prefill, dp2/tp1 decode);
# each engine only validates its own role's block
# (vllm_ascend/utils.py:1601-1604), so the asymmetric pair is legal.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"
lwd_runtime_env

export VLLM_HOST_IP=76.76.26.195
export MASTER_ADDR=76.76.26.195
export ASCEND_RT_VISIBLE_DEVICES=0,1

lwd_require_ucx_runtime

lwd_run "$LWD_PYTHON" -m vllm.entrypoints.cli.main serve /home/weight/Qwen3.6-27B/ \
  --host 0.0.0.0 \
  --port 8600 \
  --master-addr 76.76.26.195 --master-port 29525 \
  --pipeline-parallel-size 1 \
  --tensor-parallel-size 2 \
  --additional-config '{"enable_cpu_binding":true,"lwd_config":{"path":'"${LWD_TOPOLOGY_JSON}"',"role":"cloud","instance_id":0}}' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_rank":0,"kv_port":14570,"kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":2},"decode":{"dp_size":2,"tp_size":1}}}' \
  --enforce-eager \
  --trust-remote-code --served-model-name qwen3.6 \
  --max-model-len "${LWD_MAX_MODEL_LEN:-32768}" --max-num-seqs 128 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --async-scheduling \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"enforce_eager":true}'
