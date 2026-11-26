#!/usr/bin/env bash
# Prefill cloud (76.76.26.195, NPU 0,1) — cloud instance 0, DP=2 x TP=1.
#
# Multi-DP variant of pds/30rc_1124/sidecar/cloud_p.sh: the same two cards now
# host TWO data-parallel replicas of the prefill stage (one card each) instead of
# one TP=2 replica. The topology in this directory declares one dp: entry per
# rank, so each rank binds its own ctrl_port / data_port / coord_port.
#
# kv_port is a *base*: the connector derives the handshake port as
#   kv_port + data_parallel_rank * tp_size * pp_size
# (vllm_ascend/.../mooncake_connector.py:1762-1768, 2199-2200), so DP=2 TP=1
# yields 14570 and 14571. The decode cloud sits on 14580/14581 on the same host,
# hence the gap between the two bases.
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
  --tensor-parallel-size 1 \
  --data-parallel-size 2 \
  --data-parallel-address 76.76.26.195 --data-parallel-rpc-port 29600 \
  --additional-config '{"enable_cpu_binding":true,"lwd_config":{"path":'"${LWD_TOPOLOGY_JSON}"',"role":"cloud","instance_id":0}}' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_rank":0,"kv_port":14570,"kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":1},"decode":{"dp_size":2,"tp_size":1}}}' \
  --enforce-eager \
  --trust-remote-code --served-model-name qwen3.6 \
  --max-model-len 65536 --max-num-seqs 128 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --async-scheduling \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"enforce_eager":true}'
