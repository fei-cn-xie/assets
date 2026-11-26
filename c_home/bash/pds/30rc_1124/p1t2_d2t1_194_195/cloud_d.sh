#!/usr/bin/env bash
# Decode cloud (76.76.26.195, NPU 2,3) — cloud instance 1, DP=2 x TP=1.
#
# The decode side is where multi-DP is safe: MooncakeConnectorV1's consumer only
# creates a zmq.REQ with bind=False (mooncake_connector.py:1546-1551), so the two
# replicas never fight over a port. The prefill side cannot do this -- see
# cloud_p.sh.
#
# TP=1 means each replica packs the FULL 64-layer model (50.2 GB ≈ 46.8 GiB) on
# one 64 GB card, so memory is tight:
#   * --gpu-memory-utilization 0.95 leaves ~6.7 GiB for KV (0.90 left 3.3 GiB,
#     which failed the 65536-context check);
#   * this model has GDN (Mamba) layers: every decode sequence occupies one Mamba
#     cache block, and 6.7 GiB yields only ~123 blocks, so --max-num-seqs must
#     stay below that (64 here; 128 failed with "max_num_seqs exceeds available
#     Mamba cache blocks");
#   * no --speculative-config: with DP>1 the MTP drafter rebuilds a VllmConfig at
#     DP=1 and vllm-ascend's check_kv_extra_config rejects decode.dp_size=2
#     ("Expected 1, but got 2").
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"
lwd_runtime_env

export VLLM_HOST_IP=76.76.26.195
export MASTER_ADDR=76.76.26.195
export ASCEND_RT_VISIBLE_DEVICES=2,3

lwd_require_ucx_runtime

lwd_run "$LWD_PYTHON" -m vllm.entrypoints.cli.main serve /home/weight/Qwen3.6-27B/ \
  --host 0.0.0.0 \
  --port 8601 \
  --master-addr 76.76.26.195 --master-port 29526 \
  --pipeline-parallel-size 1 \
  --tensor-parallel-size 1 \
  --data-parallel-size 2 \
  --data-parallel-address 76.76.26.195 --data-parallel-rpc-port 29601 \
  --additional-config '{"enable_cpu_binding":true,"lwd_config":{"path":'"${LWD_TOPOLOGY_JSON}"',"role":"cloud","instance_id":1}}' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_rank":1,"kv_port":14580,"kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":2},"decode":{"dp_size":2,"tp_size":1}}}' \
  --trust-remote-code --served-model-name qwen3.6 \
  --max-model-len "${LWD_MAX_MODEL_LEN:-32768}" --max-num-seqs "${LWD_MAX_NUM_SEQS:-64}" --max-num-batched-tokens 8192 \
  --gpu-memory-utilization "${LWD_GPU_MEM_UTIL:-0.95}" \
  --async-scheduling \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
