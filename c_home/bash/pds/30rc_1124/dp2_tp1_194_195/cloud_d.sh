#!/usr/bin/env bash
# Decode cloud (76.76.26.195, NPU 2,3) — cloud instance 1, DP=2 x TP=1.
#
# Mirrors cloud_p.sh: two independent replicas, one card each, KV extra-config
# compared against the preserved deployment DP, and a per-rank kv_port base
# derived from data_parallel_index (the consumer only connects, so 14580/14581
# never have to be free).
#
# Memory: same TP=1 pressure as the prefill side (full 64-layer model per card).
# With MTP the KV pool only covers ~32-63 GDN(Mamba) blocks, one per decode
# sequence, so max_num_seqs is 32 (64 was rejected at startup).
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
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_rank":1,"kv_port":14580,"kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":1},"decode":{"dp_size":2,"tp_size":1}}}' \
  --trust-remote-code --served-model-name qwen3.6 \
  --max-model-len "${LWD_MAX_MODEL_LEN:-32768}" --max-num-seqs "${LWD_MAX_NUM_SEQS:-32}" --max-num-batched-tokens "${LWD_MAX_NUM_BATCHED_TOKENS:-8192}" \
  --gpu-memory-utilization "${LWD_GPU_MEM_UTIL:-0.95}" \
  --async-scheduling \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"enforce_eager":true}'
