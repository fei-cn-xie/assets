#!/usr/bin/env bash
# Prefill cloud (76.76.26.195, NPU 0,1) — cloud instance 0, DP=2 x TP=1.
#
# Each DP rank is one replica on one card. For a dense model every rank runs as an
# independent engine whose ParallelConfig collapses to DP=1, so the two things
# that follow depend on fix/lwd-multidp-dense-pd:
#   * the KV extra-config DP check compares against the preserved deployment DP
#     (vllm ParallelConfig.data_parallel_size_global);
#   * the KV handshake port is derived from data_parallel_index, giving this
#     instance ports 14570 (dp 0) and 14571 (dp 1) instead of colliding on 14570.
#
# TP=1 means each replica holds the FULL 64-layer model (50.2 GB ≈ 46.8 GiB) on a
# single 64 GB card, so the KV budget is the binding constraint: 0.95 utilization
# leaves ≈6.7 GiB, while max_num_seqs is capped by the GDN(Mamba) block count
# (≈123 blocks at that pool size) — 64 stays clear of it.
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
  --trust-remote-code --served-model-name qwen3.6 \
  --max-model-len "${LWD_MAX_MODEL_LEN:-32768}" --max-num-seqs "${LWD_MAX_NUM_SEQS:-64}" --max-num-batched-tokens 8192 \
  --gpu-memory-utilization "${LWD_GPU_MEM_UTIL:-0.95}" \
  --async-scheduling \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"enforce_eager":true}'
