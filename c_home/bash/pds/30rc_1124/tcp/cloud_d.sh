#!/usr/bin/env bash
# Cloud B / decode (cloud instance 1). Start after cloud_p.sh.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"
lwd_transport_env cloud

# ---- instance-local placement -------------------------------------------------
export VLLM_HOST_IP=76.76.26.189
# D uses NPU 4,5; P owns 0,1 on the same host.
export ASCEND_RT_VISIBLE_DEVICES=4,5
export GLOO_SOCKET_IFNAME=enp189s0f0
export TP_SOCKET_IFNAME=enp189s0f0
export HCCL_SOCKET_IFNAME=enp189s0f0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}"
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export HCCL_BUFFSIZE=1024
# Prepended AFTER lwd_transport_env so the UCX libucp path stays in the list.
export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:${LD_LIBRARY_PATH:-}"

# profiling configuration
#export SERVICE_PROF_CONFIG_PATH=/bash_dir/profiling/ms_service_profiler_config.json
#export PROFILING_SYMBOLS_PATH=/bash_dir/profiling/prof_dir/service_profiling_symbols.0.23.0.yaml
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-ERROR}"

# Decode keeps graph mode (no --enforce-eager); itertools-free capture list.
# kv_port 24680 pairs with P's 24679 (see cloud_p.sh for the ephemeral-range caveat).
lwd_run vllm serve /weight/Qwen3.6-27B/ \
  --host 0.0.0.0 --port 8561 \
  --served-model-name qwen3.6 \
  --tensor-parallel-size 2 \
  --data-parallel-size 1 \
  --data-parallel-size-local 1 \
  --distributed-executor-backend mp \
  --enable-prefix-caching \
  --async-scheduling \
  --max-model-len 65536 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100]}' \
  --additional-config '{"lwd_config":{"path":'"${LWD_TOPOLOGY_JSON}"',"role":"cloud","instance_id":1}}' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_rank":1,"engine_id":"1","kv_port":24680,"kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":2},"decode":{"dp_size":1,"tp_size":2}}}' \
  --override-generation-config '{"temperature": 0, "topk": 1}'
