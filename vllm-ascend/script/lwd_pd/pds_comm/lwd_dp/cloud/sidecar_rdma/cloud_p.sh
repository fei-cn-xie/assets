#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

export VLLM_HOST_IP=76.76.26.189
export ASCEND_RT_VISIBLE_DEVICES=0,1
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
export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:${LD_LIBRARY_PATH:-}"
#export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

lwd_run vllm serve /weight/Qwen3.6-27B/ \
  --host 0.0.0.0 --port 8560 \
  --served-model-name qwen3.6 \
  --tensor-parallel-size 2 \
  --data-parallel-size 1 \
  --data-parallel-size-local 1 \
  --distributed-executor-backend mp \
  --enable-prefix-caching \
  --enforce-eager \
  --async-scheduling \
  --max-model-len 65536 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.95 \
  --trust-remote-code \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}' \
  --additional-config '{
    "lwd_config": {
      "enabled": true,
      "role": "cloud",
      "mode": "prefill_only",
      "edge_head_tail_layers": [0, 0],
      "cloud_id": 0,
      "transport": {
        "mode": "sidecar",
        "directory_path": '"${LWD_DIRECTORY_JSON}"',
        "instance_id": "cloud-p"
      }
    },
    "lwd_coordination": {
      "enabled": true,
      "listen_host": "0.0.0.0",
      "listen_port": 8660,
      "instance_id": "cloud-p"
    },
    "lwd_pd_config": {
      "enabled": true
    }
  }' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_rank":0,"engine_id":"0","kv_port":24679,"kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":2},"decode":{"dp_size":1,"tp_size":2}}}' \
  --override-generation-config '{"temperature": 0, "topk": 1}'
