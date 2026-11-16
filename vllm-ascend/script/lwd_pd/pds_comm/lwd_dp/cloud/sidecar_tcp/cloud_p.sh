#!/usr/bin/env bash

export VLLM_HOST_IP=76.76.26.189
export ASCEND_RT_VISIBLE_DEVICES=0,1
export GLOO_SOCKET_IFNAME=enp189s0f0
export TP_SOCKET_IFNAME=enp189s0f0
export HCCL_SOCKET_IFNAME=enp189s0f0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CODEBASE_DIR=/vllm-workspace
export PYTHONPATH=$CODEBASE_DIR/vllm-ascend:$CODEBASE_DIR/vllm:$PYTHONPATH
export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}"
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export HCCL_BUFFSIZE=1024
export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:${LD_LIBRARY_PATH:-}"

# profiling configuration
export SERVICE_PROF_CONFIG_PATH=/bash_dir/profiling/ms_service_profiler_config.json     # 动态配置文件
export PROFILING_SYMBOLS_PATH=/bash_dir/profiling/prof_dir/service_profiling_symbols.0.23.0.yaml     # 埋点文件

vllm serve /weight/Qwen3.6-27B/ \
  --host 0.0.0.0 --port 8260 \
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
        "directory_path": "/bash_dir/pds_comm/sidecar_tcp/lwd_sidecar_directory.json",
        "instance_id": "cloud-p"
      }
    },
    "lwd_coordination": {
      "enabled": true,
      "listen_host": "0.0.0.0",
      "listen_port": 8160,
      "instance_id": "cloud-p"
    },
    "lwd_pd_config": {
      "enabled": true
    }
  }' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_rank":0,"engine_id":"0","kv_port":14679,"kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":2},"decode":{"dp_size":1,"tp_size":2}}}' \
  --override-generation-config '{"temperature": 0, "topk": 1}' \
