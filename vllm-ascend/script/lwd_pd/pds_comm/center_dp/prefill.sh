# 1. P 节点（卡 0,1，HTTP 8060，kv_port 30000）
export nic_name="enp189s0f0"
export local_ip="76.76.26.189"
export MODEL_DIR="/weight/Qwen3.6-27B/"
export CODEBASE_DIR="/vllm-workspace"
export PYTHONPATH="${CODEBASE_DIR}/vllm-ascend:${CODEBASE_DIR}/vllm:${PYTHONPATH}"
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}"
export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export HCCL_BUFFSIZE=1024
export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# profiling configuration
export SERVICE_PROF_CONFIG_PATH=/bash_dir/profiling/ms_service_profiler_config.json     # 佊¨彀~A轅~M置彖~G件
export PROFILING_SYMBOLS_PATH=/bash_dir/profiling/prof_dir/service_profiling_symbols.0.23.0.yaml     # 佟~K潂¹彖~G件

vllm serve "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port 8060 \
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
  --kv-transfer-config \
  '{"kv_connector": "MooncakeConnectorV1",
  "kv_role": "kv_producer",
  "kv_port": "30000",
  "engine_id": "0",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 1,
                    "tp_size": 2
            },
            "decode": {
                    "dp_size": 1,
                    "tp_size": 2
            }
    }
  }' \
 --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
 --override-generation-config '{"temperature": 0, "topk": 1}' \
