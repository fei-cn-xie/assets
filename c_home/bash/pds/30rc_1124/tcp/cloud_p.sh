#!/usr/bin/env bash
# Cloud B / prefill (cloud instance 0). Start order on B:
#   cloud_p.sh -> cloud_d.sh -> proxy.sh, then edge.sh on host A.
# v0.30.0 has no sidecar process: the data plane runs inside each engine.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"
lwd_transport_env cloud

# ---- instance-local placement -------------------------------------------------
export VLLM_HOST_IP=76.76.26.189
# Cloud TP is unrestricted in host comm; P uses NPU 0,1.
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
# Prepended AFTER lwd_transport_env so the UCX libucp path stays in the list.
export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:${LD_LIBRARY_PATH:-}"
#export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

# profiling configuration
#export SERVICE_PROF_CONFIG_PATH=/bash_dir/profiling/ms_service_profiler_config.json
#export PROFILING_SYMBOLS_PATH=/bash_dir/profiling/prof_dir/service_profiling_symbols.0.23.0.yaml
# ERROR hides the [LWD] handshake/digest lines; use INFO for first bring-up.
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-ERROR}"

# --port 8560 is the ordinary OpenAI API of this cloud engine.
# coord_port 8660 lives in the topology and is the PD registration endpoint the
# proxy posts to; ctrl_port 19180 / data_port 19600 are also topology-owned.
# kv_port 24679 keeps the reference numbering. v0.30.0's AscendDirectTransport
# picks ephemeral KV ports in [20000, 20000 + 1000*npu), so on an 8-card host
# 24679/24680 sits inside that range: if the Mooncake handshake fails to bind,
# move both sides to 28679/28680 and update kv_connector_extra_config with it.
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
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}' \
  --additional-config '{"lwd_config":{"path":'"${LWD_TOPOLOGY_JSON}"',"role":"cloud","instance_id":0}}' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_rank":0,"engine_id":"0","kv_port":24679,"kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":2},"decode":{"dp_size":1,"tp_size":2}}}' \
  --override-generation-config '{"temperature": 0, "topk": 1}'
