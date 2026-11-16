#!/usr/bin/env bash
set -e

export VLLM_HOST_IP=76.76.26.194
export ASCEND_RT_VISIBLE_DEVICES=4
export GLOO_SOCKET_IFNAME=enp189s0f0
export TP_SOCKET_IFNAME=enp189s0f0
export HCCL_SOCKET_IFNAME=enp189s0f0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CODEBASE_DIR=/vllm-workspace
export PYTHONPATH=$CODEBASE_DIR/vllm-ascend:$CODEBASE_DIR/vllm:$PYTHONPATH

# profiling configuration
export SERVICE_PROF_CONFIG_PATH=/bash_dir/profiling/ms_service_profiler_config.json     # 动态配置文件
export PROFILING_SYMBOLS_PATH=/bash_dir/profiling/prof_dir/service_profiling_symbols.0.23.0.yaml     # 埋点文件

LWD_EDGE_EPOCH="$(python -c 'import uuid; print(uuid.uuid4().hex)')"

vllm serve /weight/Qwen3.6-27B/ \
  --host 0.0.0.0 --port 9060 --served-model-name qwen3.6 \
  --tensor-parallel-size 1 --data-parallel-size 1 --data-parallel-size-local 1 \
  --distributed-executor-backend mp \
  --no-enable-prefix-caching --enforce-eager --async-scheduling \
  --max-model-len 65536 --max-num-seqs 128 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95 \
  --trust-remote-code \
  --additional-config '{"lwd_config":{"enabled":true,"role":"edge","mode":"prefill_only","edge_head_tail_layers":[0,0],"edge_id":0,"transport":{"mode":"sidecar","directory_path":"/bash_dir/pds_comm/sidecar_tcp/lwd_sidecar_directory.json","edge_epoch":"'"${LWD_EDGE_EPOCH}"'"}},"lwd_coordination":{"enabled":true,"control_url":"http://76.76.26.189:8600/v1/chat/completions","tenant_key_file":"/vllm-workspace/lwd_tenant.key","consumer_id":"edge-0"},"lwd_pd_config":{"enabled":true,"tenant_id":"edge-0"}}' \
  --override-generation-config '{"temperature": 0, "topk": 1}' \
