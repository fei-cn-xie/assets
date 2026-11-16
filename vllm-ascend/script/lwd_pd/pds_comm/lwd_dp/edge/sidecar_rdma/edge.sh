#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"
lwd_ucx_env edge

export VLLM_HOST_IP=76.76.26.194
export ASCEND_RT_VISIBLE_DEVICES=4
export GLOO_SOCKET_IFNAME=enp189s0f0
export TP_SOCKET_IFNAME=enp189s0f0
export HCCL_SOCKET_IFNAME=enp189s0f0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn

LWD_EDGE_EPOCH="$("$LWD_PYTHON" -c 'import uuid; print(uuid.uuid4().hex)')"

lwd_run vllm serve /weight/Qwen3.6-27B/ \
  --host 0.0.0.0 --port 9060 --served-model-name qwen3.6 \
  --tensor-parallel-size 1 --data-parallel-size 1 --data-parallel-size-local 1 \
  --distributed-executor-backend mp \
  --no-enable-prefix-caching --enforce-eager --async-scheduling \
  --max-model-len 65536 --max-num-seqs 128 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95 \
  --trust-remote-code \
  --additional-config '{"lwd_config":{"enabled":true,"role":"edge","mode":"prefill_only","edge_head_tail_layers":[0,0],"edge_id":0,"transport":{"mode":"sidecar","directory_path":'"${LWD_DIRECTORY_JSON}"',"edge_epoch":"'"${LWD_EDGE_EPOCH}"'"}},"lwd_coordination":{"enabled":true,"control_url":"http://76.76.26.189:8500/v1/chat/completions","tenant_key_file":"/vllm-workspace/lwd_tenant.key","consumer_id":"edge-0"},"lwd_pd_config":{"enabled":true,"tenant_id":"edge-0"}}' \
  --override-generation-config '{"temperature": 0, "topk": 1}'
