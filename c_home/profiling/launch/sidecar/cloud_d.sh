#!/usr/bin/env bash
# Cloud B / decode (76.76.26.195, NPU 2,3) — cloud instance 1.
# Copied from the curl-verified reference v30_rc/cloud_d.sh: no --enforce-eager,
# graph mode instead, and the Mooncake consumer side.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"
lwd_runtime_env

export VLLM_HOST_IP=76.76.26.195
export MASTER_ADDR=76.76.26.195
export ASCEND_RT_VISIBLE_DEVICES=2,3

lwd_require_ucx_runtime

# coord_port 8101 pairs with the proxy's decode target and the topology entry.
lwd_run "$LWD_PYTHON" -m vllm.entrypoints.cli.main serve /home/weight/Qwen3.6-27B/ \
  --host 0.0.0.0 \
  --port 8601 \
  --master-addr 76.76.26.195 --master-port 29526 \
  --pipeline-parallel-size 1 \
  --tensor-parallel-size 2 \
  --additional-config '{"enable_cpu_binding":true,"lwd_config":{"path":'"${LWD_TOPOLOGY_JSON}"',"role":"cloud","instance_id":1}}' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_rank":1,"kv_port":14580,"kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":2},"decode":{"dp_size":1,"tp_size":2}}}' \
  --trust-remote-code --served-model-name qwen3.6 \
  --max-model-len 65536 --max-num-seqs 128 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.95 \
  --async-scheduling \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"enforce_eager":true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100]}'
