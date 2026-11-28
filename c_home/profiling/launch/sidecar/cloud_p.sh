#!/usr/bin/env bash
# Cloud B / prefill (76.76.26.195, NPU 0,1) — cloud instance 0.
# Copied from the curl-verified reference v30_rc/cloud_p.sh.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"
lwd_runtime_env

export VLLM_HOST_IP=76.76.26.195
export MASTER_ADDR=76.76.26.195
export ASCEND_RT_VISIBLE_DEVICES=0,1

lwd_require_ucx_runtime

# coord_port 8100 is the PD registration endpoint the proxy posts to; it lives in
# the topology, not on this command line. ctrl_port 6100 / data_port 7200 too.
lwd_run "$LWD_PYTHON" -m vllm.entrypoints.cli.main serve /home/weight/Qwen3.6-27B/ \
  --host 0.0.0.0 \
  --port 8600 \
  --master-addr 76.76.26.195 --master-port 29525 \
  --pipeline-parallel-size 1 \
  --tensor-parallel-size 2 \
  --additional-config '{"enable_cpu_binding":true,"lwd_config":{"path":'"${LWD_TOPOLOGY_JSON}"',"role":"cloud","instance_id":0}}' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_rank":0,"kv_port":14579,"kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":2},"decode":{"dp_size":1,"tp_size":2}}}' \
  --enforce-eager \
  --trust-remote-code --served-model-name qwen3.6 \
  --max-model-len 65536 --max-num-seqs 128 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.95 \
  --async-scheduling \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"enforce_eager":true}'
