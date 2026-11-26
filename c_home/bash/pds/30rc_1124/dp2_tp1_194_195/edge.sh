#!/usr/bin/env bash
# Edge A (76.76.26.194, NPU 4) — edge instance 0 of 1.
# Flags, ports, master-port and model path are copied from the curl-verified
# reference v30_rc/edge.sh; only --additional-config's topology path differs
# (resolved from this directory instead of /vllm-workspace).
#
# Start this LAST: the edge is the only side that blocks waiting for its links
# (605s per IO, up to 4 IOs on a 1-edge/2-cloud topology).
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"
lwd_runtime_env

export VLLM_HOST_IP=76.76.26.194
export MASTER_ADDR=76.76.26.194
export ASCEND_RT_VISIBLE_DEVICES=4

lwd_require_ucx_runtime

# Host data plane constraints (v0.30.0): the edge must run TP=1 and DP=1; it
# never listens, so the topology gives it no ctrl_port/data_port/coord_port.
lwd_run "$LWD_PYTHON" -m vllm.entrypoints.cli.main serve /weight/Qwen3.6-27B/ \
  --host 0.0.0.0 \
  --port 8510 \
  --master-addr 76.76.26.194 --master-port 29527 \
  --pipeline-parallel-size 1 \
  --tensor-parallel-size 1 \
  --additional-config '{"enable_cpu_binding":true,"lwd_config":{"path":'"${LWD_TOPOLOGY_JSON}"',"role":"edge","instance_id":0}}' \
  --enforce-eager \
  --trust-remote-code --served-model-name qwen3.6 \
  --max-model-len "${LWD_MAX_MODEL_LEN:-32768}" --max-num-seqs "${LWD_MAX_NUM_SEQS:-32}" --max-num-batched-tokens "${LWD_MAX_NUM_BATCHED_TOKENS:-8192}" \
  --gpu-memory-utilization 0.90 \
  --async-scheduling
