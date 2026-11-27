#!/usr/bin/env bash
# Edge instance 0 of 1 — 76.76.26.17, NPU 0.
#
# Start this LAST: the edge is the only side that blocks waiting for its links
# (605s per IO, up to 4 IOs on a 1-edge/2-cloud topology).
#
# The edge runs TP=1 and DP=1 by construction (vllm/config/lwd topology rules and
# the [LWD] check in v1/engine/core.py): it owns the embedding/sampling ends and
# never listens, so the topology gives it no ctrl_port/data_port/coord_port.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"
lwd_runtime_env
lwd_host_env edge
lwd_mtp_args
lwd_require_ucx_runtime

lwd_preflight "$LWD_EDGE_PORT"

lwd_run "$LWD_PYTHON" -m vllm.entrypoints.cli.main serve "$LWD_MODEL" \
  --host 0.0.0.0 \
  --port "$LWD_EDGE_PORT" \
  --master-addr "$LWD_EDGE_HOST" --master-port 29527 \
  --pipeline-parallel-size 1 \
  --tensor-parallel-size 1 \
  --additional-config '{"enable_cpu_binding":true,"lwd_config":{"path":'"${LWD_TOPOLOGY_JSON}"',"role":"edge","instance_id":0}}' \
  --enforce-eager \
  --trust-remote-code --served-model-name qwen3.6 \
  --max-model-len "$LWD_MAX_MODEL_LEN" --max-num-seqs "$LWD_MAX_NUM_SEQS" --max-num-batched-tokens "$LWD_MAX_NUM_BATCHED_TOKENS" \
  --gpu-memory-utilization "$LWD_GPU_MEM_UTIL" \
  --async-scheduling \
  ${LWD_MTP_ARGS[@]+"${LWD_MTP_ARGS[@]}"}
