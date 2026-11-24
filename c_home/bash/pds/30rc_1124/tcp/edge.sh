#!/usr/bin/env bash
# Edge A (edge instance 0). Launch AFTER the cloud B chain is up: host comm has
# no cross-world barrier, the cloud only reports "listener bound" and adopts the
# edge asynchronously, while the edge waits for its links (handshake timeout +5s).
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"
lwd_transport_env edge

# ---- instance-local placement -------------------------------------------------
export VLLM_HOST_IP=76.76.26.194
export ASCEND_RT_VISIBLE_DEVICES=4
export GLOO_SOCKET_IFNAME=enp189s0f0
export TP_SOCKET_IFNAME=enp189s0f0
export HCCL_SOCKET_IFNAME=enp189s0f0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# profiling configuration
#export SERVICE_PROF_CONFIG_PATH=/bash_dir/profiling/ms_service_profiler_config.json
#export PROFILING_SYMBOLS_PATH=/bash_dir/profiling/prof_dir/service_profiling_symbols.0.23.0.yaml
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-ERROR}"

# HARD CONSTRAINTS of data_plane_comm=host on v0.30.0:
#   --tensor-parallel-size 1  (only edge TP rank 0 holds the host links; TP>1
#                              fails at the first UP submit with a missing-IO error)
#   --data-parallel-size 1    (edge DP=1; its local DP 0 fans out to every cloud DP)
# The edge never listens: no data_port, no ctrl_port, no coord_port in the topology.
# The edge runs 0 transformer layers, so it skips warmup and graph capture by
# itself: --enforce-eager is neither required nor forbidden here. --async-scheduling
# is unverified on the edge (the LWD edge hijacks execute_model and the code has no
# async_scheduling handling): if the edge fails or stalls on first bring-up, drop
# --async-scheduling here first.
lwd_run vllm serve /weight/Qwen3.6-27B/ \
  --host 0.0.0.0 --port 9060 --served-model-name qwen3.6 \
  --tensor-parallel-size 1 --data-parallel-size 1 --data-parallel-size-local 1 \
  --distributed-executor-backend mp \
  --no-enable-prefix-caching --enforce-eager --async-scheduling \
  --max-model-len 65536 --max-num-seqs 128 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --additional-config '{"lwd_config":{"path":'"${LWD_TOPOLOGY_JSON}"',"role":"edge","instance_id":0}}' \
  --override-generation-config '{"temperature": 0, "topk": 1}'
