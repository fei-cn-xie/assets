#!/usr/bin/env bash
# Decode cloud (76.76.26.195, NPU 2,3) — cloud instance 1, DP=2 x TP=1.
#
# Multi-DP variant of pds/30rc_1124/sidecar/cloud_d.sh: the same two cards now
# host TWO data-parallel replicas of the decode stage (one card each) instead of
# one TP=2 replica. The topology in this directory declares one dp: entry per
# rank, so each rank binds its own ctrl_port / data_port / coord_port.
#
# kv_port is a base: rank r binds kv_port + r * tp_size * pp_size
# (vllm_ascend/.../mooncake_connector.py:1762-1768, 2199-2200). Decode uses
# 14580/14581, clear of the prefill cloud's 14570/14571 on the same host.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"
lwd_runtime_env

export VLLM_HOST_IP=76.76.26.195
export MASTER_ADDR=76.76.26.195
export ASCEND_RT_VISIBLE_DEVICES=2,3

lwd_require_ucx_runtime

lwd_run "$LWD_PYTHON" -m vllm.entrypoints.cli.main serve /home/weight/Qwen3.6-27B/ \
  --host 0.0.0.0 \
  --port 8601 \
  --master-addr 76.76.26.195 --master-port 29526 \
  --pipeline-parallel-size 1 \
  --tensor-parallel-size 1 \
  --data-parallel-size 2 \
  --data-parallel-address 76.76.26.195 --data-parallel-rpc-port 29601 \
  --additional-config '{"enable_cpu_binding":true,"lwd_config":{"path":'"${LWD_TOPOLOGY_JSON}"',"role":"cloud","instance_id":1}}' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_rank":1,"kv_port":14580,"kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":1},"decode":{"dp_size":2,"tp_size":1}}}' \
  --trust-remote-code --served-model-name qwen3.6 \
  --max-model-len "${LWD_MAX_MODEL_LEN:-32768}" --max-num-seqs "${LWD_MAX_NUM_SEQS:-64}" --max-num-batched-tokens 8192 \
  --gpu-memory-utilization "${LWD_GPU_MEM_UTIL:-0.95}" \
  --async-scheduling \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
# NOTE: no --speculative-config here (unlike sidecar/cloud_d.sh). With DP>1 the
# MTP drafter rebuilds a VllmConfig at data_parallel_size=1 and vllm-ascend's
# check_kv_extra_config (vllm_ascend/utils.py:1580-1604) then rejects our
# kv_connector_extra_config.decode.dp_size=2 with
#   "KV transfer 'decode' config has a conflicting data parallel size.
#    Expected 1, but got 2."
# So DP>1 and MTP are mutually exclusive in this build. MTP stays available in
# ../sidecar (DP=1, TP=2).
