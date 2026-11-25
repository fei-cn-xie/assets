#!/usr/bin/env bash
# Centralized (non-PD) Qwen3.6-27B service — single engine, no edge/cloud split.
#
# Host: 76.76.26.195   NPU: 0,1,2,3 (TP=4)   port: 9060
#
# Derived from c_home/bash/pds/30rc_1124/sidecar/cloud_d.sh, which is the
# curl-verified graph-mode + MTP combination on this hardware. What is
# deliberately NOT here, and why:
#   * --additional-config lwd_config   no LWD control plane in a single engine
#   * --kv-transfer-config             no Mooncake hand-off without PD
#   * --master-addr/--master-port      single node; the default master port is fine
#
# NPU 0-3 are the same cards cloud_p.sh (0,1) and cloud_d.sh (2,3) use, so the
# PD pair must be stopped first. base_preflight() warns about that.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"
base_runtime_env

export VLLM_HOST_IP=76.76.26.195
export MASTER_ADDR=76.76.26.195
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

base_preflight 9060

base_run "$BASE_PYTHON" -m vllm.entrypoints.cli.main serve /home/weight/Qwen3.6-27B/ \
  --host 0.0.0.0 \
  --port 9060 \
  --served-model-name qwen3.6 \
  --pipeline-parallel-size 1 \
  --tensor-parallel-size 4 \
  --additional-config '{"enable_cpu_binding":true}' \
  --trust-remote-code \
  --max-model-len 65536 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --async-scheduling \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"enforce_eager":true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
