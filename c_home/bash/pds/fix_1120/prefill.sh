#!/usr/bin/env bash
# 集中式 PD（2M 对齐修复验证版）: 188 Prefill, A2 卡0-7, DP1/TP8
#
# 与原脚本 private_home/vllm_script/pds_main_test/prefill.sh 的差异只有：
#   1) 代码版本预检：当前 vllm-ascend 必须包含 hybrid KV 注册修复，否则直接失败
#   2) LOGDIR 默认带时间戳，避免覆盖历史日志
#   3) 清理更彻底（api server + VllmWorker 子进程）
#   4) 去掉 vLLM 不识别的 VLLM_RPC_TIMEOUT（原来只有一条 Unknown env 警告）
# vLLM 启动参数与原脚本保持一致，便于对比验证。
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

MODEL=${MODEL:-/weight/DeepSeek-V4-Flash-w8a8-mtp}
CODEBASE_DIR=${CODEBASE_DIR:-/vllm-workspace}
HOST=76.76.26.188
NIC=${NIC:-enp189s0f0}
LOGDIR=${LOGDIR:-$HERE/logs/$(date +%m%d-%H%M)-P}
MAXLEN=${MAXLEN:-65536}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_HOST_IP=$HOST
export HCCL_IF_IP=$HOST
export GLOO_SOCKET_IFNAME=$NIC
export TP_SOCKET_IFNAME=$NIC
export HCCL_SOCKET_IFNAME=$NIC
export HCCL_INTRA_ROCE_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE="AIV"
export TORCH_DISTRIBUTED_DETAIL=DEBUG
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export ACL_OP_INIT_MODE=1
export PYTHONPATH=$CODEBASE_DIR/vllm-ascend:$CODEBASE_DIR/vllm:${PYTHONPATH:-}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:${LD_LIBRARY_PATH:-}

# ---- 代码版本预检 -----------------------------------------------------------
CONNECTOR=$CODEBASE_DIR/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py
if [ ! -f "$CONNECTOR" ]; then
  echo "[fatal] 找不到 $CONNECTOR" >&2
  exit 1
fi
if ! grep -q "has_mamba_group or self.use_hybrid" "$CONNECTOR"; then
  echo "[fatal] 当前 vllm-ascend 不含 hybrid KV 注册修复。" >&2
  echo "        请在两台机上 git checkout fix/pd-sep-hybrid-kv-register 后重试。" >&2
  exit 1
fi
if ! grep -Fq "remote_block_sizes=[group.kv_cache_spec.block_size for group in self.kv_cache_groups]" "$CONNECTOR"; then
  echo "[fatal] 当前 vllm-ascend 不含 DSV4 per-group block size 修复。" >&2
  echo "        请在两台机上 git checkout fix/pd-sep-hybrid-kv-register 后重试。" >&2
  exit 1
fi

mkdir -p "$LOGDIR"
LOG=$LOGDIR/prefill.log
ASCEND_BRANCH=$(git -C "$CODEBASE_DIR/vllm-ascend" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')
ASCEND_SHA=$(git -C "$CODEBASE_DIR/vllm-ascend" rev-parse --short HEAD 2>/dev/null || echo '?')
VLLM_BRANCH=$(git -C "$CODEBASE_DIR/vllm" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')
VLLM_SHA=$(git -C "$CODEBASE_DIR/vllm" rev-parse --short HEAD 2>/dev/null || echo '?')
{
  echo "[code] vllm-ascend: $ASCEND_BRANCH $ASCEND_SHA"
  echo "[code] vllm:        $VLLM_BRANCH $VLLM_SHA"
  echo "[code] PD fixes: hybrid registration + per-group block sizes"
} | tee "$LOG.runinfo"

# ---- 清理旧进程 -------------------------------------------------------------
if [ -z "${SKIP_CLEAN:-}" ]; then
  pkill -f "vllm.entrypoints.cli.main serve" 2>/dev/null || true
  pkill -f "VllmWorker" 2>/dev/null || true
  pkill -f lwd_host_comm 2>/dev/null || true
  for _ in $(seq 1 20); do
    pgrep -f "vllm.entrypoints.cli.main serve" >/dev/null 2>&1 || break
    sleep 1
  done
  sleep 3
fi

export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}"

nohup python3 -m vllm.entrypoints.cli.main serve "$MODEL" \
  --host 0.0.0.0 --port 9730 --served-model-name dsv4 --seed 1024 \
  --pipeline-parallel-size 1 --tensor-parallel-size 8 \
  --api-server-count 1 --distributed-executor-backend mp \
  --additional-config '{"enable_cpu_binding":true}' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_rank":0,"engine_id":"0","kv_port":14579,"kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":8},"decode":{"dp_size":1,"tp_size":8}}}' \
  --trust-remote-code \
  --max-model-len "$MAXLEN" --max-num-seqs 128 \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching --async-scheduling \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1,"enforce_eager":true}' \
  --enable-expert-parallel \
  --quantization ascend --block-size 128 --tokenizer-mode deepseek_v4 \
  --attention-config '{"indexer_kv_dtype":"int8"}' \
  --safetensors-load-strategy prefetch \
  --enforce-eager > "$LOG" 2>&1 &
PID=$!
echo "prefill host=$HOST cards=0-7 http=9730 kv_base=14579 pid=$PID"
echo "log=$LOG"
echo "runinfo=$LOG.runinfo"
echo "下一步：等 3~6 分钟后执行  bash $HERE/check.sh prefill"
