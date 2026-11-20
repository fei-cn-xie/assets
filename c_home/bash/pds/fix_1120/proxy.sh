#!/usr/bin/env bash
# 集中式原生 Mooncake PD proxy（2M 对齐修复验证版）: 187:8108 -> 188:9730(P), 187:9740(D)
#
# 与原脚本 private_home/vllm_script/pds_main_test/proxy.sh 的差异只有：
#   1) LOGDIR 默认带时间戳
#   2) 清理更彻底
# 代理脚本与端口参数保持一致。
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

PORT=${PORT:-8108}
CODEBASE_DIR=${CODEBASE_DIR:-/vllm-workspace}
PROXY_SCRIPT=$CODEBASE_DIR/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py
LOGDIR=${LOGDIR:-$HERE/logs/$(date +%m%d-%H%M)-proxy}
export PYTHONPATH=$CODEBASE_DIR/vllm-ascend:$CODEBASE_DIR/vllm:${PYTHONPATH:-}
mkdir -p "$LOGDIR"

[ -f "$PROXY_SCRIPT" ] || { echo "[fatal] 缺 proxy: $PROXY_SCRIPT" >&2; exit 1; }

pkill -f "load_balance_proxy_server_example.py.*--port $PORT" 2>/dev/null || true
pkill -f "vllm.v1.lwd_control.pd.proxy.*--port $PORT" 2>/dev/null || true
sleep 2

LOG=$LOGDIR/proxy.log
nohup python3 "$PROXY_SCRIPT" \
  --host 0.0.0.0 --port "$PORT" \
  --prefiller-hosts 76.76.26.188 --prefiller-ports 9730 \
  --decoder-hosts 76.76.26.187 --decoder-ports 9740 > "$LOG" 2>&1 &
PID=$!
echo "proxy pid=$PID port=$PORT log=$LOG"
