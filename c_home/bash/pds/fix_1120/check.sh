#!/usr/bin/env bash
# 单节点自检：启动后确认「KV cache 2M 对齐注册」这一步是否通过。
#
# 用法（在对应节点上执行）：
#   bash check.sh prefill        # 188 节点，检查 9730
#   bash check.sh decode         # 187 节点，检查 9740
#   LOGDIR=/path/to/logs bash check.sh prefill   # 指定日志目录
#
# 退出码：0 = 该节点已就绪；1 = 启动失败（会打印判定原因）；2 = 还在加载中
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

ROLE=${1:-}
case "$ROLE" in
  prefill) PORT=9730; SUFFIX=-P ;;
  decode)  PORT=9740; SUFFIX=-D ;;
  *) echo "用法: bash check.sh <prefill|decode>" >&2; exit 1 ;;
esac

LATEST=""
if [ -z "${LOGDIR:-}" ]; then
  LATEST=$(ls -1t "$HERE/logs" 2>/dev/null | grep -- "$SUFFIX\$" | head -1)
  LOGDIR="$HERE/logs/$LATEST"
fi
LOG=$LOGDIR/$ROLE.log

if [ ! -f "$LOG" ]; then
  echo "[fatal] 找不到日志 $LOG" >&2
  echo "        先执行 bash $HERE/$ROLE.sh，或用 LOGDIR=<日志目录> 指定" >&2
  exit 1
fi

probe_health() {
  if command -v curl >/dev/null 2>&1; then
    curl -s -o /dev/null -w "%{http_code}" --max-time 5 "http://127.0.0.1:$PORT/health" || true
  else
    python3 -c "
import sys, urllib.request
try:
    with urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=5) as r:
        print(r.status)
except Exception:
    print('000')
"
  fi
}

echo "=== 1) 代码版本 ==="
cat "$LOG.runinfo" 2>/dev/null || echo "(无 runinfo，跳过)"

echo
echo "=== 2) 旧的失败签名（应无输出） ==="
if grep -n "not aligned to 2 MiB" "$LOG"; then
  echo "[FAIL] 仍然命中 2M 对齐断言：修复未生效（确认已切到 fix/pd-sep-hybrid-kv-register，且没被 site-packages 里的旧安装覆盖）" >&2
  exit 1
fi
echo "  OK: 无 'not aligned to 2 MiB'"

echo
echo "=== 3) KV cache 注册结果（本次修复新增日志） ==="
if grep -h "Mooncake registering" "$LOG" | sed 's/^/  /' | grep -q .; then
  grep -h "Mooncake registering" "$LOG" | sed 's/^/  /'
else
  echo "  (尚未打印，说明还没走到 register_kv_caches)"
fi

echo
echo "=== 4) KV cache 容量 ==="
grep -h -E "num_blocks:|NPU KV cache size|Available KV cache memory" "$LOG" | sed 's/^/  /'

echo
echo "=== 5) 进程存活 ==="
PROC_DEAD=0
if pgrep -f "vllm.entrypoints.cli.main serve" >/dev/null 2>&1; then
  echo "  OK: api server 进程在"
else
  echo "  [FAIL] api server 进程已退出"
  PROC_DEAD=1
fi

echo
echo "=== 6) 致命错误扫描 ==="
FATAL=$(grep -n -E "Engine core initialization failed|Worker failed with error|EngineDeadError|not divisible by kernel_size" "$LOG" | head -20)
if [ -n "$FATAL" ]; then
  echo "$FATAL"
  echo "[FAIL] 日志里有关键错误，见上面几行" >&2
  exit 1
fi
echo "  OK: 无 Engine core 初始化失败"
if [ "$PROC_DEAD" = "1" ]; then
  echo "[FAIL] api server 进程已退出，但日志未匹配到已知致命错误；请抓尾部日志" >&2
  tail -20 "$LOG" >&2
  exit 1
fi

echo
echo "=== 7) 就绪探测 http://127.0.0.1:$PORT/health ==="
HTTP_CODE=$(probe_health)
echo "  HTTP $HTTP_CODE"
if [ "$HTTP_CODE" = "200" ]; then
  echo
  echo "[PASS] $ROLE 节点已就绪（log=$LOG）"
  exit 0
fi

if grep -q "Application startup complete" "$LOG"; then
  echo "[WARN] 日志显示 startup complete 但 /health 非 200，稍后重试本脚本" >&2
  exit 2
fi

echo
echo "[WAIT] 还在加载（日志尾部）："
tail -5 "$LOG" | sed 's/^/  /'
exit 2
