#!/usr/bin/env bash
# 本地离线自测：验证 check.sh 的日志解析/判定分支、prefill.sh 的代码版本预检。
# 不启动 vllm，不占 NPU。仅用到 bash / grep / pgrep。
set -uo pipefail
PKG=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK=$(mktemp -d)
DISC="$PKG/logs/9900-0000-P"     # 自动发现用例用的临时目录，跑完删除
FAKE_PID=""
PASS=0
FAIL=0

ok()  { echo "  [ok]   $*"; PASS=$((PASS+1)); }
bad() { echo "  [BAD]  $*"; FAIL=$((FAIL+1)); }

cleanup() {
  [ -n "$FAKE_PID" ] && kill "$FAKE_PID" 2>/dev/null
  rm -rf "$WORK" "$DISC"; rmdir "$PKG/logs" 2>/dev/null
}
trap cleanup EXIT

# 造一个 argv 里带 vllm api server 字样的假进程，让 check.sh 的存活判定通过
start_fake_proc() {
  bash -c 'exec -a "python3 -m vllm.entrypoints.cli.main serve" sleep 60' &
  FAKE_PID=$!
  sleep 0.5
}
stop_fake_proc() {
  [ -n "$FAKE_PID" ] && kill "$FAKE_PID" 2>/dev/null
  FAKE_PID=""
  sleep 0.3
}

# 期望「仍在加载」：Linux 上假进程可见 -> exit 2；
# msys/Windows 下 pgrep 看不到假进程 -> 走 exit 1 的「进程已退出」分支，同样算通过。
expect_loading() {
  local out="$1" rc="$2"
  if [ "$rc" = "2" ]; then
    ok "exit=2（就绪探测未通过，仍在加载，符合预期）"
  elif echo "$out" | grep -q "日志未匹配到已知致命错误"; then
    ok "exit=1 且命中「进程已退出」分支（本机 pgrep 看不到假进程；Linux 上会走 exit=2）"
  else
    bad "exit=$rc; $out"
  fi
}

echo "== 1) check.sh: 找不到日志 -> exit 1 =="
out=$(LOGDIR="$WORK/nope" bash "$PKG/check.sh" prefill 2>&1); rc=$?
[ "$rc" = "1" ] && ok "exit=1" || bad "exit=$rc"
echo "$out" | grep -q "找不到日志" && ok "提示文案正确" || bad "缺提示：$out"

D="$WORK/logs/1120-1530-P"; mkdir -p "$D"

echo "== 2) check.sh: 仍命中 2M 断言 -> exit 1 =="
cat > "$D/prefill.log" <<'LOG'
INFO 11-20 03:21:46 [mooncake_connector.py:2625] num_blocks: 5096
RuntimeError: Tensor start addr 20737952704512 is not aligned to 2 MiB.
LOG
out=$(LOGDIR="$D" bash "$PKG/check.sh" prefill 2>&1); rc=$?
[ "$rc" = "1" ] && ok "exit=1" || bad "exit=$rc"
echo "$out" | grep -q "仍然命中 2M 对齐断言" && ok "判定文案正确" || bad "缺判定：$out"

echo "== 3) check.sh: 修复生效但未 ready -> exit 2 =="
cat > "$D/prefill.log" <<'LOG'
INFO 11-20 15:31:02 [mooncake_connector.py:2625] num_blocks: 5096
INFO 11-20 15:31:03 [mooncake_connector.py:2658] Mooncake registering 1 KV cache buffer region(s), 16.13 GiB.
INFO 11-20 15:31:10 [kv_cache_utils.py:2395] NPU KV cache size: 215,326 tokens
LOG
printf '[code] vllm-ascend: fix/pd-sep-hybrid-kv-register 877555720c\n' > "$D/prefill.log.runinfo"
start_fake_proc
out=$(LOGDIR="$D" bash "$PKG/check.sh" prefill 2>&1); rc=$?
stop_fake_proc
expect_loading "$out" "$rc"
echo "$out" | grep -q "Mooncake registering 1 KV cache buffer region(s), 16.13 GiB." && ok "打印注册结果" || bad "缺注册结果：$out"
echo "$out" | grep -q "877555720c" && ok "打印代码版本" || bad "缺代码版本"
echo "$out" | grep -q "OK: 无 'not aligned to 2 MiB'" && ok "旧签名扫描通过" || bad "旧签名扫描异常"

echo "== 4) check.sh: 未指定 LOGDIR 时自动挑最新 -P 目录 =="
mkdir -p "$DISC"
cp "$D/prefill.log" "$DISC/prefill.log"
cp "$D/prefill.log.runinfo" "$DISC/prefill.log.runinfo"
start_fake_proc
out=$(bash "$PKG/check.sh" prefill 2>&1); rc=$?
stop_fake_proc
expect_loading "$out" "$rc"
echo "$out" | grep -q "Mooncake registering" && ok "命中自动发现的日志" || bad "未命中：$out"

echo "== 5) check.sh: Engine core 初始化失败 -> exit 1 =="
cat > "$D/prefill.log" <<'LOG'
RuntimeError: Engine core initialization failed. See root cause above.
LOG
out=$(LOGDIR="$D" bash "$PKG/check.sh" prefill 2>&1); rc=$?
[ "$rc" = "1" ] && ok "exit=1" || bad "exit=$rc"
echo "$out" | grep -q "日志里有关键错误" && ok "判定文案正确" || bad "缺判定：$out"

echo "== 6) check.sh: 进程已退出且无已知错误 -> exit 1 =="
cat > "$D/prefill.log" <<'LOG'
INFO 11-20 15:31:10 [kv_cache_utils.py:2395] NPU KV cache size: 215,326 tokens
LOG
out=$(LOGDIR="$D" bash "$PKG/check.sh" prefill 2>&1); rc=$?
[ "$rc" = "1" ] && ok "exit=1" || bad "exit=$rc"
echo "$out" | grep -q "日志未匹配到已知致命错误" && ok "判定文案正确" || bad "缺判定：$out"

echo "== 6b) check.sh: per-group block size 旧失败签名 -> exit 1 =="
cat > "$D/prefill.log" <<'LOG'
AssertionError: remote_block_size(8) not divisible by kernel_size(128)
LOG
out=$(LOGDIR="$D" bash "$PKG/check.sh" prefill 2>&1); rc=$?
[ "$rc" = "1" ] && ok "exit=1" || bad "exit=$rc"
echo "$out" | grep -q "not divisible by kernel_size" && ok "识别新的失败签名" || bad "未识别：$out"

echo "== 7) prefill.sh: 代码未含修复 -> 立刻报错退出，不启动 vllm =="
FAKE="$WORK/fake-code"
mkdir -p "$FAKE/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p" "$FAKE/vllm"
printf 'if has_mamba_group:\n    pass\n' > "$FAKE/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py"
out=$(CODEBASE_DIR="$FAKE" LOGDIR="$WORK/pre-out" bash "$PKG/prefill.sh" 2>&1); rc=$?
[ "$rc" = "1" ] && ok "exit=1" || bad "exit=$rc"
echo "$out" | grep -q "不含 hybrid KV 注册修复" && ok "预检文案正确" || bad "缺预检：$out"
echo "$out" | grep -q "vllm.entrypoints.cli.main serve" && bad "不应启动 vllm" || ok "未启动 vllm"

echo "== 8) prefill.sh: 只有注册修复、缺 per-group 修复 -> 立刻报错退出 =="
printf 'if has_mamba_group or self.use_hybrid:\n    pass\n' > "$FAKE/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py"
out=$(CODEBASE_DIR="$FAKE" LOGDIR="$WORK/pre-out" bash "$PKG/prefill.sh" 2>&1); rc=$?
[ "$rc" = "1" ] && ok "exit=1" || bad "exit=$rc"
echo "$out" | grep -q "不含 DSV4 per-group block size 修复" && ok "第二项预检文案正确" || bad "缺第二项预检：$out"
echo "$out" | grep -q "vllm.entrypoints.cli.main serve" && bad "不应启动 vllm" || ok "未启动 vllm"

echo
echo "结果: PASS=$PASS FAIL=$FAIL"
[ "$FAIL" = "0" ]
