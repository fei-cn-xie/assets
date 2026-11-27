#!/usr/bin/env bash
# Read-only post-startup check that this deployment really runs 2 DP x 4 TP per
# engine on the cards it claims.
#
#   bash verify_dp.sh            # everything
#   bash verify_dp.sh --quick    # skip the request through the proxy
#
# Run it after prefill.sh, decode.sh and start_proxy.sh printed
# "Application startup complete". It writes nothing and starts or stops nothing.
#
# Why these checks: a DP=1 x TP=8 deployment would also answer requests, and so
# would a 4 DP x 2 TP one. The counts below are what separates 2x4 from those:
#   * 2 EngineCore processes per engine  (one per DP rank)
#   * 4 workers per EngineCore           (TP=4 inside each rank)
#   * 8 KV handshake ports per producer engine, owned by exactly 2 PIDs
#     (kv_port + dp_rank * tp + device)
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

QUICK=0
for arg in "$@"; do
  case $arg in
    --quick) QUICK=1 ;;
    *) printf 'Usage: bash %s [--quick]\n' "$0" >&2; exit 2 ;;
  esac
done
# env.sh validates the caller's positional parameters, so clear them first.
set --
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

pass=0
fail=0
ok() { printf '  [ OK ] %s\n' "$*"; pass=$((pass + 1)); }
bad() { printf '  [FAIL] %s\n' "$*"; fail=$((fail + 1)); }
note() { printf '  ...    %s\n' "$*"; }

expected_cores=$((PD_DP_SIZE * 2))
expected_workers=$((PD_DP_SIZE * PD_TP_SIZE * 2))

echo "== 1. engine core processes =="
cores=$(pgrep -af 'EngineCore_DP[0-9]+' 2>/dev/null | grep -c 'EngineCore_DP')
if [[ $cores -eq $expected_cores ]]; then
  ok "${cores} EngineCore processes (2 per engine: prefill DP0/DP1 + decode DP0/DP1)"
else
  bad "expected ${expected_cores} EngineCore processes, found ${cores}"
  note "a missing rank means its handshake failed: grep the engine log for"
  note "  'Did not receive response from front-end process' (engine identity collision)"
fi

echo "== 2. worker processes (TP=4 inside each DP rank) =="
workers=$(pgrep -af 'VLLM::Worker' 2>/dev/null | wc -l)
if [[ $workers -ge $expected_workers ]]; then
  ok "${workers} VLLM::Worker processes (expected >= ${expected_workers} = 2 engines x 2 DP x 4 TP)"
else
  bad "expected at least ${expected_workers} worker processes, found ${workers}"
fi

echo "== 3. KV handshake ports (one per dp x tp, producer binds) =="
ss_port() {
  local out
  out=$(ss -ltnpH "sport = :$1" 2>/dev/null)
  if [[ -z $out ]]; then
    out=$(ss -ltnp "sport = :$1" 2>/dev/null | tail -n +2)
  fi
  printf '%s' "$out"
}

if ! command -v ss >/dev/null 2>&1; then
  note "ss not available, port checks skipped"
else
  declare -A owners=()
  listening=0
  for offset in $(seq 0 $((PD_DP_SIZE * PD_TP_SIZE - 1))); do
    port=$((PD_PREFILL_KV_PORT + offset))
    line=$(ss_port "$port" | head -n 1)
    if [[ -n $line ]]; then
      listening=$((listening + 1))
      owners[$port]=$(grep -o 'pid=[0-9]*' <<<"$line" | head -n 1)
    fi
  done
  if [[ $listening -eq $((PD_DP_SIZE * PD_TP_SIZE)) ]]; then
    ok "prefill listens on all $((PD_DP_SIZE * PD_TP_SIZE)) KV ports ${PD_PREFILL_KV_PORT}-$((PD_PREFILL_KV_PORT + PD_DP_SIZE * PD_TP_SIZE - 1))"
  else
    bad "only ${listening}/$((PD_DP_SIZE * PD_TP_SIZE)) prefill KV ports are listening"
    note "on a branch without the KV-port fix every DP rank reuses one port, so"
    note "only one would answer and the others would report 'Address already in use'"
  fi
  distinct_pids=$(printf '%s\n' "${owners[@]}" | sort -u | wc -l)
  if [[ ${#owners[@]} -gt 0 && $distinct_pids -eq $PD_DP_SIZE ]]; then
    ok "${distinct_pids} distinct PIDs own the KV ports (one per DP rank)"
  elif [[ ${#owners[@]} -gt 0 ]]; then
    bad "expected ${PD_DP_SIZE} distinct PIDs across the KV ports, got ${distinct_pids}"
  fi
  if [[ -n $(ss_port "$PD_DECODE_KV_PORT") ]]; then
    bad "decode binds ${PD_DECODE_KV_PORT}: it is the consumer and must only connect"
  else
    ok "decode does not bind ${PD_DECODE_KV_PORT} (consumer connects only)"
  fi
fi

echo "== 4. API endpoints =="
# vLLM engines answer /v1/models, while the proxy example only registers
# /healthcheck -- probing /health on the proxy would report a healthy proxy dead.
probe() {
  local host=$1 port=$2
  shift 2
  local path
  for path in "$@"; do
    if curl -sf -m 5 "http://${host}:${port}${path}" >/dev/null 2>&1; then
      printf '%s' "$path"
      return 0
    fi
  done
  return 1
}

for endpoint in "$PD_PREFILL_PORT" "$PD_DECODE_PORT"; do
  if hit=$(probe "$PD_HOST" "$endpoint" /v1/models); then
    ok "engine ${endpoint} answers ${hit}"
  else
    bad "engine ${endpoint} did not answer /v1/models"
  fi
done
if hit=$(probe "$PD_HOST" "$PD_PROXY_PORT" /healthcheck /v1/models); then
  ok "proxy ${PD_PROXY_PORT} answers ${hit}"
else
  bad "proxy ${PD_PROXY_PORT} did not answer /healthcheck or /v1/models"
fi

if [[ $QUICK -eq 0 ]]; then
  echo "== 5. one request through the proxy =="
  reply=$(curl -s -m 180 "http://${PD_HOST}:${PD_PROXY_PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.6","messages":[{"role":"user","content":"ping"}],"max_tokens":8}' 2>/dev/null)
  if grep -q '"choices"' <<<"$reply"; then
    ok "proxy answered: $(head -c 120 <<<"$reply" | tr -d '\n')"
  else
    bad "proxy did not answer with choices: $(head -c 200 <<<"$reply")"
  fi
fi

echo
printf '== %d ok, %d failed ==\n' "$pass" "$fail"
[[ $fail -eq 0 ]]
