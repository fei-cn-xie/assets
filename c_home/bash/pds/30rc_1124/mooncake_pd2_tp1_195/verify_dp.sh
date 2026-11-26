#!/usr/bin/env bash
# Read-only post-startup check that this deployment really runs 2 DP x 1 TP.
#
#   bash verify_dp.sh            # check everything
#   bash verify_dp.sh --quick    # skip the request through the proxy
#
# Run it after prefill.sh, decode.sh and start_proxy.sh have printed
# "Application startup complete". It writes nothing and starts or stops nothing,
# so it is safe to run at any time, including on a broken deployment.
#
# Why these checks: a DP=1 x TP=2 deployment answers requests just as well as
# this one, so "the request succeeded" proves nothing about the layout. The
# quantities below are the ones that differ between the two.
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

QUICK=0
for arg in "$@"; do
  case $arg in
    --quick) QUICK=1 ;;
    *) printf 'Usage: bash %s [--quick]\n' "$0" >&2; exit 2 ;;
  esac
done
# env.sh validates the caller's positional parameters (it accepts only
# --dry-run), so clear them before sourcing.
set --

# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

pass=0
fail=0
ok() { printf '  [ OK ] %s\n' "$*"; pass=$((pass + 1)); }
bad() { printf '  [FAIL] %s\n' "$*"; fail=$((fail + 1)); }
note() { printf '  ...    %s\n' "$*"; }

echo "== 1. engine core processes =="
# One EngineCore per DP rank: 2 per engine, 4 for the deployment.
cores=$(pgrep -af 'EngineCore_DP[0-9]+' 2>/dev/null | grep -c 'EngineCore_DP')
if [[ $cores -eq 4 ]]; then
  ok "4 EngineCore processes (prefill DP0/DP1 + decode DP0/DP1)"
else
  bad "expected 4 EngineCore processes, found ${cores}"
  note "one rank missing means the handshake failed for it: grep the engine log for"
  note "  'Did not receive response from front-end process' (engine identity collision)"
fi

echo "== 2. worker processes (one independent engine per DP rank) =="
workers=$(pgrep -af 'VLLM::Worker' 2>/dev/null | wc -l)
if [[ $workers -ge 4 ]]; then
  ok "${workers} VLLM::Worker processes (>= 1 per DP rank)"
else
  bad "expected at least 4 worker processes, found ${workers}"
fi

echo "== 3. KV handshake ports belong to different DP ranks =="
if ! command -v ss >/dev/null 2>&1; then
  note "ss not available, port checks skipped"
else
  # kv_port = base + data_parallel_index and only the producer binds, so prefill
  # must own two adjacent ports on two distinct PIDs.
  declare -A owner=()
  for port in "$PD_PREFILL_KV_PORT" $(((PD_PREFILL_KV_PORT) + 1)); do
    line=$(ss -ltnpH "sport = :${port}" 2>/dev/null | head -n 1)
    if [[ -n $line ]]; then
      pid=$(grep -o 'pid=[0-9]*' <<<"$line" | head -n 1)
      owner[$port]=$pid
      ok "prefill listens on ${port} (${pid:-pid unknown})"
    else
      bad "nothing listens on ${port}"
      note "both DP ranks sharing one kv_port, or the base is wrong"
    fi
  done
  if [[ ${#owner[@]} -eq 2 && ${owner[$PD_PREFILL_KV_PORT]} == "${owner[$(((PD_PREFILL_KV_PORT) + 1))]}" ]]; then
    bad "both KV ports belong to the same PID: that is one rank, not two"
  fi
  if [[ -n $(ss -ltnH "sport = :${PD_DECODE_KV_PORT}" 2>/dev/null) ]]; then
    bad "decode binds ${PD_DECODE_KV_PORT}: it is the consumer and must only connect"
  else
    ok "decode does not bind ${PD_DECODE_KV_PORT} (consumer connects only)"
  fi
fi

echo "== 4. API endpoints =="
for endpoint in "${PD_HOST}:${PD_PREFILL_PORT}" "${PD_HOST}:${PD_DECODE_PORT}" "${PD_HOST}:${PD_PROXY_PORT}"; do
  if curl -sf -m 5 "http://${endpoint}/health" >/dev/null 2>&1; then
    ok "http://${endpoint}/health"
  else
    bad "http://${endpoint}/health did not answer"
  fi
done

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
