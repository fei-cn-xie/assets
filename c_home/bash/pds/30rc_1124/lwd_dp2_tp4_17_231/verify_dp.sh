#!/usr/bin/env bash
# Read-only post-startup check for the edge/cloud deployment.
#
#   bash verify_dp.sh              # auto-detect the host (17 = edge, 231 = cloud host)
#   bash verify_dp.sh --role cloud # force a role
#   bash verify_dp.sh --quick      # skip the request through the proxy
#
# Run it after the cloud engines (and, on 231, the proxy) printed
# "Application startup complete". It writes nothing and starts or stops nothing.
#
# What separates 2 DP x 4 TP from a look-alike:
#   * 2 EngineCore processes per cloud engine (one per DP rank), 4 for both clouds
#   * >= 16 workers on the cloud host (2 engines x 2 DP x 4 TP) and 1 on the edge
#   * all 8 producer KV ports bound, owned by exactly 2 PIDs (= 2 DP ranks)
#   * the four coord_port endpoints from pd_host_topology.yaml are listening
#
# Note: the LWD proxy only registers POST /v1/chat/completions — it has no
# /health or /v1/models — so a TCP connect is the only cheap liveness probe for it.
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

ROLE=""
QUICK=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --role) ROLE=${2:-}; shift 2 ;;
    --role=*) ROLE=${1#*=}; shift ;;
    --quick) QUICK=1; shift ;;
    *) printf 'Usage: bash %s [--role edge|cloud] [--quick]\n' "$0" >&2; exit 2 ;;
  esac
done
set --
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

pass=0
fail=0
ok() { printf '  [ OK ] %s\n' "$*"; pass=$((pass + 1)); }
bad() { printf '  [FAIL] %s\n' "$*"; fail=$((fail + 1)); }
note() { printf '  ...    %s\n' "$*"; }

if [[ -z $ROLE ]]; then
  local_ips=$(hostname -I 2>/dev/null || true)
  if [[ $local_ips == *"$LWD_CLOUD_HOST"* ]]; then
    ROLE=cloud
  elif [[ $local_ips == *"$LWD_EDGE_HOST"* ]]; then
    ROLE=edge
  else
    printf '[verify] cannot tell whether this host is %s (cloud) or %s (edge);\n' \
      "$LWD_CLOUD_HOST" "$LWD_EDGE_HOST" >&2
    printf '[verify] hostname -I reports: %s\n' "${local_ips:-<empty>}" >&2
    printf '[verify] pass --role cloud (or edge) explicitly.\n' >&2
    exit 2
  fi
fi
echo "== host role: ${ROLE} =="

tcp_open() { # host port -> 0 when something accepts a TCP connection
  timeout 3 bash -c "exec 3<>/dev/tcp/$1/$2" 2>/dev/null
}

if [[ $ROLE == cloud ]]; then
  expected_cores=$((LWD_DP_SIZE * 2))
  expected_workers=$((LWD_DP_SIZE * LWD_TP_SIZE * 2))

  echo "== 1. cloud engine core processes =="
  cores=$(pgrep -af 'EngineCore_DP[0-9]+' 2>/dev/null | grep -c 'EngineCore_DP')
  if [[ $cores -eq $expected_cores ]]; then
    ok "${cores} EngineCore processes (2 per cloud engine)"
  else
    bad "expected ${expected_cores} EngineCore processes, found ${cores}"
    note "a missing rank means its handshake failed: look for"
    note "  'Did not receive response from front-end process' in the engine log"
  fi

  echo "== 2. worker processes (TP=4 inside each DP rank) =="
  workers=$(pgrep -af 'VLLM::Worker' 2>/dev/null | wc -l)
  if [[ $workers -ge $expected_workers ]]; then
    ok "${workers} VLLM::Worker processes (expected >= ${expected_workers})"
  else
    bad "expected at least ${expected_workers} worker processes, found ${workers}"
  fi

  echo "== 3. producer KV handshake ports =="
  ss_port() {
    local out
    out=$(ss -ltnpH "sport = :$1" 2>/dev/null)
    [[ -z $out ]] && out=$(ss -ltnp "sport = :$1" 2>/dev/null | tail -n +2)
    printf '%s' "$out"
  }
  if ! command -v ss >/dev/null 2>&1; then
    note "ss not available, port checks skipped"
  else
    total=$((LWD_DP_SIZE * LWD_TP_SIZE))
    declare -A owners=()
    listening=0
    for offset in $(seq 0 $((total - 1))); do
      port=$((LWD_PREFILL_KV_PORT + offset))
      line=$(ss_port "$port" | head -n 1)
      if [[ -n $line ]]; then
        listening=$((listening + 1))
        owners[$port]=$(grep -o 'pid=[0-9]*' <<<"$line" | head -n 1)
      fi
    done
    if [[ $listening -eq $total ]]; then
      ok "prefill binds all ${total} KV ports ${LWD_PREFILL_KV_PORT}-$((LWD_PREFILL_KV_PORT + total - 1))"
    else
      bad "only ${listening}/${total} prefill KV ports are listening"
      note "without the multi-DP KV-port fix every DP rank reuses one port"
    fi
    distinct=$(printf '%s\n' "${owners[@]}" | sort -u | wc -l)
    if [[ ${#owners[@]} -gt 0 && $distinct -eq $LWD_DP_SIZE ]]; then
      ok "${distinct} distinct PIDs own the KV ports (one per DP rank)"
    elif [[ ${#owners[@]} -gt 0 ]]; then
      bad "expected ${LWD_DP_SIZE} distinct PIDs across the KV ports, got ${distinct}"
    fi
    if [[ -n $(ss_port "$LWD_DECODE_KV_PORT") ]]; then
      bad "decode binds ${LWD_DECODE_KV_PORT}: it is the consumer and must only connect"
    else
      ok "decode does not bind ${LWD_DECODE_KV_PORT} (consumer connects only)"
    fi
  fi

  echo "== 4. cloud endpoints =="
  for endpoint in "$LWD_PREFILL_PORT" "$LWD_DECODE_PORT"; do
    if curl -sf -m 5 "http://${LWD_CLOUD_HOST}:${endpoint}/v1/models" >/dev/null 2>&1; then
      ok "cloud engine ${endpoint} answers /v1/models"
    else
      bad "cloud engine ${endpoint} did not answer /v1/models"
    fi
  done

  echo "== 5. topology coord ports (one per cloud DP) and the proxy =="
  coords=$(grep -o 'coord_port: [0-9]*' "$LWD_TOPOLOGY" | grep -o '[0-9]*')
  for port in $coords; do
    if tcp_open "$LWD_CLOUD_HOST" "$port"; then
      ok "coord_port ${port} accepts connections"
    else
      bad "coord_port ${port} is not listening (cloud DP not up?)"
    fi
  done
  if tcp_open "$LWD_CLOUD_HOST" "$LWD_PROXY_PORT"; then
    ok "LWD proxy ${LWD_PROXY_PORT} accepts connections (it only serves POST /v1/chat/completions)"
  else
    bad "LWD proxy ${LWD_PROXY_PORT} is not listening"
  fi

  if [[ $QUICK -eq 0 ]]; then
    echo "== 6. one request through the proxy =="
    reply=$(curl -s -m 240 "http://${LWD_CLOUD_HOST}:${LWD_PROXY_PORT}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d '{"model":"qwen3.6","messages":[{"role":"user","content":"ping"}],"max_tokens":8}' 2>/dev/null)
    if grep -q '"choices"' <<<"$reply"; then
      ok "proxy answered: $(head -c 120 <<<"$reply" | tr -d '\n')"
    else
      bad "proxy did not answer with choices: $(head -c 200 <<<"$reply")"
    fi
  fi
else
  echo "== edge checks (host ${LWD_EDGE_HOST}) =="
  workers=$(pgrep -af 'VLLM::Worker' 2>/dev/null | wc -l)
  if [[ $workers -ge 1 ]]; then
    ok "${workers} VLLM::Worker process(es) on the edge (TP=1)"
  else
    bad "no VLLM::Worker process found: the edge engine is not running"
  fi
  if curl -sf -m 5 "http://${LWD_EDGE_HOST}:${LWD_EDGE_PORT}/v1/models" >/dev/null 2>&1; then
    ok "edge ${LWD_EDGE_PORT} answers /v1/models"
  else
    bad "edge ${LWD_EDGE_PORT} did not answer /v1/models"
  fi
  if pgrep -af 'lwd_control|EngineCore' 2>/dev/null | grep -q .; then
    ok "edge control/engine processes present"
  else
    bad "no edge engine process found"
  fi
  if [[ $QUICK -eq 0 ]]; then
    echo "== one request straight to the edge =="
    reply=$(curl -s -m 240 "http://${LWD_EDGE_HOST}:${LWD_EDGE_PORT}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d '{"model":"qwen3.6","messages":[{"role":"user","content":"ping"}],"max_tokens":8}' 2>/dev/null)
    if grep -q '"choices"' <<<"$reply"; then
      ok "edge answered: $(head -c 120 <<<"$reply" | tr -d '\n')"
    else
      bad "edge did not answer with choices: $(head -c 200 <<<"$reply")"
    fi
  fi
fi

echo
printf '== %d ok, %d failed ==\n' "$pass" "$fail"
[[ $fail -eq 0 ]]
