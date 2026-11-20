#!/usr/bin/env bash
# 集中式 PD 请求冒烟：通过 proxy 发一条真实请求并校验 OpenAI 响应。
set -euo pipefail

PROXY_URL=${PROXY_URL:-http://127.0.0.1:8108}
MODEL=${MODEL:-dsv4}
MAX_TOKENS=${MAX_TOKENS:-64}
PROMPT=${PROMPT:-用一句话说明什么是 KV cache}
export MODEL MAX_TOKENS PROMPT
BODY_FILE=$(mktemp)
trap 'rm -f "$BODY_FILE"' EXIT

PAYLOAD=$(python3 -c '
import json, os
print(json.dumps({
    "model": os.environ["MODEL"],
    "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
    "max_tokens": int(os.environ["MAX_TOKENS"]),
    "temperature": 0,
}, ensure_ascii=False))
' )

HTTP_CODE=$(curl -sS -o "$BODY_FILE" -w "%{http_code}" --max-time 180 \
  -H 'Content-Type: application/json' \
  -d "$PAYLOAD" \
  "$PROXY_URL/v1/chat/completions") || {
  echo "[FAIL] 无法连接 $PROXY_URL，或请求在 180 秒内未完成" >&2
  cat "$BODY_FILE" >&2 || true
  exit 1
}

if [ "$HTTP_CODE" != "200" ]; then
  echo "[FAIL] HTTP $HTTP_CODE" >&2
  cat "$BODY_FILE" >&2
  exit 1
fi

python3 - "$BODY_FILE" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    response = json.load(f)
try:
    choice = response["choices"][0]
    content = choice["message"]["content"]
except (KeyError, IndexError, TypeError) as exc:
    print(f"[FAIL] 响应缺少 choices[0].message.content: {response}", file=sys.stderr)
    raise SystemExit(1) from exc
if not content or not content.strip():
    print(f"[FAIL] 返回内容为空: {response}", file=sys.stderr)
    raise SystemExit(1)
print(f"[PASS] PD 请求成功，finish_reason={choice.get('finish_reason')}")
print(content)
PY
