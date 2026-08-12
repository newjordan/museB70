#!/usr/bin/env bash
# End-to-end smoke test for serve-muse.sh. The serving entry point owns the
# GPU lock for its full lifetime.
set -euo pipefail
umask 077

MUSE=/home/frosty40/turbo/muse
PORT=${SERVE_PORT:-8095}
LOG=${SERVE_LOG:-$MUSE/results/smoke-server.log}
OUT=${SMOKE_OUT:-$MUSE/results/smoke-$(date -u +%Y%m%dT%H%M%SZ).md}
CLIENT_API_KEY=${SMOKE_API_KEY:-${SERVE_API_KEY:-}}
EXPECTED_SYCL_SHA8=${SMOKE_EXPECTED_SYCL_SHA8:-aa90882d}
EXPECTED_SERVER_SHA8=${SMOKE_EXPECTED_SERVER_SHA8:-e5bfa43f}
EXPECTED_MANIFEST_SHA8=${SMOKE_EXPECTED_MANIFEST_SHA8:-53e20fe1}
EXPECTED_MODEL_SHA8=${SMOKE_EXPECTED_MODEL_SHA8:-7e9b74b7}
TMP=$(mktemp -d /tmp/muse-smoke.XXXXXX)
AUTH_HEADER=()
[[ -n "$CLIENT_API_KEY" ]] && AUTH_HEADER=(-H "Authorization: Bearer $CLIENT_API_KEY")

SERVER_PID=""
cleanup() {
  [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null && wait "$SERVER_PID" 2>/dev/null || true
  rm -rf "$TMP"
}
trap cleanup EXIT

rm -f "$LOG"
SERVE_LOG="$LOG" SERVE_PORT="$PORT" bash "$MUSE/serve-muse.sh" &
SERVER_PID=$!

for i in $(seq 1 120); do
  curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && break
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "FATAL: server died during load"; tail -20 "$LOG"; exit 1; }
  sleep 2
done
curl -sf "localhost:$PORT/health" >/dev/null || { echo "FATAL: no health after 240s"; tail -20 "$LOG"; exit 1; }
ui_status=$(curl -s -o /dev/null -w '%{http_code}' "localhost:$PORT/" "${AUTH_HEADER[@]}")
[[ "$ui_status" == 404 ]] || { echo "FATAL: API-only root returned HTTP $ui_status, expected 404"; exit 1; }
if [[ -n "$CLIENT_API_KEY" ]]; then
  unauth_status=$(curl -s -o /dev/null -w '%{http_code}' \
    "localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"muse-glimmer-30b-q4","messages":[{"role":"user","content":"auth probe"}],"max_tokens":1}')
  [[ "$unauth_status" == 401 ]] || { echo "FATAL: unauthenticated request returned HTTP $unauth_status, expected 401"; exit 1; }
fi
remote_cors=$(curl -s -D - -o /dev/null -X OPTIONS \
  "localhost:$PORT/v1/chat/completions" -H 'Origin: https://untrusted.example' \
  -H 'Access-Control-Request-Method: POST')
if grep -qi '^Access-Control-Allow-Origin:' <<<"$remote_cors"; then
  echo "FATAL: untrusted browser origin received Access-Control-Allow-Origin" >&2
  exit 1
fi
local_cors=$(curl -s -D - -o /dev/null -X OPTIONS \
  "localhost:$PORT/v1/chat/completions" -H 'Origin: http://localhost:3000' \
  -H 'Access-Control-Request-Method: POST')
grep -qi '^Access-Control-Allow-Origin: http://localhost:3000' <<<"$local_cors" || {
  echo "FATAL: localhost browser origin was not allowed" >&2
  exit 1
}
template_check=$(curl -s "localhost:$PORT/apply-template" \
  -H 'Content-Type: application/json' "${AUTH_HEADER[@]}" -d '{
    "messages":[
      {"role":"user","content":"first"},
      {"role":"assistant","reasoning_content":"MUSE-OLD-REASONING-741","content":"visible one"},
      {"role":"user","content":"second"},
      {"role":"assistant","reasoning_content":"MUSE-RECENT-REASONING-852","content":"visible two"},
      {"role":"user","content":"final"}
    ]
  }')
python3 -c 'import json,sys; p=json.load(sys.stdin)["prompt"]; a="MUSE-OLD-REASONING-741"; b="MUSE-RECENT-REASONING-852"; assert p.count(a)==1 and p.count(b)==1 and p.index(a)<p.index(b), p' <<<"$template_check"

curl -s "localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model":"muse-glimmer-30b-q4",
  "messages":[{"role":"user","content":"Reply with exactly: MUSE-OK"}],
  "max_tokens":128,"temperature":0}' "${AUTH_HEADER[@]}" >"$TMP/turn1.json"
python3 - "$TMP/turn1.json" "$TMP/turn2-request.json" <<'PY'
import json
import pathlib
import sys

turn1 = json.loads(pathlib.Path(sys.argv[1]).read_text())
message = turn1["choices"][0]["message"]
assert message.get("reasoning_content"), message
request = {
    "model": "muse-glimmer-30b-q4",
    "messages": [
        {"role": "user", "content": "Reply with exactly: MUSE-OK"},
        {
            "role": "assistant",
            "reasoning_content": message["reasoning_content"],
            "content": message["content"],
        },
        {"role": "user", "content": "Now reply with exactly: SECOND-OK"},
    ],
    "max_tokens": 128,
    "temperature": 0,
}
pathlib.Path(sys.argv[2]).write_text(json.dumps(request, separators=(",", ":")))
PY
curl -s "localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  --data-binary "@$TMP/turn2-request.json" "${AUTH_HEADER[@]}" >"$TMP/turn2.json"

{
echo "# Muse serve smoke — $(date -u +%FT%TZ)"
echo
echo "## browser origin policy"
echo "untrusted origin rejected; localhost origin allowed"
echo "Web UI disabled: authenticated root returned HTTP 404"
echo "old and recent assistant reasoning preserved once in rendered history"
if [[ -n "$CLIENT_API_KEY" ]]; then
  echo
  echo "## authentication"
  echo "unauthenticated chat HTTP 401; authenticated chat accepted"
fi
echo
echo "## turn 1 (short)"
python3 -c 'import json,sys; r=json.load(sys.stdin); m=r["choices"][0]["message"]; c=m["content"]; assert c.strip()=="MUSE-OK", repr(c); assert "to=" not in c and "analysis" not in c, repr(c); print("content:",repr(c)); rc=m.get("reasoning_content") or ""; assert rc; print("reasoning (first 120ch):",repr(rc[:120])); t=r.get("timings",{}); p=t.get("prompt_n"); assert isinstance(p,int) and 0 < p < 256, p; print("prompt_n=%s prompt_ms=%.0f predicted_per_sec=%.1f" % (p, t.get("prompt_ms",0), t.get("predicted_per_second",0)))' <"$TMP/turn1.json"
echo
echo "## turn 2 (actual reasoning+content echoed; cache reuse check)"
python3 -c 'import json,sys; r=json.load(sys.stdin); c=r["choices"][0]["message"]["content"]; assert c.strip()=="SECOND-OK", repr(c); assert "to=" not in c and "analysis" not in c, repr(c); print("content:",repr(c)); t=r.get("timings",{}); p=t.get("prompt_n"); assert isinstance(p,int) and 0 < p < 128, p; print("prompt_n=%s prompt_ms=%.0f predicted_per_sec=%.1f" % (p, t.get("prompt_ms",0), t.get("predicted_per_second",0)))' <"$TMP/turn2.json"
echo
echo "## template sanity (server log lines)"
grep -iE 'chat.template|jinja|common_chat' "$LOG" | head -5 || true
echo
echo "## slots / ctx at load"
grep -iE 'n_slots|n_ctx_slot' "$LOG" | head -3 || true
} | tee "$OUT"
grep -Fq 'n_slots = 1, n_ctx_slot = 131072' "$LOG"
grep -Fq 'using full-size SWA cache' "$LOG"
grep -Eq "manifest_sha=$EXPECTED_MANIFEST_SHA8 server_sha=$EXPECTED_SERVER_SHA8 so_sha=$EXPECTED_SYCL_SHA8 model_sha=$EXPECTED_MODEL_SHA8 .*ctx=131072 np=1 b=4096 ub=4096 swa_full=1 pb=24 mask_bounds=1" "$LOG"
echo; echo "smoke receipts -> $OUT"
