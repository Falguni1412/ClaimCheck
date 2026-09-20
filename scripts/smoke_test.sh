#!/usr/bin/env bash
# Smoke-test every ClaimCheck endpoint against a running server.
#
#   ./scripts/smoke_test.sh                      # localhost:8000
#   ./scripts/smoke_test.sh https://your.app     # deployed instance
#
# Exits non-zero if any check fails, so it works as a post-deploy gate.
set -uo pipefail

BASE="${1:-http://localhost:8000}"
PASS=0; FAIL=0

check() { # name  expected_status  curl-args...
  local name="$1" expected="$2"; shift 2
  local code
  code=$(curl -s -o /tmp/cc_body -w '%{http_code}' --max-time 120 "$@")
  if [[ "$code" == "$expected" ]]; then
    printf '  \033[32mok\033[0m    %-42s %s\n' "$name" "$code"; PASS=$((PASS+1))
  else
    printf '  \033[31mFAIL\033[0m  %-42s got %s want %s\n' "$name" "$code" "$expected"
    head -c 300 /tmp/cc_body; echo; FAIL=$((FAIL+1))
  fi
}

json() { curl -s -H 'Content-Type: application/json' "$@"; }

PAYLOAD='{"answer":"Metformin should be taken with meals. Nausea is a common side effect.","sources":["Take metformin with meals to reduce stomach upset. Common side effects include nausea and diarrhea."],"top_k":2}'

echo "ClaimCheck smoke test -> $BASE"
echo
echo "System:"
check "GET  /"                      200 "$BASE/"
check "GET  /health"                200 "$BASE/health"
check "GET  /api/health"            200 "$BASE/api/health"
check "GET  /ready"                 200 "$BASE/ready"
check "GET  /metrics"               200 "$BASE/metrics"
check "GET  /api/docs"              200 "$BASE/api/docs"
check "GET  /openapi.json"          200 "$BASE/openapi.json"

echo
echo "Verification:"
check "POST /verify"                200 -X POST -H 'Content-Type: application/json' -d "$PAYLOAD" "$BASE/verify"
check "POST /verify (empty answer)" 422 -X POST -H 'Content-Type: application/json' -d '{"answer":"","sources":["x"]}' "$BASE/verify"
check "POST /verify (no sources)"   422 -X POST -H 'Content-Type: application/json' -d '{"answer":"hello there","sources":[]}' "$BASE/verify"
check "POST /verify/batch"          200 -X POST -H 'Content-Type: application/json' -d "{\"items\":[$PAYLOAD]}" "$BASE/verify/batch"
check "POST /batch/validate"        200 -X POST -H 'Content-Type: application/json' -d "{\"items\":[$PAYLOAD]}" "$BASE/batch/validate"
check "POST /verify/stream"         200 -X POST -H 'Content-Type: application/json' -d "$PAYLOAD" "$BASE/verify/stream"

echo
echo "History & admin:"
check "GET  /history/"              200 "$BASE/history/"
check "GET  /history/{unknown}"     404 "$BASE/history/no-such-id"
check "GET  /admin/stats"           200 "$BASE/admin/stats"
check "GET  /webhooks/"             200 "$BASE/webhooks/"
check "GET  /not-a-route"           404 "$BASE/not-a-route"

echo
echo "Error envelope shape:"
BODY=$(json -X POST -d '{"answer":"","sources":["x"]}' "$BASE/verify")
if grep -q '"error"' <<<"$BODY" && grep -q '"detail"' <<<"$BODY" && grep -q '"path"' <<<"$BODY"; then
  printf '  \033[32mok\033[0m    %s\n' "error/detail/path present"; PASS=$((PASS+1))
else
  printf '  \033[31mFAIL\033[0m  %s\n' "unexpected error body: $BODY"; FAIL=$((FAIL+1))
fi

echo
echo "Latency (cold vs warm /verify):"
for i in 1 2; do
  t=$(curl -s -o /dev/null -w '%{time_total}' -X POST -H 'Content-Type: application/json' \
      -d "$PAYLOAD" --max-time 120 "$BASE/verify")
  echo "  request $i: ${t}s"
done

echo
echo "passed: $PASS   failed: $FAIL"
[[ $FAIL -eq 0 ]]
