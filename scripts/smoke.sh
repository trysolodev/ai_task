#!/usr/bin/env bash
# Minimal smoke test from the spec (§7). Usage: scripts/smoke.sh [BASE_URL]
set -euo pipefail
BASE="${1:-http://localhost:8080}"
AUTH_HEADER=()
[ -n "${MEMORY_AUTH_TOKEN:-}" ] && AUTH_HEADER=(-H "Authorization: Bearer ${MEMORY_AUTH_TOKEN}")

echo "# waiting for health"
until curl -sf "${AUTH_HEADER[@]}" "$BASE/health" >/dev/null; do sleep 1; done
curl -s "${AUTH_HEADER[@]}" "$BASE/health" | jq . 2>/dev/null || curl -s "$BASE/health"; echo

echo "# POST /turns"
curl -s "${AUTH_HEADER[@]}" -X POST "$BASE/turns" \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "smoke-1",
    "user_id": "user-1",
    "messages": [
      {"role": "user", "content": "I just moved to Berlin from NYC last month. Loving it so far."},
      {"role": "assistant", "content": "That sounds exciting! Berlin is a great city. How are you settling in?"}
    ],
    "timestamp": "2025-03-15T10:30:00Z",
    "metadata": {}
  }'; echo

echo "# POST /recall (different session, same user — should mention Berlin / note NYC)"
curl -s "${AUTH_HEADER[@]}" -X POST "$BASE/recall" \
  -H 'Content-Type: application/json' \
  -d '{"query":"Where does this user live?","session_id":"smoke-2","user_id":"user-1","max_tokens":512}'; echo

echo "# GET /users/user-1/memories (structured, typed)"
curl -s "${AUTH_HEADER[@]}" "$BASE/users/user-1/memories"; echo

echo "# cleanup"
curl -s -o /dev/null -w "delete user-1: %{http_code}\n" "${AUTH_HEADER[@]}" -X DELETE "$BASE/users/user-1"
