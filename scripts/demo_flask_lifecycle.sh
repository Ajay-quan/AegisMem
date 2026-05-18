#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-http://127.0.0.1:8000}
API_KEY=${API_KEY:-}
if [[ -n "$API_KEY" ]]; then
  AUTH_HEADER=( -H "X-API-Key: $API_KEY" )
else
  AUTH_HEADER=()
fi

echo "Health"
curl -s "$BASE/health"; echo

echo "Ingest first memory"
MEM1=$(curl -s -X POST "$BASE/api/v1/memories" \
  -H "Content-Type: application/json" "${AUTH_HEADER[@]}" \
  -d '{"user_id":"alice","key":"python-pref","content":"Alice prefers Python and FAISS for memory retrieval.","importance_score":0.9}' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["memory"]["memory_id"])')
echo "$MEM1"

echo "Ingest related memory"
MEM2=$(curl -s -X POST "$BASE/api/v1/memories" \
  -H "Content-Type: application/json" "${AUTH_HEADER[@]}" \
  -d "{\"user_id\":\"alice\",\"key\":\"aws-pref\",\"content\":\"Alice deploys portfolio demos on AWS Free Tier.\",\"related_memory_ids\":[\"$MEM1\"]}" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["memory"]["memory_id"])')
echo "$MEM2"

echo "Semantic retrieval"
curl -s -X POST "$BASE/api/v1/retrieve" \
  -H "Content-Type: application/json" "${AUTH_HEADER[@]}" \
  -d '{"user_id":"alice","query":"FAISS retrieval","top_k":5}'; echo

echo "Hash-key lookup"
curl -s "${AUTH_HEADER[@]}" "$BASE/api/v1/memories/key/alice/python-pref"; echo

echo "Graph traversal"
curl -s "${AUTH_HEADER[@]}" "$BASE/api/v1/graph/$MEM1?depth=2"; echo

echo "Update"
curl -s -X PATCH "$BASE/api/v1/memories/$MEM1" \
  -H "Content-Type: application/json" "${AUTH_HEADER[@]}" \
  -d '{"content":"Alice prefers Flask APIs and FAISS vector retrieval."}'; echo

echo "Delete"
curl -s -X DELETE "${AUTH_HEADER[@]}" "$BASE/api/v1/memories/$MEM1"; echo
