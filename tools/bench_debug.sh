#!/usr/bin/env bash
# Like bench.sh but logs every FN/FP entry's id + payload + expected vs got.
set -euo pipefail
N="${1:-2000}"
URL="${URL:-http://127.0.0.1:8088/fraud-score}"
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT
jq -r --argjson n "$N" '.entries[0:$n] | .[] | (.request|tostring) + "\t" + (.expected_approved|tostring) + "\t" + (.request.id)' \
  ~/rinha-de-backend-2026/test/test-data.json > "$TMP"
python3 - "$TMP" "$URL" <<'PY'
import sys, json, urllib.request
path, url = sys.argv[1], sys.argv[2]
fp=fn=err=0
with open(path) as f:
    for line in f:
        parts = line.rstrip('\n').split('\t')
        body, expected, tx_id = parts[0], parts[1], parts[2]
        expected = (expected == 'true')
        req = urllib.request.Request(url, data=body.encode(), headers={'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                resp = json.loads(r.read())
        except Exception as e:
            err += 1; print("ERR", tx_id, e); continue
        got = resp['approved']
        if got != expected:
            if got:
                fn += 1
                print(f"FN id={tx_id} expected=DENY got=APPROVE score={resp['fraud_score']}")
            else:
                fp += 1
                print(f"FP id={tx_id} expected=APPROVE got=DENY score={resp['fraud_score']}")
print(f"--- fp={fp} fn={fn} err={err}")
PY
