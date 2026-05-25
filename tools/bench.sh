#!/usr/bin/env bash
# Quick correctness + latency check.
# Reads test-data.json entries, fires N requests, computes FP/FN/Err + p50/p99.
set -euo pipefail
N="${1:-500}"
URL="${URL:-http://127.0.0.1:8088/fraud-score}"
JQ_FILTER='.entries[0:'$N'] | .[] | (.request|tostring) + "\t" + (.expected_approved|tostring)'

TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

jq -r "$JQ_FILTER" ~/rinha-de-backend-2026/test/test-data.json > "$TMP"

python3 - "$TMP" "$URL" <<'PY'
import sys, json, urllib.request, time, statistics
path, url = sys.argv[1], sys.argv[2]
lat = []
tp=tn=fp=fn=err=0
with open(path) as f:
    for line in f:
        body, expected = line.rstrip('\n').split('\t')
        expected = (expected == 'true')
        req = urllib.request.Request(url, data=body.encode(), headers={'Content-Type':'application/json'})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                resp = json.loads(r.read())
        except Exception:
            err += 1; continue
        t1 = time.perf_counter()
        lat.append((t1-t0)*1000)
        got = resp['approved']
        if got == expected:
            if got: tn += 1
            else:   tp += 1
        else:
            if got: fn += 1   # missed fraud
            else:   fp += 1   # blocked legit
total = tp+tn+fp+fn+err
fail_rate = (fp+fn+err)/total if total else 0
print(f"n={total}  tp={tp}  tn={tn}  fp={fp}  fn={fn}  err={err}")
print(f"failure_rate={fail_rate*100:.3f}%")
if lat:
    lat.sort()
    p50 = lat[len(lat)//2]
    p99 = lat[int(len(lat)*0.99)]
    print(f"latency  p50={p50:.2f}ms  p99={p99:.2f}ms  mean={statistics.mean(lat):.2f}ms  max={max(lat):.2f}ms")
PY
