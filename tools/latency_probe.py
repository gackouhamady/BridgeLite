
"""
BridgeLite — Latency probe (p50, p95, p99) for /predict

Usage (PowerShell or bash):
  python latency_probe.py --url http://127.0.0.1:8000/predict --n 300
"""
import time
import json
import argparse
import statistics
import http.client
from urllib.parse import urlparse

EXAMPLES = [
    {"tx_id":"t1","raw_label":"CB CARREFOUR 75 PARIS TPE1245","channel":"CB","mcc":5411},
    {"tx_id":"t2","raw_label":"PRLV EDF FACTURE 07/2025","channel":"PRLV","mcc":4900},
    {"tx_id":"t3","raw_label":"CB ZXQW PLORP STORE 9999 XJ-REF88","channel":"CB"},
]

def post_json(url, payload):
    u = urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=5)
    body = json.dumps(payload)
    headers = {"Content-Type": "application/json"}
    t0 = time.perf_counter()
    conn.request("POST", u.path, body=body, headers=headers)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    dt = (time.perf_counter() - t0) * 1000.0
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {data[:200]}")
    return dt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/predict")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    lats = []
    for i in range(args.n):
        payload = {"transactions": [EXAMPLES[i % len(EXAMPLES)]]}
        try:
            dt = post_json(args.url, payload)
            lats.append(dt)
        except Exception as e:
            print("Error:", e)

    if not lats:
        print("No successful requests.")
        return

    lats_sorted = sorted(lats)
    def p(x): 
        k = max(0, min(len(lats_sorted)-1, int(round(x*(len(lats_sorted)-1)))))
        return lats_sorted[k]
    print(f"count={len(lats_sorted)} p50={p(0.50):.1f} ms  p95={p(0.95):.1f} ms  p99={p(0.99):.1f} ms  max={max(lats_sorted):.1f} ms")
if __name__ == "__main__":
    main()
