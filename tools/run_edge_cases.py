import json, sys, http.client
from pathlib import Path
from urllib.parse import urlparse

URL = "http://127.0.0.1:8000/predict"
EDGE = Path("reports/demo/edge_cases.json")
OUT  = Path("reports/demo/edge_cases_results.json")

def post_json(url, payload):
    u = urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=5)
    body = json.dumps(payload)
    conn.request("POST", u.path, body=body, headers={"Content-Type":"application/json"})
    r = conn.getresponse()
    data = r.read().decode("utf-8")
    conn.close()
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {data[:200]}")
    return json.loads(data)

def main():
    assert EDGE.exists(), f"Missing {EDGE}"
    payload = json.loads(EDGE.read_text(encoding="utf-8"))
    res = post_json(URL, payload)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    # Print a tiny summary
    results = res.get("results", [])
    n = len(results)
    n_llm = sum(1 for r in results if "llm" in (r.get("router") or ""))
    cov = res.get("metrics", {}).get("coverage", None)
    print(f"edge-cases: n={n}  fallback={n_llm}  coverage={cov}")

if __name__ == "__main__":
    main()
