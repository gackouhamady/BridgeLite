
"""
BridgeLite — τ sweep script (choose threshold to hit coverage target)

Usage:
  python tau_sweep.py --data data/production_sim.csv --bundle app/model_sklearn.pkl --target-cov 0.9
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, joblib
from app.preproc import preprocess_record

def build_text(row):
    pre = preprocess_record({"raw_label": row.get("raw_label",""), "channel": row.get("channel", None)})
    op = pre.operation_type or "unknown"
    mcc = row.get("mcc", "")
    mcc_tok = f"mcc:{int(mcc)}" if str(mcc).strip().isdigit() else "mcc:none"
    return f"{pre.masked} op:{op} {mcc_tok}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--bundle", default="app/model_sklearn.pkl")
    ap.add_argument("--target-cov", type=float, default=0.9)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    bundle = joblib.load(args.bundle)
    vec = bundle["vectorizer"]; clf = bundle["model"]; le = bundle["label_encoder"]

    texts = df.apply(build_text, axis=1).astype(str).tolist()
    X = vec.transform(texts)
    proba = clf.predict_proba(X)
    pmax = proba.max(axis=1)

    taus = np.linspace(0.4, 0.9, 26)
    rows = []
    for t in taus:
        cov = float((pmax >= t).mean())
        rows.append({"tau": float(t), "coverage": cov})
    # pick smallest tau achieving target coverage
    feasible = [r for r in rows if r["coverage"] >= args.target_cov]
    rec = min(feasible, key=lambda r: r["tau"]) if feasible else max(rows, key=lambda r: r["coverage"])

    print(json.dumps({"recommendation": rec, "grid": rows}, indent=2))

if __name__ == "__main__":
    main()
