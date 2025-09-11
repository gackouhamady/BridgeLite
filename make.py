# This script creates a ready-to-use evaluation notebook and two helper scripts
# in the notebook environment. You can download them and place them into your repo.
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

# 1) Title / goals
cells.append(nbf.v4.new_markdown_cell("""
# BridgeLite — Offline Evaluation Notebook

This notebook evaluates the exported model bundle on **train** and **production_sim** datasets, produces:
- macro-F1, accuracy, coverage@τ
- confusion matrices (saved as PNGs)
- a τ sweep (F1 vs coverage) plot to help choose the operating threshold
- a qualitative sample of predictions

> Expected repo layout:
> - `app/model_sklearn.pkl` (bundle with `vectorizer`, `model`, `label_encoder`)
> - `data/transactions_mock.csv`, `data/production_sim.csv`
> - `app/preproc.py` that defines `preprocess_record`

**Tip:** If running inside Docker, ensure these files exist inside the container.
"""))

# 2) Imports
cells.append(nbf.v4.new_code_cell("""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

# Import your preprocessor from the app package
from app.preproc import preprocess_record

# Save plots into repo-relative paths as well
REPORT_DIR = Path("reports/eval")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("Environment OK. Reports will be saved to:", REPORT_DIR.resolve())
"""))

# 3) Config
cells.append(nbf.v4.new_code_cell("""
# ---- Configuration ----
MODEL_PATH = Path("app/model_sklearn.pkl")
TRAIN_CSV  = Path("data/transactions_mock.csv")
PROD_CSV   = Path("data/production_sim.csv")

TAU = 0.6            # operating threshold for coverage calculation
N_SAMPLES_QUAL = 12  # qualitative sample size

assert MODEL_PATH.exists(), f"Missing model bundle: {MODEL_PATH}"
assert TRAIN_CSV.exists(), f"Missing train CSV: {TRAIN_CSV}"
assert PROD_CSV.exists(),  f"Missing prod CSV:  {PROD_CSV}"

bundle = joblib.load(MODEL_PATH)
vec = bundle["vectorizer"]
clf = bundle["model"]
le  = bundle["label_encoder"]
labels = list(le.classes_)

print("Loaded bundle with classes:", labels[:10], "..." if len(labels)>10 else "")
"""))

# 4) Helpers
cells.append(nbf.v4.new_code_cell("""
def build_text(row: pd.Series) -> str:
    \"\"\"Build the exact text the model expects: masked + op/mcc tokens.\"\"\"
    pre = preprocess_record({"raw_label": row.get("raw_label", ""), "channel": row.get("channel", None)})
    op = pre.operation_type or "unknown"
    mcc = row.get("mcc", "")
    mcc_tok = f"mcc:{int(mcc)}" if str(mcc).strip().isdigit() else "mcc:none"
    return f"{pre.masked} op:{op} {mcc_tok}"


def predict_batch(texts: list[str]):
    X = vec.transform(texts)
    proba = clf.predict_proba(X)
    idx = np.argmax(proba, axis=1)
    pred = le.inverse_transform(idx)
    pmax = proba.max(axis=1)
    return pred, pmax, proba


def eval_split(df: pd.DataFrame, tau: float = 0.6, name: str = "eval"):
    texts = df.apply(build_text, axis=1).astype(str).tolist()
    y_true = None
    if "category_label" in df.columns:
        y_true = le.transform(df["category_label"].astype(str).values)
    pred, pmax, proba = predict_batch(texts)
    if y_true is not None:
        y_pred = le.transform(pred)
        f1 = float(f1_score(y_true, y_pred, average="macro"))
        acc = float(accuracy_score(y_true, y_pred))
    else:
        f1, acc = None, None
    coverage = float(np.mean(pmax >= tau))
    return {
        "pred": pred, "pmax": pmax, "proba": proba, "y_true": y_true,
        "f1": f1, "acc": acc, "coverage": coverage, "name": name
    }


def plot_confusion(y_true, y_pred, classes: list[str], title: str, out_png: Path):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    for (i,j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png
"""))

# 5) Load data & evaluate train
cells.append(nbf.v4.new_code_cell("""
df_train = pd.read_csv(TRAIN_CSV)
print("Train shape:", df_train.shape)
res_train = eval_split(df_train, tau=TAU, name="train")
print({k: res_train[k] for k in ["f1","acc","coverage"]})

if res_train["y_true"] is not None:
    y_pred_train = le.transform(res_train["pred"])
    png = REPORT_DIR / "confusion_train.png"
    plot_confusion(res_train["y_true"], y_pred_train, labels, "Confusion (train)", png)
    print("Saved:", png)
"""))

# 6) Load data & evaluate prod
cells.append(nbf.v4.new_code_cell("""
df_prod = pd.read_csv(PROD_CSV)
print("Prod shape:", df_prod.shape)
res_prod = eval_split(df_prod, tau=TAU, name="prod")
print({k: res_prod[k] for k in ["f1","acc","coverage"]})

if res_prod["y_true"] is not None:
    y_pred_prod = le.transform(res_prod["pred"])
    png = REPORT_DIR / "confusion_prod.png"
    plot_confusion(res_prod["y_true"], y_pred_prod, labels, "Confusion (prod)", png)
    print("Saved:", png)
"""))

# 7) Tau sweep
cells.append(nbf.v4.new_code_cell("""
taus = np.linspace(0.4, 0.9, 26)  # step 0.02
f1s, covs = [], []
# Use prod if it has labels, else train
target = res_prod if res_prod["y_true"] is not None else res_train

for t in taus:
    cov = float(np.mean(target["pmax"] >= t))
    if target["y_true"] is not None:
        idx = np.argmax(target["proba"], axis=1)
        y_pred = idx  # doesn't change with tau; tau only affects coverage
        f1 = float(f1_score(target["y_true"], y_pred, average="macro"))
    else:
        f1 = np.nan
    covs.append(cov); f1s.append(f1)

best_idx = int(np.nanargmax([f for f, c in zip(f1s, covs) if c >= 0.90] or [np.nan]))
best_tau = float(taus[best_idx]) if f1s else None

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111)
ax.plot(taus, covs, label="coverage")
ax.plot(taus, f1s, label="macro-F1")
ax.set_title("τ sweep — coverage vs macro-F1")
ax.set_xlabel("τ")
ax.set_ylabel("value")
ax.legend()
fig.tight_layout()
out_png = REPORT_DIR / "tau_sweep.png"
fig.savefig(out_png, dpi=150)
plt.close(fig)

summary_tau = {"taus": taus.tolist(), "coverage": covs, "macro_f1": f1s, "recommended_tau_if_cov90": best_tau}
print("Saved τ sweep:", out_png, "  recommended τ (coverage≥0.90):", best_tau)
"""))

# 8) Qualitative examples
cells.append(nbf.v4.new_code_cell("""
# Show some qualitative examples with predictions
sample_df = (df_prod if len(df_prod) else df_train).sample(min(N_SAMPLES_QUAL, len(df_prod) or len(df_train)), random_state=7)
texts = sample_df.apply(build_text, axis=1).astype(str).tolist()
pred, pmax, _ = predict_batch(texts)
out = sample_df.copy()
out["pred"] = pred
out["conf"] = pmax
if "category_label" in out.columns:
    out["true"] = out["category_label"]
display_cols = [c for c in ["raw_label","channel","mcc","true","pred","conf"] if c in out.columns]
out_sorted = out[display_cols].sort_values("conf", ascending=False)
out_sorted.head(N_SAMPLES_QUAL)
"""))

# 9) Save compact summary json
cells.append(nbf.v4.new_code_cell("""
summary = {
    "train": {k: res_train[k] for k in ["f1","acc","coverage"]},
    "prod":  {k: res_prod[k]  for k in ["f1","acc","coverage"]},
    "tau": float(TAU),
    "recommended_tau_if_cov90": summary_tau.get("recommended_tau_if_cov90", None),
    "classes": labels,
}
(REPORT_DIR / "offline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
"""))

nb["cells"] = cells

out_path = Path("data/01_eval_offline.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out_path)

# Also create two helper scripts the user can download:
latency_probe = Path("data/latency_probe.py")
latency_probe.write_text(r'''
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
''', encoding="utf-8")

tau_sweep = Path("data/tau_sweep.py")
tau_sweep.write_text(r'''
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
''', encoding="utf-8")

print("Created files:")
print("01_eval_offline.ipynb")
print("latency_probe.py")
print("tau_sweep.py")
