# training/eval.py
"""
BridgeLite — Offline evaluation for the exported bundle

What this script does
---------------------
1) Loads a CSV (test or validation-like) and the model bundle exported by training.
2) Rebuilds text via `preproc.preprocess_record` and the same metadata tokens.
3) Computes probabilities, top-1 predictions, macro-F1, accuracy.
4) Computes **coverage@τ** (share with max proba >= τ) and **fallback_rate = 1 - coverage**.
5) Writes a small JSON summary and a confusion matrix PNG.

CLI
---
python training/eval.py \
  --test data/transactions_mock.csv \
  --model app/model_sklearn.pkl \
  --tau 0.6 \
  --out-dir reports/eval

Outputs
-------
- reports/eval/summary.json
- reports/eval/confusion_matrix.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # force non-GUI backend for PNG export
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from app.preproc import preprocess_record


def _build_text(row: pd.Series) -> str:
    pre = preprocess_record({"raw_label": row["raw_label"], "channel": row.get("channel", None)})
    op = pre.operation_type or "unknown"
    mcc = row.get("mcc", "")
    mcc_tok = f"mcc:{int(mcc)}" if str(mcc).strip().isdigit() else "mcc:none"
    return f"{pre.masked} op:{op} {mcc_tok}"


def evaluate(test_csv: str, model_path: str, tau: float, out_dir: str) -> dict:
    df = pd.read_csv(test_csv)
    if not {"raw_label", "category_label"}.issubset(df.columns):
        raise ValueError("CSV must contain 'raw_label' and 'category_label'.")

    # Load bundle
    bundle = joblib.load(model_path)
    vec = bundle["vectorizer"]
    clf = bundle["model"]
    le = bundle["label_encoder"]
    classes = list(le.classes_)

    # Texts & labels
    texts = df.apply(_build_text, axis=1).values
    y_true = le.transform(df["category_label"].astype(str).values)

    X = vec.transform(texts)
    y_pred = clf.predict(X)
    y_proba = getattr(clf, "predict_proba")(X)

    # Metrics
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    acc = float(accuracy_score(y_true, y_pred))
    pmax = np.max(y_proba, axis=1)
    coverage = float(np.mean(pmax >= tau))
    fallback_rate = float(1.0 - coverage)

    # Confusion matrix (absolute counts)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))

    # Plot confusion (matplotlib only, no seaborn)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    fig.tight_layout()
    cm_path = Path(out_dir) / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    summary = {
        "macro_f1": round(f1, 4),
        "accuracy_top1": round(acc, 4),
        "coverage_at_tau": round(coverage, 4),
        "fallback_rate": round(fallback_rate, 4),
        "tau": tau,
        "classes": classes,
        "confusion_matrix_png": str(cm_path),
        "model_metadata": bundle.get("metadata", {}),
    }
    with open(Path(out_dir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate a trained BridgeLite bundle.")
    ap.add_argument("--test", required=True, help="CSV with raw_label & category_label")
    ap.add_argument("--model", default="app/model_sklearn.pkl", help="Path to model bundle")
    ap.add_argument("--tau", type=float, default=0.6, help="Confidence threshold for coverage")
    ap.add_argument("--out-dir", default="reports/eval", help="Output directory for artifacts")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse()
    evaluate(args.test, args.model, args.tau, args.out_dir)
