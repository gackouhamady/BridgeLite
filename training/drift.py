# training/drift.py
"""
BridgeLite — Drift detection: PSI/KS on char n-grams + OOV, simple HTML/PNG report.

What this script does
---------------------
1) Loads a TRAIN csv and a PROD csv (drifted).
2) Builds the SAME normalized/masked text as the model uses:
     preproc.preprocess_record -> masked text + op:<...> + mcc:<...>
3) Uses the model bundle's TF-IDF analyzer/vocab if available (best), else falls back
   to a fresh TfidfVectorizer(analyzer="char", ngram_range=(1,5)).
4) Computes:
   - Top-K char n-gram frequency distributions (train vs prod) → PSI
   - OOV rate (share of grams in prod not present in train vocab)
   - Optional: category mix shift (uses labels if present; otherwise model predictions)
   - Optional: KS test on confidence distributions (if bundle/model provided)
5) Saves:
   - reports/drift/top_ngrams_delta.png     (largest frequency deltas)
   - reports/drift/confidence_hist.png      (if model available)
   - reports/drift/label_mix.png            (if labels or predictions available)
   - reports/drift/summary.json
   - reports/drift/index.html               (tiny HTML report)
6) Prints a JSON summary with `psi_topk` and `alert` flags.

CLI
---
python training/drift.py \
  --train data/transactions_mock.csv \
  --prod  data/production_sim.csv \
  --bundle app/model_sklearn.pkl \
  --topk 500 \
  --out-dir reports/drift

Alert policy (defaults)
-----------------------
- alert = True if psi_topk > 0.20 or any |Δ category proportion| > 0.15
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from scipy.stats import ks_2samp

# Headless plotting
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.preproc import preprocess_record


# ------------------------------ utils -----------------------------------------

def psi(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Population Stability Index between two discrete distributions (aligned)."""
    p = p.astype(float); q = q.astype(float)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    return float(np.sum((q - p) * np.log((q + eps) / (p + eps))))


def build_model_text(row: pd.Series) -> str:
    """Masked text + light metadata tokens (must match training/serving)."""
    pre = preprocess_record({"raw_label": row["raw_label"], "channel": row.get("channel", None)})
    op = pre.operation_type or "unknown"
    mcc = row.get("mcc", "")
    mcc_tok = f"mcc:{int(mcc)}" if str(mcc).strip().isdigit() else "mcc:none"
    return f"{pre.masked} op:{op} {mcc_tok}"


def count_char_ngrams(texts: List[str], analyzer, limit_vocab: Optional[Dict[str, int]] = None) -> Counter:
    """
    Counts char n-grams over a list of texts using a provided analyzer.
    If limit_vocab is given, only grams in that vocab are counted (aligns spaces).
    """
    c = Counter()
    for t in texts:
        grams = analyzer(t)
        if limit_vocab is None:
            c.update(grams)
        else:
            c.update(g for g in grams if g in limit_vocab)
    return c


def oov_rate_char_ngrams(texts: List[str], analyzer, train_vocab: Dict[str, int]) -> float:
    """Share of grams in PROD not present in TRAIN vocab (0..1)."""
    total = 0
    oov = 0
    for t in texts:
        grams = analyzer(t)
        total += len(grams)
        oov += sum(1 for g in grams if g not in train_vocab)
    if total == 0:
        return 0.0
    return float(oov / total)


def proportions_from_labels(series: pd.Series) -> Dict[str, float]:
    """Returns a normalized frequency dict for categorical labels."""
    vc = series.astype(str).value_counts(dropna=False)
    s = vc / max(1, vc.sum())
    return {k: float(v) for k, v in s.items()}


def proportions_from_predictions(texts: List[str], bundle) -> Dict[str, float]:
    """Uses the model bundle to get predicted class mix for the corpus."""
    vec = bundle["vectorizer"]
    clf = bundle["model"]
    le = bundle["label_encoder"]
    X = vec.transform(texts)
    y_pred = clf.predict(X)
    labels = le.inverse_transform(y_pred)
    return proportions_from_labels(pd.Series(labels))


# ------------------------------ main ------------------------------------------

def run(
    train_csv: str,
    prod_csv: str,
    out_dir: str = "reports/drift",
    bundle_path: Optional[str] = "app/model_sklearn.pkl",
    topk: int = 500,
    psi_alert: float = 0.20,
    shift_alert: float = 0.15,
) -> Dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    # Load data
    df_train = pd.read_csv(train_csv)
    df_prod = pd.read_csv(prod_csv)
    for need in ["raw_label"]:
        if need not in df_train.columns or need not in df_prod.columns:
            raise ValueError(f"Both CSVs must contain column '{need}'")

    # Try to load the bundle to reuse analyzer/vocab and (optionally) confidence/preds
    bundle = None
    vec = None
    analyzer = None
    train_vocab = None
    have_confidences = False

    if bundle_path and Path(bundle_path).exists():
        bundle = joblib.load(bundle_path)
        vec = bundle["vectorizer"]
        analyzer = vec.build_analyzer()
        train_vocab = vec.vocabulary_
    else:
        # Fallback: fit a temp vectorizer on train to define the vocabulary
        vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 5), lowercase=False, dtype=np.float32, max_features=200_000)
        analyzer = vec.build_analyzer()
        vec.fit(df_train["raw_label"].astype(str).tolist())
        train_vocab = vec.vocabulary_

    # Build normalized/masked texts + metadata (exactly like serving)
    train_texts = df_train.apply(build_model_text, axis=1).astype(str).tolist()
    prod_texts  = df_prod.apply(build_model_text, axis=1).astype(str).tolist()

    # Count n-grams aligned to train vocab
    train_counts = count_char_ngrams(train_texts, analyzer, limit_vocab=train_vocab)
    prod_counts  = count_char_ngrams(prod_texts,  analyzer, limit_vocab=train_vocab)

    # Pick Top-K grams by TRAIN frequency
    top_items = sorted(train_counts.items(), key=lambda x: x[1], reverse=True)[:max(1, topk)]
    vocab_top = [g for g, _ in top_items]
    p = np.array([train_counts.get(g, 0) for g in vocab_top], dtype=float)
    q = np.array([prod_counts.get(g, 0) for g in vocab_top], dtype=float)

    psi_topk = psi(p, q)

    # OOV on PROD vs TRAIN vocab
    oov = oov_rate_char_ngrams(prod_texts, analyzer, train_vocab)

    # Confidence KS (if bundle/model available)
    ks_stat = None
    ks_p = None
    if bundle is not None:
        try:
            Xtr = vec.transform(train_texts)
            Xpr = vec.transform(prod_texts)
            proba_tr = bundle["model"].predict_proba(Xtr)
            proba_pr = bundle["model"].predict_proba(Xpr)
            pmax_tr = np.max(proba_tr, axis=1)
            pmax_pr = np.max(proba_pr, axis=1)
            ks_stat, ks_p = ks_2samp(pmax_tr, pmax_pr)
        except Exception:
            ks_stat, ks_p = None, None

    # Label mix (use ground-truth if present; else predictions if bundle)
    mix_train = None
    mix_prod = None
    if "category_label" in df_train.columns and "category_label" in df_prod.columns:
        mix_train = proportions_from_labels(df_train["category_label"])
        mix_prod  = proportions_from_labels(df_prod["category_label"])
    elif bundle is not None:
        mix_train = proportions_from_predictions(train_texts, bundle)
        mix_prod  = proportions_from_predictions(prod_texts,  bundle)

    # Largest per-class shifts (if we computed mixes)
    max_abs_shift = 0.0
    largest_shift = None
    if mix_train and mix_prod:
        classes = sorted(set(mix_train) | set(mix_prod))
        for c in classes:
            a = mix_train.get(c, 0.0)
            b = mix_prod.get(c, 0.0)
            d = abs(b - a)
            if d > max_abs_shift:
                max_abs_shift = d
                largest_shift = (c, a, b, d)

    # ------------------------ plots ------------------------

    # 1) Top-ngrams delta
    deltas = [(g, (prod_counts.get(g, 0) - train_counts.get(g, 0))) for g in vocab_top]
    top_delta = sorted(deltas, key=lambda x: abs(x[1]), reverse=True)[:25]
    if top_delta:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        grams = [g for g, _ in top_delta]
        vals  = [v for _, v in top_delta]
        ax.bar(range(len(vals)), vals)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(grams, rotation=90)
        ax.set_title("Top char n-grams by absolute frequency delta (prod - train)")
        ax.set_ylabel("Δ count")
        fig.tight_layout()
        (Path(out_dir) / "top_ngrams_delta.png").parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(out_dir) / "top_ngrams_delta.png", dpi=150)
        plt.close(fig)

    # 2) Confidence histograms (if available)
    if ks_stat is not None:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        ax.hist(pmax_tr, bins=30, alpha=0.6, label="train")
        ax.hist(pmax_pr, bins=30, alpha=0.6, label="prod")
        ax.set_title(f"Confidence hist (KS={ks_stat:.3f}, p={ks_p:.2e})")
        ax.set_xlabel("max predicted probability")
        ax.set_ylabel("count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(Path(out_dir) / "confidence_hist.png", dpi=150)
        plt.close(fig)

    # 3) Label mix stacked bars (if available)
    if mix_train and mix_prod:
        all_classes = sorted(set(mix_train) | set(mix_prod))
        tr = [mix_train.get(c, 0.0) for c in all_classes]
        pr = [mix_prod.get(c, 0.0) for c in all_classes]
        x = np.arange(len(all_classes))
        width = 0.38
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.bar(x - width/2, tr, width, label="train")
        ax.bar(x + width/2, pr, width, label="prod")
        ax.set_xticks(x)
        ax.set_xticklabels(all_classes, rotation=45, ha="right")
        ax.set_ylabel("proportion")
        ax.set_title("Category mix (train vs prod)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(Path(out_dir) / "label_mix.png", dpi=150)
        plt.close(fig)

    # ------------------------ summary & HTML ------------------------

    summary = {
        "psi_topk": round(float(psi_topk), 4),
        "oov_rate": round(float(oov), 4),
        "ks_confidence_stat": None if ks_stat is None else round(float(ks_stat), 4),
        "ks_confidence_pvalue": None if ks_p   is None else float(ks_p),
        "largest_label_shift": {
            "class": largest_shift[0],
            "train": round(largest_shift[1], 4),
            "prod":  round(largest_shift[2], 4),
            "abs_delta": round(largest_shift[3], 4),
        } if largest_shift else None,
        "alerts": {
            "psi_topk_gt_threshold": bool(psi_topk > psi_alert),
            "label_shift_gt_threshold": bool(max_abs_shift > shift_alert),
        },
        "thresholds": {"psi_alert": psi_alert, "shift_alert": shift_alert},
        "artifacts": {
            "top_ngrams_delta_png": str(Path(out_dir) / "top_ngrams_delta.png"),
            "confidence_hist_png":  str(Path(out_dir) / "confidence_hist.png") if ks_stat is not None else None,
            "label_mix_png":        str(Path(out_dir) / "label_mix.png") if mix_train and mix_prod else None,
        },
    }

    with open(Path(out_dir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Minimal HTML
    html = [
        "<html><head><meta charset='utf-8'><title>BridgeLite Drift Report</title></head><body>",
        "<h1>BridgeLite — Drift Report</h1>",
        f"<p><b>PSI (Top-K n-grams)</b>: {summary['psi_topk']} (threshold {psi_alert})</p>",
        f"<p><b>OOV rate</b>: {summary['oov_rate']}</p>",
    ]
    if ks_stat is not None:
        html.append(f"<p><b>Confidence KS</b>: stat={summary['ks_confidence_stat']}, p={summary['ks_confidence_pvalue']:.2e}</p>")
    if largest_shift:
        c, a, b, d = largest_shift
        html.append(f"<p><b>Largest label shift</b>: {c} — train={a:.3f}, prod={b:.3f}, Δ={d:.3f} (threshold {shift_alert})</p>")

    if Path(out_dir, "top_ngrams_delta.png").exists():
        html.append("<h2>Top n-grams by Δ</h2>")
        html.append("<img src='top_ngrams_delta.png' style='max-width:100%;'>")
    if Path(out_dir, "confidence_hist.png").exists():
        html.append("<h2>Confidence histogram</h2>")
        html.append("<img src='confidence_hist.png' style='max-width:100%;'>")
    if Path(out_dir, "label_mix.png").exists():
        html.append("<h2>Category mix</h2>")
        html.append("<img src='label_mix.png' style='max-width:100%;'>")

    alerts = summary["alerts"]
    html.append("<h2>Alerts</h2><ul>")
    html.append(f"<li>PSI > {psi_alert}: <b>{alerts['psi_topk_gt_threshold']}</b></li>")
    html.append(f"<li>|Δ label mix| > {shift_alert}: <b>{alerts['label_shift_gt_threshold']}</b></li>")
    html.append("</ul>")
    html.append("</body></html>")
    (Path(out_dir) / "index.html").write_text("\n".join(html), encoding="utf-8")

    print(json.dumps({"psi_topk": summary["psi_topk"], "oov_rate": summary["oov_rate"], "alert": any(alerts.values())}))
    return summary


def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="BridgeLite drift detection and simple report.")
    ap.add_argument("--train", required=True, help="Training CSV with raw_label (and optional category_label)")
    ap.add_argument("--prod", required=True, help="Production CSV with raw_label (and optional category_label)")
    ap.add_argument("--bundle", default="app/model_sklearn.pkl", help="Path to model bundle (optional, recommended)")
    ap.add_argument("--topk", type=int, default=500, help="Top-K n-grams from training to monitor")
    ap.add_argument("--out-dir", default="reports/drift", help="Directory to write report files")
    ap.add_argument("--psi-alert", type=float, default=0.20, help="Alert threshold for PSI")
    ap.add_argument("--shift-alert", type=float, default=0.15, help="Alert threshold for label proportion shift")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse()
    run(
        train_csv=args.train,
        prod_csv=args.prod,
        out_dir=args.out_dir,
        bundle_path=args.bundle,
        topk=args.topk,
        psi_alert=args.psi_alert,
        shift_alert=args.shift_alert,
    )
