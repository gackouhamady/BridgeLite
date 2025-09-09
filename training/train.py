# training/train.py
"""
BridgeLite — Training script (TF-IDF char + XGBoost/RandomForest)

What this script does
---------------------
1) Loads the training CSV (expects columns: raw_label, category_label, channel, mcc).
2) Preprocesses labels with `app.preproc.preprocess_record` (normalize + PII mask).
   It also injects light metadata tokens (op:<type>, mcc:<code>) to help the model.
3) Vectorizes text with char-level TF-IDF (1–5).
4) Trains a classifier:
      - XGBoost (default)  OR
      - RandomForest (fallback if xgboost unavailable or via --model rf)
5) (Optional) Probability calibration via CalibratedClassifierCV.
6) Evaluates on a validation split and prints macro-F1 & accuracy.
7) Exports a single joblib bundle:
      app/model_sklearn.pkl  →  {"vectorizer", "model", "label_encoder", "metadata"}

CLI
---
python training/train.py \
  --train data/transactions_mock.csv \
  --out app/model_sklearn.pkl \
  --model xgb --val-size 0.2 --seed 7 \
  --ngram-min 1 --ngram-max 5 --max-features 200000 \
  --calibrate none  # or sigmoid|isotonic

Notes
-----
- Uses masked text (`preproc.masked`) to avoid leaking PII.
- Adds "op:<card|direct_debit|transfer|unknown>" and "mcc:<code>" as tokens.
- If xgboost is not installed, the script falls back to RandomForest automatically.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

# Try XGBoost, else fallback
try:
    from xgboost import XGBClassifier  # type: ignore
    _HAVE_XGB = True
except Exception:
    _HAVE_XGB = False

from sklearn.ensemble import RandomForestClassifier

from app.preproc import preprocess_record


@dataclass
class TrainArtifacts:
    vectorizer: TfidfVectorizer
    model: object
    label_encoder: LabelEncoder
    metadata: dict


def _build_text(row: pd.Series) -> str:
    """Return the model text: masked label + tiny metadata tokens."""
    pre = preprocess_record({"raw_label": row["raw_label"], "channel": row.get("channel", None)})
    op = pre.operation_type or "unknown"
    mcc = row.get("mcc", "")
    mcc_tok = f"mcc:{int(mcc)}" if str(mcc).strip().isdigit() else "mcc:none"
    return f"{pre.masked} op:{op} {mcc_tok}"


def _make_model(kind: str, seed: int):
    """Create the classifier (XGB or RF) with sensible defaults for text."""
    if kind == "xgb" and _HAVE_XGB:
        # Lightweight, fast to train on char-ngrams
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=seed,
            tree_method="hist",
        )
    # Fallback RF
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        n_jobs=-1,
        random_state=seed,
        class_weight=None,
    )


def train(
    train_csv: str,
    out_path: str,
    model_kind: str = "xgb",
    ngram_min: int = 1,
    ngram_max: int = 5,
    max_features: int | None = 200_000,
    val_size: float = 0.2,
    seed: int = 42,
    calibrate: str = "none",  # "none" | "sigmoid" | "isotonic"
) -> TrainArtifacts:
    """Train the model and export a single joblib bundle."""
    df = pd.read_csv(train_csv)
    needed = {"raw_label", "category_label"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Build text corpus
    text = df.apply(_build_text, axis=1)
    y = df["category_label"].astype(str).values

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        text.values, y_enc, test_size=val_size, random_state=seed, stratify=y_enc
    )

    # Vectorizer (char n-grams)
    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(ngram_min, ngram_max),
        lowercase=False,  # already normalized in preproc
        max_features=max_features,
        dtype=np.float32,
    )
    Xtr = vec.fit_transform(X_train)
    Xva = vec.transform(X_val)

    # Model
    clf = _make_model(model_kind, seed)

    # Optional calibration (wraps an unfitted base estimator)
    if calibrate in {"sigmoid", "isotonic"}:
        clf = CalibratedClassifierCV(base_estimator=clf, method=calibrate, cv=3)

    # Fit
    clf.fit(Xtr, y_train)

    # Quick val metrics
    yhat = clf.predict(Xva)
    f1 = f1_score(y_val, yhat, average="macro")
    acc = accuracy_score(y_val, yhat)
    print(f"[VAL] macro-F1={f1:.4f}  acc@top1={acc:.4f}  classes={list(le.classes_)}")

    # Export one bundle
    bundle = {
        "vectorizer": vec,
        "model": clf,
        "label_encoder": le,
        "metadata": {
            "model_kind": model_kind if (model_kind == "xgb" and _HAVE_XGB) else "rf",
            "ngram_range": (ngram_min, ngram_max),
            "max_features": max_features,
            "val_size": val_size,
            "seed": seed,
            "calibrate": calibrate,
        },
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    print(f"[OK] Exported bundle → {out.resolve()}")

    return TrainArtifacts(vec, clf, le, bundle["metadata"])


def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train TF-IDF (char) + XGB/RF and export bundle.")
    ap.add_argument("--train", required=True, help="CSV with raw_label & category_label")
    ap.add_argument("--out", default="app/model_sklearn.pkl", help="Output joblib bundle")
    ap.add_argument("--model", choices=["xgb", "rf"], default="xgb", help="Classifier kind")
    ap.add_argument("--ngram-min", type=int, default=1)
    ap.add_argument("--ngram-max", type=int, default=5)
    ap.add_argument("--max-features", type=int, default=200_000)
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--calibrate", choices=["none", "sigmoid", "isotonic"], default="none")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse()
    train(
        train_csv=args.train,
        out_path=args.out,
        model_kind=args.model,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        max_features=args.max_features or None,
        val_size=args.val_size,
        seed=args.seed,
        calibrate=args.calibrate,
    )
