# app/service.py
"""
BridgeLite — Routing service (rules → gazetteer → model → LLM fallback)

What this module provides
-------------------------
- BridgeService: loads model bundle + gazetteer and exposes:
    - classify_one(tx: dict) -> dict (single transaction)
    - classify_batch(txs: list[dict]) -> {"results": [...], "metrics": {...}}
- Prometheus metrics counters/histograms updated on each call.

Routing logic
-------------
1) Preprocess (normalize/mask, detect operation type).
2) Strong rules (MCC and keywords) → if hit, return immediately.
3) Gazetteer lookup for merchant normalization (RapidFuzz) → optional category nudge.
4) ML model (char TF-IDF + XGB/RF) → if confidence ≥ τ and OOV acceptable, accept.
5) Else call fallback LLM (strict JSON) → accept LLM output.

Config knobs
------------
- tau: confidence threshold for "coverage" (default 0.6).
- gaz_threshold: fuzzy score to accept a merchant (default 0.90).
- oov_tau: simple OOV proxy threshold (0..1). If OOV > oov_tau, prefer fallback.

Notes
-----
- Expose these methods through FastAPI in `app/api.py`.
- Gazetteer is optional; if file missing, merchant normalization is skipped.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from prometheus_client import Counter, Gauge, Histogram
from rapidfuzz import fuzz, process  # type: ignore

from .llm_fallback import classify_with_llm
from .preproc import preprocess_record
from .rules import apply_rules


# ------------------------------ Metrics --------------------------------------

PREDICT_REQUESTS = Counter("predict_requests_total", "Total prediction requests")
FALLBACK_TOTAL = Counter("fallback_total", "Total fallback LLM invocations")
COVERAGE_RATIO = Gauge("coverage_ratio", "Share of predictions with confidence >= tau")
PREDICT_LAT_MS = Histogram(
    "predict_latency_ms",
    "Prediction latency in milliseconds",
    buckets=(5, 10, 20, 40, 80, 160, 320, 640, 1280),
)

# Drift-related (updated via /drift/summary or /drift/update)
DRIFT_PSI_TOPK = Gauge("psi_topk", "Population Stability Index on top-K char n-grams")
DRIFT_OOV_RATE = Gauge("oov_rate", "Out-of-vocabulary rate on prod window")
DRIFT_ALERTS_TOTAL = Counter("drift_alerts_total", "Total number of drift alerts")


def update_drift_metrics(summary: dict) -> None:
    """
    Update drift gauges from a summary dict (see training/drift.py output).
    Safe no-op on errors (metrics must never break serving).
    """
    try:
        if "psi_topk" in summary:
            DRIFT_PSI_TOPK.set(float(summary["psi_topk"]))
        if "oov_rate" in summary:
            DRIFT_OOV_RATE.set(float(summary["oov_rate"]))
        alerts = summary.get("alerts", {})
        if isinstance(alerts, dict) and any(alerts.values()):
            DRIFT_ALERTS_TOTAL.inc()
    except Exception:
        pass


# ------------------------------ Gazetteer ------------------------------------

@dataclass
class Gazetteer:
    """Simple fuzzy gazetteer for alias→merchant mapping."""
    aliases: List[str]
    ids: List[str]
    display: List[str]

    @classmethod
    def from_csv(cls, path: str) -> "Gazetteer":
        df = pd.read_csv(path)
        need = {"alias", "merchant_id", "display_name"}
        if not need.issubset(df.columns):
            raise ValueError(f"Gazetteer must have columns {need}")
        aliases = df["alias"].astype(str).str.lower().str.strip().tolist()
        ids = df["merchant_id"].astype(str).tolist()
        display = df["display_name"].astype(str).tolist()
        return cls(aliases, ids, display)

    @classmethod
    def from_pkl(cls, path: str) -> "Gazetteer":
        b = joblib.load(path)
        return cls(
            aliases=list(b["aliases_lower"]),
            ids=list(b["merchant_ids"]),
            display=list(b["display_names"]),
        )

    def lookup(
        self, text: str, score_cut: float = 0.90
    ) -> Tuple[Optional[str], Optional[str], float, str]:
        """
        Fuzzy lookup; returns (merchant_id, display_name, score, reason).
        Score is normalized to 0..1 (RapidFuzz native scale is 0..100).
        """
        if not self.aliases:
            return None, None, 0.0, "gazetteer-empty"

        item = process.extractOne(
            query=(text or "").lower(),
            choices=self.aliases,
            scorer=fuzz.WRatio,  # robust composite scorer
        )
        if not item:
            return None, None, 0.0, "no-match"

        alias, score, idx = item[0], float(item[1]), int(item[2])
        score01 = score / 100.0
        if score01 >= score_cut:
            return self.ids[idx], self.display[idx], score01, f"alias:{alias}|score:{score01:.2f}"
        return None, None, score01, f"weak-alias:{alias}|score:{score01:.2f}"


# ------------------------------ Helpers --------------------------------------

def _build_text_for_model(masked_text: str, op_type: str, mcc: Optional[int]) -> str:
    """Compose the exact text representation used by the vectorizer at training time."""
    mcc_tok = f"mcc:{int(mcc)}" if (mcc is not None and str(mcc).isdigit()) else "mcc:none"
    return f"{masked_text} op:{op_type or 'unknown'} {mcc_tok}"


def _oov_rate_char_ngrams(text: str, vectorizer) -> float:
    """
    Very simple OOV proxy for char n-grams: share of n-grams not in the vocab.
    Returns 0.0 on any error to avoid blocking routing.
    """
    try:
        analyzer = vectorizer.build_analyzer()
        grams = analyzer(text or "")
        if not grams:
            return 1.0
        vocab = vectorizer.vocabulary_
        seen = sum(1 for g in grams if g in vocab)
        return float(1.0 - (seen / max(1, len(grams))))
    except Exception:
        return 0.0


# ------------------------------ Service --------------------------------------

class BridgeService:
    """
    Main runtime router. Prefers:
      rules → gazetteer → model → LLM
    """

    def __init__(
        self,
        model_path: str = "app/model_sklearn.pkl",
        gazetteer_path: Optional[str] = "data/gazetteer_merchants.csv",
        tau: float = 0.6,
        gaz_threshold: float = 0.90,
        oov_tau: float = 0.55,
        llm_mode: str = "stub",  # "stub" | "auto" | "hf-zero-shot"
    ):
        self.tau = float(tau)
        self.gaz_threshold = float(gaz_threshold)
        self.oov_tau = float(oov_tau)
        self.llm_mode = llm_mode

        # --- Load model bundle if available ---
        self.vectorizer = None
        self.model = None
        self.le = None
        self.labels: List[str] = []

        mp = Path(model_path)
        if mp.exists():
            bundle = joblib.load(mp)
            self.vectorizer = bundle.get("vectorizer")
            self.model = bundle.get("model")
            self.le = bundle.get("label_encoder")
            if self.le is not None:
                self.labels = list(self.le.classes_)

        # --- Load gazetteer (prefer artifact .pkl, else CSV) ---
        self.gazetteer: Optional[Gazetteer] = None
        if gazetteer_path:
            pkl = Path("app/gazetteer.pkl")
            csv = Path(gazetteer_path)
            try:
                if pkl.exists():
                    self.gazetteer = Gazetteer.from_pkl(str(pkl))
                elif csv.exists():
                    self.gazetteer = Gazetteer.from_csv(str(csv))
            except Exception:
                # Do not crash if gazetteer is malformed; just skip it
                self.gazetteer = None

    # -------------------------- Routing core --------------------------

    def classify_one(self, tx: Dict) -> Dict:
        """
        Route a single transaction through rules → gazetteer → model → LLM.
        Returns the final structured dict expected by /predict.
        """
        t0 = time.perf_counter()
        PREDICT_REQUESTS.inc()

        # Preprocess
        pre = preprocess_record(tx)
        explanation_parts: List[str] = []

        # 1) Rules
        r = apply_rules(pre, mcc=tx.get("mcc"))
        if r["hit"]:
            category = r["category"]
            conf = 0.80  # heuristic default for strong rules
            explanation_parts.append(f"rules:{r['reason']}")

            # Optional: still try merchant normalization
            merch_name = None
            if self.gazetteer:
                _, disp, s, why = self.gazetteer.lookup(pre.normalized, self.gaz_threshold)
                if disp and s >= self.gaz_threshold:
                    merch_name = disp
                    explanation_parts.append(f"gazetteer:{why}")

            lat_ms = (time.perf_counter() - t0) * 1000.0
            PREDICT_LAT_MS.observe(lat_ms)
            return {
                "tx_id": tx.get("tx_id"),
                "operation_type": pre.operation_type,
                "normalized_merchant": merch_name,
                "budget_category": category,
                "confidence": round(conf, 4),
                "explanation": " | ".join(explanation_parts),
                "router": "rules" + (">gazetteer" if merch_name else ""),
                "lat_ms": round(lat_ms, 2),
            }

        # 2) Gazetteer (merchant normalization)
        merch_name = None
        if self.gazetteer:
            _, disp, s, why = self.gazetteer.lookup(pre.normalized, self.gaz_threshold)
            if disp and s >= self.gaz_threshold:
                merch_name = disp
                explanation_parts.append(f"gazetteer:{why}")

        # 3) Model prediction (if model available)
        result_category = None
        result_conf = 0.0
        route = "model" if merch_name else "model"  # default route name

        use_model = False
        if self.vectorizer is not None and self.model is not None and self.le is not None:
            model_text = _build_text_for_model(pre.masked, pre.operation_type, tx.get("mcc"))
            X = self.vectorizer.transform([model_text])
            proba = self.model.predict_proba(X)[0]
            idx = int(np.argmax(proba))
            p = float(proba[idx])
            category = self.le.inverse_transform([idx])[0]
            explanation_parts.append(f"model:pmax={p:.3f}")

            oov = _oov_rate_char_ngrams(pre.masked, self.vectorizer)
            explanation_parts.append(f"oov={oov:.2f}")

            use_model = (p >= self.tau) and (oov <= self.oov_tau)
            result_category = category
            result_conf = p
            if merch_name:
                route = "gazetteer>model"
        else:
            explanation_parts.append("model:unavailable")
            use_model = False  # force LLM if we have no model

        # 4) Fallback LLM if needed
        if not use_model:
            FALLBACK_TOTAL.inc()
            labels_for_llm = self.labels or []  # may be empty; stub will handle
            llm = classify_with_llm(pre.normalized, labels_for_llm, mode=self.llm_mode)
            result_category = llm.get("category", result_category or "Other")
            # show the best confidence we have (model pmax vs llm confidence)
            result_conf = float(max(result_conf, float(llm.get("confidence", 0.0))))
            explanation_parts.append(f"llm:{llm.get('reason', 'fallback')}")

            route = ("gazetteer>llm" if merch_name else "llm") if "model" not in route else (route + ">llm")

        lat_ms = (time.perf_counter() - t0) * 1000.0
        PREDICT_LAT_MS.observe(lat_ms)

        return {
            "tx_id": tx.get("tx_id"),
            "operation_type": pre.operation_type,
            "normalized_merchant": merch_name,
            "budget_category": result_category,
            "confidence": round(result_conf, 4),
            "explanation": " | ".join(explanation_parts),
            "router": route,
            "lat_ms": round(lat_ms, 2),
        }

    # -------------------------- Batch helper --------------------------

    def classify_batch(self, txs: List[Dict]) -> Dict:
        """
        Classify a batch and compute simple request-level metrics (coverage, fallback rate).
        The API layer will set COVERAGE_RATIO gauge based on this.
        """
        results = [self.classify_one(tx) for tx in txs]
        confs = [r["confidence"] for r in results]
        coverage = float(np.mean([c >= self.tau for c in confs])) if confs else 0.0
        fallback_rate = float(np.mean(["llm" in r["router"] for r in results])) if results else 0.0
        COVERAGE_RATIO.set(coverage)
        return {
            "results": results,
            "metrics": {"coverage": round(coverage, 4), "fallback_rate": round(fallback_rate, 4)},
        }


# -------------------------- Quick smoke (optional) ----------------------------

if __name__ == "__main__":
    # Minimal manual test without FastAPI
    svc = BridgeService(
        model_path="app/model_sklearn.pkl",
        gazetteer_path="data/gazetteer_merchants.csv",
        tau=0.6,
        gaz_threshold=0.90,
        oov_tau=0.55,
        llm_mode="stub",  # use "stub" unless transformers/torch are installed
    )
    batch = {
        "transactions": [
            {"tx_id": "t1", "raw_label": "CB CARREFOUR 75 PARIS TPE1245", "channel": "CB", "mcc": 5411},
            {"tx_id": "t2", "raw_label": "PRLV EDF FACTURE 07/2025 RUM1234-5678", "channel": "PRLV", "mcc": 4900},
            {"tx_id": "t3", "raw_label": "ACHAT ?????", "channel": "CB"},
        ]
    }
    out = svc.classify_batch(batch["transactions"])
    from pprint import pprint
    pprint(out)
