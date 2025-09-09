# app/llm_fallback.py
"""
BridgeLite — Fallback LLM (strict-JSON output)

Purpose
-------
Provide a last-resort classifier for ambiguous / low-confidence cases.
The output is strictly constrained JSON so callers can safely parse it:

    {"category": "<one-of-labels>", "confidence": 0.0..1.0, "reason": "..."}

Modes
-----
- "stub" (default): lightweight heuristic using keywords + fuzzy matching
  (no heavy dependencies; good enough for a fallback gate).
- "hf-zero-shot": optional zero-shot classification using Hugging Face
  (requires `transformers` and `torch`; auto-detected if installed).

Public API
----------
classify_with_llm(text: str, labels: list[str], mode: str = "auto") -> dict
- mode="auto" tries "hf-zero-shot" if available, otherwise falls back to "stub".
- Always validates output to ensure category ∈ labels and confidence ∈ [0,1].

Notes
-----
- We intentionally keep this deterministic and safe: no network calls.
- If you later host a remote LLM, keep the **same strict JSON contract**.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import math
import re

try:
    # Optional deps — only used if present
    from transformers import pipeline  # type: ignore
    _HAVE_TRANSFORMERS = True
except Exception:
    _HAVE_TRANSFORMERS = False

try:
    from rapidfuzz import fuzz, process  # type: ignore
    _HAVE_RAPIDFUZZ = True
except Exception:
    _HAVE_RAPIDFUZZ = False


# ---------------------------------------------------------------------------

# Minimal keywords per common category name. Extend as needed.
DEFAULT_CATEGORY_HINTS: Dict[str, List[str]] = {
    "Groceries": ["carrefour", "auchan", "intermarche", "monoprix", "casino", "super", "market"],
    "Restaurants": ["mcdo", "mcdonald", "burger", "kfc", "restaurant", "pizza", "cafe"],
    "Transport": ["ratp", "sncf", "uber", "bolt", "total", "station", "essence", "tpe", "toll"],
    "Utilities": ["edf", "engie", "orange", "sfr", "free", "facture"],
    "Rent": ["loyer", "foncia", "nexity"],
    "Salary": ["salaire", "payroll", "paie", "salary"],
}

_WORD = re.compile(r"[a-z0-9]+")


def _normalize(s: str) -> str:
    s = s or ""
    s = s.lower()
    # Keep it simple: ascii-like only; accents should already be removed upstream
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _stub_score(text: str, labels: List[str]) -> Tuple[str, float, str]:
    """
    Heuristic scorer:
    - keyword hits from DEFAULT_CATEGORY_HINTS
    - fuzzy match against label names
    Produces a softmax-like prob for the winning label.
    """
    t = _normalize(text)
    toks = set(_WORD.findall(t))

    scores = []
    reasons = {}

    for lab in labels:
        hints = DEFAULT_CATEGORY_HINTS.get(lab, [])
        hit = len([h for h in hints if h in toks])
        kw_score = 0.6 * min(1.0, hit / 2.0)  # up to 0.6 from keywords

        # Fuzzy against the label itself (fallback if hints are missing)
        fuzz_score = 0.0
        if _HAVE_RAPIDFUZZ:
            fuzz_score = 0.01 * max(
                fuzz.partial_ratio(t, lab.lower()),
                max((fuzz.partial_ratio(t, h) for h in hints), default=0),
            )  # up to ~1.0 * 0.01 = 0.01 .. we keep it small
        else:
            # naive contains as a tiny signal
            fuzz_score = 0.05 if lab.lower() in t else 0.0

        s = kw_score + fuzz_score
        scores.append(s)
        reasons[lab] = f"kw_hits={hit}, fuzzy={fuzz_score:.3f}"

    # Normalize to [0,1] probs
    # Softmax with small temperature to separate ties
    mx = max(scores) if scores else 0.0
    exp = [math.exp((s - mx) / 0.3) for s in scores]
    Z = sum(exp) or 1.0
    probs = [e / Z for e in exp]
    if not labels:
        return "", 0.0, "no-labels"

    idx = int(probs.index(max(probs)))
    best = labels[idx]
    return best, float(probs[idx]), reasons[best]


def _hf_zero_shot(text: str, labels: List[str]) -> Tuple[str, float, str]:
    """
    Zero-shot classification using transformers (if available).
    We pick the top label and its score, pass through as-is.
    """
    if not _HAVE_TRANSFORMERS:
        raise RuntimeError("Transformers not installed")

    # Select a compact model if present; otherwise pipeline will download a default.
    # IMPORTANT: Prefer local cached models to avoid network calls in prod.
    clf = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1")
    out = clf(text, candidate_labels=labels, multi_label=False)
    # 'out' schema: {'sequence':..., 'labels':[...], 'scores':[...]}
    top_label = out["labels"][0]
    top_score = float(out["scores"][0])
    return top_label, top_score, "zero-shot mnli"


def _validate(labels: List[str], category: str, confidence: float, reason: str) -> Dict:
    """
    Enforce strict JSON contract and clip invalid values.
    """
    if not labels:
        return {"category": "", "confidence": 0.0, "reason": "no-labels"}

    if category not in labels:
        # Fallback deterministically to first label to avoid schema breaks
        category = labels[0]
        reason = (reason + " | coerced-to-valid").strip()

    confidence = float(max(0.0, min(1.0, confidence)))
    if not isinstance(reason, str):
        reason = str(reason)
    return {"category": category, "confidence": confidence, "reason": reason}


def classify_with_llm(text: str, labels: List[str], mode: str = "auto") -> Dict:
    """
    Classify free text into one of `labels` and return strict JSON.

    Parameters
    ----------
    text : str
        Raw/normalized transaction label.
    labels : list[str]
        Category names allowed by the system (exact strings).
    mode : {"auto", "stub", "hf-zero-shot"}
        - "auto": use hf-zero-shot if available; otherwise stub.
        - "stub": lightweight heuristic (default safe option).
        - "hf-zero-shot": force transformers pipeline (if installed).

    Returns
    -------
    dict
        {"category": <label>, "confidence": float in [0,1], "reason": <str>}
    """
    try:
        if mode == "hf-zero-shot" or (mode == "auto" and _HAVE_TRANSFORMERS):
            cat, prob, why = _hf_zero_shot(text, labels)
        else:
            cat, prob, why = _stub_score(text, labels)
    except Exception as e:
        # Any failure → deterministic safe fallback
        cat, prob, why = _stub_score(text, labels)
        why = f"fallback-stub ({e}) | " + why

    return _validate(labels, cat, prob, why)
