# app/rules.py
"""
BridgeLite — Rule-Based Categorization
======================================

Purpose
-------
Provide fast, deterministic categorization signals prior to/or alongside the ML model:
1) **MCC → Category** mapping when the merchant category code is present.
2) **Keyword patterns** in normalized label text (e.g., EDF/ENGIE → Utilities, LOYER/FONCIA → Rent).
3) A few **heuristics** (e.g., "facture" + known operator → Utilities).

These rules boost precision on obvious cases and yield interpretable explanations.
They also help the router decide whether to trust rules, gazetteer, or the model.

Important note
--------------
The dataset mimics **French** banking labels; keywords are French (e.g., "loyer", "facture", "salaire").
Docstrings and code comments are in English to maintain clarity for international teams.

Public API
----------
- `category_from_mcc(mcc: int | None) -> str | None`
- `apply_rules(preproc: PreprocResult, mcc: int | None = None) -> dict`

`apply_rules` returns:
    {
      "hit": bool,               # whether a rule fired
      "category": str | None,    # proposed category
      "reason": str              # short explanation (e.g., "mcc:5411", "keyword:edf")
    }
"""

from __future__ import annotations

from typing import Optional

import regex as re

from .preproc import PreprocResult


# ---------------------------------------------------------------------------
# MCC → Category
# ---------------------------------------------------------------------------

MCC_TO_CATEGORY = {
    5411: "Groceries",    # Supermarkets
    5814: "Restaurants",  # Fast Food / Restaurants
    4111: "Transport",    # Commuter Transport
    4900: "Utilities",    # Utilities
    6513: "Rent",         # Real estate / Rentals (proxy for "loyer")
    6011: "Salary",       # Financial institutions (used as a salary proxy here)
}


def category_from_mcc(mcc: Optional[int]) -> Optional[str]:
    """
    Map MCC to a budget category if recognized.

    Parameters
    ----------
    mcc : int | None
        Merchant Category Code.

    Returns
    -------
    str | None
        Budget category name if known, else None.

    Examples
    --------
    >>> category_from_mcc(5411)
    'Groceries'
    >>> category_from_mcc(9999) is None
    True
    """
    if mcc is None:
        return None
    try:
        return MCC_TO_CATEGORY.get(int(mcc))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Keyword patterns (on normalized, lowercase ASCII text)
# ---------------------------------------------------------------------------

# Direct keyword → category mapping
KEYWORDS_CATEGORY = {
    # Rent / housing
    "loyer": "Rent",
    "foncia": "Rent",
    "nexity": "Rent",
    # Utilities
    "edf": "Utilities",
    "engie": "Utilities",
    "orange": "Utilities",
    "sfr": "Utilities",
    "free": "Utilities",
    # Transport
    "ratp": "Transport",
    "sncf": "Transport",
    "uber": "Transport",
    "bolt": "Transport",
    "total": "Transport",          # TotalEnergies (stations)
    "totalenergies": "Transport",
    # Groceries (retail chains)
    "carrefour": "Groceries",
    "auchan": "Groceries",
    "intermarche": "Groceries",
    "monoprix": "Groceries",
    "casino": "Groceries",
    # Restaurants
    "mcdo": "Restaurants",
    "mcdonald": "Restaurants",
    "burger": "Restaurants",       # part of "burger king"
    "kfc": "Restaurants",
    "starbucks": "Restaurants",
    "quick": "Restaurants",
    # Salary / payroll
    "salaire": "Salary",
    "payroll": "Salary",
}

# Auxiliary patterns for specific phrases
RE_FACTURE = re.compile(r"\bfacture\b")                      # invoice
RE_RENT = re.compile(r"\bloyer\b")                           # rent
RE_SALARY = re.compile(r"\bsalaire\b|\bpayroll\b")           # salary/payroll


def apply_rules(preproc: PreprocResult, mcc: Optional[int] = None) -> dict:
    """
    Apply rule-based categorization using MCC codes and normalized-label keywords.

    Strategy
    --------
    1) If MCC is present and recognized, return the mapped category immediately.
    2) Else scan normalized text for direct keywords (EDF, RATP, CARREFOUR, LOYER...).
    3) Else apply a couple of heuristic patterns:
       - presence of 'facture' + a known telecom/utility operator → 'Utilities'
       - 'salaire'/'payroll' → 'Salary'
       - 'loyer' → 'Rent'
    4) If no rule fires, return `{"hit": False, "category": None, "reason": ""}`.

    Parameters
    ----------
    preproc : PreprocResult
        Output from `preprocess_record`, providing normalized text and tokens.
    mcc : int | None
        MCC code from the transaction, if available.

    Returns
    -------
    dict
        Rule decision with `hit`, `category`, `reason`.

    Examples
    --------
    >>> from .preproc import preprocess_record
    >>> out = apply_rules(preprocess_record({"raw_label": "PRLV EDF FACTURE 07/2025"}))
    >>> out["hit"], out["category"]
    (True, 'Utilities')
    """
    text = preproc.normalized

    # 1) MCC → category
    cat = category_from_mcc(mcc)
    if cat:
        return {"hit": True, "category": cat, "reason": f"mcc:{mcc}"}

    # 2) Direct keyword mapping
    for kw, c in KEYWORDS_CATEGORY.items():
        if kw in text:
            return {"hit": True, "category": c, "reason": f"keyword:{kw}"}

    # 3) Heuristics
    # "facture" + known operator → Utilities
    if RE_FACTURE.search(text):
        for kw in ("edf", "engie", "orange", "sfr", "free"):
            if kw in text:
                return {"hit": True, "category": "Utilities", "reason": "pattern:facture+operator"}

    # explicit salary
    if RE_SALARY.search(text):
        return {"hit": True, "category": "Salary", "reason": "pattern:salaire"}

    # explicit rent
    if RE_RENT.search(text):
        return {"hit": True, "category": "Rent", "reason": "pattern:loyer"}

    # 4) No rule fired
    return {"hit": False, "category": None, "reason": ""}
