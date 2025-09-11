# tests/test_rules.py
from __future__ import annotations
from app.preproc import preprocess_record
from app.rules import apply_rules

def _pre(label: str, channel: str | None = None):
    return preprocess_record({"raw_label": label, "channel": channel})

def test_rules_mcc_maps_category():
    pre = _pre("CB CARREFOUR 75 PARIS TPE1245", "CB")
    out = apply_rules(pre, mcc=5411)  # 5411 = groceries in our mapping
    assert out["hit"] is True
    assert out["category"].lower() in {"groceries", "alimentation", "food"}

def test_rules_keyword_utilities():
    pre = _pre("PRLV EDF FACTURE 07/2025", "PRLV")
    out = apply_rules(pre, mcc=4900)
    assert out["hit"] is True
    assert out["category"].lower() in {"utilities"}

def test_rules_rent_pattern():
    pre = _pre("PRLV LOYER APPARTEMENT SEPT", "PRLV")
    out = apply_rules(pre, mcc=None)
    assert out["hit"] is True
    assert out["category"].lower() in {"rent"}
