# tests/test_preproc.py
from __future__ import annotations
from app.preproc import preprocess_record

def test_preprocess_normalizes_and_masks():
    tx = {"raw_label": "PRLV EDF FACTURE 07/2025 RUM FR12-ABC", "channel": "PRLV"}
    pre = preprocess_record(tx)

    # basic structure
    assert hasattr(pre, "normalized")
    assert hasattr(pre, "masked")
    assert hasattr(pre, "operation_type")

    # normalization: lowercase, trimmed
    assert pre.normalized == pre.normalized.lower().strip()
    assert "edf" in pre.normalized

    # masking: IBAN/RUM/long ids should be replaced
    assert "rum" in pre.masked  # token preserved
    assert "FR12" not in pre.masked  # masked away

    # operation type from channel/text
    assert pre.operation_type == "direct_debit"  # PRLV → direct_debit
