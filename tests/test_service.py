# tests/test_service.py
from __future__ import annotations
import pytest
from pathlib import Path
from app.service import BridgeService

MODEL_PATH = Path("app/model_sklearn.pkl")

@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Train a model first (app/model_sklearn.pkl)")
def test_service_model_path_without_llm():
    svc = BridgeService(model_path=str(MODEL_PATH), gazetteer_path=None, tau=0.3, llm_mode="stub")
    # Easy sample where model should be confident
    tx = {"tx_id": "t1", "raw_label": "CB CARREFOUR 75 PARIS TPE1245", "channel": "CB", "mcc": 5411}
    out = svc.classify_one(tx)
    assert out["tx_id"] == "t1"
    assert "budget_category" in out
    assert "llm" not in out["router"]  # model should suffice at low tau

@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Train a model first (app/model_sklearn.pkl)")
def test_service_forces_llm_by_tau():
    svc = BridgeService(model_path=str(MODEL_PATH), gazetteer_path=None, tau=0.99, oov_tau=0.0, llm_mode="stub")
    tx = {"tx_id": "t3", "raw_label": "CB ZXQW PLORP STORE 9999 XJ-REF88", "channel": "CB"}
    out = svc.classify_one(tx)
    assert "llm" in out["router"]  # forced by high tau
