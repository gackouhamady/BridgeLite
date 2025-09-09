# tests/test_api.py
"""
BridgeLite — API tests for /health, /predict, /metrics

- Verifies health endpoint
- Checks /predict single & batch schema
- Forces LLM fallback path deterministically
- Confirms Prometheus text on /metrics
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.api as api
from app.service import BridgeService

# Skip all tests if the trained artifact is missing (dev convenience)
if not Path("app/model_sklearn.pkl").exists():
    pytest.skip("Missing app/model_sklearn.pkl — run training first.", allow_module_level=True)


@pytest.fixture(scope="module")
def client() -> TestClient:
    """FastAPI test client over the real app instance."""
    return TestClient(api.app)


def test_health_ok(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_single_happy_path(client: TestClient):
    payload = {
        "transactions": [
            {
                "tx_id": "t1",
                "raw_label": "CB CARREFOUR 75 PARIS TPE1245",
                "channel": "CB",
                "mcc": 5411,
            }
        ]
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "results" in body and isinstance(body["results"], list)
    assert "metrics" in body and isinstance(body["metrics"], dict)

    item = body["results"][0]
    for k in ["tx_id", "operation_type", "budget_category", "confidence", "router", "lat_ms"]:
        assert k in item
    assert isinstance(item["confidence"], (int, float))
    assert 0.0 <= body["metrics"].get("coverage", 0.0) <= 1.0


def test_predict_batch_metrics(client: TestClient):
    payload = {
        "transactions": [
            {"tx_id": "t1", "raw_label": "CB CARREFOUR 75 PARIS TPE1245", "channel": "CB", "mcc": 5411},
            {"tx_id": "t2", "raw_label": "PRLV EDF FACTURE 07/2025", "channel": "PRLV", "mcc": 4900},
        ]
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    m = r.json()["metrics"]
    assert 0.0 <= m["coverage"] <= 1.0
    assert 0.0 <= m["fallback_rate"] <= 1.0


def test_predict_validation_error_422(client: TestClient):
    # Missing "transactions" → should fail validation
    r = client.post("/predict", json={"foo": "bar"})
    assert r.status_code == 422


def test_predict_forces_llm_fallback(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    """
    Force the LLM route by:
      - disabling gazetteer,
      - setting tau very high so the model path is rejected,
      - keeping llm_mode='stub' (no heavy deps required).
    """
    api.svc = BridgeService(
        model_path="app/model_sklearn.pkl",
        gazetteer_path=None,  # disable gazetteer to avoid accidental matches
        tau=0.99,             # require near-perfect model confidence
        oov_tau=0.0,          # don't block on OOV gate; tau will force fallback
        llm_mode="stub",
    )
    noisy = {
        "transactions": [
            {"tx_id": "t3", "raw_label": "CB ZXQW PLORP STORE 9999 XJ-REF88", "channel": "CB"}
        ]
    }
    r = client.post("/predict", json=noisy)
    assert r.status_code == 200
    out = r.json()
    assert out["results"], "Empty results list"
    item = out["results"][0]
    assert "llm" in item["router"], f"Router did not include LLM: {item['router']}"


def test_metrics_prometheus_text(client: TestClient):
    r = client.get("/metrics")
    assert r.status_code == 200
    txt = r.text
    assert "predict_requests_total" in txt
    assert re.search(r"#\s*HELP\s+predict_requests_total", txt) is not None
