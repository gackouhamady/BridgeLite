# app/api.py (tiny sketch)
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from .service import BridgeService
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
# app/api.py
from pathlib import Path
import json

app = FastAPI(title="BridgeLite API")
svc = BridgeService()

class Tx(BaseModel):
    tx_id: Optional[str] = None
    date: Optional[str] = None
    amount: Optional[float] = None
    currency: Optional[str] = None
    raw_label: str
    channel: Optional[str] = None
    mcc: Optional[int] = None

class PredictRequest(BaseModel):
    transactions: List[Tx]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    batch = [t.dict() for t in req.transactions]
    return svc.classify_batch(batch)

@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

@app.get("/drift/summary")
def drift_summary():
    p = Path("reports/drift/summary.json")
    if not p.exists():
        return {"status": "no_report"}
    s = json.loads(p.read_text(encoding="utf-8"))
    # Update gauges as a side effect (optional)
    try:
        from prometheus_client import Gauge, Counter
        from .service import DRIFT_PSI_TOPK, DRIFT_OOV_RATE, DRIFT_ALERTS_TOTAL
        DRIFT_PSI_TOPK.set(s.get("psi_topk", 0.0))
        DRIFT_OOV_RATE.set(s.get("oov_rate", 0.0))
        if any(s.get("alerts", {}).values()):
            DRIFT_ALERTS_TOTAL.inc()
    except Exception:
        pass
    return s