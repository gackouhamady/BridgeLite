# app/api.py (tiny sketch)
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from .service import BridgeService
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

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
