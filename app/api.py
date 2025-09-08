"""
app/api.py (excerpt)

Show how the service-layer function plugs into FastAPI.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from app.service_merchants import normalize_merchant

app = FastAPI()


class TxIn(BaseModel):
    tx_id: str
    raw_label: str


class TxOut(BaseModel):
    tx_id: str
    normalized_merchant: str | None = None
    explanation: str | None = None
    router: str | None = None


@app.post("/predict", response_model=List[TxOut])
def predict(transactions: List[TxIn]) -> List[TxOut]:
    """
    Predict endpoint (simplified to focus on merchant normalization).

    For each transaction:
    - Attempts gazetteer normalization with the configured threshold.
    - If matched, returns display name + explanation.
    - Otherwise, leaves fields empty (your model/LLM would fill them later).
    """
    outs: List[TxOut] = []
    for tx in transactions:
        disp, mid, score = normalize_merchant(tx.raw_label, threshold=86)
        if mid:
            outs.append(
                TxOut(
                    tx_id=tx.tx_id,
                    normalized_merchant=disp,
                    explanation=f"gazetteer match '{disp}' (score={score})",
                    router="rules>gazetteer",
                )
            )
        else:
            outs.append(TxOut(tx_id=tx.tx_id, router="rules"))
    return outs
