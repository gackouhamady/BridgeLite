"""
app/service.py (excerpt)

Example of how to use `normalize_merchant()` inside your routing logic.
"""

from dataclasses import dataclass
from typing import Optional

from app.service_merchants import normalize_merchant


@dataclass
class TxRequest:
    """Minimal request model for demonstration."""
    tx_id: str
    raw_label: str
    # ... add other fields you need (date, amount, mcc, channel, etc.) ...


@dataclass
class TxResult:
    """Minimal response model for demonstration."""
    tx_id: str
    normalized_merchant: Optional[str] = None
    operation_type: Optional[str] = None
    budget_category: Optional[str] = None
    confidence: Optional[float] = None
    explanation: Optional[str] = None
    router: Optional[str] = None
    lat_ms: Optional[int] = None


def route_predict(req: TxRequest) -> TxResult:
    """
    Route a single transaction through the system.

    Strategy
    --------
    1) Try strong rules (not shown here).
    2) Try gazetteer-based merchant normalization.
    3) If still unresolved, fall back to ML model / LLM.

    Returns
    -------
    TxResult
        Populated with whatever stage produced the answer.
    """
    result = TxResult(tx_id=req.tx_id)

    # --- Gazetteer stage ---
    disp, mid, score = normalize_merchant(req.raw_label, threshold=86)
    if mid:
        # We matched a merchant using the gazetteer; record and return or
        # continue with other predictions (category, op type) as needed.
        result.normalized_merchant = disp
        result.explanation = f"gazetteer match '{disp}' (score={score})"
        result.router = "rules>gazetteer"
        # Optionally: continue to model for category prediction, etc.
        return result

    # --- Fallbacks (not implemented here) ---
    # 1) ML model for budget_category
    # 2) LLM fallback if confidence low
    # Fill result fields and router accordingly.

    result.router = "rules>model>llm"
    return result
