"""
app/service_merchants.py

Thin service-layer adapter around `app.merchants.Gazetteer`.

Responsibilities
---------------
- Load a single (process-wide) Gazetteer instance once.
- Expose `normalize_merchant()` with a clear contract for the rest of the app.

Thread-safety / performance
---------------------------
- `Gazetteer` is read-only after construction, so it is safe to reuse across
  requests/threads in typical FastAPI/Uvicorn setups.
- Loading it once avoids repeated CSV parsing and improves latency.
"""

from __future__ import annotations
from typing import Optional, Tuple

from app.merchants import Gazetteer

# Lazily-initialized singleton (created on first use).
_GAZ: Optional[Gazetteer] = None


def _get_gazetteer() -> Gazetteer:
    """
    Return the process-wide Gazetteer instance, loading it the first time.

    Returns
    -------
    Gazetteer
        The shared alias→merchant resolver.

    Notes
    -----
    - Uses the default CSV path from `app.merchants.DATA_PATH`.
    - If you need a different path (e.g., env-specific), adapt the constructor.
    """
    global _GAZ
    if _GAZ is None:
        _GAZ = Gazetteer()  # load once at startup / first call
    return _GAZ


def normalize_merchant(raw_label: str, threshold: int = 86) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Try to resolve a noisy banking `raw_label` to a canonical merchant.

    Parameters
    ----------
    raw_label : str
        The raw transaction label coming from the bank (may contain noise like
        "CB ... TPE...").
    threshold : int, default=86
        Minimum RapidFuzz score (0–100) required to accept the match.

    Returns
    -------
    (display_name, merchant_id, score) : tuple[str|None, str|None, int|None]
        - If a candidate meets the threshold, returns:
          (human-readable name, canonical ID, integer score).
        - If no candidate qualifies, returns (None, None, None).

    Examples
    --------
    >>> normalize_merchant("CB CARREFOUR 75 PARIS TPE1245", threshold=86)
    ('Carrefour', 'carrefour', 96)
    """
    hit = _get_gazetteer().best(raw_label, threshold=threshold)
    if hit:
        return hit["display_name"], hit["merchant_id"], int(hit["score"])
    return None, None, None
