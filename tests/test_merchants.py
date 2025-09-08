"""
tests/test_merchants.py

Minimal unit tests for the Gazetteer matcher.

What this test checks:
- A tiny, in-memory CSV with a couple of aliases is enough to construct a
  `Gazetteer`.
- Given a realistic raw label (with noise like "CB ... TPE..."), the matcher
  still finds the correct merchant when the threshold is reasonable.
"""

from pathlib import Path
import pandas as pd
from app.merchants import Gazetteer


def test_gazetteer_basic(tmp_path: Path):
    """
    Create a toy gazetteer CSV and verify that a noisy label maps to the
    expected merchant using `Gazetteer.best()`.

    The threshold is set low (70) so the test remains stable even if
    RapidFuzz scoring nudges slightly across versions.
    """
    csv_path = tmp_path / "gaz.csv"
    pd.DataFrame(
        [
            {"alias": "carrefour", "merchant_id": "carrefour", "display_name": "Carrefour"},
            {"alias": "mcdo", "merchant_id": "mcdonalds", "display_name": "McDonald's"},
        ]
    ).to_csv(csv_path, index=False)

    g = Gazetteer(csv_path)
    best = g.best("CB CARREFOUR 75 PARIS TPE1245", threshold=70)

    assert best is not None, "Expected at least one match"
    assert best["merchant_id"] == "carrefour", f"Wrong merchant: {best}"
