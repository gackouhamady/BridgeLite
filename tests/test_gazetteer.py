# tests/test_gazetteer.py
from __future__ import annotations
import pandas as pd
from pathlib import Path
from app.service import Gazetteer

def test_gazetteer_lookup_tmp_csv(tmp_path: Path):
    csv = tmp_path / "gaz.csv"
    pd.DataFrame(
        {
            "alias": ["carrefour", "mcdo", "edf"],
            "merchant_id": ["m1", "m2", "m3"],
            "display_name": ["Carrefour", "McDonald's", "EDF"],
        }
    ).to_csv(csv, index=False)

    g = Gazetteer.from_csv(str(csv))
    mid, disp, score, why = g.lookup("CB CARREFOUR 75 PARIS", score_cut=0.80)
    assert disp == "Carrefour"
    assert score >= 0.80
    assert "alias" in why
