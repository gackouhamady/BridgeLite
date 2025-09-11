"""
Builds a fast-load gazetteer artifact from CSV.

Input  : data/gazetteer_merchants.csv
         columns -> alias, merchant_id, display_name
Output : app/gazetteer.pkl
         dict with:
           - aliases_lower: list[str]
           - merchant_ids : list[str]
           - display_names: list[str]
           - alias_to_index: dict[str,int]  (first occurrence kept)
           - meta
"""
from __future__ import annotations
import argparse, joblib, pandas as pd
from pathlib import Path

def build(csv_path: str, out_pkl: str) -> None:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Missing gazetteer CSV: {p}")
    df = pd.read_csv(p)
    need = {"alias","merchant_id","display_name"}
    if not need.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {need}")

    # Normalize
    aliases = df["alias"].astype(str).str.strip()
    aliases_lower = aliases.str.lower().tolist()
    merchant_ids = df["merchant_id"].astype(str).tolist()
    display = df["display_name"].astype(str).tolist()

    # First occurrence index map (alias → index)
    alias_to_index = {}
    for i, a in enumerate(aliases_lower):
        alias_to_index.setdefault(a, i)

    bundle = {
        "aliases_lower": aliases_lower,
        "merchant_ids": merchant_ids,
        "display_names": display,
        "alias_to_index": alias_to_index,
        "meta": {
            "source_csv": str(p),
            "n_rows": len(df),
        },
    }
    Path(out_pkl).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_pkl)
    print(f"Wrote {out_pkl} (rows={len(df)})")

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/gazetteer_merchants.csv")
    ap.add_argument("--out", default="app/gazetteer.pkl")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse()
    build(args.csv, args.out)
