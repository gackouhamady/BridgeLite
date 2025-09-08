"""
app/merchants.py

Gazetteer-based merchant resolution for BridgeLite.

This module provides:
- A text normalizer for noisy banking labels.
- A `Gazetteer` class that loads alias→merchant mappings from CSV and
  performs fuzzy matching with RapidFuzz to find the most likely merchant.
- A small Typer sub-CLI so you can try it from the terminal:
    bridge merchants match "CB CARREFOUR 75 PARIS TPE1245" --threshold 86
    bridge merchants eval --threshold 86

Design notes:
- We normalize text (lowercase, strip accents, remove punctuation and common
  payment boilerplate) so fuzzy matching is more robust.
- We use `fuzz.token_set_ratio` which handles token reorder/duplication well.
- Decision is based on a single `threshold` integer (0–100). If the best
  candidate score ≥ threshold, we accept it; otherwise we return None.
"""

from __future__ import annotations
from pathlib import Path
import re
import unicodedata
from typing import List, Tuple, Optional, Dict

import pandas as pd
from rapidfuzz import process, fuzz
import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(no_args_is_help=True)

# Default path to the gazetteer CSV. You can override by passing a path to Gazetteer().
DATA_PATH = Path("data/gazetteer_merchants.csv")

# Precompiled regexes for fast normalization.
_WS = re.compile(r"\s+")
_NONWORD = re.compile(r"[^a-z0-9 ]+")


def normalize_text(s: str) -> str:
    """
    Normalize noisy payment labels and merchant aliases for better fuzzy matching.

    Steps:
    1) Unicode NFKD → remove diacritics (é→e).
    2) Lowercase.
    3) Keep only letters, digits and spaces.
    4) Collapse repeated whitespace and trim.
    5) Remove frequent payment boilerplate tokens (CB, TPE, etc.).

    Parameters
    ----------
    s : str
        Raw input label or alias.

    Returns
    -------
    str
        Normalized text (may be empty if input is empty/noisy).
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = _NONWORD.sub(" ", s)
    s = _WS.sub(" ", s).strip()

    # Remove typical bank/payment boilerplate you don't want to match on.
    stop_bits = {
        "cb", "tpe", "pdv", "visa", "mastercard", "transaction",
        "paiement", "sepa", "prlv", "vir", "paylib", "applepay"
    }
    tokens = [t for t in s.split() if t not in stop_bits]
    return " ".join(tokens) if tokens else s


class Gazetteer:
    """
    A simple alias→merchant resolver backed by a CSV file and RapidFuzz.

    CSV columns (required):
      - alias         : textual variant seen in bank statements (typos, spacing, etc.)
      - merchant_id   : canonical identifier (stable key in your system)
      - display_name  : nice human-readable merchant name

    Attributes
    ----------
    alias_list : list[str]
        Pre-normalized list of aliases used as the search corpus.
    meta : list[tuple[str, str, str]]
        Parallel metadata for each alias: (alias_norm, merchant_id, display_name).
    """

    def __init__(self, csv_path: Path = DATA_PATH):
        """
        Load and preprocess the gazetteer from a CSV file.

        Parameters
        ----------
        csv_path : Path, optional
            Path to the CSV (defaults to data/gazetteer_merchants.csv).

        Raises
        ------
        FileNotFoundError
            If the CSV cannot be found.
        ValueError
            If required columns are missing.
        """
        if not csv_path.exists():
            raise FileNotFoundError(f"Gazetteer CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        for col in ("alias", "merchant_id", "display_name"):
            if col not in df.columns:
                raise ValueError(f"CSV missing column: {col}")

        # Normalize once at load time for speed.
        df["alias_norm"] = df["alias"].astype(str).map(normalize_text)
        df = df[df["alias_norm"].str.len() > 0].copy()

        # Prepare lookup lists.
        self.alias_list: List[str] = df["alias_norm"].tolist()
        self.meta: List[Tuple[str, str, str]] = list(
            zip(df["alias_norm"], df["merchant_id"], df["display_name"])
        )

    def lookup(self, raw_label: str, k: int = 5) -> List[Dict[str, object]]:
        """
        Return the top-k candidate merchants for a raw bank label.

        Parameters
        ----------
        raw_label : str
            A raw transaction label from a bank statement.
        k : int, default=5
            Number of candidates to return.

        Returns
        -------
        list[dict]
            Each dict contains: merchant_id, display_name, alias_norm, score (0–100).
        """
        q = normalize_text(raw_label)
        if not q:
            return []

        matches = process.extract(
            q,
            self.alias_list,
            scorer=fuzz.token_set_ratio,  # robust to token order and duplicates
            limit=k,
        )
        out: List[Dict[str, object]] = []
        for alias_norm, score, idx in matches:
            _, merchant_id, display_name = self.meta[idx]
            out.append(
                {
                    "merchant_id": merchant_id,
                    "display_name": display_name,
                    "alias_norm": alias_norm,
                    "score": int(score),
                }
            )
        return out

    def best(self, raw_label: str, threshold: int = 86) -> Optional[Dict[str, object]]:
        """
        Return the single best merchant candidate if it clears the threshold.

        Parameters
        ----------
        raw_label : str
            Raw bank label to resolve.
        threshold : int, default=86
            Minimum RapidFuzz score to accept (0–100).

        Returns
        -------
        dict | None
            Best candidate dict if score ≥ threshold; otherwise None.
        """
        candidates = self.lookup(raw_label, k=5)
        if not candidates:
            return None
        top = candidates[0]
        return top if top["score"] >= threshold else None


# -------------------------
# Typer sub-commands
# -------------------------

@app.command()
def match(label: str, threshold: int = 86, k: int = 5):
    """
    CLI helper to try the gazetteer on a single label.

    Examples
    --------
    bridge merchants match "CB CARREFOUR 75 PARIS TPE1245" --threshold 86
    """
    g = Gazetteer(DATA_PATH)
    table = Table(title=f'Match: "{label}" (threshold {threshold})')
    table.add_column("merchant_id")
    table.add_column("display_name")
    table.add_column("score", justify="right")
    table.add_column("alias_norm")
    for row in g.lookup(label, k=k):
        table.add_row(row["merchant_id"], row["display_name"], str(row["score"]), row["alias_norm"])
    console.print(table)

    best = g.best(label, threshold=threshold)
    if best:
        console.print(f"[green]BEST ≥ {threshold}[/] → {best}")
    else:
        console.print(f"[yellow]No candidate ≥ {threshold}[/]")


@app.command()
def eval(pairs_csv: Path = Path("data/merchant_eval_pairs.csv"), threshold: int = 86):
    """
    Measure Accuracy@1 on a small labeled pairs CSV.

    CSV format
    ----------
    raw_label,merchant_id
    "CB CARREFOUR 75 PARIS TPE1245",carrefour

    Parameters
    ----------
    pairs_csv : Path, default="data/merchant_eval_pairs.csv"
        CSV file containing raw_label and expected merchant_id.
    threshold : int, default=86
        Acceptance threshold for `best()`.

    Prints
    ------
    Accuracy@1 summary like: "Accuracy@1 (threshold=86): 0.900 (9/10)"
    """
    if not pairs_csv.exists():
        typer.secho(f"Pairs file not found: {pairs_csv}", fg=typer.colors.RED)
        raise typer.Exit(1)

    g = Gazetteer(DATA_PATH)
    df = pd.read_csv(pairs_csv)
    ok = 0
    total = 0
    for _, r in df.iterrows():
        total += 1
        pred = g.best(str(r["raw_label"]), threshold=threshold)
        if pred and pred["merchant_id"] == str(r["merchant_id"]):
            ok += 1
    acc = ok / total if total else 0.0
    console.print(f"Accuracy@1 (threshold={threshold}): [bold]{acc:.3f}[/] ({ok}/{total})")
