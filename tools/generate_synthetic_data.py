# tools/generate_synthetic_data.py
"""
BridgeLite – Synthetic Data Generator

This script creates realistic, *model-friendly* mock data for banking transactions,
plus a drifted “production” split to test drift detection pipelines.

It generates up to four artifacts:

1) data/transactions_mock.csv       ← training-like dataset
2) data/production_sim.csv          ← drifted dataset (text noise + prior shift)
3) data/gazetteer_merchants.csv     ← alias → merchant_id, display_name  (for fuzzy match)
4) data/taxonomy_categories.json    ← category → seed merchant names     (for docs/rules)

CSV schema (both train and prod):
    tx_id,date,amount,currency,raw_label,channel,mcc,category_label,merchant_label

Why this fits the next stages
-----------------------------
- **Preprocessing/Rules (H2):** labels include realistic tokens (CB/PRLV/VIR, MCC-like numbers,
  IBAN/RUM fragments) to validate regexes and rule extraction.
- **Gazetteer (H3):** a small but rich merchant set with multiple alias patterns.
- **Main model (H4):** char TF-IDF + XGBoost/RF works well on these noisy strings.
- **Drift (H7):** the production split introduces category-prior shift and string perturbations
  (spacing, token insertions, typos) so PSI/KS can flag changes.
- **API & Metrics:** magnitudes and distributions are realistic enough for latency/coverage tests.

Usage
-----
CLI (default paths):
    python tools/generate_synthetic_data.py --n 6000

Custom output + stronger drift:
    python tools/generate_synthetic_data.py --n 12000 --drift-intensity 0.25 --seed 7

Programmatic:
    from tools.generate_synthetic_data import generate
    files = generate(n=8000, drift_intensity=0.2)
    print(files)

Notes
-----
- Amounts are **negative** for expenses and **positive** for salaries.
- Channels are CB (card), PRLV (direct debit), VIR (bank transfer).
- “Production” split injects drift in both *text* and *label distribution*.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple


# -----------------------------
# Domain configuration
# -----------------------------

CURRENCY_DEFAULT = "EUR"

#: Category priors for the **training** split. Sum ≈ 1.0.
CATEGORY_PRIORS: Dict[str, float] = {
    "Groceries": 0.34,
    "Restaurants": 0.20,
    "Transport": 0.14,
    "Utilities": 0.12,
    "Rent": 0.07,
    "Salary": 0.13,  # incomes (positive amounts)
}

#: Representative MCC codes per category (used by rules / feature hints).
MCC_BY_CATEGORY: Dict[str, int] = {
    "Groceries": 5411,
    "Restaurants": 5814,
    "Transport": 4111,
    "Utilities": 4900,
    "Rent": 6513,
    "Salary": 6011,
}

#: Absolute amount ranges (sign applied later per category).
AMOUNT_RANGES: Dict[str, Tuple[float, float]] = {
    "Groceries": (8, 120),
    "Restaurants": (6, 60),
    "Transport": (2, 50),
    "Utilities": (20, 140),
    "Rent": (500, 1500),
    "Salary": (1500, 3200),
}

#: Channels and their per-category priors (informative but not enforced).
CHANNELS = ["CB", "PRLV", "VIR"]
CHANNEL_PRIORS: Dict[str, Dict[str, float]] = {
    "Groceries": {"CB": 0.90, "PRLV": 0.05, "VIR": 0.05},
    "Restaurants": {"CB": 0.95, "PRLV": 0.02, "VIR": 0.03},
    "Transport": {"CB": 0.70, "PRLV": 0.15, "VIR": 0.15},
    "Utilities": {"CB": 0.05, "PRLV": 0.90, "VIR": 0.05},
    "Rent": {"CB": 0.02, "PRLV": 0.96, "VIR": 0.02},
    "Salary": {"CB": 0.00, "PRLV": 0.00, "VIR": 1.00},
}

#: Merchant seeds with label templates (drives realistic, variable raw labels).
#: Each entry: category -> List[(merchant_display_name, [templates...])]
MERCHANTS: Dict[str, List[Tuple[str, List[str]]]] = {
    "Groceries": [
        ("Carrefour", ["CB CARREFOUR {city_code} {city}", "CARREFOUR CITY {city}", "CB CARREFOUR"]),
        ("Auchan", ["CB AUCHAN", "AUCHAN SUPERMARCHE", "CB AUCHAN {city_code}"]),
        ("Intermarché", ["CB INTERMARCHE", "INTERMARCHE {city_code}", "CB INTERMARCHE {city}"]),
        ("Monoprix", ["CB MONOPRIX", "MONOPRIX {city}", "CB MONOPRIX {city_code}"]),
        ("Casino", ["CB CASINO SUPERMARCHE", "CASINO {city}"]),
    ],
    "Restaurants": [
        ("McDonald's", ["CB MCDO {city}", "MCDONALD {city}", "CB MCDONALD"]),
        ("KFC", ["CB KFC", "KFC {city_code}", "CB KFC {city}"]),
        ("Burger King", ["CB BURGER KING", "BK {city}"]),
        ("Starbucks", ["CB STARBUCKS {city}", "STARBUCKS {city_code}"]),
        ("Quick", ["CB QUICK {city}", "QUICK RESTO {city_code}"]),
    ],
    "Transport": [
        ("RATP", ["PRLV RATP", "RATP PASS NAVIGO", "PRLV RATP {month}"]),
        ("SNCF", ["CB SNCF BILLET", "SNCF TGV {city_code}", "CB SNCF {city}"]),
        ("Uber", ["UBER *TRIP {city}", "CB UBER BV"]),
        ("Bolt", ["BOLT RIDE {city}", "CB BOLT EU"]),
        ("TotalEnergies", ["CB TOTAL {city_code}", "TOTAL STATION {city}"]),
    ],
    "Utilities": [
        ("EDF", ["PRLV EDF FACTURE {month}/{year}", "EDF FACTURE {month}/{year}"]),
        ("Engie", ["PRLV ENGIE FACTURE {month}/{year}", "ENGIE FACTURE {month}/{year}"]),
        ("Orange", ["PRLV ORANGE FACTURE {month}/{year}"]),
        ("SFR", ["PRLV SFR FACTURE {month}/{year}"]),
        ("Free", ["PRLV FREE FACTURE {month}/{year}"]),
    ],
    "Rent": [
        ("Foncia", ["PRLV FONCIA LOYER {month}/{year}", "LOYER FONCIA {month}/{year}"]),
        ("Nexity", ["PRLV NEXITY LOYER {month}/{year}", "LOYER NEXITY {month}/{year}"]),
    ],
    "Salary": [
        ("ACME", ["VIR SALAIRE ACME {month}/{year}", "VIR PAYROLL ACME {month}/{year}"]),
        ("Globex", ["VIR SALAIRE GLOBEX {month}/{year}"]),
        ("Initech", ["VIR SALAIRE INITECH {month}/{year}"]),
    ],
}

#: Cities / codes to inject variability in labels.
CITY_CODES = ["75", "92", "93", "94", "69", "13", "31", "33", "59", "67"]
CITIES = [
    "PARIS",
    "LA DEFENSE",
    "AUBERVILLIERS",
    "CRETEIL",
    "LYON",
    "MARSEILLE",
    "TOULOUSE",
    "BORDEAUX",
    "LILLE",
    "STRASBOURG",
]


# -----------------------------
# Utilities
# -----------------------------

def _weighted_choice(weights: Dict[str, float]) -> str:
    """
    Draw one key from a dict of {key: weight} using proportional sampling.

    Parameters
    ----------
    weights : Dict[str, float]
        Keys with non-negative weights. Sum does not need to be 1.

    Returns
    -------
    str
        A sampled key.
    """
    keys = list(weights.keys())
    vals = list(weights.values())
    total = sum(vals) or 1.0
    r = random.random() * total
    acc = 0.0
    for k, v in zip(keys, vals):
        acc += v
        if r <= acc:
            return k
    return keys[-1]


def _random_date(start: dt.date, end: dt.date) -> dt.date:
    """
    Sample a random date between two inclusive endpoints.

    Parameters
    ----------
    start : datetime.date
    end   : datetime.date

    Returns
    -------
    datetime.date
    """
    delta = (end - start).days
    return start + dt.timedelta(days=random.randint(0, delta))


def _month_year() -> Tuple[str, str]:
    """
    Random month/year pair as strings (MM, YYYY).

    Returns
    -------
    (str, str)
        Month '01'..'12' and year '2024'..'2026'.
    """
    m = f"{random.randint(1,12):02d}"
    y = str(random.randint(2024, 2026))
    return m, y


def _render_label(template: str) -> str:
    """
    Fill a label template with city and date tokens, plus optional terminal artifacts.

    Parameters
    ----------
    template : str
        Template containing placeholders: {city}, {city_code}, {month}, {year}

    Returns
    -------
    str
        A rendered raw label string.
    """
    month, year = _month_year()
    city_code = random.choice(CITY_CODES)
    city = random.choice(CITIES)
    s = template.format(month=month, year=year, city_code=city_code, city=city)
    if random.random() < 0.08:
        s += f" TPE{random.randint(1000, 9999)}"
    return s


def _amount_for_category(cat: str) -> float:
    """
    Sample a transaction amount (with correct sign) for a category.

    Parameters
    ----------
    cat : str
        Category name.

    Returns
    -------
    float
        Positive for salaries, negative for expenses.
    """
    low, high = AMOUNT_RANGES[cat]
    amt = random.uniform(low, high)
    sign = 1.0 if cat == "Salary" else -1.0
    return round(sign * amt, 2)


def _inject_text_noise(label: str, intensity: float) -> str:
    """
    Apply text perturbations to simulate drift (spacing, extra tokens, typos).

    Parameters
    ----------
    label : str
        Original raw label.
    intensity : float
        0..1 magnitude. Higher = more/noisier edits.

    Returns
    -------
    str
        Noisy label.
    """
    s = label
    if random.random() < intensity:
        s = s.replace("CB ", "CB TPE ")
    if random.random() < intensity * 0.7:
        s = s.replace("SNCF", "S N C F")
    if random.random() < intensity * 0.6:
        s = re.sub(r"\bPRLV\b", "PRLV AUTO", s)
    if random.random() < intensity * 0.3:
        s = s.replace("CARREFOUR", random.choice(["CARREFORU", "CARRREFOUR", "CARREF0UR"]))
    if random.random() < intensity * 0.2:
        s += f" REF{random.randint(100000,999999)}"
    return s


def _maybe_iban_rum(label: str) -> str:
    """
    Occasionally append IBAN/RUM-like tokens to direct debit or transfer strings.

    Parameters
    ----------
    label : str

    Returns
    -------
    str
        Label with optional IBAN or RUM fragments.
    """
    if label.startswith("PRLV ") and random.random() < 0.10:
        label += f" RUM{random.randint(10000000, 99999999)}-{random.randint(1000,9999)}"
    if label.startswith("VIR ") and random.random() < 0.05:
        # Not a real IBAN, but realistic enough for regex tests
        label += f" IBAN FR{random.randint(10,99)}{random.randint(10**9,10**10-1)}{random.randint(10**9,10**10-1)}"
    return label


# -----------------------------
# Row assembly
# -----------------------------

def _make_row(i: int, date: dt.date, cat: str, currency: str) -> Dict[str, str]:
    """
    Compose one transaction row with a merchant, a rendered label, and coherent fields.

    Parameters
    ----------
    i : int
        Row index (used to build a tx_id).
    date : datetime.date
        Transaction date.
    cat : str
        Category name.
    currency : str
        ISO-like currency code (e.g., "EUR").

    Returns
    -------
    Dict[str, str]
        Populated CSV row (all values stringified for easy writing).
    """
    merch_name, templates = random.choice(MERCHANTS[cat])
    label = _render_label(random.choice(templates))
    label = _maybe_iban_rum(label)

    # Try to keep channel consistent with prefix, else sample from priors
    if label.startswith("CB "):
        channel = "CB"
    elif label.startswith("PRLV "):
        channel = "PRLV"
    elif label.startswith("VIR "):
        channel = "VIR"
    else:
        channel = _weighted_choice(CHANNEL_PRIORS[cat])

    mcc = MCC_BY_CATEGORY.get(cat, "")
    amount = _amount_for_category(cat)

    return {
        "tx_id": f"tx_{i:07d}",
        "date": str(date),
        "amount": f"{amount:.2f}",
        "currency": currency,
        "raw_label": label,
        "channel": channel,
        "mcc": mcc,
        "category_label": cat,
        "merchant_label": merch_name if cat != "Salary" else "",
    }


def _shift_priors(base: Dict[str, float], drift_intensity: float) -> Dict[str, float]:
    """
    Shift category priors to create a *production* distribution different from training.

    Strategy: move probability mass from Groceries/Restaurants to Transport/Utilities
    to emulate seasonality or behavior changes.

    Parameters
    ----------
    base : Dict[str, float]
        Original priors (sum ≈ 1.0).
    drift_intensity : float
        0..1 scalar controlling the magnitude of the shift.

    Returns
    -------
    Dict[str, float]
        New priors normalized to sum to 1.0.
    """
    shift = drift_intensity * 0.5
    p = base.copy()

    for src in ["Groceries", "Restaurants"]:
        p[src] = max(0.01, p[src] - shift * p[src])

    add_mass = (base.get("Groceries", 0.0) + base.get("Restaurants", 0.0)) * shift
    for dst in ["Transport", "Utilities"]:
        p[dst] = p.get(dst, 0.0) + add_mass * 0.5

    s = sum(p.values()) or 1.0
    return {k: v / s for k, v in p.items()}


# -----------------------------
# Writers
# -----------------------------

def _write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    """
    Write a list of dict rows to CSV with the expected schema.

    Parameters
    ----------
    path : pathlib.Path
        Output file path.
    rows : List[Dict[str, str]]
        Rows to write (keys must match the schema).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tx_id",
                "date",
                "amount",
                "currency",
                "raw_label",
                "channel",
                "mcc",
                "category_label",
                "merchant_label",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_gazetteer(path: Path) -> None:
    """
    Create a small gazetteer (alias → merchant_id, display_name).

    Heuristic:
      - Derive aliases from templates by removing prefixes and placeholders.
      - Add common short aliases (e.g., 'mcdo', 'mcdonald', 'total').

    Parameters
    ----------
    path : pathlib.Path
        Output CSV path.
    """
    entries: List[Tuple[str, str, str]] = []
    for _cat, pairs in MERCHANTS.items():
        for merchant, templates in pairs:
            aliases = set()
            for t in templates:
                a = (
                    t.replace("CB ", "")
                    .replace("PRLV ", "")
                    .replace("VIR ", "")
                    .replace("{city}", "")
                    .replace("{city_code}", "")
                    .replace("{month}", "")
                    .replace("{year}", "")
                    .strip()
                    .lower()
                )
                a = re.sub(r"\s+", " ", a).strip()
                if a:
                    aliases.add(a)

            base = merchant.lower().replace("'", "")
            aliases.add(base)
            if merchant.lower().startswith("mc"):
                aliases.add("mcdo")
                aliases.add("mcdonald")
            if merchant.lower() == "totalenergies":
                aliases.add("total")

            for alias in sorted(aliases):
                entries.append((alias, base, merchant))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["alias", "merchant_id", "display_name"])
        w.writerows(entries)


def _write_taxonomy(path: Path) -> None:
    """
    Save a minimal taxonomy mapping: category → seed merchant list.

    Parameters
    ----------
    path : pathlib.Path
        Output JSON path.
    """
    tax = {cat: [m for (m, _templates) in MERCHANTS[cat]] for cat in MERCHANTS.keys()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tax, indent=2, ensure_ascii=False), encoding="utf-8")


# -----------------------------
# Public API
# -----------------------------

def generate(
    n: int = 6000,
    out_train: str = "data/transactions_mock.csv",
    out_prod: str = "data/production_sim.csv",
    currency: str = CURRENCY_DEFAULT,
    drift_intensity: float = 0.15,
    seed: int = 42,
    write_gazetteer: bool = True,
    write_taxonomy: bool = True,
    start_date: str = "2025-01-01",
    end_date: str = "2025-08-31",
) -> Dict[str, str]:
    """
    Generate training and *drifted* production datasets, plus optional gazetteer/taxonomy.

    Parameters
    ----------
    n : int, default=6000
        Number of rows to generate for **each** split (train and production).
    out_train : str
        Output CSV path for the training-like dataset.
    out_prod : str
        Output CSV path for the drifted production dataset.
    currency : str
        Currency code (e.g., "EUR").
    drift_intensity : float, default=0.15
        0..1 magnitude controlling:
          - label text noise (insertions, spacing, typos),
          - categorical prior shift for production.
    seed : int, default=42
        Random seed (reproducibility).
    write_gazetteer : bool, default=True
        If True, also write `data/gazetteer_merchants.csv`.
    write_taxonomy : bool, default=True
        If True, also write `data/taxonomy_categories.json`.
    start_date : str, default="2025-01-01"
        Lower bound for random dates (inclusive).
    end_date : str, default="2025-08-31"
        Upper bound for random dates (inclusive).

    Returns
    -------
    Dict[str, str]
        Paths of created files, keyed by {"train_csv","prod_csv","gazetteer_csv","taxonomy_json"}.

    Examples
    --------
    >>> from tools.generate_synthetic_data import generate
    >>> paths = generate(n=2000, drift_intensity=0.2, seed=123)
    >>> print(paths["train_csv"])
    data/transactions_mock.csv
    """
    random.seed(seed)

    start = dt.date.fromisoformat(start_date)
    end = dt.date.fromisoformat(end_date)

    # Training distribution
    rows_train: List[Dict[str, str]] = []
    for i in range(n):
        cat = _weighted_choice(CATEGORY_PRIORS)
        date = _random_date(start, end)
        rows_train.append(_make_row(i, date, cat, currency))

    # Production: shift priors and inject text noise
    prod_priors = _shift_priors(CATEGORY_PRIORS, max(0.0, min(1.0, drift_intensity)))
    rows_prod: List[Dict[str, str]] = []
    for i in range(n):
        cat = _weighted_choice(prod_priors)
        date = _random_date(start, end)
        row = _make_row(i, date, cat, currency)
        row["raw_label"] = _inject_text_noise(row["raw_label"], drift_intensity)
        # Occasionally hide merchant (long-tail/unseen)
        if random.random() < max(0.02, drift_intensity * 0.2):
            row["merchant_label"] = ""
        rows_prod.append(row)

    # Write files
    out_paths: Dict[str, str] = {}
    p_train = Path(out_train)
    p_prod = Path(out_prod)
    _write_csv(p_train, rows_train)
    _write_csv(p_prod, rows_prod)
    out_paths["train_csv"] = str(p_train)
    out_paths["prod_csv"] = str(p_prod)

    if write_gazetteer:
        p_gaz = Path("data/gazetteer_merchants.csv")
        _write_gazetteer(p_gaz)
        out_paths["gazetteer_csv"] = str(p_gaz)

    if write_taxonomy:
        p_tax = Path("data/taxonomy_categories.json")
        _write_taxonomy(p_tax)
        out_paths["taxonomy_json"] = str(p_tax)

    return out_paths


# -----------------------------
# CLI
# -----------------------------

def _parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for the generator.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with defaults.
    """
    ap = argparse.ArgumentParser(
        description="Generate synthetic banking transactions for BridgeLite."
    )
    ap.add_argument(
        "--n",
        type=int,
        default=6000,
        help="Rows for each dataset (train and production).",
    )
    ap.add_argument(
        "--out-train",
        type=str,
        default="data/transactions_mock.csv",
        help="Output CSV for training-like split.",
    )
    ap.add_argument(
        "--out-prod",
        type=str,
        default="data/production_sim.csv",
        help="Output CSV for drifted production split.",
    )
    ap.add_argument(
        "--currency",
        type=str,
        default=CURRENCY_DEFAULT,
        help='Currency code (e.g., "EUR").',
    )
    ap.add_argument(
        "--drift-intensity",
        type=float,
        default=0.15,
        help="0..1 magnitude for text noise + prior shift (production).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    ap.add_argument(
        "--no-gazetteer",
        action="store_true",
        help="Do not write data/gazetteer_merchants.csv",
    )
    ap.add_argument(
        "--no-taxonomy",
        action="store_true",
        help="Do not write data/taxonomy_categories.json",
    )
    ap.add_argument(
        "--start-date",
        type=str,
        default="2025-01-01",
        help="Start date YYYY-MM-DD (inclusive).",
    )
    ap.add_argument(
        "--end-date",
        type=str,
        default="2025-08-31",
        help="End date YYYY-MM-DD (inclusive).",
    )
    return ap.parse_args()


def main() -> None:
    """
    Entry-point for CLI execution.

    Prints a small JSON summary of all generated file paths so the script can be
    used in CI logs or external runners.
    """
    args = _parse_args()
    paths = generate(
        n=args.n,
        out_train=args.out_train,
        out_prod=args.out_prod,
        currency=args.currency,
        drift_intensity=max(0.0, min(1.0, args.drift_intensity)),
        seed=args.seed,
        write_gazetteer=not args.no_gazetteer,
        write_taxonomy=not args.no_taxonomy,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(json.dumps({"ok": True, "files": paths}, indent=2))


if __name__ == "__main__":
    main()
