# app/preproc.py
"""
BridgeLite — Text Preprocessing
===============================

Purpose
-------
Preprocess raw banking labels so the rest of the pipeline (rules, gazetteer, ML model)
receives clean and consistent inputs. The module also masks sensitive patterns such as
IBAN and RUM that might appear in free text, and infers the operation type
(card / direct debit / bank transfer).

Design goals
------------
1) **Normalization**: lowercase, de-accent (ASCII), collapse whitespace.
2) **PII masking**: replace IBAN/RUM-like substrings with neutral tokens so they do not leak
   into logs, metrics, or model features.
3) **Operation type detection**: derive from explicit channel when provided, otherwise
   infer from the label prefix (e.g., "CB ", "PRLV ", "VIR ").
4) **Lightweight tokenization**: simple alphanumeric token stream suitable for TF-IDF.

Public API
----------
- `normalize_text(s: Optional[str]) -> str`
- `mask_pii(s: str) -> tuple[str, dict[str, list[str]]]`
- `detect_operation_type(raw_label: Optional[str], explicit_channel: Optional[str]) -> str`
- `tokenize(s: str) -> list[str]`
- `preprocess_record(tx: dict) -> PreprocResult`

Notes
-----
- Uses `regex` (a drop-in replacement for `re`) and `unidecode` for accent stripping.
- All docstrings are written in English; **keywords in labels remain French** (e.g., "PRLV", "FACTURE"),
  because the synthetic dataset mimics French banking statements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import regex as re
from unidecode import unidecode


# ---------------------------------------------------------------------------
# PII-like patterns (approximate on purpose; good enough for detection/masking)
# ---------------------------------------------------------------------------

# IBAN: country code (2 letters) + 2 digits + 11..30 alphanumeric chars.
# We detect in UPPERCASE; the dataset often contains "IBAN FR...".
RE_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")

# RUM (SEPA mandate reference): prefix RUM + 8+ alphanum/- chars. Case-insensitive.
RE_RUM = re.compile(r"\bRUM[\w-]{8,}\b", re.IGNORECASE)

# Simple tokenizer: alphanumeric words after normalization
RE_TOKEN = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------
# Operation types
# ---------------------------------------------------------------------------

OperationType = str  # Literal["card", "direct_debit", "transfer", "unknown"]

OP_CARD = "card"
OP_DD = "direct_debit"
OP_TRANSFER = "transfer"
OP_UNKNOWN = "unknown"


@dataclass
class PreprocResult:
    """
    Container object returned by `preprocess_record`.

    Attributes
    ----------
    raw : str
        Original raw label as provided in the transaction.
    normalized : str
        Lowercased, de-accented (ASCII), whitespace-collapsed version of `raw`.
    masked : str
        Same as `normalized` but with IBAN/RUM substrings replaced by neutral tokens.
    tokens : list[str]
        Alphanumeric tokens extracted from `masked`; suitable input for TF-IDF.
    has_iban : bool
        Whether an IBAN-like pattern was detected in the original label.
    has_rum : bool
        Whether a RUM-like pattern was detected in the original label.
    operation_type : OperationType
        One of: "card" | "direct_debit" | "transfer" | "unknown".
    """

    raw: str
    normalized: str
    masked: str
    tokens: List[str]
    has_iban: bool
    has_rum: bool
    operation_type: OperationType


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def normalize_text(s: Optional[str]) -> str:
    """
    Normalize a string for downstream processing.

    Steps
    -----
    1) Handle `None` as empty string; strip leading/trailing whitespace.
    2) Lowercase, de-accent to ASCII via `unidecode`.
    3) Collapse any run of whitespace to a single space.

    Parameters
    ----------
    s : Optional[str]
        Input text (can be None).

    Returns
    -------
    str
        Normalized ASCII, lowercase, single-spaced string.

    Examples
    --------
    >>> normalize_text("  Café   ÉTÉ  ")
    'cafe ete'
    """
    s = s or ""
    s = s.strip().lower()
    s = unidecode(s)
    s = re.sub(r"\s+", " ", s)
    return s


def mask_pii(s: str) -> Tuple[str, Dict[str, List[str]]]:
    """
    Replace IBAN/RUM substrings with neutral tokens.

    Detection is intentionally approximate yet practical. We first look for IBAN patterns
    in UPPERCASE (common in exports), and RUM patterns case-insensitively. Matches are
    replaced 1:1 in the given string.

    Parameters
    ----------
    s : str
        Input text (preferably normalized, but raw also works).

    Returns
    -------
    tuple[str, dict]
        A tuple `(masked, info)` where:
        - `masked` is the string with PII-like substrings replaced by "IBAN_TOKEN"/"RUM_TOKEN".
        - `info` is a dict `{"iban": [...], "rum": [...]}` listing the original matches.

    Examples
    --------
    >>> mask_pii("PRLV EDF RUM12345678-4321 IBAN FR7612345678901234567890123")
    ('prlv edf RUM_TOKEN IBAN IBAN_TOKEN', {'iban': ['FR7612345678901234567890123'], 'rum': ['RUM12345678-4321']})
    """
    # Detect with the same case strategy used in training labels
    found_iban = RE_IBAN.findall(s.upper())
    found_rum = RE_RUM.findall(s)

    masked = s
    for ib in set(found_iban):
        masked = masked.replace(ib, "IBAN_TOKEN")
    for rm in set(found_rum):
        masked = masked.replace(rm, "RUM_TOKEN")

    return masked, {"iban": found_iban, "rum": found_rum}


def detect_operation_type(
    raw_label: Optional[str],
    explicit_channel: Optional[str] = None,
) -> OperationType:
    """
    Infer the operation type using the explicit channel when present,
    otherwise fallback to the label prefix.

    Rules
    -----
    - If `explicit_channel` is provided (case-insensitive):
        - "CB"   → "card"
        - "PRLV" → "direct_debit"
        - "VIR"  → "transfer"
    - Else, inspect the start of `raw_label` (uppercased) for the same prefixes.

    Parameters
    ----------
    raw_label : Optional[str]
        Raw label string.
    explicit_channel : Optional[str]
        Channel provided in the input transaction, if any.

    Returns
    -------
    OperationType
        "card" | "direct_debit" | "transfer" | "unknown"

    Examples
    --------
    >>> detect_operation_type("CB CARREFOUR 75 PARIS")
    'card'
    >>> detect_operation_type("PRLV EDF FACTURE 07/2025")
    'direct_debit'
    >>> detect_operation_type("VIR SALAIRE ACME")
    'transfer'
    """
    ch = (explicit_channel or "").strip().upper()
    if ch == "CB":
        return OP_CARD
    if ch == "PRLV":
        return OP_DD
    if ch == "VIR":
        return OP_TRANSFER

    s = (raw_label or "").strip().upper()
    if s.startswith("CB "):
        return OP_CARD
    if s.startswith("PRLV "):
        return OP_DD
    if s.startswith("VIR "):
        return OP_TRANSFER

    return OP_UNKNOWN


def tokenize(s: str) -> List[str]:
    """
    Extract a simple alphanumeric token stream from a normalized/masked string.

    Intended for lightweight vectorizers (e.g., TF-IDF; though the project mainly uses
    character n-grams, keeping tokens is useful for rules and gazetteer lookups).

    Parameters
    ----------
    s : str
        Input text (prefer `masked` produced by `mask_pii`).

    Returns
    -------
    list[str]
        Alphanumeric tokens in order of appearance.

    Examples
    --------
    >>> tokenize("cb carrefour 75 paris tpe1245")
    ['cb', 'carrefour', '75', 'paris', 'tpe1245']
    """
    return RE_TOKEN.findall(s)


def preprocess_record(tx: Dict) -> PreprocResult:
    """
    Preprocess a single transaction record.

    Steps
    -----
    1) Read `raw_label` and `channel` from the dict.
    2) Normalize `raw_label`.
    3) Mask PII-like substrings in the normalized string.
    4) Detect operation type (uses explicit channel first, then label prefix).
    5) Tokenize the masked string.

    Parameters
    ----------
    tx : dict
        Example: {"raw_label": "...", "channel": "CB", "mcc": 5411, ...}

    Returns
    -------
    PreprocResult
        Rich object with normalized/masked text, tokens, PII flags, and operation type.

    Examples
    --------
    >>> tx = {"raw_label": "PRLV EDF FACTURE 07/2025 RUM12345678-4321"}
    >>> pre = preprocess_record(tx)
    >>> pre.operation_type
    'direct_debit'
    >>> pre.has_rum
    True
    """
    raw = tx.get("raw_label", "") or ""
    normalized = normalize_text(raw)
    masked, info = mask_pii(normalized)
    op_type = detect_operation_type(raw_label=raw, explicit_channel=tx.get("channel"))
    toks = tokenize(masked)

    return PreprocResult(
        raw=raw,
        normalized=normalized,
        masked=masked,
        tokens=toks,
        has_iban=bool(info["iban"]),
        has_rum=bool(info["rum"]),
        operation_type=op_type,
    )
