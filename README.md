# BridgeLite

[![License](https://img.shields.io/badge/License-MIT-black.svg)](#license)
[![Python](https://img.shields.io/badge/Python-3.10%20|%203.11-3776AB?logo=python&logoColor=white)](#requirements)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](#api)
[![Build CI](https://img.shields.io/github/actions/workflow/status/<owner>/<repo>/ci.yml?label=CI)](.github/workflows/ci.yml)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?logo=docker&logoColor=white)](#docker)
[![Prometheus](https://img.shields.io/badge/Monitoring-Prometheus-E6522C?logo=prometheus&logoColor=white)](#monitoring--metrics)
[![Evidently](https://img.shields.io/badge/Drift-Evidently-3C3C3C)](#drift-detection)
[![Tests](https://img.shields.io/badge/Tests-Pytest-0A9EDC?logo=pytest&logoColor=white)](#tests)
![Progress](https://img.shields.io/badge/progress-0%25-blue)


> **BridgeLite — Banking transaction categorization & merchant normalization with drift detection and a real-time API.**  
> Turn raw banking labels into structured outputs: `normalized_merchant`, `operation_type`, `budget_category`, plus confidence & explanation.  
> Ships with a FastAPI endpoint, monitoring (Prometheus text), drift checks, Docker, and minimal CI.

---

## Table of Contents

- [Project Goals](#project-goals)
- [Tech Stack (icons)](#tech-stack-icons)
- [Architecture Overview](#architecture-overview)
- [Data](#data)
- [Quickstart](#quickstart)
- [API](#api)
- [Training & Evaluation](#training--evaluation)
- [Drift Detection](#drift-detection)
- [Monitoring & Metrics](#monitoring--metrics)
- [Quality Gates (Acceptance)](#quality-gates-acceptance)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Roadmap / Stretch Goals](#roadmap--stretch-goals)
- [Security & PII](#security--pii)
- [Docker](#docker)
- [Tests](#tests)
- [License](#license)

---

## Project Goals

- Transform raw banking labels into:
  - `normalized_merchant`
  - `operation_type`
  - `budget_category`
  - confidence score + human-readable explanation
- Fallback LLM for ambiguous/long-tail cases (local lightweight model or stub).
- Real-time API via FastAPI (`POST /predict`, `GET /metrics`).
- Quality monitoring (macro-F1, coverage, latency) and **drift detection**.
- Minimal CI/CD, Docker image, and Prometheus-style metrics.

---

## Tech Stack (icons)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EB5757)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/HF%20Transformers-FF6F00?logo=huggingface&logoColor=white)
![Sentence-Transformers](https://img.shields.io/badge/Sentence--Transformers-333333)
![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?logo=prometheus&logoColor=white)
![Evidently](https://img.shields.io/badge/Evidently-3C3C3C)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![Black](https://img.shields.io/badge/Black-000000)
![Ruff](https://img.shields.io/badge/Ruff-2B90D9)
![Pytest](https://img.shields.io/badge/pytest-0A9EDC?logo=pytest&logoColor=white)

> Note: PyTorch/Transformers/Sentence-Transformers are **optional** for the LLM fallback (install later if needed).

---

## Architecture Overview

- **Main model**: Char-level TF-IDF + **XGBoost** (or RandomForest) for `budget_category`.
- **Merchant normalization**: fuzzy match (RapidFuzz) + optional sentence embeddings.
- **Fallback LLM**: small local transformer (or strict JSON stub) if `confidence < τ` or `OOV_rate > τ`.
- **Routing**: `rules → gazetteer → model → llm`.
- **API**: FastAPI with `/predict`, `/metrics`, `/health`.
- **Monitoring**: Prometheus counters/histograms; latency, coverage, fallback rate; drift alerts.
- **CI/CD**: minimal GitHub Actions (lint + tests); Docker image build.

---

## Data

- `data/transactions_mock.csv` (~10–15k rows):  
  `tx_id,date,amount,currency,raw_label,channel,mcc(optional),category_label,merchant_label`
- `data/gazetteer_merchants.csv` (~2k alias → `merchant_id`):  
  `alias,merchant_id,display_name`
- `data/taxonomy_categories.json` (30–50 categories)
- `data/production_sim.csv` (drifted): new label variants & shifted category proportions
- If no real dataset, generate: `tools/generate_synthetic_data.py`

---

## Quickstart

### 1) Clone & create a virtual environment
```bash
python -m venv venv
# Linux/macOS: source venv/bin/activate
# Windows PowerShell:
.\venv\Scripts\Activate
```

### 2) Install dependencies (light first) :
```bash
python -m pip install --upgrade pip
pip install -r requirements/core.txt
# Optional:
# pip install -r requirements/notebooks.txt
# pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu -r requirements/llm.txt
```
To keep Hugging Face model cache off your system drive (Windows):
setx HF_HOME "D:\huggingface" then in current shell $env:HF_HOME = "D:\huggingface".

### 3) Generate synthetic data, train, run API :
``` bash
python tools/generate_synthetic_data.py --n 6000
python training/train.py --train data/transactions_mock.csv --out app/model_sklearn.pkl
uvicorn app.api:app --host 0.0.0.0 --port 8000
```
### 4) Smoke test
``` bash
# Health
curl http://127.0.0.1:8000/health

# Predict (example)
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "transactions":[
          {"tx_id":"abc123","raw_label":"CB CARREFOUR 75 PARIS TPE1245","channel":"CB","mcc":5411}
        ]
      }'

# Metrics (Prometheus text)
curl http://127.0.0.1:8000/metrics
```

### API
```bash
POST /predict

Request

{
  "transactions": [
    {
      "tx_id": "abc123",
      "date": "2025-08-30",
      "amount": -54.3,
      "currency": "EUR",
      "raw_label": "CB CARREFOUR 75 PARIS TPE1245",
      "channel": "CB",
      "mcc": 5411
    }
  ]
}


Response

{
  "results": [
    {
      "tx_id": "abc123",
      "operation_type": "card",
      "normalized_merchant": "Carrefour",
      "budget_category": "Groceries",
      "confidence": 0.91,
      "explanation": "MCC=5411 + gazetteer match 'carrefour'",
      "router": "rules>gazetteer>model",
      "lat_ms": 17
    }
  ],
  "metrics": { "fallback_rate": 0.08, "coverage": 0.93 }
}
```

### GET /metrics :
- Exports text format counters/histograms consumable by Prometheus:

- predict_requests_total

- predict_latency_ms_bucket

- fallback_total

- coverage_ratio

- drift_alerts_total

### Training & Evaluation

- Training pipeline: training/train.py
- Steps: clean → TF-IDF (chars 1–5) → XGBoost (or RF) → export app/model_sklearn.pkl

- Evaluation: training/eval.py (macro-F1, confusion, top-k, calibration)
- Notebook: notebooks/01_eval_offline.ipynb (macro-F1, accuracy@top1, coverage, simulated latency)

#### Command

- python training/train.py --train data/transactions_mock.csv --out app/model_sklearn.pkl

- Drift Detection

- Script: training/drift.py

- PSI / KS on character n-grams (via TF-IDF) and optional OOV rate

- Generates a simple report (console JSON; extend to HTML/PNG)

#### Command

- python training/drift.py --train data/transactions_mock.csv --prod data/production_sim.csv --topk 500
- # → {"psi_topk": 0.23, "alert": true}


- Rule of thumb: PSI > 0.2 → moderate drift, > 0.3 → strong.

#### Monitoring & Metrics

- Latency histogram (predict_latency_ms_bucket)

- Coverage gauge (coverage_ratio) — share of predictions with confidence ≥ τ

- Fallback counter (fallback_total)

- Requests counter (predict_requests_total)

- Drift alerts counter (drift_alerts_total)

- Hook these into a Prometheus + Grafana stack; endpoint is /metrics.

- Quality Gates (Acceptance)

- Categorization: macro-F1 ≥ 0.82, coverage ≥ 90% (for confidence ≥ τ)

- Merchant normalization: pairwise F1 ≥ 0.90 on known aliases

- Latency: /predict p95 ≤ 80 ms (excluding Docker cold start)

- Fallback share ≤ 15% of requests

- Drift alert if PSI > 0.2 on top-50 n-grams or category shift > 15%

### Project Structure
```bash BridgeLite/
├─ app/
│  ├─ api.py                # FastAPI: /predict, /metrics, /health
│  ├─ service.py            # load models & gazetteer, routing logic
│  ├─ preproc.py            # text cleaning, tokenization, features
│  ├─ rules.py              # strong rules: operation type, MCC, IBAN/RUM
│  ├─ model_sklearn.pkl     # main model (XGBoost/RF)
│  ├─ gazetteer.pkl         # alias → merchant_id (optional artifact)
│  └─ llm_fallback.py       # local transformer or strict-JSON stub
├─ training/
│  ├─ train.py              # training pipeline + export
│  ├─ eval.py               # metrics, confusion, top-k
│  └─ drift.py              # KS/PSI, n-grams shift + reports
├─ tools/
│  └─ generate_synthetic_data.py
├─ tests/
│  ├─ test_preproc.py
│  └─ test_api.py
├─ data/
│  ├─ transactions_mock.csv
│  ├─ production_sim.csv
│  ├─ gazetteer_merchants.csv
│  └─ taxonomy_categories.json
├─ notebooks/
│  └─ 01_eval_offline.ipynb
├─ requirements/
│  ├─ core.txt
│  ├─ llm.txt
│  ├─ notebooks.txt
│  ├─ dev.txt
│  └─ all.txt
├─ Dockerfile
├─ .github/workflows/ci.yml
├─ requirements.txt
└─ README.md
```

### Development Workflow

#### 0–1 h — Setup

- Scaffold repo, choose libs (XGBoost, FastAPI, pytest, black/ruff).

- Generate synthetic data if needed.

#### 1–2 h — Preprocessing & Rules

- preproc.py: Unicode/accents, lowercase, regex (IBAN/RUM), op type (CB/PRLV/VIR).

- rules.py: MCC→category map (if present); patterns (LOYER/EDF/SALAIRE…).

#### 2–3 h — Gazetteer & Entity Resolution

- Load gazetteer_merchants.csv, fuzzy match (Jaro-Winkler/RapidFuzz) + threshold.

- Optionally embeddings; unit tests.

#### 3–4 h — Main Model

- train.py: TF-IDF char (1–5), XGBoost, export .pkl.

- eval.py: macro-F1, confusion, top-k, calibration.

#### 4–5 h — Fallback LLM

- llm_fallback.py: classify_with_llm(text, labels) -> {category, prob, reason} (strict JSON).

- Routing in service.py: rules → gazetteer → model → LLM.

#### 5–6 h — API

- api.py: POST /predict (single & batch), GET /health, GET /metrics.

- Add API tests.

#### 6–7 h — Drift & Monitoring

- drift.py: PSI/KS on TF-IDF n-grams; OOV rate.

- Simple HTML/PNG drift report comparing train vs production_sim.

### 7–8 h — Docker & CI

- Dockerfile (multi-stage, slim).

- ci.yml: lint + tests.

- Scripts to run (make or PowerShell/bash).

#### 8–9 h — Notebook & README

- 01_eval_offline.ipynb: plots, tables, annotated examples.

- README: quickstart, endpoints, cURL examples, decisions.

- 9–10 h — Polish & Demo

- 20 edge-cases; check local p95 latency; screenshots of drift report.

- Sanity check metrics vs thresholds.

### Roadmap / Stretch Goals

- Light Mixture-of-Experts router (rules → XGB → small transformer vs gazetteer-heavy).

- Interpretability: SHAP for XGBoost (top n-grams per category).

- Minimal feature store (Parquet + versioning).

- LRU cache to deduplicate repeated labels.

- Optional LLM service via Hugging Face Inference API or self-hosted microservice.

### Security & PII

- Do not log full raw labels if they contain PII (IBAN/RUM).

- Redact or hash sensitive substrings before exporting metrics or sending to external services.

- Configure outbound access and store any API tokens as environment secrets.

### Docker

- Build & run:

- docker build -t bridgelite .
- docker run -p 8000:8000 bridgelite


### Health:

- curl http://127.0.0.1:8000/health

- Tests
- pytest -q


- CI runs on each push/PR via .github/workflows/ci.yml (ruff + black check + pytest).
- Replace <owner>/<repo> in the CI badge at the top with your repository slug.

### License

- MIT. See LICENSE : for details.

