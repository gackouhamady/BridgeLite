# ---- Base (slim) ----
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    TZ=Europe/Paris

# Minimal OS deps (xgboost wheel needs libgomp1; curl for HEALTHCHECK)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements/ ./requirements/
# (Optional) keep root requirements.txt if present
COPY requirements.txt ./requirements.txt

# Install only the core runtime deps (no notebooks / LLM heavy)
RUN python -m pip install --upgrade pip && \
    pip install -r requirements/core.txt

# Copy app code
COPY app ./app
COPY training ./training
COPY tools ./tools
# Provide empty dirs (dockerignore keeps local files out)
RUN mkdir -p data reports/drift

# Build a small demo model in-image (RandomForest → faster, portable)
# You can switch to XGBoost later if your infra allows wheels.
RUN python tools/generate_synthetic_data.py --n 4000 && \
    python training/train.py --train data/transactions_mock.csv --out app/model_sklearn.pkl --model rf --val-size 0.2 --seed 7

EXPOSE 8000

# Healthcheck pings /health
HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD curl -fsS http://localhost:8000/health || exit 1

# Run API
CMD ["uvicorn","app.api:app","--host","0.0.0.0","--port","8000"]
