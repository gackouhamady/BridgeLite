# ============================
# BridgeLite: Console Setup Script (PowerShell)
# ============================

# 0) From your repo root, ensure venv is active if you have one:
# .\venv\Scripts\Activate

# 1) Create requirements/ folder
New-Item -ItemType Directory requirements -Force | Out-Null

# 2) Create modular requirements files
# -- core.txt (small, install first)
Set-Content -Path requirements\core.txt -Encoding UTF8 -Value @'
# Core ML (small)
scikit-learn==1.5.1
xgboost==2.1.1
numpy==1.26.4
pandas==2.2.2
scipy==1.13.1

# Text processing
regex==2024.7.24
python-levenshtein==0.25.1
rapidfuzz==3.9.7
unidecode==1.3.8

# API
fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.9.2

# Monitoring
prometheus-client==0.21.0

# Drift detection
evidently==0.4.37

# Testing (light)
pytest==8.3.3
httpx==0.27.2
pytest-asyncio==0.24.0
'@

# -- llm.txt (heavy; install later)
Set-Content -Path requirements\llm.txt -Encoding UTF8 -Value @'
# LLM / embeddings (heavy)
sentence-transformers==3.0.1
transformers==4.44.2
tokenizers==0.19.1
torch==2.4.0
'@

# -- notebooks.txt (optional)
Set-Content -Path requirements\notebooks.txt -Encoding UTF8 -Value @'
matplotlib==3.9.2
seaborn==0.13.2
jupyter==1.0.0
notebook==7.2.2
'@

# -- dev.txt (optional)
Set-Content -Path requirements\dev.txt -Encoding UTF8 -Value @'
black==24.8.0
ruff==0.6.9
'@

# -- packaging.txt (optional)
Set-Content -Path requirements\packaging.txt -Encoding UTF8 -Value @'
poetry-core>=1.9.0
'@

# -- all.txt (install everything)
Set-Content -Path requirements\all.txt -Encoding UTF8 -Value @'
-r core.txt
-r llm.txt
-r notebooks.txt
-r dev.txt
-r packaging.txt
'@

# 3) Minimal root requirements.txt (some tools expect this path)
Set-Content -Path requirements.txt -Encoding UTF8 -Value @'
-r requirements/core.txt
'@

# 4) Install only what you need, when you need it

# A) First install (lightweight)
python -m pip install --upgrade pip
pip install -r requirements\core.txt

# B) Notebooks/plots (optional)
# pip install -r requirements\notebooks.txt

# C) LLM stack (HEAVY) — Option 1: CPU-only torch (saves space)
# pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu -r requirements\llm.txt

#    LLM stack — Option 2: default index (larger)
# pip install --no-cache-dir -r requirements\llm.txt

# D) Dev tools (formatter + linter)
# pip install -r requirements\dev.txt

# E) Everything (on a machine with enough space)
# pip install --no-cache-dir -r requirements\all.txt


# 5) Move Hugging Face model cache off system drive (saves lots of space)
# Persistent for future terminals:
#   setx HF_HOME "D:\huggingface"
# For current session:
#   $env:HF_HOME = "D:\huggingface"


# 6) Verify imports

# Core stack
python -c "import sklearn, xgboost, fastapi, pandas, numpy, evidently; print('Core OK')"

# LLM stack (run only after installing llm.txt)
# python -c "import torch, transformers, sentence_transformers; print('LLM OK:', torch.__version__)"

# Notebooks/plots (run only after installing notebooks.txt)
# python -c "import matplotlib, notebook; print('Notebooks OK')"


# 7) If you hit 'No space left on device'

# Clear pip cache
# pip cache purge

# Recreate a clean venv (optional)
# deactivate 2>$null
# rmdir /s /q venv
# python -m venv venv
# .\venv\Scripts\Activate
# python -m pip install --upgrade pip
# pip install -r requirements\core.txt

# Install heavy libs to another folder and import via PYTHONPATH
# New-Item -ItemType Directory D:\pydeps -Force | Out-Null
# pip install --no-cache-dir --target D:\pydeps -r requirements\llm.txt --index-url https://download.pytorch.org/whl/cpu
# $env:PYTHONPATH = "D:\pydeps"
# python -c "import torch, transformers; print('Heavy deps via PYTHONPATH OK')"
