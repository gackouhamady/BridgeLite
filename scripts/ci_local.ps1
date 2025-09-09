# Lint + train fast + tests (what CI does)
$ErrorActionPreference = "Stop"
cd $PSScriptRoot\..
.\venv\Scripts\Activate

ruff check .; if ($LASTEXITCODE -ne 0) { exit 1 }
black --check .; if ($LASTEXITCODE -ne 0) { exit 1 }

python tools/generate_synthetic_data.py --n 3000
python training/train.py --train data/transactions_mock.csv --out app/model_sklearn.pkl --model rf --val-size 0.2 --seed 7

pytest -q
