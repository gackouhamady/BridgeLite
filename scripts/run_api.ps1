# Run API locally (venv)
$ErrorActionPreference = "Stop"
cd $PSScriptRoot\..
.\venv\Scripts\Activate
uvicorn app.api:app --host 0.0.0.0 --port 8000
