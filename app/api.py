# FastAPI endpoints /predict, /metrics, /health
from fastapi import FastAPI

app = FastAPI()

@app.get('/health')
def health():
    return {'status': 'ok'}
