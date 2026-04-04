import time
import os
import threading
from fastapi import FastAPI, Request
from .middleware import RateLimitMiddleware

app = FastAPI(title="LoadGuard Vulnerable API")

app.add_middleware(RateLimitMiddleware)

# Global concurrent requests counter
active_requests = 0
counter_lock = threading.Lock()

@app.get("/predict")
async def predict():
    time.sleep(0.05)
    return {"status": "success", "prediction": "dog"}

@app.get("/health")
async def health():
    return {"status": "healthy", "active_requests": active_requests}