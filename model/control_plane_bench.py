import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import asyncio, time
import pandas as pd
import numpy as np
import redis.asyncio as aioredis
import torch
import os
import pickle
from datetime import datetime
from lstm_model import CrashPredictorLSTM

LOG_FILE = "data/latency_log.csv"

async def control_plane_worker():
    print("Starting VeloGuard AI Control Plane (benchmark mode)...")
    while (not os.path.exists("data/model.pth") or
           not os.path.exists("data/scaler.pkl") or
           not os.path.exists("data/threshold.txt")):
        print("Waiting for model.pth, scaler.pkl, and threshold.txt...")
        await asyncio.sleep(5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CrashPredictorLSTM(input_size=4, hidden_size=128, num_layers=2).to(device)
    model.load_state_dict(torch.load("data/model.pth", map_location=device, weights_only=True))
    model.eval()
    with open("data/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    base_threshold      = float(open("data/threshold.txt").read().strip())
    warn_threshold      = base_threshold
    critical_threshold  = min(base_threshold + 0.15, 0.90)
    blackhole_threshold = 0.92

    redis_client = aioredis.from_url("redis://localhost:6379", decode_responses=True)
    log_path     = "data/api_traffic.log"
    lookback     = 60
    column_names = ['timestamp', 'req_count', 'latency_ms', 'cpu_usage', 'mem_usage']
    features     = ['rps', 'avg_latency', 'cpu_usage', 'mem_usage']

    await redis_client.set("global_ai_limit", 100)

    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("timestamp,rps,agg_ms,inference_ms,redis_ms,total_ms\n")

    print("\nControl plane (benchmark mode) live. Logging to", LOG_FILE, "\n")
    while True:
        try:
            if not os.path.exists(log_path):
                await asyncio.sleep(1); continue
            try:
                df = pd.read_csv(log_path, names=column_names)
            except pd.errors.EmptyDataError:
                await asyncio.sleep(1); continue
            if df.empty:
                await asyncio.sleep(1); continue

            t0 = time.perf_counter()
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp', 'cpu_usage', 'mem_usage'])
            df['timestamp_sec'] = df['timestamp'].astype(int)
            df = df.sort_values('timestamp_sec')
            intervals = []
            for ts, group in df.groupby('timestamp_sec'):
                rps = len(group)
                avg_latency = pd.to_numeric(group['latency_ms'], errors='coerce').mean()
                avg_cpu     = pd.to_numeric(group['cpu_usage'],  errors='coerce').mean()
                avg_mem     = pd.to_numeric(group['mem_usage'],  errors='coerce').mean()
                intervals.append({'timestamp_sec': ts, 'rps': float(rps), 'avg_latency': avg_latency, 'cpu_usage': avg_cpu, 'mem_usage': avg_mem})
            df_agg = pd.DataFrame(intervals).dropna()
            if len(df_agg) < lookback:
                await asyncio.sleep(1); continue

            last_window = df_agg.tail(lookback).copy()
            current_rps = float(last_window['rps'].iloc[-1])
            last_window[features] = scaler.transform(last_window[features])
            X_tensor = torch.tensor(last_window[features].values, dtype=torch.float32).unsqueeze(0).to(device)
            t1 = time.perf_counter()

            with torch.no_grad():
                prob = torch.sigmoid(model(X_tensor)).item()
            t2 = time.perf_counter()

            if prob >= blackhole_threshold: limit, state = 0,   "BLACKHOLE"
            elif prob >= critical_threshold: limit, state = 20,  "CRITICAL"
            elif prob >= warn_threshold: limit, state = 60,  "WARNING"
            else: limit, state = 100, "SAFE"
            await redis_client.set("global_ai_limit", limit)
            t3 = time.perf_counter()

            agg_ms, inf_ms, redis_ms, total_ms = (t1-t0)*1000, (t2-t1)*1000, (t3-t2)*1000, (t3-t0)*1000
            with open(LOG_FILE, "a") as f:
                f.write(f"{time.time()},{current_rps},{agg_ms:.3f},{inf_ms:.3f},{redis_ms:.3f},{total_ms:.3f}\n")

            ts_str = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts_str}] [{state}] rps={current_rps:.0f} total={total_ms:.1f}ms (agg={agg_ms:.1f} inf={inf_ms:.1f} redis={redis_ms:.1f})")
        except Exception as e:
            print(f"Control plane error: {e}")
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(control_plane_worker())
