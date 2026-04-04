import pandas as pd
import numpy as np
import torch
import random
import os

# Reproducibility — makes dataset generation deterministic
random.seed(42)
np.random.seed(42)

def prepare_dataset(log_path="data/api_traffic.log", lookback=60, lookahead=3):
    """
    Returns raw, UNSCALED tensors. Scaling happens in train.py after the split.

    Columns: timestamp, req_count, latency_ms, cpu_usage, mem_usage
    Features: rps, avg_latency, cpu_usage, mem_usage
    Label: 1 if ANY crash occurs in next lookahead seconds, else 0
    """
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return None, None

    column_names = ['timestamp', 'req_count', 'latency_ms', 'cpu_usage', 'mem_usage']

    try:
        df = pd.read_csv(log_path, names=column_names)
    except pd.errors.EmptyDataError:
        print("Log file is empty.")
        return None, None

    if df.empty:
        return None, None

    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp', 'cpu_usage', 'mem_usage'])
    df['timestamp_sec'] = df['timestamp'].astype(int)
    df = df.sort_values('timestamp_sec')

    intervals = []
    for ts, group in df.groupby('timestamp_sec'):
        rps         = len(group)
        avg_latency = pd.to_numeric(group['latency_ms'], errors='coerce').mean()
        avg_cpu     = pd.to_numeric(group['cpu_usage'],  errors='coerce').mean()
        avg_mem     = pd.to_numeric(group['mem_usage'],  errors='coerce').mean()
        crash       = 1.0 if (avg_latency > 1000 or avg_cpu > 90) else 0.0

        intervals.append({
            'timestamp_sec':   ts,
            'rps':             float(rps),
            'avg_latency':     avg_latency,
            'cpu_usage':       avg_cpu,
            'mem_usage':       avg_mem,
            'crash_indicator': crash,
        })

    df_agg = pd.DataFrame(intervals).dropna()

    min_rows = lookback + lookahead
    if len(df_agg) <= min_rows:
        print(f"Not enough data. Need > {min_rows} rows, got {len(df_agg)}.")
        return None, None

    features = ['rps', 'avg_latency', 'cpu_usage', 'mem_usage']

    X, y = [], []
    for i in range(len(df_agg) - lookback - lookahead + 1):
        window = df_agg.iloc[i : i + lookback][features].values
        future = df_agg.iloc[
            i + lookback : i + lookback + lookahead
        ]['crash_indicator'].values
        label  = float(future.max())
        X.append(window)
        y.append(label)

    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)

    crash_count = int(y_tensor.sum().item())
    safe_count  = len(y_tensor) - crash_count
    print(f"Dataset ready: {len(X_tensor)} windows | Safe: {safe_count} | Crash: {crash_count}")
    print(f"Crash rate: {crash_count/len(y_tensor)*100:.1f}%")
    return X_tensor, y_tensor


if __name__ == "__main__":
    print("Preparing dataset (raw, unscaled)...")
    X, y = prepare_dataset()
    if X is not None:
        print(f"X: {X.shape} | y: {y.shape}")
        torch.save((X, y), "data/processed_tensors.pt")
        print("Saved to data/processed_tensors.pt")