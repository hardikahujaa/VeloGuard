import pandas as pd

df = pd.read_csv("data/latency_log.csv").dropna()
print(f"Total samples logged: {len(df)}")

df = df.sort_values('rps').reset_index(drop=True)
n = len(df)
groups = {
    "Low":    df.iloc[:n // 3],
    "Medium": df.iloc[n // 3: 2 * n // 3],
    "High":   df.iloc[2 * n // 3:],
}

rows = []
for name, d in groups.items():
    rows.append({
        "load": name, "n": len(d),
        "rps_min": d['rps'].min(), "rps_max": d['rps'].max(),
        "agg_ms": d['agg_ms'].mean(), "inference_ms": d['inference_ms'].mean(),
        "redis_ms": d['redis_ms'].mean(), "total_ms": d['total_ms'].mean(),
        "total_max_ms": d['total_ms'].max(),
    })
print(pd.DataFrame(rows).round(2).to_string(index=False))
print(f"\nOverall mean total latency: {df['total_ms'].mean():.2f} ms")
print(f"Overall max total latency:  {df['total_ms'].max():.2f} ms")
