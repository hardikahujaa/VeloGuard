import pandas as pd

df = pd.read_csv("data/latency_log.csv").dropna()
print(f"Total samples logged: {len(df)}")

q1, q2 = df['rps'].quantile([0.33, 0.66])
def bucket(r):
    if r <= q1: return "Low"
    elif r <= q2: return "Medium"
    else: return "High"
df['load'] = df['rps'].apply(bucket)

summary = df.groupby('load').agg(
    n=('rps', 'count'), rps_min=('rps', 'min'), rps_max=('rps', 'max'),
    agg_ms=('agg_ms', 'mean'), inference_ms=('inference_ms', 'mean'),
    redis_ms=('redis_ms', 'mean'), total_ms=('total_ms', 'mean'),
    total_max_ms=('total_ms', 'max')
).reindex(["Low", "Medium", "High"])

print(summary.round(2).to_string())
print(f"\nOverall mean total latency: {df['total_ms'].mean():.2f} ms")
print(f"Overall max total latency:  {df['total_ms'].max():.2f} ms")
