# D-2: holes >= 6h in the cached Binance spot 1H archive before 2020 (warm-up rows only) and the price change across them;
# reference distribution: |open(t+k+1)/close(t)-1| for ordinary k-hour spans of the same 2019 spot data.
import _setup
import numpy as np
import pandas as pd
from _setup import IS_START
s = pd.read_parquet("data/processed/BTCUSDT_spot_1h.parquet")
s = s[s.index < IS_START]
H = pd.Timedelta(hours=1)
grid = pd.date_range(s.index.min(), IS_START - H, freq="h", tz="UTC")
miss = grid.difference(s.index)
runs = []
if len(miss):
    st = miss[0]; pv = miss[0]; n = 1
    for t in miss[1:]:
        if t - pv == H:
            pv, n = t, n + 1
        else:
            runs.append((st, pv, n)); st, pv, n = t, t, 1
    runs.append((st, pv, n))
print("2019 spot: rows", len(s), "missing hours", len(miss), "runs", len(runs))
for a, b, n in runs:
    cb = s.loc[a - H, "close"] if (a - H) in s.index else np.nan
    oa = s.loc[b + H, "open"] if (b + H) in s.index else np.nan
    print(f"  hole {a} .. {b} ({n} h): close before {cb}, open after {oa}, change {(oa / cb - 1) * 1e4:+.2f} bp")
zero = (s["volume"] == 0).sum()
print("zero-volume rows in 2019 spot:", int(zero))
# reference: ordinary spans of k missing hours -> |open(t+k+1) / close(t) - 1| over all 2019 positions
c = s["close"]; o = s["open"]
full = s.reindex(grid)
for k in (6, 8, 10):
    ratio = (full["open"].shift(-(k + 1)) / full["close"] - 1).dropna().abs() * 1e4
    print(f"  ordinary {k}h spans: n={len(ratio)}, share with |change| < 0.05 bp: {(ratio < 0.05).mean():.4f}, "
          f"< 1 bp: {(ratio < 1).mean():.4f}, < 10 bp: {(ratio < 10).mean():.4f}, median {ratio.median():.1f} bp")
# adjacent-hour equality rate for comparison
adj = (full["open"].shift(-1) == full["close"]).sum() / full["close"].notna().sum()
print("  share of hours where next open == this close exactly (normal 1h boundary): %.3f" % adj)
