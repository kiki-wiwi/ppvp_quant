# Holes >= 1h in the locally cached Binance BTCUSDT spot 1H archive before 2020 (warm-up period only).
import pandas as pd
sp = pd.read_parquet("/home/user/ppvp_quant/data/processed/BTCUSDT_spot_1h.parquet")
sp = sp[sp.index < pd.Timestamp("2020-01-01", tz="UTC")]          # truncate immediately
H = pd.Timedelta(hours=1)
grid = pd.date_range("2019-01-01", "2019-12-31 23:00", freq="h", tz="UTC")
miss = grid.difference(sp.index)
runs, start, prev = [], None, None
for t in miss:
    if start is None or t - prev != H:
        if start is not None:
            runs.append((start, prev))
        start = t
    prev = t
if start is not None:
    runs.append((start, prev))
for s, e in runs:
    n = int((e - s) / H) + 1
    before = sp.loc[s - H, "close"]; after = sp.loc[e + H, "open"]
    print("%s .. %s  %2d h  %s  close_before %.2f  open_after %.2f  move %+.1f bp" % (
        s, e, n, ">=6h" if n >= 6 else "    ", before, after, (after / before - 1) * 1e4))
print("total missing hours", len(miss))
