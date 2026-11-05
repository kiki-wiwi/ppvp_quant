# IS-only 1H / 4H data integrity checks through the sanctioned entry point.
import numpy as np
import pandas as pd
from ppvp_audit.common import load_is_market
from config import IS_START, IS_END
from ppvp_engine import build_4h
from data_loader import resample_ohlcv

df1h, fund, spot = load_is_market()
H = pd.Timedelta(hours=1)
print("pandas", pd.__version__)
print("1H rows", len(df1h), "first", df1h.index[0], "last", df1h.index[-1], "index dtype", df1h.index.dtype)
grid = pd.date_range(IS_START, IS_END - H, freq="h", tz="UTC")
print("expected", len(grid), "missing", len(grid.difference(df1h.index)), "off-grid", len(df1h.index.difference(grid)),
      "dups", int(df1h.index.duplicated().sum()), "monotonic", df1h.index.is_monotonic_increasing)
print("columns", list(df1h.columns))
ct = pd.DatetimeIndex(df1h["close_time"])
print("close_time == open_time + 1h for all rows:", bool((ct == df1h.index + H).all()), "close_time dtype", df1h["close_time"].dtype)
print("fill_source", df1h["fill_source"].value_counts().to_dict())
v = df1h["volume"]
flat = (df1h["open"] == df1h["high"]) & (df1h["high"] == df1h["low"]) & (df1h["low"] == df1h["close"])
print("zero-volume rows", int((v == 0).sum()), "flat O=H=L=C rows", int(flat.sum()), "NaN in OHLCV",
      int(df1h[["open", "high", "low", "close", "volume"]].isna().sum().sum()))
print("NaN count/taker_buy_volume", int(df1h["count"].isna().sum()), int(df1h["taker_buy_volume"].isna().sum()))
bad = (df1h["high"] < df1h[["open", "close"]].max(axis=1)) | (df1h["low"] > df1h[["open", "close"]].min(axis=1))
print("bad OHLC rows", int(bad.sum()), "non-positive prices", int((df1h[["open","high","low","close"]] <= 0).any(axis=1).sum()))

# Partial / thin hours (no row-level flag exists): low trade count or volume vs a trailing 168h median.
med_v = v.rolling(168, min_periods=24).median().shift(1)
med_n = df1h["count"].rolling(168, min_periods=24).median().shift(1)
rv = v / med_v
rn = df1h["count"] / med_n
thin = df1h.assign(vol_vs_med=rv, count_vs_med=rn)[["open", "high", "low", "close", "volume", "count", "vol_vs_med", "count_vs_med"]]
print("\nthinnest 12 IS hours by volume / trailing-168h median volume:")
print(thin.nsmallest(12, "vol_vs_med").to_string())
print("hours with vol < 10% of trailing median:", int((rv < 0.10).sum()), " < 20%:", int((rv < 0.20).sum()))

# open vs previous close (continuity)
gap_bp = (df1h["open"] / df1h["close"].shift(1) - 1) * 1e4
print("\n|open_i / close_{i-1} - 1| in bp: max %.2f, p99.9 %.3f, count > 5bp: %d, > 20bp: %d" % (
    gap_bp.abs().max(), gap_bp.abs().quantile(0.999), int((gap_bp.abs() > 5).sum()), int((gap_bp.abs() > 20).sum())))
print(gap_bp.abs().nlargest(8).to_string())

# 4H aggregation: UTC boundaries, n_bars, available_time
d4 = build_4h(df1h)
print("\n4H rows", len(d4), "n_bars value counts", d4["n_bars"].value_counts().to_dict())
print("4H open hours", sorted(set(d4.index.hour)), "minutes", sorted(set(d4.index.minute)))
print("available_time == open+4h", bool((pd.DatetimeIndex(d4["available_time"]) == d4.index + pd.Timedelta(hours=4)).all()),
      "close_time == open+4h", bool((pd.DatetimeIndex(d4["close_time"]) == d4.index + pd.Timedelta(hours=4)).all()))
print("last 4H open", d4.index[-1], "available", d4["available_time"].iloc[-1])
# independent re-aggregation by explicit floor('4h') of UTC timestamps (not every-four-rows)
key = df1h.index.floor("4h")
g = df1h.groupby(key)
ind = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                    "close": g["close"].last(), "volume": g["volume"].sum(), "n": g["close"].count()})
same_idx = ind.index.equals(d4.index)
print("independent floor('4h') aggregation index equal:", same_idx)
for c in ["open", "high", "low", "close"]:
    print("  ", c, "mismatches", int((ind[c].to_numpy() != d4[c].to_numpy()).sum()))
print("   volume max rel diff", float(((ind["volume"] - d4["volume"]).abs() / d4["volume"]).max()))

# official 4H (cross-check only), IS rows only
off = pd.read_parquet("/home/user/ppvp_quant/data/processed/BTCUSDT_perp_4h_official.parquet")
off = off[off.index < IS_END]
common = d4.index.intersection(off.index)
print("\nofficial 4H IS rows", len(off), "common", len(common), "only local", len(d4.index.difference(off.index)),
      "only official", len(off.index.difference(d4.index)))
a, b = d4.loc[common], off.loc[common]
for c in ["open", "high", "low", "close"]:
    mm = a.index[(a[c] != b[c]).to_numpy()]
    print("  ", c, "mismatch bars", len(mm), list(map(str, mm)))
rel = (a["volume"] - b["volume"]).abs() / b["volume"]
print("   volume rel diff > 1e-6:", int((rel > 1e-6).sum()), "bars:", [(str(t), float(r)) for t, r in rel[rel > 1e-6].items()])

# Independent 1H-only adjudication of 2023-11-10 12:00 using Binance SPOT 1H (independent market), no 1m data
spot1h = pd.read_parquet("/home/user/ppvp_quant/data/processed/BTCUSDT_spot_1h.parquet")
spot1h = spot1h[spot1h.index < IS_END]
t = pd.Timestamp("2023-11-10 12:00", tz="UTC")
win = pd.date_range(t - pd.Timedelta(hours=4), t + pd.Timedelta(hours=7), freq="h", tz="UTC")
cmp_ = pd.DataFrame({"perp_close": df1h["close"].reindex(win), "spot_close": spot1h["close"].reindex(win),
                     "perp_vol": df1h["volume"].reindex(win), "perp_count": df1h["count"].reindex(win)})
cmp_["basis_bp"] = (cmp_["perp_close"] / cmp_["spot_close"] - 1) * 1e4
print("\n2023-11-10 perp vs spot 1H around the disputed 4H bar:")
print(cmp_.to_string())
s15 = spot1h.loc[pd.Timestamp("2023-11-10 15:00", tz="UTC"), "close"]
print("4H close candidates vs spot 15:00 close: local 1H-aggregate 37092.6 -> %.2f bp ; official 4H 37118.4 -> %.2f bp" % (
      (37092.6 / s15 - 1) * 1e4, (37118.4 / s15 - 1) * 1e4))
nxt = df1h.loc[pd.Timestamp("2023-11-10 16:00", tz="UTC"), "open"]
print("perp 16:00 open (next bar)", nxt, " local close vs next open bp %.2f, official close vs next open bp %.2f" % (
      (37092.6 / nxt - 1) * 1e4, (37118.4 / nxt - 1) * 1e4))
basis_all = ((df1h["close"] / spot1h["close"].reindex(df1h.index)) - 1) * 1e4
print("IS perp-spot basis bp: median %.2f p1 %.2f p99 %.2f (n=%d)" % (basis_all.median(), basis_all.quantile(.01),
      basis_all.quantile(.99), basis_all.notna().sum()))
