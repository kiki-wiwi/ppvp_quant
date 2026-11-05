# Synthetic / IS-copy code-path tests (no network: the OKX fetcher is replaced by a local stub).
import warnings
import numpy as np
import pandas as pd
import data_loader as DL
from ppvp_audit.common import load_is_market
from ppvp_engine import build_4h, build_boxes, detect_pivots
from config import BASELINE_STRATEGY as S

df1h, _, _ = load_is_market()
H = pd.Timedelta(hours=1)

# ---------- (A) 4H buckets with missing 1H (code path; IS itself has none) ----------
x = df1h.loc["2020-01-01":"2020-01-31"].copy()
drop_one = pd.Timestamp("2020-01-10 05:00", tz="UTC")                 # inside bucket 04:00 -> n_bars = 3
drop_bucket = pd.date_range("2020-01-15 08:00", periods=4, freq="h", tz="UTC")   # whole 08:00 bucket missing
y = x.drop(index=[drop_one, *drop_bucket])
d4 = build_4h(y)
print("(A) synthetic gaps in Jan-2020 copy: 4H rows", len(d4), "n_bars counts", d4["n_bars"].value_counts().to_dict())
print("    bucket 2020-01-10 04:00 ->", d4.loc[pd.Timestamp("2020-01-10 04:00", tz="UTC"), ["high", "low", "n_bars", "available_time"]].to_dict())
print("    bucket 2020-01-15 08:00 present:", pd.Timestamp("2020-01-15 08:00", tz="UTC") in d4.index)
print("    columns carried to engine:", list(d4.columns))
bx, _ = build_boxes(y, S)
print("    boxes columns:", list(bx.columns), "-> no n_bars / partial flag in boxes")
# row-based pivot windows: calendar span of an L=5/R=5 window that crosses the missing bucket
i = d4.index.get_loc(pd.Timestamp("2020-01-15 04:00", tz="UTC"))
print("    calendar span of 11-row pivot window around the hole:", d4.index[i + 5] - d4.index[i - 5], "(40h without the hole)")
# trailing partial bucket at the data end is dropped
z = x[x.index < pd.Timestamp("2020-01-31 02:00", tz="UTC")]
print("    data ending 2020-01-31 01:00: last 4H kept", build_4h(z).index[-1], "(partial 00:00 bucket dropped)")

# ---------- (B) >=6h placeholder run is re-filled as an 'archive gap' ----------
calls = []
def okx_stub(symbol, start, end, bar="1H"):
    calls.append((start, end))
    idx = pd.date_range(start, end - H, freq="h", tz="UTC")
    return pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 10.0, "quote_volume": 10.0},
                        index=pd.DatetimeIndex(idx, name="open_time"))
DL.fetch_okx_klines = okx_stub
DL.RESEARCH_END = pd.Timestamp("2020-02-01", tz="UTC")      # shrink the grid to the synthetic month
raw = x[["open", "high", "low", "close", "volume", "quote_volume", "count", "taker_buy_volume"]].copy()
ph = pd.date_range("2020-01-20 00:00", periods=6, freq="h", tz="UTC")
for t in ph:   # make a 6-hour maintenance placeholder run: volume 0 and O=H=L=C
    p = raw.loc[t - H, "close"] if (t - H) in raw.index else raw.loc[t, "open"]
    raw.loc[t, ["open", "high", "low", "close"]] = p
    raw.loc[t, ["volume", "quote_volume", "count", "taker_buy_volume"]] = 0.0
cleaned, dropped = DL.drop_placeholder_bars(raw)
filled, rec = DL.fill_archive_gaps(cleaned, "BTCUSDT")
print("\n(B) 6 placeholder hours dropped:", len(dropped), "-> fill records:", [(r["gap_start"], r["missing_hours"], r["source"]) for r in rec])
print("    fill_source of the 6 maintenance hours after cleaning:", filled.loc[ph, "fill_source"].unique().tolist(),
      " volume there:", filled.loc[ph, "volume"].round(3).unique().tolist())
# a 5-hour placeholder run is left missing
raw5 = raw.copy()
raw5 = raw5.drop(index=ph[-1])
raw5.loc[ph[-1]] = x.loc[ph[-1], raw5.columns]
c5, d5 = DL.drop_placeholder_bars(raw5)
f5, r5 = DL.fill_archive_gaps(c5, "BTCUSDT")
print("    5-hour placeholder run: dropped", len(d5), "fill records", len(r5), "rows", len(f5))

# ---------- (C) gap at the very start of the research window: empty scale window -> NaN volumes ----------
raw_c = x[["open", "high", "low", "close", "volume", "quote_volume", "count", "taker_buy_volume"]].copy()
raw_c = raw_c[raw_c.index >= pd.Timestamp("2020-01-01 06:00", tz="UTC")]
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    fc, rc = DL.fill_archive_gaps(raw_c, "BTCUSDT")
print("\n(C) 6h archive gap at window start: ratio", rc[0]["volume_scale_ratio_binance_over_okx"], "window hours",
      rc[0]["scale_ratio_window_hours"], "| filled volume NaN rows:", int(fc["volume"].isna().sum()),
      "| warnings:", sorted(set(type(i.message).__name__ for i in w)))
print("    no network calls made: stub called", len(calls), "times")
