# Synthetic code-path checks on a copy of IS Jan-2020 1H data (network blocked; OKX fetcher replaced by a local stub).
import _setup
import warnings
import numpy as np
import pandas as pd
import data_loader as DL
from ppvp_audit.common import load_is_market
from ppvp_engine import build_4h, build_boxes, volume_profile
from config import BASELINE_STRATEGY as S

df1h, _, _ = load_is_market()
H = pd.Timedelta(hours=1)
RAWCOLS = ["open", "high", "low", "close", "volume", "quote_volume", "count", "taker_buy_volume"]
x = df1h.loc["2020-01-01":"2020-01-31", RAWCOLS].copy()

calls = []
def okx_stub(symbol, start, end, bar="1H"):
    """Deterministic local stand-in for OKX: price = Binance close of the same hour (or 1.0), volume = 2.0."""
    calls.append((str(start), str(end)))
    idx = pd.date_range(start, end - H, freq="h", tz="UTC")
    px = df1h["close"].reindex(idx).fillna(1.0).to_numpy()
    return pd.DataFrame({"open": px, "high": px * 1.001, "low": px * 0.999, "close": px, "volume": 2.0,
                         "quote_volume": 2.0 * px}, index=pd.DatetimeIndex(idx, name="open_time"))
DL.fetch_okx_klines = okx_stub
DL.RESEARCH_END = pd.Timestamp("2020-02-01", tz="UTC")   # grid restricted to the synthetic month

# ------------- D-1: placeholder run of n hours (volume 0, O=H=L=C) through the build_cache cleaning order ------------
def placeholder_case(n):
    raw = x.copy()
    ph = pd.date_range("2020-01-20 00:00", periods=n, freq="h", tz="UTC")
    p = raw.loc[ph[0] - H, "close"]
    raw.loc[ph, ["open", "high", "low", "close"]] = p
    raw.loc[ph, ["volume", "quote_volume", "count", "taker_buy_volume"]] = 0.0
    cleaned, dropped = DL.drop_placeholder_bars(raw)        # build_cache line 528
    filled, rec = DL.fill_archive_gaps(cleaned, "BTCUSDT")  # build_cache line 529
    back = filled.reindex(ph)
    return len(dropped), rec, back["fill_source"].tolist(), back["volume"].round(4).tolist()

for n in (5, 6, 8):
    nd, rec, fs, vol = placeholder_case(n)
    print(f"D-1 placeholder run {n}h: dropped {nd}; fill records {[(r['gap_start'], r['missing_hours']) for r in rec]}; "
          f"fill_source of those hours {sorted(set(map(str, fs)))}; volumes {sorted(set(vol))[:3]}")
# same hours genuinely missing from the archive (no placeholder rows) behave identically -> indistinguishable
raw_m = x.drop(index=pd.date_range("2020-01-20 00:00", periods=6, freq="h", tz="UTC"))
f_m, r_m = DL.fill_archive_gaps(raw_m, "BTCUSDT")
print("D-1 same 6 hours absent from archive: fill records", [(r["gap_start"], r["missing_hours"]) for r in r_m])

# ------------- D-5: 6h gap at the very start of the window: empty pre-gap scale window -------------
raw_c = x[x.index >= pd.Timestamp("2020-01-01 06:00", tz="UTC")].copy()
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    fc, rc = DL.fill_archive_gaps(raw_c, "BTCUSDT")
print("\nD-5 start-of-window gap: ratio", rc[0]["volume_scale_ratio_binance_over_okx"], "window hours",
      rc[0]["scale_ratio_window_hours"], "NaN volume rows", int(fc["volume"].isna().sum()),
      "warnings", sorted({f"{type(i.message).__name__}: {i.message}" for i in w}))
# gap starting 3h into the window: 3 common hours
raw_d = x.drop(index=pd.date_range("2020-01-01 03:00", periods=6, freq="h", tz="UTC"))
fd, rd = DL.fill_archive_gaps(raw_d, "BTCUSDT")
print("D-5 gap starting 2020-01-01 03:00: window hours", rd[0]["scale_ratio_window_hours"], "ratio %.4f" % rd[0]["volume_scale_ratio_binance_over_okx"])
# QA would not flag NaN volume
q = DL.qa_ohlcv(fc.drop(columns=["fill_source"]), "1h", pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2020-02-01", tz="UTC"))
print("D-5 qa_ohlcv on the filled frame: zero_volume_rows", q["zero_volume_rows"], "negative_volume_rows", q["negative_volume_rows"],
      "missing_bars", q["missing_bars"], "-> NaN volume not flagged")
# volume profile with one NaN volume bar
lo = np.array([100.0, 101.0, 102.0, 101.5]); hi = np.array([101.0, 103.0, 103.0, 102.5])
vf = np.array([10.0, 20.0, 30.0, 15.0]); vn = vf.copy(); vn[1] = np.nan
pf = volume_profile(lo, hi, vf, 50, 0.7); pn = volume_profile(lo, hi, vn, 50, 0.7)
print("D-5 volume_profile finite: POC %.3f VAL %.3f VAH %.3f | with one NaN: POC %.3f VAL %.3f VAH %.3f, NaN bins %d/50"
      % (pf["poc"], pf["val"], pf["vah"], pn["poc"], pn["val"], pn["vah"], int(np.isnan(pn["hist"]).sum())))
# downstream: a box whose window includes a NaN-volume hour, built on the start-gap frame
fcb = fc.drop(columns=["fill_source"])
bx, d4 = build_boxes(fcb, S)
bad = bx[~np.isfinite(bx["total_volume"])]
print("D-5 boxes on start-gap frame:", len(bx), "boxes; with NaN total_volume:", len(bad))
if len(bad):
    b0 = bad.iloc[0]
    print("    first such box window", b0["window_start"], "->", b0["window_end"], " VAL", b0["VAL"], "P_low", b0["P_low"],
          " VAH", b0["VAH"], " POC", b0["POC"], " (VAL==P_low:", b0["VAL"] == b0["P_low"], ")")
vs = pd.Series(fcb["volume"].to_numpy()).rolling(20, min_periods=20).mean()
print("D-5 20-bar vol SMA NaN rows (first 40 bars):", int(vs.iloc[:40].isna().sum()))

# ------------- D-9: partial / missing 4H buckets -------------
drop_one = pd.Timestamp("2020-01-10 05:00", tz="UTC")
drop_bucket = pd.date_range("2020-01-15 08:00", periods=4, freq="h", tz="UTC")
y = df1h.loc["2020-01-01":"2020-01-31"].drop(index=[drop_one, *drop_bucket])
d4 = build_4h(y)
print("\nD-9 4H rows", len(d4), "n_bars counts", d4["n_bars"].value_counts().to_dict())
b = pd.Timestamp("2020-01-10 04:00", tz="UTC")
print("    partial bucket", b, d4.loc[b, ["high", "low", "n_bars", "available_time"]].to_dict())
full = build_4h(df1h.loc["2020-01-01":"2020-01-31"])
print("    same bucket with all 4 hours: high", full.loc[b, "high"], "low", full.loc[b, "low"])
print("    missing bucket 2020-01-15 08:00 present:", pd.Timestamp("2020-01-15 08:00", tz="UTC") in d4.index)
i = d4.index.get_loc(pd.Timestamp("2020-01-15 04:00", tz="UTC"))
print("    calendar span of an 11-row window centred next to the hole:", d4.index[i + 5] - d4.index[i - 5])
z = df1h.loc["2020-01-01":"2020-01-31"]; z = z[z.index < pd.Timestamp("2020-01-31 02:00", tz="UTC")]
print("    data ending with 1H open 2020-01-31 01:00 -> last 4H kept", build_4h(z).index[-1])
print("\nstub calls", len(calls), "| network attempts", len(_setup.NET_ATTEMPTS))
