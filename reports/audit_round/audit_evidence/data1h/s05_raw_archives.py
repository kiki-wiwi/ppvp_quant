# Raw local Binance Vision zips (BTCUSDT, IS months only; spot 2019 only): timestamps, headers, funding jitter,
# and bit-for-bit agreement with the processed parquet rows used by the engine. No network access.
import io, zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from data_loader import parse_kline_csv, parse_funding_csv, _has_header, month_range
from config import IS_END

RAW = Path("/home/user/ppvp_quant/data/raw")
IS_MONTHS = month_range("2020-01", "2023-12")

def csv_of(p):
    with zipfile.ZipFile(p) as z:
        names = z.namelist()
        assert len(names) == 1, names
        return z.read(names[0])

# ---------- perp 1H ----------
frames, hdr, ct_bad, ncols = [], {}, 0, set()
for ym in IS_MONTHS:
    b = csv_of(RAW / f"futures/um/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-{ym}.zip")
    hdr[ym] = _has_header(b)
    f = parse_kline_csv(b)
    exp = ((f.index - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)).to_numpy() + 3_600_000 - 1
    ct_bad += int((f["raw_close_time_ms"].to_numpy() != exp).sum())
    frames.append(f)
raw = pd.concat(frames).sort_index()
print("perp 1H raw IS rows", len(raw), "dups", int(raw.index.duplicated().sum()),
      "raw close_time != open+1h-1ms rows:", ct_bad)
print("months with header:", [m for m, h in hdr.items() if h][:3], "...", sum(hdr.values()), "of", len(hdr),
      "; first headered month:", min([m for m, h in hdr.items() if h], default=None))
proc = pd.read_parquet("/home/user/ppvp_quant/data/processed/BTCUSDT_perp_1h.parquet")
proc = proc[proc.index < IS_END]
print("processed IS rows", len(proc), "index equal raw:", proc.index.equals(raw.index))
for c in ["open", "high", "low", "close", "volume", "quote_volume", "count", "taker_buy_volume"]:
    print("   %-16s exact-equal: %s" % (c, bool(np.array_equal(proc[c].to_numpy(), raw[c].to_numpy()))))

# ---------- perp 4H official (cross-check only) ----------
fr4 = [parse_kline_csv(csv_of(RAW / f"futures/um/monthly/klines/BTCUSDT/4h/BTCUSDT-4h-{ym}.zip")) for ym in IS_MONTHS]
r4 = pd.concat(fr4).sort_index()
exp4 = ((r4.index - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)).to_numpy() + 4 * 3_600_000 - 1
print("official 4H raw IS rows", len(r4), "close_time mismatches", int((r4["raw_close_time_ms"].to_numpy() != exp4).sum()),
      "open hours", sorted(set(r4.index.hour)))

# ---------- funding ----------
rows, heads, jit = [], {}, []
for ym in IS_MONTHS:
    b = csv_of(RAW / f"futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-{ym}.zip")
    first = b.split(b"\n", 1)[0].decode().strip()
    heads.setdefault(first, []).append(ym)
    d = pd.read_csv(io.BytesIO(b), header=0 if _has_header(b) else None)
    ms = d.iloc[:, 0].astype("int64").to_numpy()
    rows.append(parse_funding_csv(b))
    step = 8 * 3_600_000
    nearest = np.round(ms / step).astype(np.int64) * step
    jit.append(pd.DataFrame({"raw_ms": ms, "dev_ms": ms - nearest, "month": ym,
                             "interval_raw": d.iloc[:, 1].to_numpy(), "rate_raw": d.iloc[:, 2].to_numpy()}))
print("\nfunding CSV header lines (first line of each IS file):")
for k, v in heads.items():
    print("   %r : %d files (%s .. %s)" % (k, len(v), v[0], v[-1]))
J = pd.concat(jit, ignore_index=True)
print("funding IS raw rows", len(J), "dev from 8h grid (ms): min", J["dev_ms"].min(), "max", J["dev_ms"].max(),
      "abs>1000ms:", int((J["dev_ms"].abs() > 1000).sum()), "abs>30000ms:", int((J["dev_ms"].abs() > 30000).sum()))
print("   dev quantiles ms:", J["dev_ms"].quantile([0, .5, .9, .99, 1]).to_dict())
print("   rows with dev != 0:", int((J["dev_ms"] != 0).sum()))
print("   interval_raw values:", J["interval_raw"].value_counts().to_dict())
F = pd.concat(rows).sort_index()
on_grid = (F.index.as_unit("ns").asi8 % (8 * 3600 * 10**9) == 0)
print("parsed funding IS rows", len(F), "dups", int(F.index.duplicated().sum()), "all on 8h grid after rounding:", bool(on_grid.all()))
grid = pd.date_range("2020-01-01", "2023-12-31 16:00", freq="8h", tz="UTC")
print("8h grid points in IS", len(grid), "missing", len(grid.difference(F.index)), "extra", len(F.index.difference(grid)))
pf = pd.read_parquet("/home/user/ppvp_quant/data/processed/BTCUSDT_funding.parquet")
pf = pf[pf.index < IS_END]
print("processed funding IS rows", len(pf), "index equal", pf.index.equals(F.index),
      "rate exact-equal", bool(np.array_equal(pf["funding_rate"].to_numpy(), F["funding_rate"].to_numpy())),
      "interval values", pf["funding_interval_hours"].value_counts().to_dict())
print("IS funding rate: mean %.6g  min %.6g  max %.6g  share>0 %.3f" % (pf["funding_rate"].mean(), pf["funding_rate"].min(),
      pf["funding_rate"].max(), (pf["funding_rate"] > 0).mean()))

# ---------- spot 2019 (warm-up) ----------
sp = []
for ym in month_range("2019-01", "2019-12"):
    b = csv_of(RAW / f"spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-{ym}.zip")
    sp.append(parse_kline_csv(b))
S = pd.concat(sp).sort_index()
print("\nspot 2019 raw rows", len(S), "expected", 365 * 24, "first", S.index[0], "last", S.index[-1])
ps = pd.read_parquet("/home/user/ppvp_quant/data/processed/BTCUSDT_spot_1h.parquet")
ps = ps[ps.index < pd.Timestamp("2020-01-01", tz="UTC")]
print("processed spot <2020 rows", len(ps), "close exact-equal raw:", ps.index.equals(S.index) and bool(np.array_equal(ps["close"].to_numpy(), S["close"].to_numpy())))
last_hour = S.groupby(S.index.floor("D")).apply(lambda x: x.index.max().hour)
print("2019 spot days", len(last_hour), "days whose last available hour != 23:", int((last_hour != 23).sum()))
