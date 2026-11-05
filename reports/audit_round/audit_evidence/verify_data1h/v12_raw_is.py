# Local raw Binance perp 1H archives for the 48 IS months (no network): placeholders / holes / identity with the parquet;
# official 4H parquet truncated to IS immediately: differences vs the local 1H aggregation (D-8).
import _setup
import io, zipfile
import numpy as np
import pandas as pd
import data_loader as DL
from _setup import IS_END, IS_START
from ppvp_audit.common import load_is_market
from ppvp_engine import build_4h

df, _, _ = load_is_market()
frames = []
for ym in DL.month_range("2020-01", "2023-12"):
    p = DL.RAW_DIR / DL._vision_path("futures/um", "klines", "BTCUSDT", ym, "1h")
    with zipfile.ZipFile(io.BytesIO(p.read_bytes())) as z:
        frames.append(DL.parse_kline_csv(z.read(z.namelist()[0])))
raw = pd.concat(frames).sort_index()
assert raw.index.max() < IS_END
grid = pd.date_range(IS_START, IS_END - pd.Timedelta(hours=1), freq="h", tz="UTC")
ph = (raw.volume == 0) & (raw.open == raw.high) & (raw.high == raw.low) & (raw.low == raw.close)
print("raw IS archive rows", len(raw), "dups", int(raw.index.duplicated().sum()), "missing vs grid", len(grid.difference(raw.index)),
      "placeholder rows", int(ph.sum()), "zero-volume rows", int((raw.volume == 0).sum()))
cols = ["open", "high", "low", "close", "volume", "quote_volume", "count", "taker_buy_volume"]
same_idx = raw.index.equals(df.index)
print("raw index == parquet IS index:", same_idx, "| values bit-identical:",
      bool(same_idx and all(np.array_equal(raw[c].to_numpy(), df[c].to_numpy()) for c in cols)))

off = pd.read_parquet("data/processed/BTCUSDT_perp_4h_official.parquet")
off = off[off.index < IS_END]
loc = build_4h(df)
common = loc.index.intersection(off.index)
print("official 4H IS rows", len(off), "local", len(loc), "common", len(common))
for c in ["open", "high", "low", "close"]:
    bad = common[(loc.loc[common, c] != off.loc[common, c]).to_numpy()]
    print(f"  {c} mismatches: {len(bad)} {[str(t) for t in bad]}")
rel = ((loc.loc[common, "volume"] - off.loc[common, "volume"]).abs() / off.loc[common, "volume"])
print("  volume rel diff > 1e-6:", [(str(t), round(float(v), 6)) for t, v in rel[rel > 1e-6].items()])
