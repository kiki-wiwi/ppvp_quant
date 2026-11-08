"""Phase 0 data counts and Phase 2 buy-and-hold check from tracked parquet files.
Usage: python -I p0_data_check.py <repo_root>"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
proc = root / "data/processed"
man = json.loads((root / "data/manifest.json").read_text())
print("manifest top keys", list(man.keys())[:20])

for f in sorted(proc.glob("*.parquet")):
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    df = pd.read_parquet(f)
    idx = df.index if isinstance(df.index, pd.DatetimeIndex) else None
    print(f.name, len(df), h[:16], (idx.min(), idx.max()) if idx is not None else df.columns.tolist()[:6])

btc = pd.read_parquet(proc / "BTCUSDT_perp_1h.parquet")
print(btc.columns.tolist())
print("fill_source counts", btc["fill_source"].value_counts().to_dict() if "fill_source" in btc else "n/a")
full = pd.date_range("2020-01-01", "2026-10-31 23:00", freq="1h", tz="UTC")
missing = full.difference(btc.index)
print("expected", len(full), "missing", len(missing), list(missing[:5]))

fund = pd.read_parquet(proc / "BTCUSDT_funding.parquet")
print("funding", len(fund), fund.index.min(), fund.index.max(), "mean rate", fund.iloc[:, 0].mean() if fund.shape[1] else None, fund.columns.tolist())


def bh(seg_start, seg_end):
    # daily mark at D+1 00:00 = close of 23:00 bar of day D; start anchor = previous day's close or first open
    s = btc[(btc.index >= seg_start) & (btc.index < seg_end)]
    first_open = s["open"].iloc[0]
    last_close = s["close"].iloc[-1]
    # previous close before seg start
    prev = btc[btc.index < seg_start]
    prev_close = prev["close"].iloc[-1] if len(prev) else np.nan
    daily = s["close"].resample("1D").last()
    r = daily.pct_change()
    # first day return relative to anchor
    anchor = prev_close if not np.isnan(prev_close) else first_open
    r.iloc[0] = daily.iloc[0] / anchor - 1
    sh = r.mean() / r.std(ddof=1) * np.sqrt(365)
    return dict(first_open=first_open, prev_close=prev_close, last_close=last_close,
                tr_from_open=last_close / first_open - 1, tr_from_prev_close=last_close / prev_close - 1 if not np.isnan(prev_close) else None,
                sharpe=sh, days=len(daily))


print("IS B&H", bh(pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC")))
print("VAL B&H", bh(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-07-01", tz="UTC")))
