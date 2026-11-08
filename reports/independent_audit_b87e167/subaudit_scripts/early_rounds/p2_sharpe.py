"""Approximate reconstruction of PPVP baseline IS hourly mark-to-market equity from the B_audited ledger
(entry/exit bars, qty, fees, funding) and IS 1H closes; then daily Sharpe (D marked at D+1 00:00).
Usage: python -I p2_sharpe.py <repo_root>"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
IS0 = pd.Timestamp("2020-01-01", tz="UTC"); IS1 = pd.Timestamp("2024-01-01", tz="UTC")
df = pd.read_parquet(root / "data/processed/BTCUSDT_perp_1h.parquet")
df = df[(df.index >= IS0) & (df.index < IS1)]
fund = pd.read_parquet(root / "data/processed/BTCUSDT_funding.parquet")
fund = fund[(fund.index >= IS0) & (fund.index <= IS1)]
t = pd.read_csv(root / "reports/audit_round/experiments/rematerialised/B_audited_trades.csv")
c = df["close"].to_numpy(); o = df["open"].to_numpy()
n = len(df)
eq = np.full(n, np.nan)
realized = 0.0
TAKER = 0.0005
tr = t.sort_values("entry_bar").reset_index(drop=True)
k = 0
open_tr = None
# funding settlement times mapped to bar index whose open == T
f_idx = np.searchsorted(df.index.as_unit("ns").asi8, fund.index.as_unit("ns").asi8)
f_rate = fund["funding_rate"].to_numpy()
fmap = {}
for i_bar, rate in zip(f_idx, f_rate):
    if i_bar < n:
        fmap[int(i_bar)] = rate
for i in range(n):
    # entries happen at close of entry_bar; exits at exit_bar (intrabar or close)
    if open_tr is None and k < len(tr) and tr.loc[k, "entry_bar"] == i:
        open_tr = dict(tr.loc[k]); open_tr["fund_acc"] = 0.0; k += 1
        eq[i] = 1e5 + realized - TAKER * open_tr["qty"] * open_tr["entry_px"] + open_tr["qty"] * (c[i] - open_tr["entry_px"])
        continue
    if open_tr is not None:
        if i in fmap and i > open_tr["entry_bar"]:
            open_tr["fund_acc"] += open_tr["qty"] * o[i] * fmap[i]
        if i == open_tr["exit_bar"]:
            realized += open_tr["pnl"]
            open_tr = None
            eq[i] = 1e5 + realized
            continue
        eq[i] = 1e5 + realized - TAKER * open_tr["qty"] * open_tr["entry_px"] + open_tr["qty"] * (c[i] - open_tr["entry_px"]) - open_tr["fund_acc"]
    else:
        eq[i] = 1e5 + realized
s = pd.Series(eq, index=df.index + pd.Timedelta(hours=1))  # equity at bar close time
daily = s[s.index.hour == 0]  # D+1 00:00 marks for day D
daily = daily[(daily.index > IS0) & (daily.index <= IS1)]
r = np.diff(np.concatenate([[1e5], daily.values])) / np.concatenate([[1e5], daily.values])[:-1]
print("days", len(r), "final eq", daily.values[-1], "cum", daily.values[-1] / 1e5 - 1)
print("Sharpe", r.mean() / r.std(ddof=1) * np.sqrt(365))
mdd = (daily.values / np.maximum.accumulate(np.concatenate([[1e5], daily.values]))[1:] - 1).min()
print("daily-close MDD", mdd, " hourly MDD", (s.values / np.maximum.accumulate(np.concatenate([[1e5], s.values]))[1:] - 1).min())
print("exposure (bars in position share)", ((tr.exit_bar - tr.entry_bar).sum()) / n)
