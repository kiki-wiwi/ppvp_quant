"""Recompute Channel vs Matched Random headline numbers and Webb p-values from archived files.
Usage: python -I chrand_verify.py <repo_root>"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
d = root / "reports/channel_random_round/results"
W = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
for c in ("gross_0", "main_17", "harsh_34"):
    sd = d / f"S_channel_unfiltered__{c}"
    t = pd.read_csv(sd / "trades.csv.gz", parse_dates=["entry_time", "exit_time"])
    dd = pd.read_csv(sd / "daily.csv", index_col=0, parse_dates=True)
    eqcol = "equity" if "equity" in dd else dd.columns[0]
    eq = dd[eqcol]
    r = dd["return"] if "return" in dd else eq.pct_change().fillna(eq.iloc[0] / 1e5 - 1)
    sh = r.mean() / r.std(ddof=1) * np.sqrt(365)
    mdd = (eq / np.maximum.accumulate(np.concatenate([[1e5], eq.values]))[1:] - 1).min()
    mon = t.entry_time.dt.strftime("%Y-%m")
    print(f"S {c}: n={len(t)} months={mon.nunique()} tot={eq.iloc[-1]/1e5-1:+.4%} sharpe={sh:.6f} mdd={mdd:.4%} net_bps={t.net_bps.mean():.6f} "
          f"net_R={t.net_R.mean():.6f} gross_bps={t.gross_bps.mean():.6f} final={eq.iloc[-1]:.6f} hold_min={t.minutes_held.mean() if 'minutes_held' in t else float('nan'):.6f}")
    if c == "main_17":
        h1 = t[t.entry_time < "2022-01-01"].net_bps.mean(); h2 = t[t.entry_time >= "2022-01-01"].net_bps.mean()
        ms = t.groupby(mon).net_bps.sum().sort_values(ascending=False)
        top3 = ms.index[:3].tolist(); rest = t[~mon.isin(top3)]
        print(f"   halves {h1:.6f}/{h2:.6f}; top3 {top3} sums {ms.iloc[:3].round(6).tolist()}; rest n={len(rest)} mean={rest.net_bps.mean():.6f}")
        print("   by year", t.groupby(t.entry_time.dt.year).net_bps.agg(["size", "mean"]).round(6).to_dict("index"))
        Sr = r
# random daily returns
rr = pd.read_csv(d / "random_daily_returns.csv.gz", index_col=0, parse_dates=True)
print("random paths", rr.shape)
Rbar = rr.mean(axis=1)
dly = pd.read_csv(d / "daily.csv", index_col=0, parse_dates=True)
print("Rbar max abs diff vs daily.csv", np.abs(Rbar.values - dly["Rbar"].values).max(), "S diff", np.abs(Sr.values - dly["S"].values).max())
D = dly["S"] - dly["Rbar"]
print(f"S mean {dly['S'].mean()*1e4:.6f} bp/d, Rbar {dly['Rbar'].mean()*1e4:.6f}, D {D.mean()*1e4:.6f}")
rs = pd.read_csv(d / "random_summary.csv", index_col=0)
Stot = 0.076466
print("share random total_return > S:", (rs.total_return > (pd.read_csv(d / 'S_channel_unfiltered__main_17/daily.csv', index_col=0).iloc[-1, 0] / 1e5 - 1)).mean(),
      "share random daily_mean > S:", (rs.daily_mean > dly["S"].mean()).mean(),
      "median tot", rs.total_return.median(), "q025", rs.total_return.quantile(0.025), "q975", rs.total_return.quantile(0.975),
      "median trades", rs.trade_count.median())
print("MCSE_R candidate sd(daily_mean)/sqrt(200):", rs.daily_mean.std(ddof=1) / np.sqrt(len(rs)))
# Webb
mon = dly.index.strftime("%Y-%m"); G = len(np.unique(mon)); N = len(dly)
w = np.random.default_rng(2026110902).choice(W, size=(99999, G))
print("Webb sha", hashlib.sha256(np.ascontiguousarray(w).tobytes()).hexdigest())
for name, x in (("S", dly["S"]), ("D", D)):
    g = pd.DataFrame({"x": x.values, "m": mon}).groupby("m", sort=True)["x"].agg(["sum", "count"])
    S = g["sum"].values; K = g["count"].values
    mu = S.sum() / N; se = np.sqrt(G / (G - 1) * ((S - mu * K) ** 2).sum() / N ** 2); tt = mu / se
    Ss = w * S; mus = Ss.sum(1) / N
    ses = np.sqrt(G / (G - 1) * ((Ss - mus[:, None] * K) ** 2).sum(1) / N ** 2)
    ts = np.where(ses == 0, 0, mus / np.where(ses == 0, 1, ses))
    p = (1 + (ts >= tt).sum()) / 100000
    print(f"{name}: t={tt:.6f} p={p:.5f}")
h1 = D[D.index < "2022-01-01"].mean() * 1e4; h2 = D[D.index >= "2022-01-01"].mean() * 1e4
print(f"D halves {h1:.6f}/{h2:.6f}")
meta = json.loads((d / "statistical_metadata.json").read_text())
print("metadata keys", list(meta.keys())[:30])
for f in sorted((d / "statistical_matrices").iterdir()):
    print(f.name, f.stat().st_size, hashlib.sha256(f.read_bytes()).hexdigest()[:16])
