"""Recompute Trend Dip Limit headline numbers and Webb p-values from archived files.
Usage: python -I dip_verify.py <repo_root>"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
d = root / "reports/trend_dip_limit_round/results"
W = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
combos = ["M_market__gross", "M_market__main", "M_market__harsh", "L_limit__gross", "L_limit__main", "L_limit__harsh",
          "L_limit__main__stress", "L_limit__harsh__stress"]
for c in combos:
    t = pd.read_csv(d / f"{c}_trades.csv.gz", parse_dates=["entry_time", "exit_time"])
    dd = pd.read_csv(d / f"{c}_daily.csv", index_col=0, parse_dates=True)
    r = dd["return"]; eq = dd["equity"]
    sh = r.mean() / r.std(ddof=1) * np.sqrt(365)
    mon = t.entry_time.dt.strftime("%Y-%m")
    print(f"{c}: n={len(t)}/{mon.nunique()}m gross={t.gross_bps.mean():.2f} net={t.net_bps.mean():.2f} netR={t.net_R.mean():.3f} tot={eq.iloc[-1]/1e5-1:+.2%} "
          f"ledger={t.net_pnl.sum()/1e5:+.2%} sharpe={sh:.3f} budget_exceeded={t.risk_budget_exceeded.sum()}")
    if c in ("M_market__main", "L_limit__main"):
        h1 = t[t.entry_time < "2022-01-01"].net_bps.mean(); h2 = t[t.entry_time >= "2022-01-01"].net_bps.mean()
        ms = t.groupby(mon).net_bps.sum().sort_values(ascending=False)
        top3 = ms.index[:3].tolist(); rest = t[~mon.isin(top3)]
        print(f"   halves {h1:.2f}/{h2:.2f} top3 {top3} rest n={len(rest)} mean={rest.net_bps.mean():.2f}")
        yr = t.groupby(t.entry_time.dt.year).agg(n=("net_bps", "size"), m=("net_bps", "mean"))
        cal = eq.groupby(eq.index.year).last() / pd.concat([pd.Series([1e5]), eq.groupby(eq.index.year).last()[:-1]]).values - 1
        print("   by year", {y: (int(yr.loc[y, "n"]), round(yr.loc[y, "m"], 2), f"{cal.loc[y]:+.2%}") for y in yr.index})
# Webb
Mr = pd.read_csv(d / "M_market__main_daily.csv", index_col=0, parse_dates=True)["return"]
Lr = pd.read_csv(d / "L_limit__main_daily.csv", index_col=0, parse_dates=True)["return"]
D = Lr - Mr
mon = Mr.index.strftime("%Y-%m"); G = len(np.unique(mon)); N = len(Mr)
w = np.random.default_rng(2026110801).choice(W, size=(99999, G))
print("Webb sha", hashlib.sha256(np.ascontiguousarray(w).tobytes()).hexdigest())
mm = np.random.default_rng(2026110802).integers(0, G, size=(10000, G))
print("month matrix sha", hashlib.sha256(np.ascontiguousarray(mm).tobytes()).hexdigest(), mm.dtype)
ps = {}
for name, x in (("M", Mr), ("L", Lr), ("D", D)):
    g = pd.DataFrame({"x": x.values, "m": mon}).groupby("m", sort=True)["x"].agg(["sum", "count"])
    S = g["sum"].values; K = g["count"].values
    mu = S.sum() / N; se = np.sqrt(G / (G - 1) * ((S - mu * K) ** 2).sum() / N ** 2); tt = mu / se
    Ss = w * S; mus = Ss.sum(1) / N
    ses = np.sqrt(G / (G - 1) * ((Ss - mus[:, None] * K) ** 2).sum(1) / N ** 2)
    ts = np.where(ses == 0, 0, mus / np.where(ses == 0, 1, ses))
    p = (1 + ((np.abs(ts) >= abs(tt)) if name == "D" else (ts >= tt)).sum()) / 100000
    ps[name] = p
    print(f"{name}: mean bp/day={mu*1e4:.4f} p={p:.5f}")
order = sorted(ps, key=lambda k: (ps[k], "MLD".index(k))); run = 0
for i, k in enumerate(order):
    run = max(run, min(1, (3 - i) * ps[k])); print("holm", k, round(run, 5))
pe = pd.read_csv(d / "paired_events.csv.gz")
print("paired events n", len(pe), "filled", pe.L_filled.sum(), "mean diff", pe.difference_bps.mean(),
      "filled-subset contribution", pe.difference_bps[pe.L_filled].sum() / len(pe), "unfilled contribution", pe.difference_bps[~pe.L_filled].sum() / len(pe),
      "unfilled M mean", pe.M_net_bps[~pe.L_filled].mean(), "price improvement filled mean", pe.entry_price_improvement_bps[pe.L_filled].mean(),
      "L mean (filled, unfilled=0?)", pe.L_net_bps.fillna(0).mean(), "wait mean", pe.L_wait_minutes[pe.L_filled].mean())
fs = pd.read_csv(d / "father_signals.csv.gz")
print("father signal rows", len(fs), "eligible", fs.eligible.sum())
