"""Spot-check audit-round decision_report numbers against archived per-trade CSVs.
Usage: python -I audit_round_verify.py <repo_root>"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
ex = root / "reports/audit_round/experiments"
res = json.loads((ex / "results.json").read_text())
base = pd.read_csv(ex / "rematerialised/B_audited_trades.csv", parse_dates=["entry_time", "exit_time"])
p2 = pd.read_csv(root / "reports/phase2_is_trades.csv", parse_dates=["entry_time"])
print("B_audited n", len(base), "sum pnl", base.pnl.sum(), "equal entry times to phase2:", (base.entry_time.values == p2.entry_time.values).all(), "pnl max abs diff", np.abs(base.pnl.values - p2.pnl.values).max())
WM = ["2020-02", "2020-03", "2021-04", "2021-05", "2021-06"]
for v in ["A1", "A2", "A3", "B1", "B2", "C1", "D1", "D2", "E1"]:
    t = pd.read_csv(ex / f"{v}_trades.csv", parse_dates=["entry_time", "exit_time"])
    m = res[v]["metrics"]
    eq_end = 1e5 + t.pnl.sum()
    # compounding identity: product of (1+ret_on_equity)
    prod = np.prod(1 + t.ret_on_equity) if "ret_on_equity" in t else np.nan
    ym = t.entry_time.dt.strftime("%Y-%m")
    wm = t[ym.isin(WM)]
    wins = (t.pnl > 0).mean()
    print(f"{v}: n={len(t)} (json fills {m['fills']}) exit={t.exit_reason.value_counts().to_dict()} win={wins:.4f} (json {m['win_rate']:.4f}) "
          f"meanR={t.r_multiple.mean():.4f} (json mean_net_R {m['mean_net_R']:.4f}) sumpnl/1e5={t.pnl.sum()/1e5:.4f} prod-1={prod-1:.4f} (json tot {m['total_net_return']:.4f}) "
          f"worst5: n={len(wm)} pnl={wm.pnl.sum():.0f}; avg win usd={t.pnl[t.pnl>0].mean():.1f}")
# base worst5 and 2022, 2022-06
ym = base.entry_time.dt.strftime("%Y-%m")
print("base worst5 n", ym.isin(WM).sum(), "pnl", base.pnl[ym.isin(WM)].sum(), "share", base.pnl[ym.isin(WM)].sum() / base.pnl.sum(),
      "rest", base.pnl[~ym.isin(WM)].sum())
print("base 2022 n", (base.entry_time.dt.year == 2022).sum(), "pnl", base.pnl[base.entry_time.dt.year == 2022].sum())
print("base 2022-06 pnl (entry month)", base.pnl[ym == "2022-06"].sum(), "n", (ym == "2022-06").sum())
print("base avg win usd", base.pnl[base.pnl > 0].mean())
# negative months share
print("C1 worst5 pnl:", end=" ")
c1 = pd.read_csv(ex / "C1_trades.csv", parse_dates=["entry_time"])
cym = c1.entry_time.dt.strftime("%Y-%m")
print(c1.pnl[cym.isin(WM)].sum(), (cym.isin(WM)).sum(), "rest", c1.pnl[~cym.isin(WM)].sum())
for v in ["D1", "D2"]:
    t = pd.read_csv(ex / f"{v}_trades.csv", parse_dates=["entry_time"])
    yy = t.entry_time.dt.strftime("%Y-%m")
    print(v, "worst5 pnl", t.pnl[yy.isin(WM)].sum(), "partial count", (t.partial_bar >= 0).sum(), "exit reasons", t.exit_reason.value_counts().to_dict())
# Holm check
ps = {v: res[v]["paired_vs_base"]["p_one_sided"] for v in ["A1", "A2", "A3", "B1", "B2", "C1", "D1", "D2", "E1"]}
order = sorted(ps, key=ps.get)
m = len(order); run = 0; adj = {}
for i, k in enumerate(order):
    run = max(run, min(1.0, (m - i) * ps[k]))
    adj[k] = run
print("holm recomputed", adj)
