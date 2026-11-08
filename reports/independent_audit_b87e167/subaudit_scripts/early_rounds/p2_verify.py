"""Independent spot-check of Phase 2 IS headline numbers from reports/phase2_is_trades.csv.
Usage: python -I p2_verify.py <repo_root>
Read-only; prints results."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

root = Path(sys.argv[1])
t = pd.read_csv(root / "reports/phase2_is_trades.csv", parse_dates=["entry_time", "exit_time"])
print("n trades", len(t))
print("exit reasons", t.exit_reason.value_counts().to_dict())
print("entry range", t.entry_time.min(), t.entry_time.max())
print("exit range", t.exit_time.min(), t.exit_time.max())
print("win rate (pnl>0)", (t.pnl > 0).mean(), (t.pnl > 0).sum())
print("mean R", t.r_multiple.mean(), " mean R ex funding", t.r_ex_funding.mean())
gp = t.pnl[t.pnl > 0].sum(); gl = -t.pnl[t.pnl < 0].sum()
print("profit factor", gp / gl)
print("avg hold bars", t.hold_bars.mean())
print("sum pnl", t.pnl.sum())
# compounding identity
prod = np.prod(1 + t.ret_on_base)
print("prod(1+ret_on_base)", prod, "-> cum return", prod - 1)
print("sum pnl / 100000", t.pnl.sum() / 1e5)
# winners/losers R
print("mean R winners", t.r_multiple[t.pnl > 0].mean(), "losers", t.r_multiple[t.pnl <= 0].mean())
print("median rr_fill winners", t.rr_fill[t.pnl > 0].median(), "losers", t.rr_fill[t.pnl <= 0].median())
# random-walk expected TP
tp = (t.exit_reason == "TP").sum()
mu = t.p_rw.sum(); sd = np.sqrt((t.p_rw * (1 - t.p_rw)).sum())
print("TP", tp, "E[TP]", mu, "sd", sd, "z", (tp - mu) / sd)
# exact Poisson-binomial lower tail P(X <= tp)
p = t.p_rw.values
dist = np.zeros(len(p) + 1); dist[0] = 1.0
for pi in p:
    dist[1:] = dist[1:] * (1 - pi) + dist[:-1] * pi
    dist[0] = dist[0] * (1 - pi)
print("exact P(X<=TP)", dist[: tp + 1].sum())
# by year
t["year"] = t.entry_time.dt.year
g = t.groupby("year").agg(n=("pnl", "size"), win=("pnl", lambda s: (s > 0).mean()), meanR=("r_multiple", "mean"), pnl=("pnl", "sum"))
print(g)
# worst months by entry? report says worst 5 months 2021-05,2020-03,2021-06,2020-02,2021-04
wm = ["2021-05", "2020-03", "2021-06", "2020-02", "2021-04"]
t["ym"] = t.entry_time.dt.strftime("%Y-%m")
sub = t[t.ym.isin(wm)]
print("trades entered in worst5 months", len(sub), "meanR", sub.r_multiple.mean(), "pnl", sub.pnl.sum(), "all SL?", (sub.exit_reason == "SL").all(), "median d_sl", sub.d_sl.median())
rest = t[~t.ym.isin(wm)]
print("rest", len(rest), "meanR", rest.r_multiple.mean(), "median d_sl", rest.d_sl.median())
print("R share worst5", sub.r_multiple.sum(), "/", t.r_multiple.sum())
# biggest d_sl trade
i = t.d_sl.idxmax()
print("max d_sl", t.loc[i, ["entry_time", "d_sl", "ret_on_base", "r_multiple"]].to_dict())
# near-cost trades
near = t[t.d_tp < 3 * 0.0011]
print("d_tp<33bp", len(near), "TP with pnl<0", ((t.exit_reason == "TP") & (t.pnl < 0)).sum())
# first-bar TP
print("TP with hold_bars==1", ((t.exit_reason == "TP") & (t.hold_bars == 1)).sum())
print("hold<=4h", (t.hold_bars <= 4).sum(), "hold>12", (t.hold_bars > 12).sum(), "hold>24", (t.hold_bars > 24).sum(), "hold>48", (t.hold_bars > 48).sum())
# share of net loss by hold>24
tot = t.pnl.sum()
for h in (12, 24, 48):
    s = t[t.hold_bars > h]
    print(f"hold>{h}: n={len(s)} share_netloss={s.pnl.sum()/tot:.3f} meanR={s.r_multiple.mean():.3f} rest meanR={t[t.hold_bars<=h].r_multiple.mean():.3f}")
# SMA200 below & slope down
ds = t[(t.above_sma200 == 0) & (t.sma200_up == 0)]
print("daily_sma flagged", len(ds), "share net loss", ds.pnl.sum() / tot)
m = t[t.mom20d < 0]
print("mom20d<0", len(m), "share net loss", m.pnl.sum() / tot)
# rr<0.5 group
g05 = t[t.rr_fill < 0.5]
print("rr<0.5 n", len(g05), "mean r_ex_funding", g05.r_ex_funding.mean(), "pnl", g05.pnl.sum(), "TP hit", (g05.exit_reason == "TP").mean(), "p_rw", g05.p_rw.mean())
# cost decomposition
print("cost_r mean", t.cost_r.mean())

# JSON headline
j = json.loads((root / "reports/phase2_baseline.json").read_text())
print(json.dumps(j["headline"], indent=1)[:3000])
