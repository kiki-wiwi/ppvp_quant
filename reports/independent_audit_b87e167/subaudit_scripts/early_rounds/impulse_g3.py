"""Recompute G3 wild cluster restricted bootstrap-t p-values (Webb weights) from archived matched files,
using the frozen seeds SEED_BASE+350+v (plan section 8.6). Also G2 using SEED_BASE+250+v.
Usage: python -I impulse_g3.py <repo_root>"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
d = root / "reports/impulse_vs_box_round/cp3"
summ = json.loads((d / "summary.json").read_text())
SEED_BASE = 20261106
VERS = ["A1_box_opt", "A2_box_origin", "A3_box_half", "A4_box_flush", "A5_box_origin_z",
        "B1_pure_long", "B2_pure_long_sl", "B3_pure_bi_mom", "B4_pure_1bar", "B5_pure_bi_mom_z"]
W = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def wild_p(x, months, seed, B=99999):
    df = pd.DataFrame({"x": x, "m": months})
    g = df.groupby("m", sort=True)["x"].agg(["sum", "count"])
    S = g["sum"].values; K = g["count"].values; N = K.sum(); G = len(S)
    xb = S.sum() / N
    se = np.sqrt(G / (G - 1) * ((S - xb * K) ** 2).sum() / N ** 2)
    t = xb / se
    rng = np.random.default_rng(seed)
    w = rng.choice(W, size=(B, G))
    pcount = 0
    for i in range(0, B, 4096):
        ww = w[i:i + 4096]
        Ss = ww * S
        xs = Ss.sum(1) / N
        ses = np.sqrt(G / (G - 1) * ((Ss - xs[:, None] * K) ** 2).sum(1) / N ** 2)
        ts = np.where(ses == 0, 0.0, xs / np.where(ses == 0, 1, ses))
        pcount += (ts >= t).sum()
    return xb, t, pcount / B, G


for v, name in enumerate(VERS):
    m = pd.read_csv(d / f"matched_{name}.csv.gz", parse_dates=["entry_time"])
    x = m["excess_main_17bp"].dropna()
    months = m.loc[x.index, "entry_time"].dt.strftime("%Y-%m").values
    xb, t, p, G = wild_p(x.values, months, SEED_BASE + 350 + v)
    js = summ["versions"][name]["G3"]
    print(f"G3 {name}: mean={xb:.6f} (json {js['mean']:.6f}) t={t:.4f} (json {js['t']:.4f}) p={p:.6f} (json {js['p']:.6f}) G={G} (json {js['n_clusters']}) n={len(x)} (json {js['n']})")
# G2 for a couple of versions
for v in (0, 6):
    name = VERS[v]
    t_ = pd.read_csv(d / f"{name}__main_17bp__risk_0.5pct_trades.csv.gz", parse_dates=["entry_time"])
    xb, t, p, G = wild_p(t_.net_bps.values, t_.entry_time.dt.strftime("%Y-%m").values, SEED_BASE + 250 + v)
    js = summ["versions"][name]["G2"]
    print(f"G2 {name}: mean={xb:.6f} (json {js['mean']:.6f}) t={t:.4f} (json {js['t']:.4f}) p={p:.6f} (json {js['p']:.6f})")
