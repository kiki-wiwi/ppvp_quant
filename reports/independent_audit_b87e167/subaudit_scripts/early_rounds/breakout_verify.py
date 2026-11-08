"""Recompute Channel vs VAH headline numbers and Webb p-values from archived daily/trade files.
Usage: python -I breakout_verify.py <repo_root>"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
d = root / "reports/channel_vs_vah_round/results"
W = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def load(v, c):
    t = pd.read_csv(d / f"{v}__{c}_trades.csv.gz", parse_dates=["entry_time", "exit_time"])
    dd = pd.read_csv(d / f"{v}__{c}_daily.csv", index_col=0, parse_dates=True)
    return t, dd


for v in ("P_channel", "V_vah"):
    for c in ("gross_0", "main_17", "harsh_34"):
        t, dd = load(v, c)
        eq = dd["equity"]
        r = dd["return"]
        sh = r.mean() / r.std(ddof=1) * np.sqrt(365)
        mdd = (eq / np.maximum.accumulate(np.concatenate([[1e5], eq.values]))[1:] - 1).min()
        months = t.entry_time.dt.strftime("%Y-%m")
        print(f"{v} {c}: n={len(t)} months={months.nunique()} tot={eq.iloc[-1]/1e5-1:+.5%} final={eq.iloc[-1]:.2f} ledger_tot={t.net_pnl.sum()/1e5:+.5%} "
              f"gross_bps={t.gross_bps.mean():.2f} net_bps={t.net_bps.mean():.2f} netR={t.net_R.mean():.4f} sharpe={sh:.3f} mdd={mdd:.3%} ndays={len(r)} "
              f"fees={t.fees.sum():.2f} funding={t.funding.sum():.2f}")
        if c == "main_17":
            h1 = t[t.entry_time < "2022-01-01"].net_bps.mean(); h2 = t[t.entry_time >= "2022-01-01"].net_bps.mean()
            ms = t.groupby(months).net_bps.sum().sort_values(ascending=False)
            top3 = ms.index[:3].tolist()
            rest = t[~months.isin(top3)]
            yr = t.groupby(t.entry_time.dt.year).net_bps.mean().round(2).to_dict()
            print(f"   halves net_bps {h1:.2f} / {h2:.2f}; top3 months {top3}; rest n={len(rest)} mean={rest.net_bps.mean():.2f}; by year {yr}; "
                  f"loss>risk budget n={(t.risk_budget_exceeded).sum()}; cross-month trades={(t.entry_time.dt.strftime('%Y-%m')!=t.exit_time.dt.strftime('%Y-%m')).sum()}")

# Webb p-values, plan section 6
P = pd.read_csv(d / "P_channel__main_17_daily.csv", index_col=0, parse_dates=True)["return"]
V = pd.read_csv(d / "V_vah__main_17_daily.csv", index_col=0, parse_dates=True)["return"]
assert (P.index == V.index).all()
D = V - P
mon = P.index.strftime("%Y-%m")
G = len(np.unique(mon)); N = len(P)
rng = np.random.default_rng(2026110701)
w = rng.choice(W, size=(99999, G))
print("Webb matrix sha256 (float64 C-order bytes):", hashlib.sha256(np.ascontiguousarray(w).tobytes()).hexdigest())


def tstat(S, K):
    mu = S.sum() / N
    se = np.sqrt(G / (G - 1) * ((S - mu * K) ** 2).sum() / N ** 2)
    return mu, se


res = {}
for name, x in (("P", P), ("V", V), ("D", D)):
    g = pd.DataFrame({"x": x.values, "m": mon}).groupby("m", sort=True)["x"].agg(["sum", "count"])
    S = g["sum"].values; K = g["count"].values
    mu, se = tstat(S, K)
    t = mu / se
    Ss = w * S
    mus = Ss.sum(1) / N
    ses = np.sqrt(G / (G - 1) * ((Ss - mus[:, None] * K) ** 2).sum(1) / N ** 2)
    ts = np.where(ses == 0, 0, mus / np.where(ses == 0, 1, ses))
    if name == "D":
        p = (1 + (np.abs(ts) >= abs(t)).sum()) / 100000
    else:
        p = (1 + (ts >= t).sum()) / 100000
    res[name] = p
    print(f"{name}: mean daily bps={mu*1e4:.4f} t={t:.4f} p={p:.5f}")
# Holm with fixed size 3
order = sorted(res, key=lambda k: (res[k], "PVD".index(k)))
run = 0
for i, k in enumerate(order):
    run = max(run, min(1, (3 - i) * res[k]))
    print("holm", k, round(run, 5))
h1 = D[D.index < "2022-01-01"].mean() * 1e4; h2 = D[D.index >= "2022-01-01"].mean() * 1e4
print(f"D halves {h1:.4f} / {h2:.4f}")
s = json.loads((d / "summary.json").read_text())
print(json.dumps(s.get("conclusions", s.get("decisions", {})), ensure_ascii=False)[:1500])
