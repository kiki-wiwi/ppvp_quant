"""Recompute Impulse CP3 main-config metrics from archived ledgers and daily equity.
Usage: python -I impulse_verify.py <repo_root>"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
d = root / "reports/impulse_vs_box_round/cp3"
mc = pd.read_csv(d / "main_comparison.csv").set_index("version")
a80 = pd.read_csv(d / "all80_comparison.csv")
print("all80 rows", len(a80), "columns", a80.columns.tolist()[:12])
summ = json.loads((d / "summary.json").read_text())
rows = []
for v in mc.index:
    t = pd.read_csv(d / f"{v}__main_17bp__risk_0.5pct_trades.csv.gz", parse_dates=["entry_time", "trigger_time", "exit_time"])
    eq = pd.read_csv(d / f"{v}__main_17bp__risk_0.5pct_daily_equity.csv", parse_dates=["utc_day"]).set_index("utc_day")["equity"]
    # daily returns: equity rows include initial anchor at 2020-03-01 00:00? check count
    r = eq.pct_change().dropna()
    sh = r.mean() / r.std(ddof=1) * np.sqrt(365)
    tr = eq.iloc[-1] / 1e5 - 1
    netsum = t.net_pnl.sum()
    rows.append(dict(version=v, n=len(t), n_json=mc.loc[v, "n"], net_bps=t.net_bps.mean(), net_bps_json=mc.loc[v, "net_bps"],
                     win=(t.net_pnl > 0).mean(), win_json=mc.loc[v, "win_rate"], sharpe=sh, sharpe_json=mc.loc[v, "sharpe"],
                     n_ret=len(r), tot=tr, tot_json=mc.loc[v, "total_return"], ledger_tot=netsum / 1e5,
                     trig_min=t.trigger_time.min(), exit_max=t.exit_time.max(),
                     reasons=t.reason_name.value_counts().to_dict()))
out = pd.DataFrame(rows).set_index("version")
pd.set_option("display.width", 250)
print(out[["n", "n_json", "net_bps", "net_bps_json", "win", "win_json", "sharpe", "sharpe_json", "n_ret", "tot", "tot_json", "ledger_tot"]].round(6).to_string())
print(out[["trig_min", "exit_max"]].to_string())
for v in out.index:
    print(v, out.loc[v, "reasons"], "| json", mc.loc[v, "exit_reasons"])
print("gates in main_comparison:")
print(mc[["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8z", "G2_raw_p", "G3_raw_p"]].to_string())
print("decisions:", json.dumps(summ["decisions"], ensure_ascii=False)[:3000])
