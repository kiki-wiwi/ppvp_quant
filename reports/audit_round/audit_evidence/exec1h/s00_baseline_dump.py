"""Baseline (B_original) IS run via the sanctioned entry point; dump objects for the exec/accounting audit."""
import pickle, sys
sys.path.insert(0, "/home/user/ppvp_quant")
from pathlib import Path
import numpy as np, pandas as pd
from ppvp_audit.common import is_run, load_is_market
from config import IS_END, IS_START, BASELINE_STRATEGY, BASELINE_BACKTEST

D = Path(__file__).resolve().parent
out = is_run()
df1h, fund, spot = load_is_market()
assert df1h.index.max() < IS_END and fund.index.max() <= IS_END
res = out["result"]
tr = res.trades
print("bars", len(out["df1h"]), out["df1h"].index[0], out["df1h"].index[-1])
print("funding rows", len(fund), fund.index[0], fund.index[-1])
print("trades", len(tr), tr["exit_reason"].value_counts().to_dict())
print("IS sharpe", out["report"]["IS"]["sharpe"], "total", out["report"]["IS"]["total_return"],
      "open_at_end", out["report"]["IS"]["trades_open_at_end"])
print("final equity", res.equity.iloc[-1], res.equity.index[-1])
pickle.dump({"df1h": out["df1h"], "df4h": out["df4h"], "boxes": out["boxes"], "signals": out["signals"],
             "equity": res.equity, "position": res.position, "trades": tr, "levels": res.levels,
             "report": out["report"], "fund": fund, "spot": spot, "start_time": res.start_time},
            open(D / "baseline.pkl", "wb"))
