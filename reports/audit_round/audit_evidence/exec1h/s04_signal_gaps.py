"""s04: baseline-only structural facts relevant to the C1 order bookkeeping (no variant is run):
gaps between consecutive baseline tradable/raw signals; parity of engine_path (market mode) with backtester on the baseline."""
import pickle, sys
sys.path.insert(0, "/home/user/ppvp_quant")
from pathlib import Path
import numpy as np, pandas as pd
from config import BASELINE_BACKTEST as B
from backtester import run_backtest
from ppvp_audit.engine_path import run_path

D = Path(__file__).resolve().parent
X = pickle.load(open(D / "baseline.pkl", "rb"))
sig, df, fund = X["signals"], X["df1h"], X["fund"]
t = np.flatnonzero(sig["signal"].to_numpy()); r = np.flatnonzero(sig["sig_raw"].to_numpy())
g = np.diff(t); gr = np.diff(r)
print("tradable signals", len(t), " min gap", g.min(), " gaps<=3:", int((g <= 3).sum()), " gaps<=4:", int((g <= 4).sum()),
      " gap value counts (<=10):", pd.Series(g[g <= 10]).value_counts().sort_index().to_dict())
print("raw signals", len(r), " min gap", gr.min(), " gaps<=3:", int((gr <= 3).sum()))
# engine_path market mode vs backtester on the baseline (B_original only)
a = run_backtest(df, sig, B, fund)
b = run_path(df, sig, B, fund)
print("engine_path(market) equity identical:", bool(np.array_equal(a.equity.to_numpy(), b["equity"].to_numpy())))
for col in ("entry_bar", "exit_bar", "entry_px", "exit_px", "qty", "fees", "funding", "pnl"):
    print(f"  {col}: identical={bool(np.array_equal(a.trades[col].to_numpy(), b['trades'][col].to_numpy()))}")
print("  hold_bars identical:", bool(np.array_equal(a.trades["hold_bars"].to_numpy(), b["trades"]["hold_bars"].to_numpy())))
print("  orders status counts (market mode):", b["orders"]["status"].value_counts().to_dict())
