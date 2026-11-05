"""s02b: resolve the two 'False' checks of s02 (CSV %.10g rounding) and quantify the gross-R denominator nuance."""
import pickle, sys
sys.path.insert(0, "/home/user/ppvp_quant")
from pathlib import Path
import numpy as np, pandas as pd
from config import BASELINE_BACKTEST as B
D = Path(__file__).resolve().parent
X = pickle.load(open(D / "baseline.pkl", "rb")); I = pickle.load(open(D / "s01_indep.pkl", "rb"))
L = pd.read_csv("/home/user/ppvp_quant/reports/audit_round/trades_B_original.csv")
T2 = I["indep_trades"]; sig = X["signals"]; tr = X["trades"]
rel = lambda a, b: float(np.max(np.abs(np.asarray(a, float) / np.asarray(b, float) - 1)))
print("exit_ref rel diff (CSV vs independent):", rel(L["exit_ref"], T2["exit_ref"]))
print("SL rel diff (CSV vs signal SL):", rel(L["sl"], sig["sl"].to_numpy()[L["entry_bar"]]))
print("SL exact (in-memory trades vs signal SL):", bool(np.array_equal(tr["sl"].to_numpy(), sig["sl"].to_numpy()[tr["entry_bar"]])))
c = X["df1h"]["close"].to_numpy(float)
q, ep, sl = T2["qty"].to_numpy(), T2["entry_px"].to_numpy(), T2["sl"].to_numpy()
gref = q * (T2["exit_ref"].to_numpy() - T2["entry_ref"].to_numpy())
r_fill = gref / (q * (ep - sl)); r_ref = gref / (q * (T2["entry_ref"].to_numpy() - sl))
slm = (T2["reason"] == "SL").to_numpy()
print(f"gross R before costs, fill-based denominator (published): mean {r_fill.mean():.4f}; SL trades range [{r_fill[slm].min():.4f}, {r_fill[slm].max():.4f}]")
print(f"gross R before costs, reference denominator q*(close-SL):  mean {r_ref.mean():.4f}; SL trades range [{r_ref[slm].min():.4f}, {r_ref[slm].max():.4f}]")
d = (ep - sl) / ep
print(f"stop distance d_sl: min {d.min():.5f} median {np.median(d):.5f}; trades with d_sl < 0.5%: {int((d<0.005).sum())}")
