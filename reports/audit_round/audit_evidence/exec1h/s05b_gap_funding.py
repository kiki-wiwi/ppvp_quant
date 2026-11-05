"""s05b: engine_path funding boundary uses a sub-bar index rule (charge_from = entry_index + 2) instead of the documented
'entry time < T'. Synthetic: one settlement inside a data gap right after a close entry. Plus IS gap count (baseline data)."""
import sys, pickle
sys.path.insert(0, "/home/user/ppvp_quant")
from pathlib import Path
import numpy as np, pandas as pd
from config import BASELINE_BACKTEST as B
from backtester import run_backtest
from ppvp_audit.engine_path import run_path
H = pd.Timedelta(hours=1); T0 = pd.Timestamp("2021-03-01 00:00", tz="UTC")
rows = [(100,101,99,100)]*3 + [(100,101,99,100.6)] + [(100.6,100.8,100.2,100.4)]*3
idx = list(pd.date_range(T0, periods=4, freq="h", tz="UTC")) + [T0 + 10*H, T0 + 11*H, T0 + 12*H]   # gap 04:00 -> 10:00 covers 08:00
df = pd.DataFrame(np.asarray(rows, float), columns=["open","high","low","close"], index=pd.DatetimeIndex(idx)); df["volume"] = 100.0
s = pd.DataFrame({"signal": False, "sl": np.nan, "tp": np.nan, "poc_tp": np.nan}, index=df.index); s.iloc[3, :3] = [True, 98.0, 110.0]
F = pd.DataFrame({"funding_rate": [0.001], "funding_interval_hours": 8}, index=pd.DatetimeIndex([T0 + 8*H], name="funding_time"))
a = run_backtest(df, s, B, F); r = run_path(df, s, B, F)
print(f"entry {a.trades['entry_time'].iat[0]} (close of 03:00 bar); next bar opens {df.index[4]}; settlement 08:00 held -> should be charged")
print(f"  1H engine funding {a.trades['funding'].iat[0]:.6f} (n={a.trades['n_funding'].iat[0]}); engine_path funding {r['trades']['funding'].iat[0]:.6f}")
D = Path(__file__).resolve().parent
X = pickle.load(open(D / "baseline.pkl", "rb"))
gaps = np.diff(X["df1h"].index.as_unit("ns").asi8) != 3600 * 10**9
print("IS 1H bars with a gap to the next bar:", int(gaps.sum()))
