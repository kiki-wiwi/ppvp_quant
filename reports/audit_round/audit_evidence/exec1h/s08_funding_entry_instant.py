"""s08: size of the documented 'close fill happens just after the close instant' funding convention (IS baseline)."""
import pickle, sys
sys.path.insert(0, "/home/user/ppvp_quant")
from pathlib import Path
import numpy as np, pandas as pd
D = Path(__file__).resolve().parent
X = pickle.load(open(D / "baseline.pkl", "rb"))
df, tr, fund = X["df1h"], X["trades"], X["fund"]
o = df["open"].to_numpy(float)
rate = fund["funding_rate"]
hit = tr[tr["entry_time"].isin(rate.index)]
amt = []
for _, t in hit.iterrows():
    T = t["entry_time"]; i = df.index.get_loc(T)          # bar opening at T (first bar after entry)
    amt.append(t["qty"] * o[i] * rate.loc[T])
amt = np.array(amt)
print(f"trades entering exactly at a settlement: {len(hit)}; the settlement they skip totals {amt.sum():+.2f} USDT "
      f"(per trade {np.round(amt, 2).tolist()}); total baseline funding {tr['funding'].sum():.2f}; net pnl {tr['pnl'].sum():.2f}")
print("entry times:", [str(x) for x in hit["entry_time"]])
