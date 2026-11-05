"""v01: run the IS baseline through the sanctioned entry point and dump the objects (IS only)."""
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/ppvp_quant")
from ppvp_audit.common import is_run, load_is_market
from config import IS_END, BASELINE_STRATEGY as S
V = Path(__file__).resolve().parent
out = is_run()
df1h, df4h, boxes, sig, res = out["df1h"], out["df4h"], out["boxes"], out["signals"], out["result"]
assert df1h.index.max() < IS_END and boxes["available_time"].max() <= IS_END
print("1H", len(df1h), df1h.index.min(), df1h.index.max())
d = np.diff(df1h.index.asi8) if df1h.index.asi8.dtype == np.int64 else None
print("1H contiguous hourly:", bool((pd.Series(df1h.index).diff().dropna() == pd.Timedelta(hours=1)).all()))
print("4H", len(df4h), "contiguous 4h:", bool((pd.Series(df4h.index).diff().dropna() == pd.Timedelta(hours=4)).all()),
      "n_bars unique", sorted(df4h["n_bars"].unique()) if "n_bars" in df4h else "n/a")
print("boxes", len(boxes), "raw signals", int(sig["sig_raw"].sum()), "trades", len(res.trades))
print("IS sharpe", out["report"]["IS"]["sharpe"])
pickle.dump({"df1h": df1h, "df4h": df4h, "boxes": boxes, "sig": sig, "trades": res.trades}, open(V / "is_objs.pkl", "wb"))
print("trade columns:", list(res.trades.columns))
