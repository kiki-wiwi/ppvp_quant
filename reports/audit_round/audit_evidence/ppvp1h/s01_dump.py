"""s01: run the baseline on IS via the sanctioned entry point and dump IS-only objects to scratch."""
import os, sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/ppvp_quant")
from ppvp_audit.common import is_run, load_is_market
from config import IS_END, BASELINE_STRATEGY

OUT = Path(__file__).resolve().parent
out = is_run()
df1h, fund, spot = load_is_market()
assert df1h.index.max() < IS_END
boxes, df4h, sig, res = out["boxes"], out["df4h"], out["signals"], out["result"]
print("1H bars", len(df1h), df1h.index.min(), df1h.index.max())
print("4H bars", len(df4h), df4h.index.min(), df4h.index.max(), "n_bars uniq", df4h["n_bars"].unique())
print("boxes", len(boxes), boxes["available_time"].min(), boxes["available_time"].max())
print("trades", len(res.trades), "raw signals", int(sig["sig_raw"].sum()))
print("report IS", {k: v for k, v in out["report"]["IS"].items() if k in ("sharpe", "n_trades")})
with open(OUT / "is_objects.pkl", "wb") as f:
    pickle.dump({"df1h": df1h, "df4h": df4h, "boxes": boxes, "sig": sig, "trades": res.trades}, f)
print(boxes.head(3).T)
