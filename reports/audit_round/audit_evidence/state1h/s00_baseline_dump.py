"""s00: run the BASELINE only via ppvp_audit.common.is_run (IS-truncated), dump objects for later scripts."""
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/user/ppvp_quant")
import numpy as np
import pandas as pd

from config import BASELINE_BACKTEST, BASELINE_STRATEGY, IS_END
from ppvp_audit.common import is_run

OUT = Path(__file__).resolve().parent
t0 = time.time()
out = is_run(BASELINE_STRATEGY, BASELINE_BACKTEST)
print("is_run seconds", round(time.time() - t0, 1))
df1h, sig, boxes, res = out["df1h"], out["signals"], out["boxes"], out["result"]
assert df1h.index.max() < IS_END and sig.index.max() < IS_END
print("1H bars", len(df1h), df1h.index.min(), df1h.index.max())
print("boxes", len(boxes), "last available", boxes["available_time"].max())
print("sig_raw", int(sig["sig_raw"].sum()), "signal", int(sig["signal"].sum()))
tr = res.trades
print("trades", len(tr), tr["exit_reason"].value_counts().to_dict())
print("IS report keys", list(out["report"].keys()))
rep = out["report"]["IS"] if "IS" in out["report"] else out["report"]
print({k: rep[k] for k in list(rep)[:12]})
with open(OUT / "baseline_out.pkl", "wb") as f:
    pickle.dump({"df1h": df1h, "df4h": out["df4h"], "boxes": boxes, "signals": sig, "trades": tr}, f)
print("dumped")
