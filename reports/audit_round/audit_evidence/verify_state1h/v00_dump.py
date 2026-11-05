"""v00: BASELINE only, via ppvp_audit.common.is_run (IS-truncated). Dump objects + the exact kernel args."""
import pickle, sys, time
from pathlib import Path
sys.path.insert(0, "/home/user/ppvp_quant")
import numpy as np, pandas as pd
from config import BASELINE_BACKTEST, BASELINE_STRATEGY, IS_END, IS_START
from ppvp_audit.common import is_run
import strategy_state_machine as ssm

OUT = Path(__file__).resolve().parent
t0 = time.time()
out = is_run(BASELINE_STRATEGY, BASELINE_BACKTEST)
df1h, df4h, boxes, sig, res = out["df1h"], out["df4h"], out["boxes"], out["signals"], out["result"]
assert df1h.index.min() >= IS_START and df1h.index.max() < IS_END
assert sig.index.max() < IS_END
assert pd.DatetimeIndex(boxes["available_time"]).max() <= IS_END
args = ssm.state_machine_args(df1h, BASELINE_STRATEGY, boxes)
print("seconds", round(time.time() - t0, 1), "| 1H", len(df1h), df1h.index.min(), df1h.index.max(), "| boxes", len(boxes))
print("sig_raw", int(sig["sig_raw"].sum()), "signal", int(sig["signal"].sum()), "trades", len(res.trades),
      res.trades["exit_reason"].value_counts().to_dict())
print("params", args[14:])
with open(OUT / "v00_objs.pkl", "wb") as f:
    pickle.dump({"df1h": df1h, "df4h": df4h, "boxes": boxes, "signals": sig, "trades": res.trades, "args": args}, f)
