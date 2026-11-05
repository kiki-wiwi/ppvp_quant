# D-6 structural sensitivity only (no backtest, no performance): drop the 2021-03-02 01:00 hour and compare boxes/signals.
import _setup
import numpy as np
import pandas as pd
from config import BASELINE_STRATEGY as S
from ppvp_audit.common import is_run
from ppvp_engine import build_boxes
from strategy_state_machine import generate_signals

out = is_run()
df, sig, boxes, tr = out["df1h"], out["signals"], out["boxes"], out["result"].trades
t = pd.Timestamp("2021-03-02 01:00", tz="UTC"); H = pd.Timedelta(hours=1)
open_pos = tr[(tr["entry_time"] <= t) & (tr["exit_time"] > t)]
print("baseline position open during 2021-03-02 01:00:", len(open_pos))
nxt = tr[tr["entry_time"] > t].head(3)[["entry_time", "exit_reason"]]
print("next baseline entries after it:", [(str(a), b) for a, b in nxt.itertuples(index=False)])
boxes_w = boxes[(boxes["window_start"] <= t) & (boxes["window_end"] > t)]
print("boxes whose profile window contains the hour:", len(boxes_w), "| hour's share of their volume: max %.5f"
      % (df.loc[t, "volume"] / boxes_w["total_volume"]).max() if len(boxes_w) else "")
st = sig.loc[t - 3 * H:t + 3 * H, ["state", "box_id", "VAL", "zone_lo", "zone_hi"]]
print(st.to_string())

d2 = df.drop(index=[t])
b2, d42 = build_boxes(d2, S)
s2 = generate_signals(d2, S, b2, d42, None)
s1 = sig.drop(index=[t])
print("\nboxes: base", len(boxes), "dropped-hour", len(b2), "| box tables equal on VAL/VAH/POC/available_time:",
      bool(np.allclose(boxes[["VAL", "VAH", "POC"]].to_numpy(), b2[["VAL", "VAH", "POC"]].to_numpy(), rtol=0, atol=0)
           and (boxes["available_time"].to_numpy() == b2["available_time"].to_numpy()).all()) if len(boxes) == len(b2) else "n/a")
for c in ("sig_raw", "signal", "breakdown", "state", "box_id"):
    print(f"  {c} identical on all common rows:", bool((s1[c].to_numpy() == s2[c].to_numpy()).all()))
for c in ("VAL", "VAH"):
    print(f"  {c} identical:", bool(np.array_equal(s1[c].to_numpy(), s2[c].to_numpy(), equal_nan=True)))
w = slice(t + H, t + 24 * H)
rv = (s2.loc[w, "vsma"] / s1.loc[w, "vsma"] - 1).abs().max(); ra = (s2.loc[w, "atr"] / s1.loc[w, "atr"] - 1).abs().max()
print("max rel change next 24h: vsma %.4f, atr %.4f" % (rv, ra))
# breakdown candidates in the window where vsma changed: bearish close below zone_lo
o, c = df["open"], df["close"]
cand = [(str(i)) for i in s1.loc[t + H:t + 21 * H].index if (c[i] < o[i]) and (c[i] < s1.loc[i, "zone_lo"])]
print("bearish closes below zone_lo in the 20 h after the hour:", cand)
sr = s1.index[s1["signal"].to_numpy()]
d_sl = (s2.loc[sr, "sl"] / s1.loc[sr, "sl"] - 1).abs()
print("signals with any SL change:", [(str(i), float("%.2g" % v)) for i, v in d_sl[d_sl > 0].items()])
