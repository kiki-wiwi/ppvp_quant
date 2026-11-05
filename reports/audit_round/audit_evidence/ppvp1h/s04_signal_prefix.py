"""s04: signal-level prefix truncation on IS.

For every raw signal bar s (120) and, for each signal, the bars just before / at the available_time of the
boxes active at its first-test low bar, second-test start and signal bar: recompute boxes + signals on the IS
prefix ending at the cut bar (inclusive) and require the whole signal frame on [0, cut] to equal the full-IS
frame bit for bit (every column, NaN-aware). Also checks that the baseline signal frame does not depend on the
daily warm-up series (regime filter off -> no daily bar is read by any signal).
"""
import sys, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/ppvp_quant")
HERE = Path(__file__).resolve().parent
from ppvp_audit.common import load_is_market
from strategy_state_machine import generate_signals
from config import BASELINE_STRATEGY as S

d = pickle.load(open(HERE / "is_objects.pkl", "rb"))
df1h, boxes, sig = d["df1h"], d["boxes"], d["sig"]
_, _, spot = load_is_market()
H = pd.Timedelta(hours=1)
avail = pd.DatetimeIndex(boxes["available_time"])
pos = {t: i for i, t in enumerate(df1h.index)}


def frame_equal(a: pd.DataFrame, b: pd.DataFrame) -> tuple[bool, str]:
    if list(a.columns) != list(b.columns) or not a.index.equals(b.index):
        return False, "shape/index"
    for c in a.columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype != y.dtype:
            return False, f"dtype {c}"
        if x.dtype.kind == "f":
            if not np.array_equal(x, y, equal_nan=True):
                return False, c
        elif not np.array_equal(x, y):
            return False, c
    return True, ""


# daily independence
sig_nodaily = generate_signals(df1h, S, spot_daily=None)
ok, m = frame_equal(sig_nodaily, sig)
print("signal frame identical without daily warm-up series:", ok, m, " regime_raw any:", bool(sig["regime_raw"].any()))

sbars = np.flatnonzero(sig["sig_raw"].to_numpy())
t1b = sig["t1_bar"].to_numpy()[sbars]
t2s = sig["t2_start"].to_numpy()[sbars]
bid = sig["box_id"].to_numpy()
cuts = set(int(s) for s in sbars) | set(int(s) - 1 for s in sbars)
for s, a, b2 in zip(sbars, t1b, t2s):
    for bar in (a, b2, s):
        k = bid[bar]
        A = avail[k]
        if A - H in pos:
            cuts.add(pos[A - H])
        if A in pos:
            cuts.add(pos[A])
cuts = sorted(cuts)
t0 = time.time()
fails = []
for j in cuts:
    sub = df1h.iloc[: j + 1]
    st = generate_signals(sub, S, spot_daily=spot)
    ok, m = frame_equal(st, sig.iloc[: j + 1])
    if not ok:
        fails.append((j, m))
print(f"signal prefix cuts: {len(cuts)} cuts, failures {len(fails)} ({time.time()-t0:.0f}s)")
print(fails[:10])
