"""s11: worked example (trace) of anchor selection, pivot confirmation, window, binning and VA expansion for box 13,
the signal box of the first IS trade (signal bar 2020-01-22 19:00 UTC)."""
import sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from indep import hist_uniform
d = pickle.load(open(HERE / "is_objects.pkl", "rb"))
df1h, df4h, boxes = d["df1h"], d["df4h"], d["boxes"]
k = 13
b = boxes.iloc[k]
a_i, b_i = int(b.pivot_a_idx), int(b.pivot_b_idx)
print(f"box {k}: pivot a = 4H#{a_i} {df4h.index[a_i]} type {b.pivot_a_type}; pivot b = 4H#{b_i} {df4h.index[b_i]} type {b.pivot_b_type}")
for j, typ in ((a_i, b.pivot_a_type), (b_i, b.pivot_b_type)):
    hh, ll = df4h["high"].to_numpy(), df4h["low"].to_numpy()
    if "H" in typ:
        print(f"  4H#{j} high {hh[j]} > left max {hh[j-5:j].max()} and >= right max {hh[j+1:j+6].max()}")
    if "L" in typ:
        print(f"  4H#{j} low {ll[j]} < left min {ll[j-5:j].min()} and <= right min {ll[j+1:j+6].min()}")
print(f"  previous box {k-1} pivots {boxes.pivot_a_time.iloc[k-1]} -> {boxes.pivot_b_time.iloc[k-1]}; next box {k+1} pivots {boxes.pivot_a_time.iloc[k+1]} -> {boxes.pivot_b_time.iloc[k+1]}")
print(f"  confirmation = close of 4H#{b_i+5} ({df4h.index[b_i+5]}) = {df4h['available_time'].iloc[b_i+5]} == available_time {b.available_time}")
m = (df1h.index >= b.window_start) & (df1h.index < b.window_end)
w = df1h[m]
print(f"  window [{b.window_start}, {b.window_end}) -> {len(w)} 1H bars ({w.index[0]} .. {w.index[-1]}); excluded post-pivot 1H bars: "
      f"{b.window_end} .. {b.available_time - pd.Timedelta(hours=1)} (20 bars)")
edges = np.linspace(w["low"].min(), w["high"].max(), 51)
hist = hist_uniform(w["low"].to_numpy(), w["high"].to_numpy(), w["volume"].to_numpy(), edges)
print(f"  P_low {edges[0]} P_high {edges[-1]} bin width {edges[1]-edges[0]:.4f}; total volume {hist.sum():.3f}")
poc = int(np.argmax(hist))
print(f"  POC bin {poc} [{edges[poc]:.4f}, {edges[poc+1]:.4f}) vol {hist[poc]:.3f} -> POC {0.5*(edges[poc]+edges[poc+1]):.4f} (engine {b.POC})")
lo = hi = poc; acc = hist[poc]; steps = []
while acc < 0.7 * hist.sum():
    up = hist[hi + 1] if hi + 1 < 50 else -1; dn = hist[lo - 1] if lo - 1 >= 0 else -1
    if up >= dn:
        hi += 1; acc += hist[hi]; steps.append(f"+{hi}")
    else:
        lo -= 1; acc += hist[lo]; steps.append(f"-{lo}")
print(f"  VA expansion ({len(steps)} steps): {' '.join(steps)}; coverage {acc/hist.sum():.4f}")
print(f"  VAL = lower edge of bin {lo} = {edges[lo]:.4f} (engine {b.VAL}); VAH = upper edge of bin {hi} = {edges[hi+1]:.4f} (engine {b.VAH})")
