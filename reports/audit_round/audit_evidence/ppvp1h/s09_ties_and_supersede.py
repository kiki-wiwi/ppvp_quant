"""s09: POC-tie and VA-tie cases among the boxes used by the 109 IS trades (levels only), and the two trades whose
signal box was superseded exactly at the decision moment (levels of the superseding box, no performance)."""
import sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from indep import hist_uniform, value_area, levels

d = pickle.load(open(HERE / "is_objects.pkl", "rb"))
df1h, boxes, sig = d["df1h"], d["boxes"], d["sig"]
pv = pd.read_csv(HERE / "profile_versions_per_signal.csv")
tr = pv[pv["status"] == "filled"].reset_index(drop=True)
l, h, v = (df1h[k].to_numpy(np.float64) for k in ("low", "high", "volume"))
idx = df1h.index
atr = sig["atr"].to_numpy()


def hist_of(k):
    b = boxes.iloc[k]
    m = (idx >= b.window_start) & (idx < b.window_end)
    edges = np.linspace(l[m].min(), h[m].max(), 51)
    return hist_uniform(l[m], h[m], v[m], edges), edges


def va_from(hist, poc, va_pct=0.7, rtol=1e-9):
    tol = rtol * hist.max()
    lo = hi = poc
    acc = hist[poc]
    while acc < va_pct * hist.sum():
        cu, cd = hi + 1 < len(hist), lo - 1 >= 0
        if cu and ((not cd) or hist[hi + 1] >= hist[lo - 1] - tol):
            hi += 1; acc += hist[hi]
        else:
            lo -= 1; acc += hist[lo]
    return lo, hi


used = sorted(set(tr.box_at_signal) | set(tr.box_at_seq_start) | set(tr.box_at_t1_low) | set(tr.box_at_t2_start) | set(tr.box_at_chain_start))
print("distinct boxes used by the 109 trades (chain start, first-test start/low, t2 start, signal):", len(used))
rows = []
for _, t in tr.iterrows():
    k, s = int(t.box_at_signal), int(t.signal_bar)
    hist, edges = hist_of(k)
    hmax = hist.max()
    tied = np.flatnonzero(hist >= hmax - 1e-9 * hmax)
    if len(tied) > 1:
        p_hi = int(tied[-1])
        lo2, hi2 = va_from(hist, p_hi)
        poc_alt, val_alt, vah_alt = levels(edges, p_hi, lo2, hi2)
        rows.append({"signal_open": t.signal_open, "box": k, "tied_bins": tied.tolist(),
                     "dPOC_if_highest_tied_atr": (poc_alt - boxes.POC[k]) / atr[s],
                     "dVAL_atr": (val_alt - boxes.VAL[k]) / atr[s], "dVAH_atr": (vah_alt - boxes.VAH[k]) / atr[s]})
x = pd.DataFrame(rows)
print("trades whose signal box has a POC tie:", len(x))
if len(x):
    print(x.to_string())

print("\nTwo trades whose signal box was superseded at the decision moment (close of the signal bar):")
for _, t in tr.iterrows():
    k, s = int(t.box_at_signal), int(t.signal_bar)
    if k + 1 < len(boxes) and boxes.available_time.iloc[k + 1] == idx[s] + pd.Timedelta(hours=1):
        b0, b1 = boxes.iloc[k], boxes.iloc[k + 1]
        a = atr[s]
        print(f"  signal bar {idx[s]} (decision {idx[s] + pd.Timedelta(hours=1)}): used box {k} VAL {b0.VAL:.1f} VAH {b0.VAH:.1f}; "
              f"box {k+1} confirmed at the decision moment: VAL {b1.VAL:.1f} VAH {b1.VAH:.1f}; "
              f"dVAL {(b1.VAL-b0.VAL)/a:+.2f} ATR, dVAH {(b1.VAH-b0.VAH)/a:+.2f} ATR; signal close {df1h.close.iloc[s]:.1f}; "
              f"new box pivots {b1.pivot_a_time} -> {b1.pivot_b_time} ({b1.pivot_a_type}{b1.pivot_b_type})")
