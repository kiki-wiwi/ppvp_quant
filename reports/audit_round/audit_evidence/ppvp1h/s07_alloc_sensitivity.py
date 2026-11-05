"""s07 (DIAGNOSTIC ONLY, not a fix, no performance): sensitivity of VAL/POC/VAH of the boxes used by the 109 IS trades
to the intrabar volume-allocation assumption, using 1H-only alternatives. Same window, same 50 bin edges
[P_low, P_high], same POC/VA rule; only the per-bar allocation changes:
  base    : uniform over [low, high] (engine)
  typical : all volume at (H+L+C)/3
  close   : all volume at the close
  body    : uniform over [min(O,C), max(O,C)] (zero-length body -> at the close)
Shifts are in units of the 1H ATR used by the state machine at the signal bar (atr_prev).
The true intrabar (tick-level) distribution cannot be recovered from 1H data; no finer data is available or used.
"""
import sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from indep import hist_uniform, hist_point, value_area, levels

d = pickle.load(open(HERE / "is_objects.pkl", "rb"))
df1h, boxes, sig = d["df1h"], d["boxes"], d["sig"]
pv = pd.read_csv(HERE / "profile_versions_per_signal.csv")
tr = pv[pv["status"] == "filled"].reset_index(drop=True)
o, h, l, c, v = (df1h[k].to_numpy(np.float64) for k in ("open", "high", "low", "close", "volume"))
idx = df1h.index
atr = sig["atr"].to_numpy()


def box_levels(k, mode):
    b = boxes.iloc[k]
    m = (idx >= b.window_start) & (idx < b.window_end)
    lo, hi, vo, cl, op = l[m], h[m], v[m], c[m], o[m]
    edges = np.linspace(lo.min(), hi.max(), 51)
    if mode == "base":
        hist = hist_uniform(lo, hi, vo, edges)
    elif mode == "typical":
        hist = hist_point((hi + lo + cl) / 3.0, vo, edges)
    elif mode == "close":
        hist = hist_point(cl, vo, edges)
    elif mode == "body":
        hist = hist_uniform(np.minimum(op, cl), np.maximum(op, cl), vo, edges)
    p, a, z = value_area(hist)
    return levels(edges, p, a, z), (edges[1] - edges[0])


rows = []
for _, t in tr.iterrows():
    k, s = int(t.box_at_signal), int(t.signal_bar)
    (poc0, val0, vah0), bw = box_levels(k, "base")
    assert np.isclose(val0, boxes.VAL[k], rtol=0, atol=0) and poc0 == boxes.POC[k] and vah0 == boxes.VAH[k]
    r = {"signal_bar": s, "box": k, "atr": atr[s], "bin_w_atr": bw / atr[s], "n_1h": int(boxes.n_1h[k]),
         "VAL": val0, "POC": poc0, "VAH": vah0}
    for mode in ("typical", "close", "body"):
        (pp, vv, hh), _ = box_levels(k, mode)
        r[f"dVAL_{mode}"] = (vv - val0) / atr[s]
        r[f"dPOC_{mode}"] = (pp - poc0) / atr[s]
        r[f"dVAH_{mode}"] = (hh - vah0) / atr[s]
    rows.append(r)
x = pd.DataFrame(rows)
x.to_csv(HERE / "alloc_sensitivity_trade_boxes.csv", index=False)
print("trades:", len(x), " unique signal boxes:", x["box"].nunique())
print("bin width / ATR_signal: median %.3f  P10 %.3f  P90 %.3f  max %.3f" % tuple(x["bin_w_atr"].quantile([.5, .1, .9, 1.0])))
print("VA width / ATR_signal: median %.2f P10 %.2f P90 %.2f" % tuple(((x.VAH - x.VAL) / x.atr).quantile([.5, .1, .9])))
for mode in ("typical", "close", "body"):
    print(f"\n[{mode}]  shifts in ATR_signal units (|.|): median / P75 / P90 / max ; share |shift|>0 ; >0.5 ATR ; >1 ATR")
    for lv in ("VAL", "POC", "VAH"):
        a = x[f"d{lv}_{mode}"].abs()
        print(f"  {lv}: {a.median():.3f} / {a.quantile(.75):.3f} / {a.quantile(.9):.3f} / {a.max():.3f} ; "
              f"{(a > 1e-12).mean():.1%} ; {(a > 0.5).mean():.1%} ; {(a > 1.0).mean():.1%}   signed median {x[f'd{lv}_{mode}'].median():+.3f}")
