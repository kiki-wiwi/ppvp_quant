"""s08: engine definition diagnostics on IS (levels only, no performance):
POC / VA ties, VA coverage, the one-bin-at-a-time VA rule vs the two-row (pair) convention, the window ending at the
pivot-b close vs at the box's available_time, pivot-pair types of trade boxes, profile staleness at the signal,
and signals whose decision moment coincides with the confirmation of a newer box."""
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
o, h, l, c, v = (df1h[k].to_numpy(np.float64) for k in ("open", "high", "low", "close", "volume"))
idx = df1h.index
atr = sig["atr"].to_numpy()
H = pd.Timedelta(hours=1)


def window_hist(ws, we):
    m = (idx >= ws) & (idx < we)
    lo, hi, vo = l[m], h[m], v[m]
    edges = np.linspace(lo.min(), hi.max(), 51)
    return hist_uniform(lo, hi, vo, edges), edges


def va_pairs(hist, va_pct=0.70):
    """two-row convention: compare the sums of the next two bins on each side, add the larger pair (ties up)."""
    nb = len(hist)
    poc = int(np.flatnonzero(hist >= hist.max() * (1 - 1e-9))[0])
    lo = hi = poc
    acc, target = hist[poc], va_pct * hist.sum()
    while acc < target and (lo > 0 or hi < nb - 1):
        up = hist[hi + 1:hi + 3].sum() if hi < nb - 1 else -1.0
        dn = hist[max(lo - 2, 0):lo].sum() if lo > 0 else -1.0
        if up >= dn:
            k = min(2, nb - 1 - hi)
            acc += hist[hi + 1:hi + 1 + k].sum(); hi += k
        else:
            k = min(2, lo)
            acc += hist[lo - k:lo].sum(); lo -= k
    return poc, lo, hi


# ---- ties and coverage over all IS boxes
poc_ties = va_ties = 0
cover = []
for k in range(len(boxes)):
    b = boxes.iloc[k]
    hist, edges = window_hist(b.window_start, b.window_end)
    hmax = hist.max()
    if (hist >= hmax - 1e-9 * hmax).sum() > 1:
        poc_ties += 1
    p, a, z = value_area(hist)
    cover.append(hist[a:z + 1].sum() / hist.sum())
    # replay expansion and count steps decided inside the tolerance
    lo_, hi_, acc = p, p, hist[p]
    tol = 1e-9 * hmax
    while acc < 0.7 * hist.sum():
        cu, cd = hi_ + 1 < 50, lo_ - 1 >= 0
        if cu and cd and abs(hist[hi_ + 1] - hist[lo_ - 1]) <= tol:
            va_ties += 1
        if cu and ((not cd) or hist[hi_ + 1] >= hist[lo_ - 1] - tol):
            hi_ += 1; acc += hist[hi_]
        else:
            lo_ -= 1; acc += hist[lo_]
cover = np.array(cover)
print(f"IS boxes: {len(boxes)}; boxes with a POC tie (>=2 bins within 1e-9 of max): {poc_ties}; VA steps decided by a tie: {va_ties}")
print(f"VA coverage (share of window volume inside [VAL,VAH] bins): min {cover.min():.4f} median {np.median(cover):.4f} max {cover.max():.4f}")

# ---- trade boxes: VA rule and window-end diagnostics (levels in ATR at the signal bar)
rows = []
for _, t in tr.iterrows():
    k, s = int(t.box_at_signal), int(t.signal_bar)
    b = boxes.iloc[k]
    hist, edges = window_hist(b.window_start, b.window_end)
    p2, a2, z2 = va_pairs(hist)
    poc2, val2, vah2 = levels(edges, p2, a2, z2)
    hist_e, edges_e = window_hist(b.window_start, b.available_time)      # window extended to the confirmation time
    pe, ae, ze = value_area(hist_e)
    poce, vale, vahe = levels(edges_e, pe, ae, ze)
    rows.append({"box": k, "pair": f"{b.pivot_a_type}{b.pivot_b_type}", "atr": atr[s],
                 "dVAL_pairs": (val2 - b.VAL) / atr[s], "dVAH_pairs": (vah2 - b.VAH) / atr[s],
                 "dVAL_to_avail": (vale - b.VAL) / atr[s], "dPOC_to_avail": (poce - b.POC) / atr[s],
                 "dVAH_to_avail": (vahe - b.VAH) / atr[s],
                 "excluded_share_of_hours": 4 * 5 / (b.n_1h + 20), "n_1h": b.n_1h,
                 "h_window_end_to_signal_close": (idx[s] + H - b.window_end) / H,
                 "h_avail_to_signal_open": (idx[s] - b.available_time) / H,
                 "next_box_avail_at_signal_close": bool(k + 1 < len(boxes) and boxes.available_time.iloc[k + 1] == idx[s] + H),
                 "signal_close_on_4h_boundary": bool(((idx[s] + H).hour % 4) == 0)})
x = pd.DataFrame(rows)
x.to_csv(HERE / "engine_diag_trade_boxes.csv", index=False)
q = lambda s: f"median {s.abs().median():.3f} P90 {s.abs().quantile(.9):.3f} max {s.abs().max():.3f} share!=0 {(s.abs()>1e-12).mean():.1%} share>0.5ATR {(s.abs()>0.5).mean():.1%}"
print("\nVA two-row (pair) rule vs engine one-bin rule, trade signal boxes (|shift|/ATR):")
print("  VAL", q(x.dVAL_pairs)); print("  VAH", q(x.dVAH_pairs))
print("Window extended to available_time (adds the 20 post-pivot 1H bars) vs engine window, trade signal boxes:")
print("  VAL", q(x.dVAL_to_avail)); print("  POC", q(x.dPOC_to_avail)); print("  VAH", q(x.dVAH_to_avail))
print("  excluded post-pivot hours as share of (window + excluded): median %.2f" % x.excluded_share_of_hours.median())
print("  hours from window_end (last profiled data) to signal close: min %.0f median %.0f max %.0f" %
      (x.h_window_end_to_signal_close.min(), x.h_window_end_to_signal_close.median(), x.h_window_end_to_signal_close.max()))
print("\npivot-pair types of the 109 trades' signal boxes:", x.pair.value_counts().to_dict(),
      " same-type (HH/LL):", int(x.pair.isin(['HH', 'LL']).sum()))
print("trades whose signal-bar close coincides with the confirmation of the NEXT box (box superseded at the decision moment):",
      int(x.next_box_avail_at_signal_close.sum()), "of", len(x), "; signal closes on a 4H boundary:", int(x.signal_close_on_4h_boundary.sum()))
print(x[x.next_box_avail_at_signal_close][["box", "h_avail_to_signal_open"]].assign(
    signal=tr.loc[x.next_box_avail_at_signal_close, "signal_open"].to_numpy()).to_string())
