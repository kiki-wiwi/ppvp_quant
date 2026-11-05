"""indep.py: independent, loop-based re-implementation of the documented PPVP rules (no engine imports).

Rules re-implemented from the documentation (ppvp_engine.py docstring / PHASE1_engine.md section 3):
* 4H = UTC-aligned groups of 4 x 1H (00/04/08/12/16/20), only complete groups whose close <= last 1H close.
* pivot high at 4H i: high[i] > max(high[i-L:i]) and high[i] >= max(high[i+1:i+R+1]); low symmetric.
  confirmed at close of 4H i+R.
* box k = consecutive pair of pivots (piv[k-1], piv[k]) (P = 2), window = 1H bars with
  open in [open(4H_a), close(4H_b)); available at close(4H_{b+R}).
* profile: 50 equal bins over [min low, max high] of the window; each 1H bar's volume spread uniformly over
  [low, high] (zero-range bar -> bin containing the price). POC = centre of max bin, ties within 1e-9*max ->
  lowest bin. VA: from POC add one bin at a time on the side whose next bin is larger (ties within tol -> up)
  until acc >= 0.7 * total. VAL = lower edge of lowest VA bin, VAH = upper edge of highest VA bin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

H1 = pd.Timedelta(hours=1)
H4 = pd.Timedelta(hours=4)


def build_4h_indep(df1h: pd.DataFrame) -> pd.DataFrame:
    g = df1h.index.floor("4h")
    rows = {}
    for key, sub in df1h.groupby(g, sort=True):
        rows[key] = {"open": sub["open"].iloc[0], "high": sub["high"].max(), "low": sub["low"].min(),
                     "close": sub["close"].iloc[-1], "volume": sub["volume"].sum(), "n": len(sub)}
    d4 = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    d4["available_time"] = d4.index + H4
    last_close = df1h.index[-1] + H1
    return d4[d4["available_time"] <= last_close]


def pivots_indep(high, low, L, R):
    n = len(high)
    ph = np.zeros(n, bool)
    pl = np.zeros(n, bool)
    for i in range(L, n - R):
        left_h = max(high[i - L:i]) if L else -np.inf
        right_h = max(high[i + 1:i + R + 1]) if R else -np.inf
        left_l = min(low[i - L:i]) if L else np.inf
        right_l = min(low[i + 1:i + R + 1]) if R else np.inf
        ph[i] = high[i] > left_h and high[i] >= right_h
        pl[i] = low[i] < left_l and low[i] <= right_l
    return ph, pl


def hist_uniform(lo, hi, vol, edges):
    nb = len(edges) - 1
    hist = np.zeros(nb)
    p_low, p_high = edges[0], edges[-1]
    width = (p_high - p_low) / nb
    for a, b, v in zip(lo, hi, vol):
        if b > a:
            for j in range(nb):
                ov = min(b, edges[j + 1]) - max(a, edges[j])
                if ov > 0:
                    hist[j] += v * ov / (b - a)
        else:
            j = min(max(int((a - p_low) / width), 0), nb - 1)
            hist[j] += v
    return hist


def hist_point(price, vol, edges):
    """all of a bar's volume at one price (diagnostic allocation alternatives)."""
    nb = len(edges) - 1
    j = np.searchsorted(edges, price, side="right") - 1
    j = np.clip(j, 0, nb - 1)
    hist = np.zeros(nb)
    np.add.at(hist, j, vol)
    return hist


def value_area(hist, va_pct=0.70, rtol=1e-9):
    nb = len(hist)
    hmax = max(hist)
    tol = rtol * hmax
    poc = min(j for j in range(nb) if hist[j] >= hmax - tol)
    total = float(sum(hist))
    target = va_pct * total
    lo = hi = poc
    acc = hist[poc]
    while acc < target:
        up_ok, dn_ok = hi + 1 < nb, lo - 1 >= 0
        if not (up_ok or dn_ok):
            break
        if up_ok and (not dn_ok or hist[hi + 1] >= hist[lo - 1] - tol):
            hi += 1
            acc += hist[hi]
        else:
            lo -= 1
            acc += hist[lo]
    return poc, lo, hi


def levels(edges, poc, lo, hi):
    return 0.5 * (edges[poc] + edges[poc + 1]), edges[lo], edges[hi + 1]


def boxes_indep(df1h, L=5, R=5, n_bins=50, va_pct=0.70):
    d4 = build_4h_indep(df1h)
    ph, pl = pivots_indep(d4["high"].to_numpy(), d4["low"].to_numpy(), L, R)
    piv = np.flatnonzero(ph | pl)
    open4 = d4.index
    avail4 = d4["available_time"]
    lo1, hi1, v1 = (df1h[c].to_numpy(np.float64) for c in ("low", "high", "volume"))
    rows = []
    for k in range(1, len(piv)):
        a, b = piv[k - 1], piv[k]
        t0, t1 = open4[a], avail4.iloc[b]
        m = (df1h.index >= t0) & (df1h.index < t1)
        lo, hi, v = lo1[m], hi1[m], v1[m]
        p_low, p_high = lo.min(), hi.max()
        edges = np.linspace(p_low, p_high, n_bins + 1)
        hist = hist_uniform(lo, hi, v, edges)
        poc_i, lo_i, hi_i = value_area(hist, va_pct)
        poc, val, vah = levels(edges, poc_i, lo_i, hi_i)
        rows.append({"available_time": avail4.iloc[b + R], "pivot_a_time": open4[a], "pivot_b_time": open4[b],
                     "pivot_a_idx": a, "pivot_b_idx": b,
                     "type_a": "HL" if ph[a] and pl[a] else "H" if ph[a] else "L",
                     "type_b": "HL" if ph[b] and pl[b] else "H" if ph[b] else "L",
                     "window_start": t0, "window_end": t1, "n_1h": int(m.sum()),
                     "VAL": val, "POC": poc, "VAH": vah, "P_low": p_low, "P_high": p_high,
                     "poc_bin": poc_i, "lo_bin": lo_i, "hi_bin": hi_i})
    return pd.DataFrame(rows), d4, ph, pl
