"""Independent PPVP helpers written from the documented rules (not imported from ppvp_engine)."""
import numpy as np, pandas as pd

H1 = np.int64(3600 * 10**9)
H4 = 4 * H1
TOL = 1e-9

def ns(idx):
    return np.asarray(pd.DatetimeIndex(idx).as_unit("ns").asi8, dtype=np.int64)

def pivots(h, l, L=5, R=5):
    n = len(h); ph = np.zeros(n, bool); pl = np.zeros(n, bool)
    for i in range(L, n - R):
        ph[i] = h[i] > h[i-L:i].max() and h[i] >= h[i+1:i+R+1].max()
        pl[i] = l[i] < l[i-L:i].min() and l[i] <= l[i+1:i+R+1].min()
    return ph, pl

def poc_va(hist, va=0.70, poc_rule="lowest", rows=1, count_ties=False):
    """documented rule: POC = lowest bin within TOL*max of max; VA grows from POC one bin at a time to the larger side
    (ties within tol go up) until >= va*total. rows=2 -> classic pair-of-rows comparison (diagnostic)."""
    nb = len(hist); hmax = hist.max(); tol = TOL * hmax
    tied = np.flatnonzero(hist >= hmax - tol)
    poc = tied[0] if poc_rule == "lowest" else tied[-1]
    target = va * hist.sum(); lo = hi = poc; acc = hist[poc]; tie_steps = 0
    while acc < target:
        up = list(range(hi + 1, min(hi + 1 + rows, nb))); dn = list(range(lo - 1, max(lo - 1 - rows, -1), -1))
        if not up and not dn:
            break
        su = hist[up].sum() if up else -np.inf; sd = hist[dn].sum() if dn else -np.inf
        if up and dn and abs(su - sd) <= tol:
            tie_steps += 1
        if up and (not dn or su >= sd - tol):
            hi = up[-1]; acc += su
        else:
            lo = dn[-1]; acc += sd
    out = (poc, lo, hi, acc / hist.sum())
    return out + (tie_steps, len(tied)) if count_ties else out

def hist_uniform(lo, hi, v, edges):
    """CDF formulation: volume share of each bar below price x = clip((x-lo)/(hi-lo),0,1)."""
    assert (hi > lo).all()
    F = np.clip((edges[None, :] - lo[:, None]) / (hi - lo)[:, None], 0.0, 1.0)
    return (v[:, None] * np.diff(F, axis=1)).sum(axis=0)

def hist_point(p, v, edges):
    nb = len(edges) - 1
    j = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, nb - 1)
    return np.bincount(j, weights=v, minlength=nb)

def hist_body(o, c, v, edges):
    a, b = np.minimum(o, c), np.maximum(o, c)
    pos = b > a
    h = np.zeros(len(edges) - 1)
    if pos.any():
        h += hist_uniform(a[pos], b[pos], v[pos], edges)
    if (~pos).any():
        h += hist_point(o[~pos], v[~pos], edges)
    return h

def tri_cdf(x, a, b, m):
    x = np.clip(x, a[:, None], b[:, None]); a_ = a[:, None]; b_ = b[:, None]; m_ = m[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        left = np.where(m_ > a_, (x - a_) ** 2 / ((b_ - a_) * (m_ - a_)), 1.0)
        right = np.where(b_ > m_, 1.0 - (b_ - x) ** 2 / ((b_ - a_) * (b_ - m_)), 1.0)
    F = np.where(x <= m_, np.where(m_ > a_, left, 0.0), right)
    return np.clip(F, 0.0, 1.0)

def hist_tri(lo, hi, mode, v, edges):
    F = tri_cdf(edges[None, :].repeat(len(lo), 0), lo, hi, mode)
    return (v[:, None] * np.diff(F, axis=1)).sum(axis=0)

def levels(hist, edges, **kw):
    r = poc_va(hist, **kw)
    poc, lo, hi = r[0], r[1], r[2]
    return {"POC": 0.5 * (edges[poc] + edges[poc + 1]), "VAL": float(edges[lo]), "VAH": float(edges[hi + 1]),
            "cov": r[3], "poc_bin": poc, "lo_bin": lo, "hi_bin": hi, **({"tie_steps": r[4], "n_tied": r[5]} if len(r) > 4 else {})}
