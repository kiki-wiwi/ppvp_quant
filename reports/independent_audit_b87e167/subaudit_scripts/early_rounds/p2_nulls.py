"""Independent re-implementation of Phase 2 random-entry nulls (pct geometry, no pool) from first principles,
using only IS 1H bars read with an IS predicate. Seeds per tools/run_phase2.py: 20261104 + j.
Usage: python -I p2_nulls.py <repo_root>"""
import json
import sys
from pathlib import Path

import numba
import numpy as np
import pandas as pd

root = Path(sys.argv[1])
IS0 = pd.Timestamp("2020-01-01", tz="UTC"); IS1 = pd.Timestamp("2024-01-01", tz="UTC")
df = pd.read_parquet(root / "data/processed/BTCUSDT_perp_1h.parquet")
df = df[(df.index >= IS0) & (df.index < IS1)]
assert len(df) == 35064
o, h, l, c = (df[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
idx_ns = df.index.as_unit("ns").asi8
tf = pd.read_csv(root / "reports/phase2_is_trades.csv")
ba = pd.read_csv(root / "reports/audit_round/experiments/rematerialised/B_audited_trades.csv")
eb = ba["entry_bar"].to_numpy(np.int64)
assert (pd.to_datetime(ba.entry_time).values == pd.to_datetime(tf.entry_time).values).all()
d_sl = tf["d_sl"].to_numpy(np.float64); d_tp = tf["d_tp"].to_numpy(np.float64)
r_act = float(tf["r_ex_funding"].mean()); w_act = int((tf["r_ex_funding"] > 0).sum())
TAKER, MAKER, SLIP = 0.0005, 0.0002, 0.0002
first, end_bar = 0, int(np.searchsorted(idx_ns, IS1.value, side="left"))
m = len(eb)


@numba.njit(cache=False)
def bracket(o, h, l, c, bars, dsl, dtp, end_bar, taker, maker, slip):
    n = bars.shape[0]
    out = np.empty(n)
    for j in range(n):
        b = bars[j]
        efill = c[b] * (1.0 + slip)
        s = efill * (1.0 - dsl[j]); t = efill * (1.0 + dtp[j])
        risk = efill - s
        px = np.nan; fee_x = taker; done = False
        for i in range(b + 1, end_bar):
            if l[i] <= s:
                px = min(o[i], s) * (1.0 - slip); done = True; break
            if h[i] >= t:
                px = t * (1.0 - slip); fee_x = maker; done = True; break
        if not done:
            px = c[end_bar - 1] * (1.0 - slip)
        out[j] = (px - efill - taker * efill - fee_x * px) / risk
    return out


def sample(rng, n_rep, window_days, side, excl):
    hi_cap = end_bar - 1
    pool = np.arange(len(idx_ns), dtype=np.int64)
    if window_days is None:
        lo = np.full(m, first, np.int64); hi = np.full(m, hi_cap, np.int64)
    else:
        w = np.int64(round(window_days * 86400 * 10**9)); t_e = idx_ns[eb]
        lo = np.maximum(np.searchsorted(idx_ns, t_e - w, side="left"), first).astype(np.int64)
        hi = np.minimum(np.searchsorted(idx_ns, t_e + w, side="right"), hi_cap).astype(np.int64)
    ps = lambda x: np.searchsorted(pool, x, side="left").astype(np.int64)
    if side == "both" and excl is None:
        p_lo, p_hi = ps(lo), ps(np.maximum(hi, lo))
        return pool[p_lo[None, :] + np.floor(rng.random((n_rep, m)) * (p_hi - p_lo)[None, :]).astype(np.int64)]
    k = 0 if excl is None else int(excl)
    l_lo, l_hi = ps(lo), ps(np.maximum(np.minimum(hi, eb - k), lo))
    r_lo, r_hi = ps(np.minimum(np.maximum(lo, eb + k + 1), hi)), ps(hi)
    n_left = np.where(side in ("both", "before"), np.maximum(l_hi - l_lo, 0), 0)
    n_right = np.where(side in ("both", "after"), np.maximum(r_hi - r_lo, 0), 0)
    tot = n_left + n_right
    u = np.floor(rng.random((n_rep, m)) * tot[None, :]).astype(np.int64)
    return pool[np.where(u < n_left[None, :], l_lo[None, :] + u, r_lo[None, :] + (u - n_left[None, :]))]


# sanity: actual entries reproduce r_ex_funding
r_chk = bracket(o, h, l, c, eb, d_sl, d_tp, end_bar, TAKER, MAKER, SLIP)
print("actual-entry replay max |dR| vs r_ex_funding:", np.nanmax(np.abs(r_chk - tf["r_ex_funding"].to_numpy())))
specs = [("global", None, "both", None), ("local_60d", 60, "both", None), ("local_15d", 15, "both", None),
         ("local_15d_x24", 15, "both", 24), ("local_3d", 3, "both", None), ("before_3d", 3, "before", None),
         ("after_3d", 3, "after", None), ("after_3d_x24", 3, "after", 24)]
arch = json.loads((root / "reports/phase2_baseline.json").read_text())["is_null"]
N = 5000
for j, (key, wd, side, excl) in enumerate(specs):
    rng = np.random.default_rng(20261104 + j)
    bars = sample(rng, N, wd, side, excl)
    dsl = np.tile(d_sl, N); dtp = np.tile(d_tp, N)
    r = bracket(o, h, l, c, bars.ravel(), dsl, dtp, end_bar, TAKER, MAKER, SLIP).reshape(N, m)
    rn = r.mean(1); wn = (r > 0).sum(1)
    p_r = (1 + (rn <= r_act + 1e-12).sum()) / (N + 1); p_w = (1 + (wn <= w_act).sum()) / (N + 1)
    a = arch[key]
    print(f"{key}: null_mean_r={rn.mean():.6f} (arch {a['null_mean_r_mean']:.6f}) p={p_r:.6f} (arch {a['p_mean_r_le']:.6f}) "
          f"wins={wn.mean():.4f} (arch {a['null_wins_mean']:.4f}) p_w={p_w:.6f} (arch {a['p_wins_le']:.6f})")
