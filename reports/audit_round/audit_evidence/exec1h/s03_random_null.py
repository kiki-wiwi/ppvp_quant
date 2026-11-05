"""s03: figure-2 random-entry null (attribution.random_entry_null) — reproduction with the Phase-2 seeds, independent
re-implementation of sampling + bracket matching from the same RNG stream, boundary/IS checks, overlap and p-value. IS only."""
import pickle, sys
sys.path.insert(0, "/home/user/ppvp_quant")
from pathlib import Path
import numpy as np, pandas as pd
import attribution as A
from config import BASELINE_BACKTEST as B, BASELINE_STRATEGY, IS_END, IS_START
from ppvp_audit.common import is_run

D = Path(__file__).resolve().parent
out = is_run()
df = out["df1h"]; assert df.index.max() < IS_END
tf = A.trade_frame(out["result"], df, out["boxes"], BASELINE_STRATEGY)
o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
n = len(df); m = len(tf)
eb = tf["entry_bar"].to_numpy(np.int64)
# 1) bracket replays the backtester on the real entries
end_bar = int(np.searchsorted(df.index, IS_END, side="left"))
xb, rs, r = A.bracket_outcomes(df, eb, tf["sl"].to_numpy(), tf["tp"].to_numpy(), end_bar, B)
print("end_bar", end_bar, "n", n, " last bar used for matching opens at", df.index[end_bar - 1])
print("[bracket on real entries] exit bars equal:", bool(np.array_equal(xb, tf["exit_bar"].to_numpy())),
      " reasons equal:", bool(np.array_equal(np.where(rs == 1, "SL", "TP"), tf["exit_reason"].to_numpy())),
      " max|R - r_ex_funding|:", float(np.max(np.abs(r - tf["r_ex_funding"].to_numpy()))))
# 2) reproduce figure 2
res = {}
for key, seed, wd, ex in (("global", 20261104, None, None), ("local_15d_x24", 20261107, 15, 24)):
    nul = A.random_entry_null(df, tf, B, IS_START, IS_END, n_rep=5000, seed=seed, window_days=wd, exclude_bars=ex)
    res[key] = nul
    print(f"[{key}] actual {nul['actual_mean_r_ex_funding']:+.4f}  null mean {nul['null_mean_r_mean']:+.4f}  p_le {nul['p_mean_r_le']:.4f} "
          f"(= (1+{int(round(nul['p_mean_r_le']*5001))-1})/5001)  unresolved {nul['unresolved_frac']:.5f}  same-bar SL&TP {nul['same_bar_sl_tp_frac']:.4f}")
# 3) independent re-implementation from the same RNG stream
def indep(seed, wd, ex, n_rep=5000):
    rng = np.random.default_rng(seed)
    u = rng.random((n_rep, m))
    first, hi_cap = 0, end_bar - 1
    t = df.index.as_unit("ns").asi8
    bars = np.empty((n_rep, m), np.int64)
    for j in range(m):
        e = eb[j]
        if wd is None:
            cand = np.arange(first, hi_cap)
        else:
            w = wd * 86400 * 10**9
            cand = np.flatnonzero((np.abs(t - t[e]) <= w))
            cand = cand[(cand >= first) & (cand < hi_cap)]
            if ex is not None:
                cand = cand[np.abs(cand - e) > ex]
        bars[:, j] = cand[np.floor(u[:, j] * len(cand)).astype(np.int64)]
    d_sl = 1 - tf["sl"].to_numpy() / tf["entry_px"].to_numpy(); d_tp = tf["tp"].to_numpy() / tf["entry_px"].to_numpy() - 1
    R = np.empty((n_rep, m)); XB = np.empty((n_rep, m), np.int64); RS = np.zeros((n_rep, m), np.int8)
    slip, tk, mk = B.slippage, B.perp_taker_fee, B.perp_maker_fee
    for j in range(m):
        for k_ in range(n_rep):
            b = bars[k_, j]; ef = c[b] * (1 + slip); s = ef * (1 - d_sl[j]); tp = ef * (1 + d_tp[j])
            px = None
            for i in range(b + 1, end_bar):
                if l[i] <= s:
                    px = min(o[i], s) * (1 - slip); fx = tk; RS[k_, j] = 1; break
                if h[i] >= tp:
                    px = tp * (1 - slip); fx = mk; RS[k_, j] = 2; break
            if px is None:
                i = end_bar - 1; px = c[i] * (1 - slip); fx = tk
            XB[k_, j] = i
            R[k_, j] = (px - ef - tk * ef - fx * px) / (ef - s)
    return bars, R, XB, RS

for key, seed, wd, ex in (("global", 20261104, None, None), ("local_15d_x24", 20261107, 15, 24)):
    nrep = 300
    bars, R, XB, RS = indep(seed, wd, ex, n_rep=nrep)
    ref = A.random_entry_null(df, tf, B, IS_START, IS_END, n_rep=nrep, seed=seed, window_days=wd, exclude_bars=ex)
    print(f"[{key}] independent first {nrep} reps: max|null-mean diff| {np.max(np.abs(R.mean(axis=1) - ref['null_r_dist'])):.2e}; "
          f"bars min {bars.min()} max {bars.max()} (allowed [0,{end_bar-2}]); max exit bar {XB.max()} (IS last bar {n-1}); "
          f"draws with |b-e|<=24: {int((np.abs(bars-eb[None,:])<=24).sum())}; with replacement: duplicates within a trade column "
          f"{int(sum(nrep-len(np.unique(bars[:,j])) for j in range(m)))}")
    # overlap of random trades within one replicate (event-level, positions may overlap)
    ov = []
    for k_ in range(nrep):
        s_, e_ = bars[k_], XB[k_]
        order = np.argsort(s_); s2, e2 = s_[order], e_[order]
        cnt = 0
        for a in range(m):
            cnt += int(np.sum(s2[a+1:] < e2[a]))
        ov.append(cnt)
    hold = (XB - bars)
    print(f"[{key}] overlapping trade pairs per replicate: mean {np.mean(ov):.1f}; share of random trades that overlap another: "
          f"~{np.mean([np.mean([np.any((bars[k_]<XB[k_][a])&(XB[k_]>bars[k_][a])&(np.arange(m)!=a)) for a in range(m)]) for k_ in range(30)]):.2%} (30 reps)"
          f"; hold bars median {np.median(hold):.0f}, p99 {np.percentile(hold,99):.0f}, max {hold.max()}")
real_hold = (tf["exit_bar"] - tf["entry_bar"]).to_numpy()
print(f"[real] hold bars median {np.median(real_hold):.0f} p99 {np.percentile(real_hold,99):.0f} max {real_hold.max()}")
# 4) p-value conventions: lower tail; also report upper-tail and two-sided for context
for key, nul in res.items():
    dist = nul["null_r_dist"]; act = nul["actual_mean_r_ex_funding"]
    p_lo = (1 + (dist <= act + 1e-12).sum()) / 5001; p_hi = (1 + (dist >= act - 1e-12).sum()) / 5001
    print(f"[{key}] p_lower {p_lo:.4f} p_upper {p_hi:.4f} two-sided(min*2) {min(1, 2*min(p_lo, p_hi)):.4f}; null sd {dist.std(ddof=1):.4f}")
pickle.dump({k: {kk: v[kk] for kk in ("null_r_dist", "per_trade_null_mean_r")} for k, v in res.items()}, open(D / "s03_null.pkl", "wb"))
