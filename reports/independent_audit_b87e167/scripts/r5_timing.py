"""AUDITOR SENSITIVITY (post hoc): R5 timing luck. Same rules as approved_plan.md, but the daily decision
boundary is shifted by h hours and the weekly budget update weekday is w. main_17 cost; BTC UM IS only.
Variant (h=0, w=Mon) must reproduce the frozen result exactly."""
import math, sys
import numpy as np, pandas as pd
sys.path.insert(0, sys.argv[3])
from cp3_stats_lib import webb

df = pd.read_parquet(sys.argv[1])
grid = pd.date_range('2020-01-01', '2024-01-01', freq='1min', inclusive='left', tz='UTC').as_unit('ns')
df = df.reindex(grid)
o, h_, l_, c, v = (df[k].to_numpy(float) for k in ('open', 'high', 'low', 'close', 'volume'))
bad = np.isnan(c) | ((v == 0) & (o == h_) & (h_ == l_) & (l_ == c))
mark = c[np.maximum.accumulate(np.where(bad, -1, np.arange(len(grid))))]
f = pd.read_parquet(sys.argv[2])['funding_rate']; f.index = f.index.round('min')
f = f[(f.index >= grid[0]) & (f.index <= grid[-1])]
rate = np.full(len(grid), np.nan); rate[grid.get_indexer(f.index)] = f.to_numpy()
FEE, BS, SS = 0.0005, 0.0005, 0.0002

def daily_path(hour, weekday, version):
    first = int(np.flatnonzero(grid == pd.Timestamp('2020-03-01', tz='UTC'))[0]) + hour * 60
    n_days = (len(grid) - first) // 1440
    end = first + n_days * 1440
    bnds = np.arange(first, end, 1440)
    allb = np.arange(first - 59 * 1440, end + 1, 1440)
    assert allb[0] - 1 >= 0
    C = np.array([mark[b - 1] for b in allb]); pos = {b: i for i, b in enumerate(allb)}
    cash, q, pend, prev_g, L = 100000.0, 0.0, False, None, None
    events = sorted(set(bnds.tolist()) | set(np.flatnonzero(np.isfinite(rate)).tolist()) | {end - 1})
    events = [e for e in events if first <= e < end]
    ev_state = []
    for m in events:
        term = m == end - 1
        ref = mark[m] if term else (mark[m - 1] if bad[m] else o[m])
        e_minus = cash + q * mark[m - 1]
        if np.isfinite(rate[m]):
            cash += -q * ref * rate[m]
        if (m - first) % 1440 == 0 and not term:
            k = pos[m]
            M = C[k - 59:k + 1].mean(); u = C[k - 19:k + 1] / C[k - 20:k] - 1
            sigma = math.sqrt(365) * u.std(ddof=1)
            upd = m == first or grid[m].weekday() == weekday
            if upd:
                L = 1.0 if sigma == 0 else min(1.0, 0.2 / sigma)
            g = True if version == 'P' else bool(C[k] > M)
            trig = m == first or upd or (g != prev_g and m != first) or pend
            if trig:
                tgt = (1.0 if g else 0.0) * L * e_minus / C[k]; pend = False
                if tgt != q:
                    if bad[m]: pend = True
                    else:
                        d = tgt - q; fill = ref * (1 + BS) if d > 0 else ref * (1 - SS)
                        cash -= d * fill + abs(d) * fill * FEE; q = tgt
            prev_g = g
        if term and q:
            fill = ref * (1 - SS); cash += q * fill * (1 - FEE); q = 0.0
        ev_state.append((m, cash, q))
    # equity at each boundary-1 minute = cash/q state of the last event strictly before that minute (or at it)
    ev_m = np.array([e[0] for e in ev_state]); ev_c = np.array([e[1] for e in ev_state]); ev_q = np.array([e[2] for e in ev_state])
    closes_at = bnds[1:].tolist() + [end]
    eqs = []
    for b in closes_at:
        t = b - 1; j = np.searchsorted(ev_m, t, side='right') - 1
        eqs.append(ev_c[j] + ev_q[j] * mark[t])
    eqs = np.array(eqs)
    rets = eqs / np.r_[100000.0, eqs[:-1]] - 1
    days = grid[bnds]
    return pd.Series(rets, index=days)

rows = []
for hour in (0, 4, 8, 12, 16, 20):
    for wd in range(7):
        T = daily_path(hour, wd, 'T'); P = daily_path(hour, wd, 'P')
        months = T.index.strftime('%Y-%m').to_numpy()
        mD, pD, G = webb((T - P).to_numpy(), months); mT, pT, _ = webb(T.to_numpy(), months)
        rows.append(dict(hour=hour, weekday=wd, days=len(T), T_bp=mT * 1e4, D_bp=mD * 1e4, pT=pT, pD=pD,
                         cumT=np.prod(1 + T) - 1, cumP=np.prod(1 + P) - 1))
r = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print(r.round(4).to_string())
print('\nD_bp: min %.3f median %.3f max %.3f; share D>0: %.2f; min pD %.4f; variants with pD<=0.025 (Holm step-1 equivalent): %d' %
      (r.D_bp.min(), r.D_bp.median(), r.D_bp.max(), (r.D_bp > 0).mean(), r.pD.min(), (r.pD <= 0.025).sum()))
print('T_bp: min %.3f median %.3f max %.3f; variants with pT<=0.025: %d' % (r.T_bp.min(), r.T_bp.median(), r.T_bp.max(), (r.pT <= 0.025).sum()))
