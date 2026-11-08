"""Independent re-computation of R5 inference (approved_plan.md section 6/7) plus robustness diagnostics.
Inputs: my own independently computed daily returns (work/r5_out/daily_returns.csv) and archived matrices."""
import hashlib, json, math, sys
import numpy as np, pandas as pd
from scipy import stats

dr = pd.read_csv(sys.argv[1], index_col=0, parse_dates=True)
arch = sys.argv[2]  # archived results dir
T, P = dr['T_main_17'].to_numpy(), dr['P_main_17'].to_numpy()
D = T - P
Th, Ph = dr['T_harsh_34'].to_numpy(), dr['P_harsh_34'].to_numpy()
months = dr.index.strftime('%Y-%m')
mlist = sorted(set(months))
G, N = len(mlist), len(T)
K = np.array([(months == m).sum() for m in mlist], float)

def msum(x):
    return np.array([x[months == m].sum() for m in mlist])

def tstat(A):
    mu = A.sum() / N
    se = math.sqrt(G / (G - 1) * ((A - mu * K) ** 2).sum()) / N
    return mu, se, mu / se

pts = np.array([-math.sqrt(1.5), -1, -math.sqrt(.5), math.sqrt(.5), 1, math.sqrt(1.5)])
W = np.random.Generator(np.random.PCG64(2026111001)).choice(pts, size=(99999, G))
M = np.random.Generator(np.random.PCG64(2026111002)).integers(0, G, size=(10000, G))
print('webb sha', hashlib.sha256(W.tobytes(order='C')).hexdigest())
print('month sha', hashlib.sha256(M.tobytes(order='C')).hexdigest())
Wa = np.load(f'{arch}/statistical_matrices/webb_weights.npy'); Ma = np.load(f'{arch}/statistical_matrices/month_indices.npy')
print('archived npy equal to regenerated:', np.array_equal(W, Wa), np.array_equal(M, Ma))

def webb_p(x):
    A = msum(x)
    mu, se, t = tstat(A)
    As = W * A
    mus = As.sum(1) / N
    dev = As - mus[:, None] * K
    ses = np.sqrt(G / (G - 1) * (dev ** 2).sum(1)) / N
    ts = np.where(ses > 0, mus / np.where(ses > 0, ses, 1), 0)
    return mu, se, t, (1 + (ts >= t).sum()) / 100000

res = {k: webb_p(x) for k, x in (('T', T), ('D', D))}
for k, (mu, se, t, p) in res.items():
    print(f'{k}: mean={mu*1e4:.4f}bp se={se*1e4:.4f}bp t={t:.4f} p_raw={p:.5f}')
ps = {k: v[3] for k, v in res.items()}
order = sorted(ps, key=lambda k: (ps[k], ('T', 'D').index(k)))
run = 0
for i, k in enumerate(order):
    run = min(1, max(run, (2 - i) * ps[k])); print(f'Holm {k}: {run:.5f}')

# month-block intervals
idx_by_month = [np.flatnonzero(months == m) for m in mlist]
def metrics(rows):
    t, p = T[rows], P[rows]
    return t.mean(), p.mean(), (t - p).mean(), math.sqrt(365) * t.mean() / t.std(ddof=1), math.sqrt(365) * p.mean() / p.std(ddof=1)
boot = np.array([metrics(np.concatenate([idx_by_month[j] for j in draw])) for draw in M])
names = ['T_mean', 'P_mean', 'D_mean', 'T_sharpe', 'P_sharpe']
iv = pd.read_csv(f'{arch}/intervals.csv', index_col=0)
for j, nm in enumerate(names):
    lo, hi = np.quantile(boot[:, j], [.025, .975])
    print(f'{nm}: [{lo:.6g}, {hi:.6g}] archived [{iv.iloc[j]["lower"]:.6g}, {iv.iloc[j]["upper"]:.6g}]')

# gates
half = dr.index < pd.Timestamp('2022-01-01', tz='UTC')
feat = pd.read_csv(sys.argv[3], index_col=0, parse_dates=True)
g = feat['g'].to_numpy().astype(bool)
print('V1 state counts', [(int(g[h].sum()), int((~g[h]).sum())) for h in (half, ~half)])
print('V4 halves T', T[half].mean(), T[~half].mean(), 'D', D[half].mean(), D[~half].mean())
Dm = pd.Series(msum(D), index=mlist)
top3 = Dm.sort_values(ascending=False, kind='stable').index[:3].tolist()
keep = ~np.isin(months, top3)
print('V5 removed', top3, 'remaining days', keep.sum(), 'T', T[keep].mean(), 'D', D[keep].mean())
print('V6 harsh T cum', np.prod(1 + Th) - 1, 'harsh D mean', (Th - Ph).mean())

# ---------- robustness diagnostics (NOT part of the frozen gates) ----------
print('\n--- diagnostics (post hoc, exploratory) ---')
# 1. minimum detectable effect at 80% power for the frozen tests (normal approx)
for k, (mu, se, t, p) in res.items():
    for a in (0.025, 0.05):
        mde = (stats.norm.ppf(1 - a) + stats.norm.ppf(0.8)) * se
        print(f'MDE {k} alpha={a}: {mde*1e4:.2f} bp/day = {mde*365*100:.1f}% simple annual')
# 2. how many top months must be removed to flip D sign / how much D comes from top months
srt = Dm.sort_values(ascending=False)
print('D monthly sums top5 (bp-sum):', (srt.head(5) * 1e4).round(1).to_dict())
print('D total sum', Dm.sum() * 1e4, 'bp; share of top1/top3:', srt.iloc[0] / Dm.sum(), srt.iloc[:3].sum() / Dm.sum())
for k in range(0, 6):
    kk = ~np.isin(months, srt.index[:k]); print(f'remove top {k}: D mean {D[kk].mean()*1e4:.3f}bp')
# 3. Newey-West HAC t for D and T at daily frequency (lags 0..60)
def nw(x, L):
    x = x - x.mean(); n = len(x); g0 = x @ x / n; s = g0
    for l in range(1, L + 1):
        s += 2 * (1 - l / (L + 1)) * (x[l:] @ x[:-l]) / n
    return math.sqrt(s / n)
for k, x in (('T', T), ('D', D)):
    print(k, 'NW t by lag:', {L: round(x.mean() / nw(x, L), 3) for L in (0, 5, 20, 60)})
# 4. paired Sharpe-difference bootstrap with the same month draws
sd = boot[:, 3] - boot[:, 4]
print('Sharpe diff T-P point', metrics(np.arange(N))[3] - metrics(np.arange(N))[4], '95% month-block interval', np.quantile(sd, [.025, .975]), 'frac<=0', (sd <= 0).mean())
# 5. exposure-matched comparison: scale P daily returns to T average exposure (crude, ignores compounding/funding nonlinearity)
lev_T, lev_P = 0.21802364939908483, 0.3951732043836017
Ps = P * lev_T / lev_P
Dx = T - Ps
mu, se, t = tstat(msum(Dx))
print(f'exposure-scaled D: mean {mu*1e4:.3f}bp t {t:.3f}')
# 6. annual and per-regime contributions
yr = dr.index.year
for y in sorted(set(yr)):
    m = yr == y
    print(y, 'T cum %.3f P cum %.3f  D mean %.2fbp  T on-days %d/%d' % (np.prod(1 + T[m]) - 1, np.prod(1 + P[m]) - 1, D[m].mean() * 1e4, g[m].sum(), m.sum()))
