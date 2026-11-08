"""Recompute CP3 paired-family Webb p-values (R5 method, seeds 2026111001/2026111002) from archived daily
equity (already verified against my independent engine) and compare with the archived statistics tables."""
import glob, json, math, sys
import numpy as np, pandas as pd

ACC = sys.argv[1]; STATS = sys.argv[2]
pts = np.array([-math.sqrt(1.5), -1, -math.sqrt(.5), math.sqrt(.5), 1, math.sqrt(1.5)])

def webb(x, months):
    ml = sorted(set(months)); G = len(ml); N = len(x)
    K = np.array([(months == m).sum() for m in ml], float)
    A = np.array([x[months == m].sum() for m in ml])
    W = np.random.Generator(np.random.PCG64(2026111001)).choice(pts, size=(99999, G))
    mu = A.sum() / N; se = math.sqrt(G / (G - 1) * ((A - mu * K) ** 2).sum()) / N; t = mu / se
    As = W * A; mus = As.sum(1) / N
    ses = np.sqrt(G / (G - 1) * ((As - mus[:, None] * K) ** 2).sum(1)) / N
    ts = np.divide(mus, ses, out=np.zeros_like(mus), where=ses > 0)
    return mu, (1 + (ts >= t).sum()) / 100000, G

def holm(pT, pD):
    if pT <= pD: return min(1, 2 * pT), min(1, max(2 * pT, pD))
    return min(1, max(2 * pD, pT)), min(1, 2 * pD)

windows = {'Post-IS': ('2024-01-01', '2026-06-01'), 'VAL': ('2024-01-01', '2025-07-01'), 'OOS': ('2025-07-01', '2026-06-01')}
pairs = {'UM': ('V1_USDM', 'P_USDM'), 'SPOT': ('V2_SPOT', 'P_SPOT'), 'CM_DeltaMatched': ('V3_CM_DeltaMatched', 'P_CM_DeltaMatched')}
rows = []
for asset in ('BTC', 'ETH'):
    for wname, (a, b) in list(windows.items()) + ([('IS-Common', ('2020-09-01', '2024-01-01'))] if asset == 'ETH' else []):
        src = 'IS-Common' if wname == 'IS-Common' else 'Post-IS'
        for fam, (tv, pv) in pairs.items():
            def load(v):
                d = pd.read_csv(f'{ACC}/{asset}__{src}__UM__{v}__main_17/daily.csv', index_col=0, parse_dates=True)
                e = d['equity']
                r = (e / e.shift(1).fillna(100000.0) - 1)
                return r[(r.index >= pd.Timestamp(a, tz='UTC')) & (r.index < pd.Timestamp(b, tz='UTC'))]
            T, P = load(tv), load(pv)
            months = T.index.strftime('%Y-%m').to_numpy()
            mT, pT, G = webb(T.to_numpy(), months); mD, pD, _ = webb((T - P).to_numpy(), months)
            hT, hD = holm(pT, pD)
            arch = f'{STATS}/{asset}__{wname}__{fam}/tests.csv'
            try:
                at = pd.read_csv(arch, index_col=0)
                apT, apD = at.loc['T', 'p_raw'], at.loc['D', 'p_raw']
            except FileNotFoundError:
                apT = apD = np.nan
            rows.append(dict(family=f'{asset}__{wname}__{fam}', months=G, T_bp=mT * 1e4, D_bp=mD * 1e4, pT=pT, pD=pD, holmT=hT, holmD=hD, arch_pT=apT, arch_pD=apD))
df = pd.DataFrame(rows).set_index('family')
pd.set_option('display.width', 200)
print(df.round(5).to_string())
print('max |p - archived|:', np.nanmax(np.abs(df.pT - df.arch_pT)), np.nanmax(np.abs(df.pD - df.arch_pD)))
print('min Holm p for D across', len(df), 'families:', df.holmD.min(), '; any D rejection at 0.05:', (df.holmD <= 0.05).any())
# Global Holm/Bonferroni across all 21 D hypotheses (illustrative)
pD = np.sort(df.pD.to_numpy()); m = len(pD)
print('global Holm over D family (m=%d): smallest adjusted %.4f' % (m, min(1, m * pD[0])))
