"""Auditor diagnostics referenced in AUDIT_REPORT.md (post hoc, descriptive; none replaces a frozen gate).
1) R5 D decomposition by trend state; start-date sensitivity.
2) 2023-11-10 monthly vs adjudicated daily archive: does the daily close change?
3) Illustrative cash-yield sensitivity for Spot accounts in Post-IS (4% simple interest on idle USDT).
Usage: python -I extra_diagnostics.py <scripts_dir> <r5_out_dir> <btc_1m_inputs_dir> <manifest_impulse_1m.json>
"""
import io, json, sys, zipfile
import numpy as np, pandas as pd
from scipy import stats

scripts, r5_out, r5_inputs, manifest = sys.argv[1:5]
sys.path.insert(0, scripts)

# 1) D decomposition
dr = pd.read_csv(f'{r5_out}/daily_returns.csv', index_col=0, parse_dates=True)
f = pd.read_csv(f'{r5_out}/features.csv', index_col=0, parse_dates=True)
T, P = dr['T_main_17'].to_numpy(), dr['P_main_17'].to_numpy(); D = T - P; g = f['g'].to_numpy().astype(bool)
print('[1] on days', int(g.sum()), 'off days', int((~g).sum()))
print('    D on-days mean %.3f bp (sum %.1f bp); D off-days mean %.3f bp (sum %.1f bp)' % (D[g].mean() * 1e4, D[g].sum() * 1e4, D[~g].mean() * 1e4, D[~g].sum() * 1e4))
t = stats.ttest_1samp(P[~g], 0)
print('    P on-days mean %.2f bp; P off-days mean %.2f bp (iid t %.3f)' % (P[g].mean() * 1e4, P[~g].mean() * 1e4, t.statistic))
m = dr.index >= pd.Timestamp('2020-04-01', tz='UTC')
print('    excluding 2020-03: D mean %.3f bp, T mean %.3f bp' % (D[m].mean() * 1e4, T[m].mean() * 1e4))

# 2) 2023-11-10 adjudication
COLS = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qv', 'n', 'tbv', 'tbqv', 'ignore']
mf = json.load(open(manifest))
def rz(sha):
    z = zipfile.ZipFile(f'{r5_inputs}/{sha}'); raw = z.read(z.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), header=0 if raw[:9] == b'open_time' else None, dtype=str); df.columns = COLS
    df.index = pd.to_datetime(df.open_time.astype('int64'), unit='ms', utc=True)
    return df[['open', 'high', 'low', 'close', 'volume']].astype(float)
mon = [x for x in mf['files']['perp_1m'] if '2023-11' in x['url']][0]
M = rz(mon['sha256']); M = M[(M.index >= '2023-11-10') & (M.index < '2023-11-11')]
Dd = rz(mf['adjudicated_days'][0]['daily_1m_archive']['sha256'])
flat = (M.volume == 0) & (M.open == M.high) & (M.high == M.low) & (M.low == M.close)
print('[2] monthly placeholders %d (%s .. %s); 23:59 close monthly %.1f daily %.1f; max |close diff| %.1f' % (
    int(flat.sum()), M.index[flat].min(), M.index[flat].max(), M.close.iloc[-1], Dd.close.iloc[-1], np.abs(M.close.values - Dd.close.values).max()))

# 3) cash-yield illustration
import venue_independent as vi
for asset, cfgf in (('BTC', f'{scripts}/cfg_btc_post.json'), ('ETH', f'{scripts}/cfg_eth_post.json')):
    cfg = json.load(open(cfgf)); start, end = (pd.Timestamp(x, tz='UTC') for x in cfg['window'])
    um = vi.grid_frame(cfg['um'], pd.Timestamp(cfg['load_start'], tz='UTC'), end, cfg['um_funding'])
    sp = vi.grid_frame(cfg['spot'], pd.Timestamp(cfg['load_start'], tz='UTC'), end)
    feat = vi.canonical_features(um, start, start)
    for v in ('V2_SPOT', 'P_SPOT'):
        eq, _ = vi.run_spot(sp, feat, v, 'main_17', start)
        gl = (feat['g'].astype(float) if v == 'V2_SPOT' else 1.0) * feat['budget']
        d = eq[1439::1440]
        extra = (d * (1 - gl.to_numpy()) * 0.04 / 365).sum()
        print('[3] %s %s Post-IS final %.0f USD; illustrative 4%% simple interest on idle cash (1-gL) %.0f USD' % (asset, v, d[-1], extra))
