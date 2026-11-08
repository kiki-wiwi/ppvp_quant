"""AUDITOR SENSITIVITY (post hoc, not a replacement result): Spot self-signal V2_SPOT vs P_SPOT on the
archived truncated Post-IS window [2024-01-01,2026-06-01) and on the originally specified window
[2024-01-01,2026-11-01), using official spot 1m archives (Jun-Oct 2026 downloaded and CHECKSUM-verified)."""
import io, math, sys, zipfile, glob
import numpy as np, pandas as pd
sys.path.insert(0, sys.argv[4])
import venue_independent as vi
from cp3_stats_lib import webb

cache, ext_dir, asset = sys.argv[1], sys.argv[2], sys.argv[3]
COLS = ['open_time','open','high','low','close','volume','close_time','qv','n','tbv','tbqv','ignore']
frames = [pd.read_parquet(cache)[['open','high','low','close','volume']]]
for f in sorted(glob.glob(f'{ext_dir}/{asset}USDT-1m-2026-*.zip')):
    raw = zipfile.ZipFile(f).read(zipfile.ZipFile(f).namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), header=0 if raw[:9] == b'open_time' else None, dtype=str); df.columns = COLS
    t = df['open_time'].astype('int64'); unit = 'us' if t.max() > 1e14 else 'ms'
    df.index = pd.DatetimeIndex(pd.to_datetime(t, unit=unit, utc=True)).as_unit('ns')
    frames.append(df[['open','high','low','close','volume']].astype(float))
full = pd.concat(frames); full = full[~full.index.duplicated()].sort_index()
tmp = f'{ext_dir}/_{asset}_spot_full.parquet'; full.to_parquet(tmp)
start, end_a, end_b = (pd.Timestamp(x, tz='UTC') for x in ('2024-01-01', '2026-06-01', '2026-11-01'))
for label, end in (('truncated', end_a), ('original_spec', end_b)):
    sp = vi.grid_frame(tmp, pd.Timestamp('2023-11-01', tz='UTC'), end)
    print(f'{asset} {label}: minutes {len(sp["index"])}, missing/bad {int(sp["bad"].sum())}')
    feat = vi.canonical_features(sp, start, start)
    out = {}
    for v in ('V2_SPOT', 'P_SPOT'):
        eq, meta = vi.run_spot(sp, feat, v, 'main_17', start)
        d = eq[1439::1440]; out[v] = pd.Series(d / np.r_[100000.0, d[:-1]] - 1, index=feat.index)
    T, P = out['V2_SPOT'], out['P_SPOT']
    for wname, a, b in (('Post-IS', start, end), ('VAL', start, pd.Timestamp('2025-07-01', tz='UTC')), ('OOS', pd.Timestamp('2025-07-01', tz='UTC'), end)):
        m = (T.index >= a) & (T.index < b)
        months = T.index[m].strftime('%Y-%m').to_numpy()
        mT, pT, G = webb(T[m].to_numpy(), months); mD, pD, _ = webb((T - P)[m].to_numpy(), months)
        cumT = np.prod(1 + T[m]) - 1; cumP = np.prod(1 + P[m]) - 1
        print(f'   {wname:8s} months={G:2d} T={mT*1e4:6.2f}bp D={mD*1e4:6.2f}bp pT={pT:.4f} pD={pD:.4f} cumT={cumT:+.2%} cumP={cumP:+.2%}')
