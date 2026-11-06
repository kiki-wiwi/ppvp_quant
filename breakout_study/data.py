"""Narrow, default-locked input adapter. No downloads or cache repair."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from .scope import RealDataLockedError, require_real_authorization, utc_index, make_context
from .config import IS_START, IS_END

ROOT = Path(__file__).resolve().parents[1]


def verify_implementation_registration(root=ROOT):
    root=Path(root)
    document=json.loads((root/'reports/channel_vs_vah_round/implementation_hashes.json').read_text())
    for name,want in document['sha256'].items():
        raw=(root/name).read_bytes()
        if name=='breakout_study/scope.py':
            # Only a separately approved one-line unlock may differ later.
            raw=raw.replace(b'REAL_DATA_UNLOCKED = True\n',b'REAL_DATA_UNLOCKED = False\n')
        if hashlib.sha256(raw).hexdigest()!=want:
            raise ValueError(f'new implementation hash mismatch: {name}')
    return document


def verify_registered_assets(root=ROOT):
    """Only source files and manifests are read, never market payloads."""
    root=Path(root)
    reg=json.loads((root/'reports/channel_vs_vah_round/registration.json').read_text())
    hashes=dict(reg['legacy_frozen_files'])
    if len(hashes)!=27:
        raise ValueError('expected 27 unchanged legacy frozen files')
    for name,want in hashes.items():
        raw=(root/name).read_bytes()
        if name=='impulse_study/engine.py':
            raw=raw.replace(b'REAL_DATA_UNLOCKED = True\n',b'REAL_DATA_UNLOCKED = False\n')
        if hashlib.sha256(raw).hexdigest()!=want:
            raise ValueError(f'frozen file mismatch: {name}')
    more={
        'reports/channel_vs_vah_round/approved_plan.md':reg['approved_plan_sha256'],
        'reports/impulse_vs_box_round/experiment_plan_impulse.md':reg['legacy_plan_sha256'],
        'data/manifest_impulse_1m.json':reg['legacy_manifest_sha256'],
    }
    for name,want in more.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=want:
            raise ValueError(f'registered asset mismatch: {name}')
    from impulse_study.versions import frozen_key
    if frozen_key()!=reg['legacy_frozen_key']:
        raise ValueError('legacy frozen_key mismatch')
    if (root/'reports/channel_vs_vah_round/implementation_hashes.json').exists():
        pinned=verify_implementation_registration(root)
        reg['new_files_verified']=len(pinned['sha256'])
    return reg


def load_funding_is(root=ROOT):
    require_real_authorization()
    df=pd.read_parquet(Path(root)/'data/processed/BTCUSDT_funding.parquet',
        filters=[('funding_time','>=',IS_START),('funding_time','<',IS_END)])
    idx=pd.DatetimeIndex(df.index)
    if idx.tz is None:
        raise ValueError('funding timestamps require timezone')
    idx=utc_index(idx.tz_convert('UTC').round('min'),'funding')
    want=pd.date_range(IS_START,IS_END,freq='8h',inclusive='left').as_unit('ns')
    if not idx.equals(want) or len(idx)!=4383:
        raise ValueError('funding must be the complete 4383-settlement IS grid')
    rates=np.asarray(df['funding_rate'],float)
    if not np.isfinite(rates).all():
        raise ValueError('invalid funding values')
    return pd.Series(rates,index=idx,name='funding_rate')


def load_perp_is(root=ROOT):
    require_real_authorization()
    from impulse_study.data_1m import content_sha256
    root=Path(root)
    manifest=json.loads((root/'data/manifest_impulse_1m.json').read_text())
    frame=pd.read_parquet(root/'data/processed_1m/BTCUSDT_perp_1m_IS.parquet',
        filters=[('open_time','>=',IS_START),('open_time','<',IS_END)])
    frame=frame.copy()
    frame.index=utc_index(frame.index,'perp minutes')
    if content_sha256(frame)!=manifest['caches']['perp_1m']['content_sha256']:
        raise ValueError('perp IS content hash differs from registered manifest')
    return frame


def archive_gap_index():
    from impulse_study.versions import ARCHIVE_GAP_MINUTES
    return pd.DatetimeIndex(pd.to_datetime(ARCHIVE_GAP_MINUTES,utc=True)).as_unit('ns')


def load_real_context(root=ROOT):
    require_real_authorization()
    verify_implementation_registration(root)
    verify_registered_assets(root)
    from impulse_study.data_1m import minute_grid
    frame=load_perp_is(root)
    funding=load_funding_is(root)
    grid=minute_grid(frame, IS_START, IS_END)
    # Missing/bad bars are never traded. Their mark/funding reference is the last
    # valid close, with no backfill from future prices.
    prior=grid['close'].where(~grid['bad']).ffill()
    for col in ('open','high','low','close'):
        grid.loc[grid['bad'],col]=prior.loc[grid['bad']]
    grid.loc[grid['bad'],'volume']=0.
    gaps=archive_gap_index()
    grid['archive_gap']=grid.index.isin(gaps)
    grid.attrs['source']='real'
    ctx=make_context(grid,funding,source='real')
    if len(ctx)!=2103840 or np.isfinite(ctx.funding_rate).sum()!=4383 or ctx.archive_gap.sum()!=10:
        raise ValueError('registered IS grid/funding/archive-gap counts differ')
    return ctx,{'minute_count':len(ctx),'funding_count':4383,'archive_gap_count':10,
               'perp_content_hash_verified':True,'source':'real','symbol':'BTCUSDT'}
