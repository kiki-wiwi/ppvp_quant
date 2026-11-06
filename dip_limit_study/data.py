"""Default-locked IS adapter: reuse immutable input validation, never cache repair."""
from pathlib import Path
import hashlib
import json
from . import scope

ROOT=Path(__file__).resolve().parents[1]
REPORT='reports/trend_dip_limit_round'

def verify_implementation_registration(root=ROOT):
    root=Path(root)
    doc=json.loads((root/REPORT/'implementation_hashes.json').read_text())
    for name,want in doc['sha256'].items():
        raw=(root/name).read_bytes()
        if name=='dip_limit_study/scope.py':
            raw=raw.replace(b'REAL_DATA_UNLOCKED = True\n',b'REAL_DATA_UNLOCKED = False\n')
        if hashlib.sha256(raw).hexdigest()!=want:
            raise ValueError('new implementation hash mismatch: '+name)
    return doc

def verify_registered_assets(root=ROOT):
    root=Path(root)
    reg=json.loads((root/REPORT/'registration.json').read_text())
    for name,want in reg['protected_sources_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=want:
            raise ValueError('protected source hash mismatch: '+name)
    if hashlib.sha256((root/REPORT/'approved_plan.md').read_bytes()).hexdigest()!=reg['approved_plan_sha256']:
        raise ValueError('approved plan hash mismatch')
    if 'legacy_frozen_key' in reg:
        from impulse_study.versions import frozen_key
        if frozen_key()!=reg['legacy_frozen_key']:
            raise ValueError('legacy frozen_key mismatch')
    result={**reg,'protected_sources_verified':len(reg['protected_sources_sha256'])}
    if (root/REPORT/'implementation_hashes.json').exists():
        result['new_files_verified']=len(verify_implementation_registration(root)['sha256'])
    return result

def load_perp_is(root=ROOT):
    scope.require_real_authorization()
    from breakout_study.data import load_perp_is as existing
    return existing(root)

def load_funding_is(root=ROOT):
    scope.require_real_authorization()
    from breakout_study.data import load_funding_is as existing
    return existing(root)

def load_real_context(root=ROOT):
    scope.require_real_authorization()
    verify_implementation_registration(root)
    verify_registered_assets(root)
    from impulse_study.data_1m import minute_grid
    from breakout_study.data import archive_gap_index
    from .config import IS_START,IS_END
    import numpy as np
    frame=load_perp_is(root)
    funding=load_funding_is(root)
    grid=minute_grid(frame,IS_START,IS_END)
    previous=grid['close'].where(~grid['bad']).ffill()
    for col in ('open','high','low','close'):
        grid.loc[grid['bad'],col]=previous.loc[grid['bad']]
    grid.loc[grid['bad'],'volume']=0.
    grid['archive_gap']=grid.index.isin(archive_gap_index())
    grid.attrs['source']='real'
    ctx=scope.make_context(grid,funding,source='real')
    if len(ctx)!=2103840 or np.isfinite(ctx.funding_rate).sum()!=4383 or ctx.archive_gap.sum()!=10:
        raise ValueError('IS input minute/funding/archive counts differ')
    return ctx,{'source':'real','symbol':'BTCUSDT','minute_count':len(ctx),
        'funding_count':4383,'archive_gap_count':10,'perp_content_hash_verified':True}
