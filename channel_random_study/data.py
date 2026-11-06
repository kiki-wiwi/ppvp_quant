"""Independent authorization and bounded BTC IS reads; never repair inputs."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import numpy as np
import pandas as pd
from . import scope, config

ROOT = Path(__file__).resolve().parents[1]
REPORT = 'reports/channel_random_round'
ENVIRONMENT = {'python':'3.14.4','pandas':'3.0.5','numpy':'2.5.4',
               'pyarrow':'26.0.1','numba':'0.68.1','scipy':'1.18.1',
               'matplotlib':'3.11.2','pytest':'9.2.0'}


def environment_versions():
    return {'python':platform.python_version(), **{name:importlib.metadata.version(name)
            for name in ENVIRONMENT if name!='python'}}


def verify_environment(actual=None):
    actual=environment_versions() if actual is None else actual
    if actual!=ENVIRONMENT:
        raise ValueError(f'registered environment mismatch: {actual}')
    return actual


def verify_implementation_registration(root=ROOT):
    root=Path(root)
    doc=json.loads((root/REPORT/'implementation_hashes.json').read_text())
    for name,want in doc['sha256'].items():
        raw=(root/name).read_bytes()
        if name=='channel_random_study/scope.py':
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
        from breakout_study.data import verify_registered_assets as verify_legacy
        from dip_limit_study.data import verify_registered_assets as verify_dip
        legacy=verify_legacy(root)
        verify_dip(root)
        if legacy['legacy_frozen_key']!=reg['legacy_frozen_key']:
            raise ValueError('legacy frozen_key mismatch')
    result={**reg,'protected_sources_verified':len(reg['protected_sources_sha256'])}
    if (root/REPORT/'implementation_hashes.json').exists():
        result['new_files_verified']=len(verify_implementation_registration(root)['sha256'])
    return result


def verify_unlock(root=ROOT):
    """The future run must start on a clean, one-line-only unlock commit."""
    scope.require_real_authorization()
    root=Path(root)
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
    if git('status','--porcelain'):
        raise ValueError('formal execution requires a completely clean worktree')
    head,parent=git('rev-parse','HEAD'),git('rev-parse','HEAD^')
    if git('rev-parse',parent+'^')!=config.BASELINE:
        raise ValueError('implementation must directly descend from registered baseline')
    expected='M\tchannel_random_study/scope.py'
    if git('diff','--name-status',parent,head)!=expected:
        raise ValueError('unlock commit must change only the independent lock file')
    old=git('show',parent+':channel_random_study/scope.py')
    new=git('show',head+':channel_random_study/scope.py')
    if old.count('REAL_DATA_UNLOCKED = False')!=1 or new!=old.replace(
            'REAL_DATA_UNLOCKED = False','REAL_DATA_UNLOCKED = True'):
        raise ValueError('unlock commit must change only False to True')
    if (root/REPORT/'results').exists():
        raise FileExistsError('formal result directory already exists; never resume or overwrite')
    return {'implementation_commit':parent,'unlock_commit':head,'starting_worktree_status':''}


def load_perp_is(root=ROOT):
    scope.require_real_authorization()
    from impulse_study.data_1m import content_sha256
    root=Path(root)
    manifest=json.loads((root/'data/manifest_impulse_1m.json').read_text())
    frame=pd.read_parquet(root/'data/processed_1m/BTCUSDT_perp_1m_IS.parquet',
        filters=[('open_time','>=',config.IS_START),('open_time','<',config.IS_END)])
    frame=frame.copy()
    frame.index=scope.utc_index(frame.index,'perp minutes')
    if content_sha256(frame)!=manifest['caches']['perp_1m']['content_sha256']:
        raise ValueError('perp IS content hash differs from registered manifest')
    return frame


def load_funding_is(root=ROOT):
    scope.require_real_authorization()
    frame=pd.read_parquet(Path(root)/'data/processed/BTCUSDT_funding.parquet',
        filters=[('funding_time','>=',config.IS_START),('funding_time','<',config.IS_END)])
    raw=pd.DatetimeIndex(frame.index)
    if raw.tz is None:
        raise ValueError('funding timestamps require timezone')
    idx=scope.utc_index(raw.tz_convert('UTC').round('min'),'funding')
    expected=pd.date_range(config.IS_START,config.IS_END,freq='8h',inclusive='left').as_unit('ns')
    if not idx.equals(expected) or len(idx)!=4383:
        raise ValueError('funding must match the complete 4383-settlement IS grid')
    values=np.asarray(frame['funding_rate'],float)
    if not np.isfinite(values).all():
        raise ValueError('funding must be finite')
    return pd.Series(values,index=idx,name='funding_rate')


def load_real_context(root=ROOT):
    scope.require_real_authorization()
    verify_environment()
    registration=verify_registered_assets(root)
    verify_implementation_registration(root)
    from impulse_study.data_1m import minute_grid
    from impulse_study.versions import ARCHIVE_GAP_MINUTES
    frame=load_perp_is(root)
    funding=load_funding_is(root)
    grid=minute_grid(frame,config.IS_START,config.IS_END)
    previous=grid['close'].where(~grid['bad']).ffill()
    for col in ('open','high','low','close'):
        grid.loc[grid['bad'],col]=previous.loc[grid['bad']]
    grid.loc[grid['bad'],'volume']=0.
    gaps=pd.DatetimeIndex(pd.to_datetime(ARCHIVE_GAP_MINUTES,utc=True)).as_unit('ns')
    grid['archive_gap']=grid.index.isin(gaps)
    grid.attrs['source']='real'
    ctx=scope.make_context(grid,funding,source='real')
    if (len(ctx)!=2103840 or np.isfinite(ctx.funding_rate).sum()!=4383
            or ctx.archive_gap.sum()!=10 or ctx.bad.sum()!=144):
        raise ValueError('registered IS minute/funding/bad/archive counts differ')
    return ctx,{'source':'real','symbol':'BTCUSDT','minute_count':len(ctx),
        'funding_count':4383,'bad_count':144,'archive_gap_count':10,
        'perp_content_hash_verified':True,'environment':environment_versions(),
        'registration':registration}
