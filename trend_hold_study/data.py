"""Independent authorization, source registration and bounded BTC IS reads."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd

from . import config, scope

ROOT = Path(__file__).resolve().parents[1]
REPORT = 'reports/trend_hold_round'
LOCK_FILE = 'trend_hold_study/scope.py'
LOCKED_LINE = b'REAL_DATA_UNLOCKED = False\n'
UNLOCKED_LINE = b'REAL_DATA_UNLOCKED = True\n'
ENVIRONMENT = {
    'python': '3.14.4', 'pandas': '3.0.5', 'numpy': '2.5.4',
    'pyarrow': '26.0.1', 'numba': '0.68.1', 'scipy': '1.18.1',
    'matplotlib': '3.11.2', 'pytest': '9.2.0',
}
MINUTE_COUNT = 2_103_840
FUNDING_COUNT = 4383
BAD_COUNT = 144
ARCHIVE_GAP_COUNT = 10


def environment_versions():
    return {
        'python': platform.python_version(),
        **{name: importlib.metadata.version(name) for name in ENVIRONMENT if name != 'python'},
    }


def verify_environment(actual=None):
    actual = environment_versions() if actual is None else actual
    if actual != ENVIRONMENT:
        raise ValueError(f'registered environment mismatch: {actual}')
    return actual


def verify_implementation_registration(root=ROOT):
    root = Path(root)
    document = json.loads((root / REPORT / 'implementation_hashes.json').read_text())
    for name, expected in document['sha256'].items():
        raw = (root / name).read_bytes()
        if name == LOCK_FILE:
            raw = raw.replace(UNLOCKED_LINE, LOCKED_LINE)
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f'new implementation hash mismatch: {name}')
    return document


def verify_registered_assets(root=ROOT):
    """Read registered source and manifest bytes only."""
    from channel_random_study.data import verify_registered_assets as verify_prior

    root = Path(root)
    folder = root / REPORT
    reg = json.loads((folder / 'registration.json').read_text())
    if reg['baseline_commit'] != config.BASELINE:
        raise ValueError('registered baseline mismatch')
    for name, expected in reg['protected_sources_sha256'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'protected source hash mismatch: {name}')
    actual_plan_hash = hashlib.sha256((folder / 'approved_plan.md').read_bytes()).hexdigest()
    if actual_plan_hash != reg['approved_plan_sha256'] or actual_plan_hash != config.APPROVED_PLAN_SHA256:
        raise ValueError('approved plan hash mismatch')
    if verify_prior(root)['legacy_frozen_key'] != reg['legacy_frozen_key']:
        raise ValueError('legacy frozen_key mismatch')

    result = {**reg, 'protected_sources_verified': len(reg['protected_sources_sha256'])}
    if (folder / 'implementation_hashes.json').exists():
        result['new_files_verified'] = len(verify_implementation_registration(root)['sha256'])
    return result


def verify_unlock(root=ROOT):
    """Require one additive implementation commit and a one-line unlock."""
    scope.require_real_authorization()
    root = Path(root)

    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root)

    if git('status', '--porcelain', '--untracked-files=all').strip():
        raise ValueError('formal execution requires a completely clean worktree')
    head_history = git('rev-list', '--parents', '-n', '1', 'HEAD').decode().split()
    if len(head_history) != 2:
        raise ValueError('unlock must have exactly one implementation parent')
    head, implementation = head_history
    impl_history = git('rev-list', '--parents', '-n', '1', implementation).decode().split()
    if impl_history != [implementation, config.BASELINE]:
        raise ValueError('implementation must directly descend from registered baseline')
    if git('diff', '--diff-filter=CDMRTUXB', '--name-only', config.BASELINE, implementation).strip():
        raise ValueError('implementation commit may only add new files')

    expected = f'M\t{LOCK_FILE}'.encode()
    if (git('diff', '--name-status', implementation, head).strip() != expected
            or git('diff', '--summary', implementation, head).strip()):
        raise ValueError('unlock commit must change only the independent lock file')
    before = git('show', f'{implementation}:{LOCK_FILE}')
    after = git('show', f'{head}:{LOCK_FILE}')
    if before.splitlines(keepends=True).count(LOCKED_LINE) != 1 or after != before.replace(
            LOCKED_LINE, UNLOCKED_LINE):
        raise ValueError('unlock commit must change only False to True')
    if os.path.lexists(root / REPORT / 'results'):
        raise FileExistsError('formal result directory already exists; never resume or overwrite')
    return {'implementation_commit': implementation, 'unlock_commit': head, 'starting_worktree_status': ''}


def load_perp_is(root=ROOT):
    scope.require_real_authorization()
    from impulse_study.data_1m import content_sha256

    root = Path(root)
    manifest = json.loads((root / 'data/manifest_impulse_1m.json').read_text())
    frame = pd.read_parquet(
        root / 'data/processed_1m/BTCUSDT_perp_1m_IS.parquet',
        filters=[('open_time', '>=', config.IS_START), ('open_time', '<', config.IS_END)],
    )
    frame = frame.copy(deep=False)
    frame.index = scope.utc_index(frame.index, 'perp minutes')
    if content_sha256(frame) != manifest['caches']['perp_1m']['content_sha256']:
        raise ValueError('perp IS content hash differs from registered manifest')
    return frame


def load_funding_is(root=ROOT):
    scope.require_real_authorization()
    frame = pd.read_parquet(
        Path(root) / 'data/processed/BTCUSDT_funding.parquet',
        filters=[('funding_time', '>=', config.IS_START), ('funding_time', '<', config.IS_END)],
    )
    raw = scope.utc_index(frame.index, 'unrounded funding')
    index = scope.utc_index(raw.round('min'), 'funding')
    expected = pd.date_range(config.IS_START, config.IS_END, freq='8h', inclusive='left').as_unit('ns')
    if not index.equals(expected) or len(index) != FUNDING_COUNT:
        raise ValueError('funding must match the complete 4383-settlement IS grid')
    rates = np.asarray(frame['funding_rate'], dtype=np.float64)
    if not np.isfinite(rates).all():
        raise ValueError('funding must be finite')
    return pd.Series(rates, index=index, name='funding_rate')


def load_real_context(root=ROOT):
    scope.require_real_authorization()
    environment = verify_environment()
    registration = verify_registered_assets(root)
    if not registration.get('new_files_verified'):
        raise ValueError('new implementation hash registration required')
    from breakout_study.data import archive_gap_index
    from impulse_study.data_1m import minute_grid

    frame = load_perp_is(root)
    funding = load_funding_is(root)
    grid = minute_grid(frame, config.IS_START, config.IS_END)
    bad = grid['bad']
    prior = grid['close'].where(~bad).ffill()
    for name in ('open', 'high', 'low', 'close'):
        grid.loc[bad, name] = prior.loc[bad]
    grid.loc[bad, 'volume'] = 0.
    grid['archive_gap'] = grid.index.isin(archive_gap_index())
    grid.attrs['source'] = 'real'
    ctx = scope.make_context(grid, funding, source='real')
    if (len(ctx) != MINUTE_COUNT or np.isfinite(ctx.funding_rate).sum() != FUNDING_COUNT
            or ctx.archive_gap.sum() != ARCHIVE_GAP_COUNT or ctx.bad.sum() != BAD_COUNT):
        raise ValueError('registered IS minute/funding/bad/archive counts differ')
    return ctx, {
        'source': 'real', 'symbol': 'BTCUSDT', 'minute_count': len(ctx),
        'funding_count': FUNDING_COUNT, 'bad_count': BAD_COUNT,
        'archive_gap_count': ARCHIVE_GAP_COUNT, 'perp_content_hash_verified': True,
        'environment': environment, 'registration': registration,
    }
