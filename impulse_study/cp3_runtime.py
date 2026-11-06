"""CP3 run controls and narrow input entry points (§10, §12).

Pre-unlock implementation choices: missing caches are rebuilt only for perp_1m
and spot_1d_warmup from the existing pinned manifest. CP4-only market archives
are not fetched. Output is create-once; failures leave all partial output and a
STOPPED record. Registration is JSON embedded in the unlock commit message.
No retries of the study, selective resume, seed options, or reduced draws exist.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import subprocess
import numpy as np
import pandas as pd
from . import data_1m as D, engine as E, features as F, versions as VS
from .scope import IS_START, IS_END, ScopeError, assert_within_is, assert_before_is

ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'b4e3fc19f7f8ece6177a8460983d68b93bd481c3'
PLAN = 'reports/impulse_vs_box_round/experiment_plan_impulse.md'
EXPECTED_KEY = 'fc0d37b9d198ff85'
REG_START = 'CP3_REGISTRATION_JSON_BEGIN'
REG_END = 'CP3_REGISTRATION_JSON_END'
REGISTER_EXISTING = (
    'tools/run_impulse_cp1.py', 'data/manifest_impulse_1m.json',
    'reports/impulse_vs_box_round/cp2/cp2_feature_checks.json', PLAN,
)


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, check=True,
                          text=True, capture_output=True).stdout.strip()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_hashes():
    table = dict(re.findall(r'^\| `([^`]+)` \| `([0-9a-f]{64})` \|$',
                           (ROOT / PLAN).read_text(), re.M))
    if len(table) != 27:
        raise ValueError('frozen section 13 must contain exactly 27 files')
    actual = {}
    for name, expected in table.items():
        b = (ROOT / name).read_bytes()
        if name == 'impulse_study/engine.py':
            b = b.replace(b'REAL_DATA_UNLOCKED = True\n', b'REAL_DATA_UNLOCKED = False\n')
        actual[name] = hashlib.sha256(b).hexdigest()
        if actual[name] != expected:
            raise ValueError(f'frozen SHA256 mismatch: {name}')
    if VS.frozen_key() != EXPECTED_KEY:
        raise ValueError('frozen_key mismatch')
    baseline_plan = subprocess.run(['git', 'show', f'{BASELINE}:{PLAN}'], cwd=ROOT,
                                   check=True, capture_output=True).stdout
    if (ROOT / PLAN).read_bytes() != baseline_plan:
        raise ValueError('frozen plan changed')
    return actual


def registration():
    message = git('log', '-1', '--format=%B')
    return json.loads(message.split(REG_START, 1)[1].split(REG_END, 1)[0])


def verify_unlock():
    """Called before any real data read; require exact second commit and clean tree."""
    frozen = frozen_hashes()
    if not E.REAL_DATA_UNLOCKED:
        raise E.EngineLockedError('CP3 requires the registered unlock commit')
    dirty = git('status', '--porcelain')
    if dirty:
        raise ValueError(f'CP3 must start on a clean tree: {dirty}')
    reg = registration()
    impl = reg['implementation_commit']
    if git('rev-parse', 'HEAD^') != impl or git('rev-parse', f'{impl}^') != BASELINE:
        raise ValueError('three-commit protocol parent chain mismatch')
    changes = git('diff', '--name-status', 'HEAD^', 'HEAD')
    if changes != 'M\timpulse_study/engine.py':
        raise ValueError('unlock commit must change engine.py only')
    before = subprocess.run(['git', 'show', 'HEAD^:impulse_study/engine.py'], cwd=ROOT,
                            check=True, capture_output=True).stdout
    after = (ROOT / 'impulse_study/engine.py').read_bytes()
    if before.replace(b'REAL_DATA_UNLOCKED = False\n', b'REAL_DATA_UNLOCKED = True\n') != after:
        raise ValueError('unlock diff is not exactly False -> True')
    added = git('diff', '--name-status', BASELINE, impl).splitlines()
    if any(not row.startswith('A\t') for row in added):
        raise ValueError('implementation commit must only add files')
    if {row[2:] for row in added} != set(reg['new_file_sha256']):
        raise ValueError('implementation registration incomplete')
    for section in ('new_file_sha256', 'existing_dependency_sha256'):
        for name, expected in reg[section].items():
            if sha256(ROOT / name) != expected:
                raise ValueError(f'registered SHA256 mismatch: {name}')
    if not set(REGISTER_EXISTING) <= set(reg['existing_dependency_sha256']):
        raise ValueError('existing dependency registration incomplete')
    return {'baseline': BASELINE, 'implementation_commit': impl,
            'unlock_commit': git('rev-parse', 'HEAD'), 'dirty_files': [],
            'frozen_key': EXPECTED_KEY, 'frozen_sha256_normalized': frozen,
            'registration': reg}


def _ensure_cp3_cache(key):
    if key not in ('perp_1m', 'spot_1d_warmup'):
        raise ScopeError('CP3 cache entry point allows only IS BTC 1m and spot daily price warmup')
    p = D.CACHE_DIR / D.CACHE_FILES[key]
    if p.exists():
        return
    man = D._manifest()
    frames = []
    for rec in man['files'][key]:
        rel = rec['url'].removeprefix(D.VISION_BASE + '/')
        D._guard_rel_path(rel)
        blob, _ = D.fetch_archive(rel, rec['sha256'])
        frames.append(D._parse(blob))
    df = pd.concat(frames).sort_index(kind='stable')
    if df.index.has_duplicates:
        raise ValueError('duplicate archive minutes')
    if key == 'perp_1m':
        assert_within_is(df.index, 'CP3 pinned BTCUSDT 1m archives')
        df = df.copy()
        df['src'] = np.int8(0)
        for rec in man['adjudicated_days']:
            if rec['verdict'] == 'replace_with_daily_archive':
                day = pd.Timestamp(rec['day'], tz='UTC')
                daily, _ = D.fetch_daily_1m(day, pinned_sha256=rec['daily_1m_archive']['sha256'])
                df = D._replace_day(df, daily, day)
    else:
        assert_before_is(df.index, 'CP3 spot daily warmup')
    if D.content_sha256(df) != man['caches'][key]['content_sha256']:
        raise ValueError(f'pinned reconstructed cache content differs: {key}')
    D._write(df, key)


def load_funding_is():
    """Predicate is applied in the parquet reader, never after a full-table read."""
    df = pd.read_parquet(ROOT / 'data/processed/BTCUSDT_funding.parquet',
                         filters=[('funding_time', '>=', IS_START),
                                  ('funding_time', '<', IS_END)])
    df = df.copy()
    df.index = pd.DatetimeIndex(df.index).tz_convert('UTC').round('min').as_unit('ns')
    assert_within_is(df.index, 'CP3 funding after nearest-minute rounding')
    if df.index.has_duplicates or not np.isfinite(df['funding_rate'].to_numpy()).all():
        raise ValueError('invalid funding rates or duplicate rounded timestamps')
    expected = pd.date_range(IS_START, IS_END, freq='8h', inclusive='left').as_unit('ns')
    if len(df) != 4383 or not df.index.equals(expected):
        raise ValueError('funding must contain the exact 4383 IS settlements')
    return df['funding_rate']


def load_real_context():
    if not E.REAL_DATA_UNLOCKED:
        raise E.EngineLockedError('real input loading is locked before the second commit')
    for key in ('perp_1m', 'spot_1d_warmup'):
        _ensure_cp3_cache(key)
    df, spot = D.load_perp_1m(), D.load_spot_warmup('1d')
    funding = load_funding_is()
    grid = D.minute_grid(df, IS_START, IS_END)
    box, states = F.s6_box_ok(df, spot, grid)
    ctx = F.make_context(grid, box, 'binance_is',
                         archive_gap_minutes=VS.ARCHIVE_GAP_MINUTES, funding=funding)
    E._require_real_inputs(ctx)
    box_sha = hashlib.sha256(np.ascontiguousarray(ctx.box_ok).tobytes()).hexdigest()
    expected = json.loads((ROOT / 'reports/impulse_vs_box_round/cp2/cp2_feature_checks.json').read_text())['box_ok_sha256']
    if len(ctx.index) != 2103840 or ctx.n_funding_mapped != 4383 or int(ctx.archive_gap.sum()) != 10:
        raise ValueError('IS input counts differ from frozen CP2 requirements')
    if box_sha != expected:
        raise ValueError('S6 box_ok SHA256 differs from CP2')
    return ctx, {'funding_count': 4383, 'archive_gap_count': 10,
                 'box_ok_sha256': box_sha, 'minute_count': len(ctx.index)}
