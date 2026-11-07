"""Validate real-data boundaries with generated metadata and mocked I/O only."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from trend_hold_study import config, data, scope


@pytest.mark.parametrize('name', ('verify_unlock', 'load_real_context', 'load_perp_is', 'load_funding_is'))
def test_real_entry_rejects_before_touching_any_path(name, monkeypatch):
    monkeypatch.setattr(scope, 'REAL_DATA_UNLOCKED', False)

    class UnreadablePath:
        def __fspath__(self):
            pytest.fail('entry touched a path before its independent lock')

    with pytest.raises(scope.RealDataLockedError):
        getattr(data, name)(UnreadablePath())


def test_registration_rejects_changed_bytes(tmp_path, monkeypatch):
    folder = tmp_path / data.REPORT
    folder.mkdir(parents=True)
    (tmp_path / 'frozen.py').write_bytes(b'original\n')
    plan = folder / 'approved_plan.md'
    plan.write_bytes(b'approved\n')
    plan_hash = hashlib.sha256(plan.read_bytes()).hexdigest()
    monkeypatch.setattr(config, 'APPROVED_PLAN_SHA256', plan_hash)
    doc = {
        'baseline_commit': config.BASELINE,
        'protected_sources_sha256': {'frozen.py': hashlib.sha256(b'original\n').hexdigest()},
        'approved_plan_sha256': plan_hash,
        'legacy_frozen_key': 'test-key',
    }
    (folder / 'registration.json').write_text(json.dumps(doc))
    from channel_random_study import data as prior
    monkeypatch.setattr(prior, 'verify_registered_assets',
                        lambda root: {'legacy_frozen_key': 'test-key'})
    assert data.verify_registered_assets(tmp_path)['protected_sources_verified'] == 1
    (tmp_path / 'frozen.py').write_bytes(b'tampered\n')
    with pytest.raises(ValueError, match='protected source'):
        data.verify_registered_assets(tmp_path)


def test_implementation_hash_only_normalizes_new_study_lock(tmp_path):
    folder = tmp_path / data.REPORT
    folder.mkdir(parents=True)
    source = tmp_path / 'trend_hold_study'
    source.mkdir()
    lock = source / 'scope.py'
    lock.write_bytes(b'REAL_DATA_UNLOCKED = False\n')
    other = source / 'config.py'
    other.write_bytes(b'pin = 123\n')
    doc = {'sha256': {str(path.relative_to(tmp_path)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in (lock, other)}}
    (folder / 'implementation_hashes.json').write_text(json.dumps(doc))
    lock.write_bytes(b'REAL_DATA_UNLOCKED = True\n')
    data.verify_implementation_registration(tmp_path)
    other.write_bytes(b'pin = 124\n')
    with pytest.raises(ValueError, match='implementation hash'):
        data.verify_implementation_registration(tmp_path)


def test_perp_read_is_bounded_and_rejects_content_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(scope, 'REAL_DATA_UNLOCKED', True)
    index = pd.date_range(config.IS_START, periods=2, freq='min').as_unit('us')
    frame = pd.DataFrame({'open': [100., 101.], 'high': [101., 102.],
                          'low': [99., 100.], 'close': [100., 101.],
                          'volume': [1., 1.], 'src': [0, 0]}, index=index)
    from impulse_study.data_1m import content_sha256
    manifest = tmp_path / 'data' / 'manifest_impulse_1m.json'
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({'caches': {'perp_1m': {'content_sha256': content_sha256(frame)}}}))

    def read(path, *, filters):
        assert path == tmp_path / 'data' / 'processed_1m' / 'BTCUSDT_perp_1m_IS.parquet'
        assert filters == [('open_time', '>=', config.IS_START), ('open_time', '<', config.IS_END)]
        return frame

    monkeypatch.setattr(pd, 'read_parquet', read)
    assert data.load_perp_is(tmp_path).index.unit == 'ns'
    frame.iloc[0, 0] = 99.
    with pytest.raises(ValueError, match='content hash'):
        data.load_perp_is(tmp_path)


def test_funding_read_is_bounded_and_requires_exact_grid(tmp_path, monkeypatch):
    monkeypatch.setattr(scope, 'REAL_DATA_UNLOCKED', True)
    expected = pd.date_range(config.IS_START, config.IS_END, freq='8h', inclusive='left').as_unit('ns')
    frame = pd.DataFrame({'funding_rate': .0001}, index=expected + pd.Timedelta(milliseconds=1))

    def read(path, *, filters):
        assert path == tmp_path / 'data' / 'processed' / 'BTCUSDT_funding.parquet'
        assert filters == [('funding_time', '>=', config.IS_START), ('funding_time', '<', config.IS_END)]
        return frame

    monkeypatch.setattr(pd, 'read_parquet', read)
    funding = data.load_funding_is(tmp_path)
    assert len(funding) == 4383 and funding.index.equals(expected)
    frame = frame.drop(frame.index[1])
    with pytest.raises(ValueError, match='4383-settlement'):
        data.load_funding_is(tmp_path)


def test_environment_pin_is_exact():
    actual = data.verify_environment()
    assert actual['pandas'] == '3.0.5'
    with pytest.raises(ValueError, match='environment'):
        data.verify_environment(dict(actual, pandas='3.0.4'))


def test_unlock_accepts_only_one_line_and_fresh_output(tmp_path, monkeypatch):
    monkeypatch.setattr(scope, 'REAL_DATA_UNLOCKED', True)
    head, implementation = '1' * 40, '2' * 40
    lock_file = 'trend_hold_study/scope.py'
    before = b'REAL_DATA_UNLOCKED = False\n'
    after = b'REAL_DATA_UNLOCKED = True\n'
    response = {
        ('status', '--porcelain', '--untracked-files=all'): b'',
        ('rev-list', '--parents', '-n', '1', 'HEAD'): f'{head} {implementation}\n'.encode(),
        ('rev-list', '--parents', '-n', '1', implementation): f'{implementation} {config.BASELINE}\n'.encode(),
        ('diff', '--diff-filter=CDMRTUXB', '--name-only', config.BASELINE, implementation): b'',
        ('diff', '--name-status', implementation, head): f'M\t{lock_file}\n'.encode(),
        ('diff', '--summary', implementation, head): b'',
        ('show', f'{implementation}:{lock_file}'): before,
        ('show', f'{head}:{lock_file}'): after,
    }

    def check_output(argv, *, cwd):
        assert cwd == Path(tmp_path) and argv[0] == 'git'
        return response[tuple(argv[1:])]

    monkeypatch.setattr(data.subprocess, 'check_output', check_output)
    assert data.verify_unlock(tmp_path)['unlock_commit'] == head
    response[('show', f'{head}:{lock_file}')] = after + b'# another change\n'
    with pytest.raises(ValueError, match='only False to True'):
        data.verify_unlock(tmp_path)
    response[('show', f'{head}:{lock_file}')] = after
    output = tmp_path / data.REPORT / 'results'
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError, match='result directory'):
        data.verify_unlock(tmp_path)
