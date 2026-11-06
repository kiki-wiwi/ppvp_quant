"""Synthetic-only authorization, bounded I/O predicates, and byte registration."""
import hashlib
import json
from pathlib import Path
import pandas as pd
import pytest


def test_each_loader_is_locked_before_io(monkeypatch):
    from channel_random_study import data, scope
    monkeypatch.setattr(pd, 'read_parquet', lambda *a, **k: pytest.fail('market read'))
    for name in ('load_real_context', 'load_perp_is', 'load_funding_is'):
        with pytest.raises(scope.RealDataLockedError):
            getattr(data,name)(Path('/must/not/be/read'))


def test_registration_checks_existing_bytes(tmp_path):
    from channel_random_study.data import verify_registered_assets, REPORT
    folder=tmp_path/REPORT;folder.mkdir(parents=True)
    (tmp_path/'old.py').write_text('old\n')
    (folder/'approved_plan.md').write_text('approved\n')
    doc={'protected_sources_sha256': {'old.py':hashlib.sha256(b'old\n').hexdigest()},
         'approved_plan_sha256':hashlib.sha256(b'approved\n').hexdigest()}
    (folder/'registration.json').write_text(json.dumps(doc))
    assert verify_registered_assets(tmp_path)['protected_sources_verified']==1
    (tmp_path/'old.py').write_text('changed\n')
    with pytest.raises(ValueError,match='protected'):verify_registered_assets(tmp_path)


def test_new_manifest_only_normalizes_own_unlock(tmp_path):
    from channel_random_study.data import verify_implementation_registration, REPORT
    folder=tmp_path/REPORT;folder.mkdir(parents=True)
    source=tmp_path/'channel_random_study';source.mkdir()
    p=source/'scope.py';p.write_text('REAL_DATA_UNLOCKED = False\n')
    q=source/'config.py';q.write_text('seed=2026110901\n')
    manifest={'sha256':{str(x.relative_to(tmp_path)):hashlib.sha256(x.read_bytes()).hexdigest() for x in (p,q)}}
    (folder/'implementation_hashes.json').write_text(json.dumps(manifest))
    p.write_text('REAL_DATA_UNLOCKED = True\n')
    verify_implementation_registration(tmp_path)
    q.write_text('seed=1\n')
    with pytest.raises(ValueError,match='implementation'):verify_implementation_registration(tmp_path)


def test_funding_input_predicates_before_load(monkeypatch,tmp_path):
    # Synthetic returned rows; no real-data file or payload is opened.
    from channel_random_study import data,scope,config
    monkeypatch.setattr(scope,'REAL_DATA_UNLOCKED',True)
    dates=pd.date_range(config.IS_START,config.IS_END,freq='8h',inclusive='left').as_unit('ns')
    def read(path,**kwargs):
        assert str(path).endswith('BTCUSDT_funding.parquet')
        assert kwargs['filters']==[('funding_time','>=',config.IS_START),('funding_time','<',config.IS_END)]
        return pd.DataFrame({'funding_rate':.0001},index=dates+pd.Timedelta(milliseconds=1))
    monkeypatch.setattr(pd,'read_parquet',read)
    got=data.load_funding_is(tmp_path)
    assert len(got)==4383 and got.index.equals(dates)


def test_environment_pin_is_read_only_and_exact():
    from channel_random_study.data import environment_versions,verify_environment
    versions=environment_versions()
    verify_environment(versions)
    wrong=dict(versions,pandas='wrong')
    with pytest.raises(ValueError,match='environment'):verify_environment(wrong)
