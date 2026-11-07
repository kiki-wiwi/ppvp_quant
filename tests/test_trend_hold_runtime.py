"""Synthetic control-flow tests; no market loads, retries or output reuse."""
from types import SimpleNamespace
import json
import pandas as pd
import pytest
from trend_hold_study import config, scope


def test_exact_six_jobs_and_order():
    from trend_hold_study.runtime import jobs
    assert jobs() == [(v,c) for v in config.VERSIONS for c in config.COSTS]
    assert len(jobs()) == 6


def test_real_cli_lock_precedes_any_loader(monkeypatch,tmp_path):
    from tools.run_trend_hold_study import main
    monkeypatch.setattr(scope,'REAL_DATA_UNLOCKED',False)
    monkeypatch.setattr(pd,'read_parquet',lambda *a,**k: pytest.fail('market read'))
    with pytest.raises(scope.RealDataLockedError):main(['--real'])


def test_incomplete_grid_rejected_before_directory(tmp_path):
    from trend_hold_study.runtime import execute
    ctx=SimpleNamespace(source='synthetic',symbol='BTCUSDT',
        index=pd.date_range('2020-03-01',periods=2,freq='min',tz='UTC').as_unit('ns'))
    with pytest.raises(ValueError,match='complete IS'):execute(ctx,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_first_failure_preserves_and_never_retries(monkeypatch,tmp_path):
    from trend_hold_study import runtime,features
    ctx=SimpleNamespace(source='synthetic',symbol='BTCUSDT',
        index=pd.date_range(config.IS_START,config.IS_END,freq='min',inclusive='left').as_unit('ns'))
    calls=[]
    def fail(context):
        calls.append('features')
        raise ValueError('deliberate synthetic failure')
    monkeypatch.setattr(features,'build_features',fail)
    out=tmp_path/'out'
    with pytest.raises(ValueError,match='deliberate'):runtime.execute(ctx,out)
    stop=json.loads((out/'STOPPED.json').read_text())
    assert stop['completed_combinations']==[] and not stop['automatic_retry']
    assert calls==['features'] and 'STOPPED' in (out/'run.log').read_text()
    before=(out/'STOPPED.json').read_bytes()
    with pytest.raises(FileExistsError):runtime.execute(ctx,out)
    assert before==(out/'STOPPED.json').read_bytes() and calls==['features']
