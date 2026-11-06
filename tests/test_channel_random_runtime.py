"""Synthetic control flow: no unlock, overwrite, retries, or hidden jobs."""
from pathlib import Path
import pandas as pd
import pytest
from channel_random_study import config,scope


def frame():
    x=pd.DataFrame(dict(open=100.,high=101.,low=99.,close=100.,volume=1.),
        index=pd.date_range('2021-01-01',periods=120,freq='min',tz='UTC'))
    x.attrs['provenance']='synthetic'
    return x


def test_real_cli_rejects_before_output_or_importing_loaders(monkeypatch):
    from tools.run_channel_random_study import main
    with pytest.raises(scope.RealDataLockedError):main(['--real'])


def test_jobs_have_exact_order_and_single_random_cost():
    from channel_random_study.runtime import jobs
    got=jobs()
    assert len(got)==203
    assert got[:3]==[(config.S_VERSION,c) for c in config.COSTS]
    assert got[3:]==[(v,'main_17') for v in config.R_VERSIONS]


def test_short_official_grid_rejected_before_output(tmp_path):
    from channel_random_study.runtime import execute
    with pytest.raises(ValueError,match='complete IS'):
        execute(scope.make_context(frame()),tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_context_snapshot_and_provenance():
    source=frame();ctx=scope.make_context(source)
    source.iloc[0,0]=99
    assert ctx.O[0]==100
    with pytest.raises(ValueError):ctx.O.setflags(write=True)
    source.attrs['source']='real'
    with pytest.raises(scope.ScopeError):scope.make_context(source)
    with pytest.raises(scope.ScopeError):scope.make_context(frame(),symbol='ETHUSDT')



def test_first_error_preserves_logs_and_does_not_retry_or_match(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from channel_random_study import runtime,features,engine
    ctx=SimpleNamespace(source='synthetic',symbol='BTCUSDT',
        index=pd.date_range(config.IS_START,config.IS_END,freq='min',inclusive='left').as_unit('ns'))
    calls=[]
    def fail(ctx):
        calls.append('feature')
        raise ValueError('deliberate synthetic feature failure')
    monkeypatch.setattr(features,'build_features',fail)
    monkeypatch.setattr(engine,'run_portfolio',lambda *a,**k:pytest.fail('matched after failure'))
    out=tmp_path/'out'
    with pytest.raises(ValueError,match='deliberate synthetic'):
        runtime.execute(ctx,out)
    import json
    stopped=json.loads((out/'STOPPED.json').read_text())
    assert stopped['completed_combinations']==[] and not stopped['automatic_retry']
    assert calls==['feature'] and 'STOPPED' in (out/'run.log').read_text()
    original=(out/'STOPPED.json').read_bytes()
    with pytest.raises(FileExistsError):runtime.execute(ctx,out)
    assert (out/'STOPPED.json').read_bytes()==original and calls==['feature']
