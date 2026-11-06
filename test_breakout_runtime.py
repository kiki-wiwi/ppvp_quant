"""End-to-end orchestration uses only generated prices and temporary outputs."""
import json
from pathlib import Path
import subprocess
import sys
import pandas as pd
import pytest


def test_runner_defaults_to_verification_and_never_unlocks(tmp_path):
    script=Path(__file__).parent/'tools/run_breakout_study.py'
    assert script.is_file()
    result=subprocess.run([sys.executable,str(script),'--check'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    report=json.loads(result.stdout)
    assert report['real_data_unlocked'] is False
    assert report['legacy_frozen_files_verified']==27
    denied=subprocess.run([sys.executable,str(script),'--real'],capture_output=True,text=True)
    assert denied.returncode!=0 and 'locked' in denied.stderr.lower()
    assert not (Path(__file__).parent/'reports/channel_vs_vah_round/results').exists()


def test_full_synthetic_six_combination_pipeline(tmp_path):
    from breakout_study.synthetic import study_context
    from breakout_study.runtime import execute
    ctx=study_context()
    output=tmp_path/'synthetic_results'
    result=execute(ctx,output,{'source':'synthetic_test'})
    assert result['source']=='synthetic'
    assert result['combination_count']==6
    assert result['statistics']['family_size']==3
    assert result['statistics']['calendar']['days']==1401
    assert len(pd.read_csv(output/'comparison.csv'))==6
    assert len(list(output.glob('*_trades.csv.gz')))==6
    assert len(list(output.glob('figure_*.png')))==3
    assert not (output/'STOPPED.json').exists()
    assert all(v>0 for v in result['trade_counts'].values())
    for v in ('P_channel','V_vah'):
        d=pd.read_csv(output/f'{v}__main_17_decisions.csv.gz')
        assert len(d)==result['signal_count_by_version'][v]
        assert (d.decision=='entered').sum()==result['trade_counts'][v+'__main_17']
    with pytest.raises(FileExistsError): execute(ctx,output)


def test_unexpected_failure_preserves_stop_log_and_never_retries(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from breakout_study import runtime
    from breakout_study.config import IS_START,IS_END
    ctx=SimpleNamespace(source='synthetic',symbol='BTCUSDT',
        index=pd.date_range(IS_START,IS_END,freq='min',inclusive='left').as_unit('ns'))
    calls=[]
    def fail(context):
        calls.append(context)
        raise ValueError('synthetic injected failure')
    monkeypatch.setattr(runtime,'build_features',fail)
    output=tmp_path/'stop_fixture'
    with pytest.raises(ValueError,match='synthetic injected failure'):
        runtime.execute(ctx,output)
    stopped=json.loads((output/'STOPPED.json').read_text())
    assert len(calls)==1 and stopped['automatic_retry'] is False
    assert stopped['completed_combinations']==[]
    assert (output/'run.log').exists() and not (output/'summary.json').exists()


def test_even_gross_only_bankruptcy_permanently_disqualifies_version():
    from types import SimpleNamespace
    from breakout_study.runtime import execution_failures
    from breakout_study.config import VERSIONS,COSTS
    results={(v,c):SimpleNamespace(metadata={'failed':v=='P_channel' and c=='gross_0'})
             for v in VERSIONS for c in COSTS}
    assert execution_failures(results)=={'P_channel':True,'V_vah':False}


def test_authorized_input_verification_error_is_archived_without_reading_market(tmp_path,monkeypatch):
    from breakout_study import scope,data
    from tools import run_breakout_study as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    # Only the check is mocked; the module's real-data unlock remains False.
    monkeypatch.setattr(scope,'require_real_authorization',lambda:None)
    def fail(): raise ValueError('synthetic frozen hash mismatch')
    monkeypatch.setattr(data,'verify_registered_assets',fail)
    def no_market(): raise AssertionError('market input called after failed verification')
    monkeypatch.setattr(data,'load_real_context',no_market)
    with pytest.raises(ValueError,match='synthetic frozen hash mismatch'): runner.main(['--real'])
    report=json.loads((tmp_path/'reports/channel_vs_vah_round/results/STOPPED.json').read_text())
    assert report['stage']=='input_verification' and report['automatic_retry'] is False
    assert scope.REAL_DATA_UNLOCKED is False
