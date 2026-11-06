"""Single full-calendar synthetic integration of all 203 fixed portfolios."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def test_complete_synthetic_fixed_batch(tmp_path):
    from channel_random_study.synthetic import study_context
    from channel_random_study.runtime import execute,jobs
    from channel_random_study import config,scope
    out=tmp_path/'synthetic_outputs'
    assert scope.REAL_DATA_UNLOCKED is False
    summary=execute(study_context(),out,{'source':'synthetic','market_reads':0})
    assert summary['status']=='completed' and summary['combination_count']==203
    assert summary['completed_combinations']==[v+'__'+c for v,c in jobs()]
    assert summary['all_schedules_frozen_at']<=summary['first_matching_at']
    assert summary['schedule_counters']['parent_count']>0
    assert not (out/'STOPPED.json').exists()
    tests=pd.read_csv(out/'tests.csv',index_col=0)
    assert list(tests.index)==['S','D'] and tests.bootstrap_replicates.eq(99999).all()
    assert len(pd.read_csv(out/'gates.csv'))==7
    for version,cost in jobs():
        folder=out/(version+'__'+cost)
        daily=pd.read_csv(folder/'daily.csv',index_col=0)
        assert len(daily)==1401 and np.isfinite(daily.to_numpy()).all()
        assert (folder/'minute_equity.parquet').exists()==(version==config.S_VERSION)
    hashes=json.loads((out/'output_hashes.json').read_text())['sha256']
    for name in ('figure_1_equity.png','figure_2_annual.png','figure_3_cost_and_months.png'):
        assert (out/name).stat().st_size>10000
        assert hashlib.sha256((out/name).read_bytes()).hexdigest()==hashes[name]
    for version,rec in zip(config.R_VERSIONS,summary['schedule_metadata']['replicates'],strict=True):
        df=pd.read_csv(out/'schedules'/f'{version}.csv.gz')
        ns=pd.DatetimeIndex(pd.to_datetime(df.entry_time,utc=True)).as_unit('ns').asi8.astype('<i8')
        assert hashlib.sha256(ns.tobytes()).hexdigest()==rec['time_sha256']
    assert scope.REAL_DATA_UNLOCKED is False
