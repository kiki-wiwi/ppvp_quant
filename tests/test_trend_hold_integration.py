"""Full registered calendar and six combinations on invented prices only."""
import hashlib
import json
import numpy as np
import pandas as pd
from trend_hold_study import config,scope


def test_complete_synthetic_fixed_batch(tmp_path):
    from trend_hold_study.synthetic import study_context
    from trend_hold_study.runtime import execute,jobs
    locked_before=scope.REAL_DATA_UNLOCKED
    out=tmp_path/'synthetic_outputs'
    summary=execute(study_context(),out,{'source':'synthetic','market_reads':0})
    assert summary['status']=='completed' and summary['combination_count']==6
    assert summary['completed_combinations']==[v+'__'+c for v,c in jobs()]
    assert summary['features_frozen_at']<=summary['first_matching_at']
    assert not (out/'STOPPED.json').exists()
    tests=pd.read_csv(out/'tests.csv',index_col=0)
    assert list(tests.index)==['T','D'] and tests.bootstrap_replicates.eq(99999).all()
    assert len(pd.read_csv(out/'gates.csv'))==6
    intervals=pd.read_csv(out/'intervals.csv',index_col=0)
    assert intervals.resamples.eq(10000).all()
    for version,cost in jobs():
        folder=out/(version+'__'+cost)
        daily=pd.read_csv(folder/'daily.csv',index_col=0)
        assert len(daily)==1401 and np.isfinite(daily.to_numpy()).all()
        assert (folder/'minute_equity.parquet').exists()
        orders=pd.read_csv(folder/'orders.csv.gz')
        assert len(orders)>0 and orders.delta_qty.ne(0).all()
        assert (pd.to_datetime(orders.time,utc=True,format='mixed')<config.IS_END).all()
        meta=json.loads((folder/'metadata.json').read_text())
        assert abs(meta['account_reconciliation_error'])<1e-6
        assert meta['funding_count']==4203
    comparison=pd.read_csv(out/'comparison.csv')
    periods=pd.read_csv(out/'periods.csv')
    required={'signal_switch_count','position_entry_count','completed_holding_intervals','held_calendar_days','adjustment_count'}
    assert required.issubset(comparison.columns) and required.issubset(periods.columns)
    assert comparison['completed_holding_intervals'].ge(0).all()
    assert (periods.order_count==periods.adjustment_count+periods.terminal_order_count).all()
    stats=json.loads((out/'statistical_metadata.json').read_text())
    for name,meta in stats['matrices'].items():
        matrix=np.load(out/'statistical_matrices'/f'{name}.npy',allow_pickle=False)
        assert list(matrix.shape)==meta['shape']
        assert hashlib.sha256(matrix.tobytes(order='C')).hexdigest()==meta['sha256']
    hashes=json.loads((out/'output_hashes.json').read_text())['sha256']
    for name in ('figure_1_equity_and_budget.png','figure_2_annual.png','figure_3_cost_and_months.png'):
        assert (out/name).stat().st_size>10000
        assert hashlib.sha256((out/name).read_bytes()).hexdigest()==hashes[name]
    assert scope.REAL_DATA_UNLOCKED==locked_before
