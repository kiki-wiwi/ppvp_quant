"""Full CP3 orchestration on synthetic IS-shaped data; no market input exists.

The formal entry guards/data IO are replaced by synthetic fixtures in this test
only. All 80 portfolios, matched draws, statistics and chart paths run unchanged.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from impulse_study import synthetic as Y, features as F, engine as E, versions as VS
from impulse_study import cp3_runtime as R
from tools import run_impulse_cp3 as run


def test_complete_cp3_pipeline_with_synthetic_is_grid(tmp_path, monkeypatch):
    unlocked_before = E.REAL_DATA_UNLOCKED
    idx = pd.date_range('2020-01-01','2024-01-01',inclusive='left',freq='min',tz='UTC').as_unit('ns')
    n = len(idx)
    C = 10000. + 20 * np.sin(np.arange(n) / 5000.)
    for month in pd.date_range('2020-01-01','2023-12-01',freq='MS',tz='UTC'):
        for day, amplitude in [(5,140.),(12,190.),(19,230.),(26,160.)]:
            i = idx.get_indexer([month + pd.Timedelta(days=day,hours=12)])[0]
            C[i:i+50] += amplitude
    O = np.r_[C[0], C[:-1]]
    H, L = np.maximum(O,C)+1, np.minimum(O,C)-1
    V = np.full(n,10.)
    grid = Y.grid_from_arrays(idx[0],O,H,L,C,V)
    # Deliberately alternate box availability so box comparisons exercise both strata.
    box = ((idx.day.to_numpy() // 7) % 2 == 0)
    funding = pd.Series(.0001,index=pd.date_range('2020-01-01','2024-01-01',freq='8h',inclusive='left',tz='UTC'))
    ctx = F.make_context(grid,box,'synthetic',archive_gap_minutes=VS.ARCHIVE_GAP_MINUTES,funding=funding)
    meta = {'baseline':R.BASELINE,'implementation_commit':'synthetic-only-implementation',
            'unlock_commit':'synthetic-only-unlock','dirty_files':[], 'frozen_key':VS.frozen_key(),
            'frozen_sha256_normalized':R.frozen_hashes(),
            'registration':{'new_file_sha256':{},'existing_dependency_sha256':{}}}
    monkeypatch.setattr(R,'verify_unlock',lambda:meta.copy())
    monkeypatch.setattr(R,'load_real_context',lambda:(ctx,{'funding_count':4383,'archive_gap_count':10,
        'box_ok_sha256':hashlib.sha256(np.ascontiguousarray(ctx.box_ok).tobytes()).hexdigest(), 'minute_count':n}))
    monkeypatch.setattr(run,'unlocked_tests',lambda:{'synthetic_integration':True})
    monkeypatch.setattr(run,'OUT',tmp_path/'synthetic_cp3')
    # Phase log creation also runs, under an isolated test destination.
    monkeypatch.setattr(run,'ROOT',tmp_path)
    (tmp_path/'docs/phase_logs').mkdir(parents=True)
    result=run.main()
    assert result['provenance']['scope_counts']=={'main_portfolios':80,'matched_version_costs':40,
        'B_appendix':5,'G8z':2,'CP4_other_sensitivities':0,'VAL_OOS_ETH':0}
    assert set(result['versions'])==set(VS.VERSIONS)
    assert result['families']['F1']['size']==10
    assert result['families']['F2']['size']==10
    out=tmp_path/'synthetic_cp3'
    assert len(list(out.glob('*__*__*_trades.csv.gz')))==80
    assert len(list(out.glob('figure*.png')))==4
    assert len(list(out.glob('onset_*.csv.gz')))==10
    assert len(pd.read_csv(out/'all80_comparison.csv'))==80
    assert not (out/'STOPPED.json').exists()
    assert E.REAL_DATA_UNLOCKED is unlocked_before
    assert (tmp_path/'docs/phase_logs/IMPULSE_CP3_results.md').is_file()
    assert json.loads((out/'summary.json').read_text())['status'].startswith('CP3 completed')
