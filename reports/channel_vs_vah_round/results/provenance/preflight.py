from pathlib import Path
import sys,json,subprocess,platform,traceback,hashlib
from datetime import datetime,timezone
import numpy as np,pandas as pd,pyarrow,numba,scipy,matplotlib,pytest
ROOT=Path('/home/ubuntu/.codex/worktrees/channel-vah-prereg/ppvp_quant')
LOG=Path('/tmp/channel_vah_run_570da83_n93t7yvq')
sys.path.insert(0,str(ROOT))
record={'started_at':datetime.now(timezone.utc).isoformat(),'stage':'pre_unlock_input_verification','strategy_matching_executed':False}
try:
    from breakout_study.data import verify_registered_assets,archive_gap_index
    from breakout_study.config import IS_START,IS_END
    from breakout_study import scope
    from impulse_study.data_1m import content_sha256,minute_grid
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
    assert head=='570da8356464af5ae678cd8a5e5e6e492e1e4f2d',head
    assert status=='',status
    assert scope.REAL_DATA_UNLOCKED is False
    assert not (ROOT/'reports/channel_vs_vah_round/results').exists()
    reg=verify_registered_assets()
    env={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'pyarrow':pyarrow.__version__,'numba':numba.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,'pytest':pytest.__version__}
    pinned=json.loads((ROOT/'reports/channel_vs_vah_round/implementation_hashes.json').read_text())
    assert env==pinned['environment'],(env,pinned['environment'])
    record.update(baseline_commit=head,starting_worktree_status=status,environment=env,registration=reg,
        implementation_hashes_file_sha256=hashlib.sha256((ROOT/'reports/channel_vs_vah_round/implementation_hashes.json').read_bytes()).hexdigest())
    price_path=ROOT/'data/processed_1m/BTCUSDT_perp_1m_IS.parquet'
    filters=[('open_time','>=',IS_START),('open_time','<',IS_END)]
    frame=pd.read_parquet(price_path,filters=filters)
    frame.index=scope.utc_index(frame.index,'perp IS')
    manifest=json.loads((ROOT/'data/manifest_impulse_1m.json').read_text())
    digest=content_sha256(frame)
    expected=manifest['caches']['perp_1m']['content_sha256']
    assert digest==expected,(digest,expected)
    record['perp']={'path':str(price_path),'resolved_path':str(price_path.resolve()),'reader_time_predicate':'2020-01-01 <= open_time < 2024-01-01 UTC','rows':len(frame),'first_open':str(frame.index[0]),'last_open':str(frame.index[-1]),'content_sha256':digest,'manifest_content_sha256':expected}
    funding_path=ROOT/'data/processed/BTCUSDT_funding.parquet'
    funds=pd.read_parquet(funding_path,filters=[('funding_time','>=',IS_START),('funding_time','<',IS_END)])
    funds=funds.copy()
    funds.index=scope.utc_index(pd.DatetimeIndex(funds.index).tz_convert('UTC').round('min'),'funding')
    expected_funding=pd.date_range(IS_START,IS_END,freq='8h',inclusive='left').as_unit('ns')
    assert len(funds)==4383 and funds.index.equals(expected_funding)
    assert np.isfinite(funds['funding_rate'].to_numpy()).all()
    record['funding']={'path':str(funding_path),'reader_time_predicate':'2020-01-01 <= funding_time < 2024-01-01 UTC','settlement_count':len(funds),'first':str(funds.index[0]),'last':str(funds.index[-1]),'rounded_IS_frame_content_sha256':content_sha256(funds)}
    grid=minute_grid(frame,IS_START,IS_END)
    bad=grid['bad'].to_numpy()
    prior=grid['close'].where(~grid['bad']).ffill()
    for col in ('open','high','low','close'): grid.loc[grid['bad'],col]=prior.loc[grid['bad']]
    grid.loc[grid['bad'],'volume']=0.
    gaps=grid.index.isin(archive_gap_index())
    assert len(grid)==2103840 and int(gaps.sum())==10
    assert np.isfinite(grid[['open','high','low','close','volume']].to_numpy()).all()
    assert (grid[['open','high','low','close']].to_numpy()>0).all()
    assert (grid['volume']>=0).all()
    valid=grid.loc[~grid['bad']]
    assert (valid.high>=valid[['open','close']].max(axis=1)).all()
    assert (valid.low<=valid[['open','close']].min(axis=1)).all()
    record['grid']={'minutes':len(grid),'bad_minutes':int(bad.sum()),'archive_gap_minutes':int(gaps.sum()),'causal_bad_price_fill':'previous valid close; no backfill'}
    assert subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)==''
    record.update(status='passed',ended_at=datetime.now(timezone.utc).isoformat(),real_unlock_flag=False,forbidden_datasets_read=[])
    (LOG/'preflight.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'status':'passed','perp_rows':len(frame),'minute_grid':len(grid),'funding':len(funds),'archive_gaps':int(gaps.sum()),'legacy_frozen_files':27,'new_registered_files':23,'perp_content_sha256':digest,'log':str(LOG/'preflight.json')},ensure_ascii=False),flush=True)
except Exception as exc:
    record.update(status='STOPPED',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),ended_at=datetime.now(timezone.utc).isoformat())
    (LOG/'STOPPED_preflight.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
    traceback.print_exc()
    raise SystemExit(1)
