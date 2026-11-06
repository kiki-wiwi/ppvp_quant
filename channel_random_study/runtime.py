"""Freeze all random opportunities first, then execute exactly 203 portfolios once."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import traceback
import numpy as np
import pandas as pd
from . import config,scope


def _safe(value):
    if isinstance(value,dict):return {str(k):_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_safe(v) for v in value]
    if isinstance(value,(pd.Timestamp,datetime,Path)):return str(value)
    if isinstance(value,(np.integer,np.bool_)):return value.item()
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    if isinstance(value,np.ndarray):return _safe(value.tolist())
    return value


def write_json(path,value):
    Path(path).write_text(json.dumps(_safe(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def jobs():
    return [(config.S_VERSION,cost) for cost in config.COSTS]+[(version,'main_17') for version in config.R_VERSIONS]


def _stamp():return datetime.now(timezone.utc).isoformat()


def execute(ctx,output,input_metadata=None):
    scope.require_synthetic(ctx)
    expected=pd.date_range(config.IS_START,config.IS_END,freq='min',inclusive='left').as_unit('ns')
    if not ctx.index.equals(expected):
        raise ValueError('study execution requires the complete IS minute grid')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    started=_stamp();completed=[];current=None
    def log(message):
        with (out/'run.log').open('a') as f:f.write(_stamp()+' '+message+'\n')
        print(message,flush=True)
    try:
        from .features import build_features,build_opportunities
        from .matching import build_schedules
        from .engine import run_portfolio
        from .statistics import analyze
        from .reporting import portfolio_summary,period_rows,execution_balance,figures
        log('source='+ctx.source+'; constructing causal candidates before any matching')
        built=build_features(ctx);opportunities=build_opportunities(ctx,built)
        built.hours.to_csv(out/'hourly_features.csv.gz',compression='gzip')
        built.references.to_csv(out/'reference_snapshots.csv.gz',compression='gzip')
        opportunities.to_csv(out/'all_opportunities.csv.gz',index=False,compression='gzip')
        batch=build_schedules(opportunities)
        batch.candidates.to_csv(out/'candidate_pool.csv.gz',index=False,compression='gzip')
        batch.parents.to_csv(out/'true_parents.csv.gz',index=False,compression='gzip')
        batch.strata.to_csv(out/'strata.csv',index=False)
        schedule_dir=out/'schedules';schedule_dir.mkdir()
        for version,schedule in zip(config.R_VERSIONS,batch.schedules,strict=True):
            schedule.to_csv(schedule_dir/f'{version}.csv.gz',index=False,compression='gzip')
        matched_at=_stamp()
        write_json(out/'schedule_registration.json',dict(metadata=batch.metadata,counters=batch.counters,
            all_schedules_frozen_at=matched_at,performance_combinations_completed=0))
        log('all 200 schedules frozen; starting fixed 203-combination order')
        first_match_at=_stamp();s_results={};random_daily={};random_equity={}
        comparison=[];periods=[];balance=[];filled_layers=[]
        for version,cost in jobs():
            current=version+'__'+cost;log('executing '+current)
            signals=batch.parents if version==config.S_VERSION else batch.schedules[int(version[2:])]
            result=run_portfolio(ctx,signals,version,cost)
            folder=out/current;folder.mkdir()
            for name,frame in [('trades',result.trades),('decisions',result.decisions),('stop_updates',result.stop_updates)]:
                frame.to_csv(folder/f'{name}.csv.gz',index=False,compression='gzip')
            result.decisions.loc[result.decisions.decision=='entered'].to_csv(folder/'orders.csv.gz',index=False,compression='gzip')
            pd.DataFrame({'equity':result.daily_equity,'return':result.daily_returns}).to_csv(folder/'daily.csv')
            write_json(folder/'metadata.json',result.metadata)
            comparison.append(portfolio_summary(result));periods.extend(period_rows(result));balance.extend(execution_balance(result))
            entered=result.decisions.loc[result.decisions.decision=='entered']
            if len(entered):
                groups=entered.groupby(['month','utc_hour','vol_bin'],sort=True).size().rename('filled_count').reset_index()
                groups['version']=version;groups['cost']=cost;filled_layers.append(groups)
            if version==config.S_VERSION:
                result.equity.to_frame().to_parquet(folder/'minute_equity.parquet')
                s_results[cost]=result
            else:
                random_daily[version]=result.daily_returns
                random_equity[version]=result.daily_equity
            completed.append(current)
        random_returns=pd.DataFrame(random_daily,columns=config.R_VERSIONS)
        random_equities=pd.DataFrame(random_equity,columns=config.R_VERSIONS)
        random_returns.to_csv(out/'random_daily_returns.csv.gz',compression='gzip')
        random_equities.to_csv(out/'random_daily_equity.csv.gz',compression='gzip')
        table=pd.DataFrame(comparison);period_table=pd.DataFrame(periods)
        table.to_csv(out/'comparison.csv',index=False);period_table.to_csv(out/'periods.csv',index=False)
        pd.DataFrame(balance).to_csv(out/'execution_balance.csv',index=False)
        (pd.concat(filled_layers,ignore_index=True) if filled_layers else pd.DataFrame(
            columns=['month','utc_hour','vol_bin','filled_count','version','cost'])).to_csv(out/'filled_strata.csv',index=False)
        random_counts=table.loc[table.version.str.startswith('R_')].set_index('version')[['trade_count','exposure_fraction']]
        log('paired daily inference: fixed 99999 Webb weights and 10000 month samples')
        stats=analyze(s_results['main_17'].daily_returns,random_returns,s_results['main_17'].trades,
            s_results['harsh_34'].daily_returns,s_results['harsh_34'].trades,batch.strata,random_counts)
        for name in ('tests','intervals','gates','daily','monthly','random_summary','random_distribution'):
            stats[name].to_csv(out/f'{name}.csv')
        matrices=out/'statistical_matrices';matrices.mkdir()
        for name,matrix in stats['matrices'].items():np.save(matrices/f'{name}.npy',matrix,allow_pickle=False)
        write_json(out/'statistical_metadata.json',stats['metadata'])
        figures(s_results,random_equities,stats,table,period_table,out)
        summary=dict(status='completed',source=ctx.source,started_at=started,ended_at=_stamp(),
            all_schedules_frozen_at=matched_at,first_matching_at=first_match_at,
            combination_count=len(completed),completed_combinations=completed,
            input_metadata=input_metadata or {},verdict=stats['verdict'],
            schedule_metadata=batch.metadata,schedule_counters=batch.counters,
            statistics_metadata=stats['metadata'],
            trade_counts={row['version']+'__'+row['cost']:row['trade_count'] for row in comparison},
            real_data_authorized=scope.REAL_DATA_UNLOCKED,
            limitations=['offline conditional schedules, not online random policy',
                'fixed 200 paths; random dispersion is not a market confidence interval',
                'IS-only exploratory study; no next-stage authorization'])
        write_json(out/'summary.json',summary)
        log('completed fixed batch; STOP before any new study or validation')
        hashes={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(out.rglob('*')) if p.is_file()}
        write_json(out/'output_hashes.json',{'sha256':hashes})
        return summary
    except Exception as exc:
        partial=getattr(exc,'partial',None)
        if partial is None and hasattr(exc,'failure_time'):
            partial={name:getattr(exc,name) for name in ('failure_time','equity','trades','decisions','position')}
        if partial is not None:
            for name,value in partial.items():
                if isinstance(value,(pd.DataFrame,pd.Series)):
                    value.to_csv(out/f'STOPPED_partial_{name}.csv.gz',compression='gzip')
            write_json(out/'STOPPED_partial_state.json',{k:v for k,v in partial.items()
                if not isinstance(v,(pd.DataFrame,pd.Series))})
        write_json(out/'STOPPED.json',dict(source=ctx.source,started_at=started,error_type=type(exc).__name__,
            error=str(exc),traceback=traceback.format_exc(),completed_combinations=completed,
            current_combination=current,automatic_retry=False))
        log('STOPPED: '+str(exc));raise
