"""One fixed six-account batch; no retries, partial reruns or parameter overrides."""
from datetime import datetime, timezone
from types import SimpleNamespace
import hashlib
import json
from pathlib import Path
import traceback
import numpy as np
import pandas as pd
from . import config, scope


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
    return [(version,cost) for version in config.VERSIONS for cost in config.COSTS]


def _stamp():return datetime.now(timezone.utc).isoformat()


def _clone_context(ctx):
    return scope.Context(ctx.index,*(getattr(ctx,k) for k in ('O','H','L','C','V','bad','archive_gap','funding_rate')),
        source=ctx.source,symbol=ctx.symbol)


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
        from . import features,engine,statistics,reporting
        log('source='+ctx.source+'; constructing causal daily features and common budget')
        daily=features.build_features(ctx)
        daily.to_csv(out/'daily_features.csv',index_label='decision_time')
        feature_hash=hashlib.sha256(daily.to_csv().encode()).hexdigest()
        frozen_at=_stamp()
        write_json(out/'feature_registration.json',dict(shared_features_sha256=feature_hash,
            features_frozen_at=frozen_at,combinations_completed=0,
            index_unit=daily.index.unit,version_order=jobs()))
        accounts={};comparison=[];periods=[];metadata={};first_match=_stamp()
        for version,cost in jobs():
            current=version+'__'+cost;log('executing '+current)
            # Each portfolio receives a new immutable context and its own feature copy.
            own=_clone_context(ctx)
            result=engine.run_portfolio(own,daily.copy(deep=True),version,config.COSTS[cost])
            if hashlib.sha256(daily.to_csv().encode()).hexdigest()!=feature_hash:
                raise ValueError('shared daily features were mutated')
            folder=out/current;folder.mkdir()
            for name in ('orders','decisions','funding'):
                getattr(result,name).to_csv(folder/f'{name}.csv.gz',index=False,compression='gzip')
            result.minute.to_parquet(folder/'minute_equity.parquet')
            pd.DataFrame({'equity':result.daily_equity,'return':result.daily_returns}).to_csv(folder/'daily.csv')
            write_json(folder/'metadata.json',result.metadata)
            comparison.append(reporting.portfolio_summary(result));periods.extend(reporting.period_rows(result))
            accounts[(version,cost)]=SimpleNamespace(daily_equity=result.daily_equity,daily_returns=result.daily_returns)
            metadata[(version,cost)]=result.metadata
            completed.append(current)
            del own,result
        table=pd.DataFrame(comparison);period_table=pd.DataFrame(periods)
        table.to_csv(out/'comparison.csv',index=False);period_table.to_csv(out/'periods.csv',index=False)
        t=accounts[(config.T_VERSION,'main_17')];p=accounts[(config.P_VERSION,'main_17')]
        ht=accounts[(config.T_VERSION,'harsh_34')];hp=accounts[(config.P_VERSION,'harsh_34')]
        log('paired inference: 99999 shared Webb draws and 10000 shared month draws')
        stats=statistics.analyze(t.daily_returns,p.daily_returns,ht.daily_returns,hp.daily_returns,
            daily,metadata[(config.T_VERSION,'main_17')]['order_count'])
        for name in ('tests','intervals','gates','monthly'):stats[name].to_csv(out/f'{name}.csv')
        pd.DataFrame({'T':t.daily_returns,'P':p.daily_returns,'D':t.daily_returns-p.daily_returns}).to_csv(out/'paired_daily_returns.csv')
        matrices=out/'statistical_matrices';matrices.mkdir()
        for name,matrix in stats['matrices'].items():np.save(matrices/f'{name}.npy',matrix,allow_pickle=False)
        write_json(out/'statistical_metadata.json',stats['metadata'])
        reporting.figures(accounts,daily,stats,table,period_table,out)
        summary=dict(status='completed',source=ctx.source,started_at=started,ended_at=_stamp(),
            features_frozen_at=frozen_at,first_matching_at=first_match,combination_count=len(completed),
            completed_combinations=completed,input_metadata=input_metadata or {},
            classification=stats['classification'],statistics_metadata=stats['metadata'],
            real_data_authorized=scope.REAL_DATA_UNLOCKED,
            limitations=['same ex-ante budget rules, not equal realized risk or exposure',
                'IS exploratory research only; no authorization of VAL/OOS/ETH or deployment'])
        write_json(out/'summary.json',summary)
        log('completed six accounts; STOP and await review')
        hashes={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.rglob('*')) if p.is_file()}
        write_json(out/'output_hashes.json',{'sha256':hashes})
        return summary
    except Exception as exc:
        partial=getattr(exc,'partial',None)
        if partial is not None:
            for name,value in partial.items():
                if isinstance(value,(pd.DataFrame,pd.Series)):
                    value.to_csv(out/f'STOPPED_partial_{name}.csv.gz',compression='gzip')
            write_json(out/'STOPPED_partial_state.json',{k:v for k,v in partial.items() if not isinstance(v,(pd.DataFrame,pd.Series))})
        write_json(out/'STOPPED.json',dict(source=ctx.source,started_at=started,error_type=type(exc).__name__,
            error=str(exc),traceback=traceback.format_exc(),completed_combinations=completed,
            current_combination=current,automatic_retry=False))
        log('STOPPED: '+str(exc));raise
