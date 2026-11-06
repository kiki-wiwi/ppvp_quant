"""One pass through the six registered combinations, with durable stop logs."""
from pathlib import Path
import json
import traceback
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from .config import IS_START, IS_END, VERSIONS, COSTS
from .scope import require_synthetic
from .features import build_features, build_signals
from .engine import run_portfolio
from .statistics import analyze
from .reporting import comparison_rows, annotate_comparison, period_rows, event_exposure, figures


def _safe(value):
    if isinstance(value,dict): return {str(k):_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_safe(v) for v in value]
    if isinstance(value,(pd.Timestamp,datetime,Path)): return str(value)
    if isinstance(value,(np.integer,np.bool_)): return value.item()
    if isinstance(value,(float,np.floating)): return float(value) if np.isfinite(value) else None
    return value


def write_json(path,value):
    Path(path).write_text(json.dumps(_safe(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def execution_failures(results):
    return {v:any(results[(v,c)].metadata['failed'] for c in COSTS) for v in VERSIONS}


def execute(ctx,output,input_metadata=None):
    require_synthetic(ctx)
    expected=pd.date_range(IS_START,IS_END,freq='min',inclusive='left').as_unit('ns')
    if not ctx.index.equals(expected):
        raise ValueError('study execution requires the complete IS minute grid; no shortened official runs')
    out=Path(output)
    out.mkdir(parents=True,exist_ok=False)
    started=datetime.now(timezone.utc).isoformat()
    def log(message):
        with (out/'run.log').open('a') as handle:
            handle.write(datetime.now(timezone.utc).isoformat()+' '+message+'\n')
        print(message,flush=True)
    results={}
    try:
        log('source='+ctx.source+'; building fixed closed-bar features')
        built=build_features(ctx)
        built.hours.to_csv(out/'hourly_features.csv.gz',compression='gzip')
        built.references.to_csv(out/'reference_snapshots.csv.gz',compression='gzip')
        signal_counts={}
        for version in VERSIONS:
            signals=build_signals(ctx,version,built)
            signal_counts[version]=len(signals)
            signals.to_csv(out/f'{version}_signals.csv.gz',index=False,compression='gzip')
            for cost in COSTS:
                key=version+'__'+cost
                log('executing '+key)
                result=run_portfolio(ctx,signals,version,cost)
                results[(version,cost)]=result
                for suffix,frame in [('trades',result.trades),('decisions',result.decisions),('stop_updates',result.stop_updates)]:
                    frame.to_csv(out/f'{key}_{suffix}.csv.gz',index=False,compression='gzip')
                result.equity.to_frame().to_parquet(out/f'{key}_minute_equity.parquet')
                pd.DataFrame({'equity':result.daily_equity,'return':result.daily_returns}).to_csv(out/f'{key}_daily.csv')
                write_json(out/f'{key}_metadata.json',result.metadata)
        def returns(cost): return pd.DataFrame({v:results[(v,cost)].daily_returns for v in VERSIONS})
        def trades(cost): return {v:results[(v,cost)].trades for v in VERSIONS}
        failed=execution_failures(results)
        log('computing registered Webb/Holm and shared-month intervals')
        stats=analyze(returns('main_17'),trades('main_17'),returns('harsh_34'),trades('harsh_34'),failed_versions=failed)
        table=annotate_comparison(comparison_rows(results),stats);periods=period_rows(results)
        table.to_csv(out/'comparison.csv',index=False);periods.to_csv(out/'periods.csv',index=False)
        event_exposure(results).to_csv(out/'event_hourly_exposure.csv',index=False)
        figures(results,table,periods,out)
        result={'status':'completed','source':ctx.source,'started_at':started,
            'ended_at':datetime.now(timezone.utc).isoformat(),'combination_count':6,
            'input_metadata':input_metadata or {},'statistics':stats,
            'signal_count_by_version':signal_counts,
            'trade_counts':{v+'__'+c:len(r.trades) for (v,c),r in results.items()}}
        write_json(out/'summary.json',result)
        log('completed all six combinations; stop before independent validation')
        return result
    except Exception as exc:
        write_json(out/'STOPPED.json',{'source':ctx.source,'started_at':started,
            'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),
            'completed_combinations':[v+'__'+c for v,c in results], 'automatic_retry':False})
        log('STOPPED: '+str(exc))
        raise
