"""One immutable eight-combination batch, preserving first-error evidence."""
from pathlib import Path
from datetime import datetime,timezone
import traceback
import pandas as pd
from .config import IS_START,IS_END,VERSIONS,COSTS
from .scope import require_synthetic
from breakout_study.runtime import write_json

def execution_failures(results):
    return {v:any(r.metadata['failed'] for (version,_,_),r in results.items() if version==v) for v in VERSIONS}

def execute(ctx,output,input_metadata=None):
    require_synthetic(ctx)
    expected=pd.date_range(IS_START,IS_END,freq='min',inclusive='left').as_unit('ns')
    if not ctx.index.equals(expected):
        raise ValueError('study execution requires the complete IS minute grid')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    started=datetime.now(timezone.utc).isoformat();results={}
    def log(message):
        with (out/'run.log').open('a') as f: f.write(datetime.now(timezone.utc).isoformat()+' '+message+'\n')
        print(message,flush=True)
    try:
        from .features import build_features,build_signals
        from .engine import run_portfolio,simulate_events
        from .statistics import analyze
        from .reporting import comparison_rows,period_rows,paired_diagnostics,figures
        log('source='+ctx.source+'; building closed-bar trend/dip features')
        built=build_features(ctx);sig=build_signals(ctx,built)
        built.hours.to_csv(out/'hourly_features.csv.gz',compression='gzip')
        built.references.to_csv(out/'trend_references.csv.gz',compression='gzip')
        sig.to_csv(out/'father_signals.csv.gz',index=False,compression='gzip')
        jobs=[(v,c,False) for v in VERSIONS for c in COSTS]+[('L_limit',c,True) for c in ('main','harsh')]
        for v,c,stress in jobs:
            key=v+'__'+c+('__stress' if stress else '')
            log('executing '+key)
            r=run_portfolio(ctx,sig,v,c,stress=stress);results[(v,c,stress)]=r
            for name,df in [('trades',r.trades),('decisions',r.decisions),('orders',r.orders)]:
                df.to_csv(out/f'{key}_{name}.csv.gz',index=False,compression='gzip')
            r.equity.to_frame().to_parquet(out/f'{key}_minute_equity.parquet')
            pd.DataFrame({'equity':r.daily_equity,'return':r.daily_returns}).to_csv(out/f'{key}_daily.csv')
            write_json(out/f'{key}_metadata.json',r.metadata)
        log('independent paired father events; unfilled limits retained as zero')
        events=simulate_events(ctx,sig)
        events.to_csv(out/'paired_events.csv.gz',index=False,compression='gzip')
        diag=paired_diagnostics(events);write_json(out/'execution_attribution.json',diag)
        returns=lambda c:pd.DataFrame({v:results[(v,c,False)].daily_returns for v in VERSIONS})
        trades=lambda c:{v:results[(v,c,False)].trades for v in VERSIONS}
        sr=pd.DataFrame({c:results[('L_limit',c,True)].daily_returns for c in ('main','harsh')})
        st={c:results[('L_limit',c,True)].trades for c in ('main','harsh')}
        log('fixed Webb/Holm inference and shared-month intervals')
        stats=analyze(returns('main'),trades('main'),returns('harsh'),trades('harsh'),sr,st,events,failed_versions=execution_failures(results))
        table=comparison_rows(results,stats);table.to_csv(out/'comparison.csv',index=False)
        period_rows(results).to_csv(out/'periods.csv',index=False)
        figures(results,table,events,out)
        result=dict(status='completed',source=ctx.source,started_at=started,
            ended_at=datetime.now(timezone.utc).isoformat(),combination_count=8,
            input_metadata=input_metadata or {},statistics=stats,execution_attribution=diag,
            decision_hour_count=len(sig),eligible_father_count=int(sig.eligible.sum()),
            trade_counts={v+'__'+c+('__stress' if s else ''):len(r.trades) for (v,c,s),r in results.items()})
        write_json(out/'summary.json',result);log('completed fixed batch; stop before further validation')
        return result
    except Exception as exc:
        write_json(out/'STOPPED.json',dict(source=ctx.source,started_at=started,error_type=type(exc).__name__,
            error=str(exc),traceback=traceback.format_exc(),completed_combinations=[list(k) for k in results],automatic_retry=False))
        log('STOPPED: '+str(exc));raise
