"""Registered tables and figures from existing ledgers; no simulations or I/O inputs."""
import numpy as np
import pandas as pd
from .config import VERSIONS,COSTS,INITIAL_CAPITAL,HALF
from breakout_study.reporting import drawdown

def paired_diagnostics(events):
    n=len(events)
    if not n:
        return dict(father_count=0,fill_rate=None,mean_difference_bps=None,
            filled_contribution_bps=None,unfilled_contribution_bps=None,
            statuses={},filled_M_mean_bps=None,unfilled_M_mean_bps=None,mean_wait_minutes=None)
    filled=events.L_filled.astype(bool)
    return dict(father_count=n,fill_rate=float(filled.mean()),
        mean_difference_bps=float(events.difference_bps.mean()),
        filled_contribution_bps=float((events.loc[filled,'L_net_bps']-events.loc[filled,'M_net_bps']).sum()/n),
        unfilled_contribution_bps=float(-events.loc[~filled,'M_net_bps'].sum()/n),
        statuses={str(k):int(v) for k,v in events.L_status.value_counts().items()},
        filled_M_mean_bps=float(events.loc[filled,'M_net_bps'].mean()) if filled.any() else None,
        unfilled_M_mean_bps=float(events.loc[~filled,'M_net_bps'].mean()) if (~filled).any() else None,
        mean_wait_minutes=float(events.loc[filled,'L_wait_minutes'].mean()) if filled.any() else None)

def comparison_rows(results,stats):
    rows=[]
    for (v,c,stress),result in results.items():
        tr=result.trades;dr=result.daily_returns;sd=dr.std(ddof=1)
        avg=lambda col:float(tr[col].mean()) if len(tr) else None
        row=dict(version=v,cost=c,stress=stress,trades=len(tr),
            entry_months=int(tr.entry_time.dt.strftime('%Y-%m').nunique()),
            gross_bps=avg('gross_bps'),net_bps=avg('net_bps'),net_R=avg('net_R'),
            total_return=float(result.daily_equity.iloc[-1]/INITIAL_CAPITAL-1),
            sharpe=float(dr.mean()/sd*np.sqrt(365)) if sd>0 else None,
            daily_max_drawdown=drawdown(result.daily_equity),minute_max_drawdown=drawdown(result.equity),
            holding_fraction=float(tr.minutes_held.sum()/len(result.equity)),
            turnover=float((tr.q*(tr.entry_price+tr.exit_price)).sum()/INITIAL_CAPITAL),
            fees=float(tr.fees.sum()),funding=float(tr.funding_cashflow.sum()),
            failed=result.metadata['failed'])
        for flag in ('risk_budget_exceeded','actual_leverage_exceeds_cap','has_archive_gap','halt_in_holding','overrun_24h'):
            row[flag+'_trades']=int(tr[flag].sum())
        if c=='main' and not stress:
            for gate,g in stats['gates'][v].items(): row[gate]=g['passed']
            h='M' if v=='M_market' else 'L'
            row.update(p_raw=stats['tests'][h]['p_raw'],p_holm=stats['tests'][h]['p_holm'],
                paired_p_holm=stats['tests']['D']['p_holm'],candidate_status=stats['conclusions']['versions'][v])
        rows.append(row)
    return pd.DataFrame(rows)

def period_rows(results):
    rows=[]
    for (v,c,stress),r in results.items():
        tr=r.trades;entry=pd.DatetimeIndex(tr.entry_time)
        selections=[(str(y),entry.year==y) for y in range(2020,2024)]+[
            ('before_2022',entry<HALF),('from_2022',entry>=HALF),
            ('2020-03',entry.strftime('%Y-%m')=='2020-03'),('2021-05',entry.strftime('%Y-%m')=='2021-05')]
        for label,mask in selections:
            x=tr.loc[mask]
            if label=='before_2022': daily=r.daily_returns.loc[r.daily_returns.index<HALF]
            elif label=='from_2022': daily=r.daily_returns.loc[r.daily_returns.index>=HALF]
            else: daily=r.daily_returns.loc[label]
            rows.append(dict(version=v,cost=c,stress=stress,period=label,trades=len(x),
                net_bps=float(x.net_bps.mean()) if len(x) else None,
                gross_bps=float(x.gross_bps.mean()) if len(x) else None,
                net_R=float(x.net_R.mean()) if len(x) else None,
                period_return=float(np.prod(1+daily)-1)))
    return pd.DataFrame(rows)

def figures(results,comparison,events,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    synthetic=next(iter(results.values())).metadata['source']=='synthetic'
    note='SYNTHETIC VALIDATION — invented prices' if synthetic else 'BTC IS only'
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True,layout='constrained')
    for v in VERSIONS:
        eq=results[(v,'main',False)].daily_equity
        axes[0].plot(eq.index,eq/INITIAL_CAPITAL,label=v)
        peaks=np.maximum.accumulate(np.r_[INITIAL_CAPITAL,eq.to_numpy()])[1:]
        axes[1].plot(eq.index,eq/peaks-1,label=v)
    axes[0].set_ylabel('Net equity / initial');axes[1].set_ylabel('Drawdown');axes[0].legend()
    fig.suptitle(note+' | main costs + funding');fig.savefig(out/'figure_1_equity_drawdown.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    diag=paired_diagnostics(events)
    n=len(events);nf=int(events.L_filled.sum()) if n else 0
    axes[0].bar(['Filled','Unfilled'],[nf,n-nf]);axes[0].set_ylabel('All eligible father signals')
    waits=events.loc[events.L_filled.astype(bool),'L_wait_minutes'].dropna()
    axes[1].hist(waits,bins=np.arange(0,65,5));axes[1].set_xlabel('Filled limit wait (minutes)')
    vals=[diag['filled_contribution_bps'],diag['unfilled_contribution_bps']]
    axes[2].bar(['Filled L−M','Unfilled −M'],[v if v is not None else 0 for v in vals])
    axes[2].axhline(0,color='black',lw=.7);axes[2].set_ylabel('Contribution bps / ALL fathers')
    if not n: axes[2].text(.5,.5,'No eligible events',ha='center',transform=axes[2].transAxes)
    fig.suptitle(note);fig.savefig(out/'figure_2_fills_opportunity_cost.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    x=np.arange(len(comparison)); labels=[r.version[0]+' '+r.cost+(' stress' if r.stress else '') for r in comparison.itertuples()]
    axes[0].bar(x,comparison.net_bps.fillna(0));axes[1].bar(x,comparison.total_return)
    for ax in axes: ax.set_xticks(x,labels,rotation=35,ha='right');ax.axhline(0,color='black',lw=.7)
    for i,r in enumerate(comparison.itertuples()):
        if pd.isna(r.net_bps): axes[0].annotate('no trades',(i,0),rotation=90,fontsize=8)
    axes[0].set_ylabel('Mean net bps / trade');axes[1].set_ylabel('Total portfolio return')
    fig.suptitle(note);fig.savefig(out/'figure_3_cost_execution_stress.png',dpi=140);plt.close(fig)
