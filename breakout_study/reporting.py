"""Tables and the three registered figures; never creates new simulations."""
import numpy as np
import pandas as pd
from .config import COSTS, VERSIONS, INITIAL_CAPITAL, HALF


def annotate_comparison(table,stats):
    out=table.copy()
    for key in ['N1','N2','N3','N4','N5','N6','p_raw','p_holm','paired_p_holm',
                'daily_mean_ci_lower','daily_mean_ci_upper','candidate_status']:
        out[key]=pd.Series([None]*len(out),dtype=object)
    for i,row in out.iterrows():
        if row.cost!='main_17': continue
        version=row.version
        for gate,result in stats['gates'][version].items(): out.at[i,gate]=result['passed']
        hypothesis='P' if version=='P_channel' else 'V'
        for key in ('p_raw','p_holm'): out.at[i,key]=stats['tests'][hypothesis][key]
        out.at[i,'paired_p_holm']=stats['tests']['D']['p_holm']
        for endpoint in ('lower','upper'):
            out.at[i,'daily_mean_ci_'+endpoint]=stats['intervals']['daily_mean'][version][endpoint]
        out.at[i,'candidate_status']=stats['conclusions']['versions'][version]
    return out


def drawdown(equity):
    values=np.r_[INITIAL_CAPITAL,np.asarray(equity,float)]
    return float(np.min(values/np.maximum.accumulate(values)-1))


def comparison_rows(results):
    rows=[]
    for (version,cost),result in results.items():
        tr=result.trades
        dr=result.daily_returns
        sd=dr.std(ddof=1)
        def avg(col): return float(tr[col].mean()) if len(tr) else None
        rows.append(dict(version=version,cost=cost,trades=len(tr),
            entry_months=int(tr.entry_time.dt.strftime('%Y-%m').nunique()),
            gross_bps=avg('gross_bps'),net_bps=avg('net_bps'),net_R=avg('net_R'),
            total_return=float(result.daily_equity.iloc[-1]/INITIAL_CAPITAL-1),
            sharpe=float(dr.mean()/sd*np.sqrt(365)) if sd>0 else None,
            daily_max_drawdown=drawdown(result.daily_equity),minute_max_drawdown=drawdown(result.equity),
            holding_fraction=float(tr.minutes_held.sum()/len(result.equity)),
            turnover=float((tr.q*(tr.entry_fill+tr.exit_fill)).sum()/INITIAL_CAPITAL),
            fees=float(tr.fees.sum()),funding=float(tr.funding.sum()),
            cap_bound_trades=int(tr.cap_bound.sum()),risk_budget_exceeded_trades=int(tr.risk_budget_exceeded.sum()),
            actual_leverage_exceeds_cap=int(tr.actual_leverage_exceeds_cap.sum()),
            maintenance_trades=int(tr.halt_in_holding.sum()),overrun_72h_trades=int(tr.overrun_72h.sum()),
            archive_gap_trades=int(tr.has_archive_gap.sum()),failed=result.metadata['failed']))
    return pd.DataFrame(rows)


def period_rows(results):
    rows=[]
    for (version,cost),result in results.items():
        tr=result.trades
        entry=pd.DatetimeIndex(tr.entry_time)
        for label,mask in [(str(year),entry.year==year) for year in range(2020,2024)]+[
                ('before_2022',entry<HALF),('from_2022',entry>=HALF),
                ('2020-03',entry.strftime('%Y-%m')=='2020-03'),('2021-05',entry.strftime('%Y-%m')=='2021-05')]:
            x=tr.loc[mask]
            if label=='before_2022': daily=result.daily_returns.loc[result.daily_returns.index<HALF]
            elif label=='from_2022': daily=result.daily_returns.loc[result.daily_returns.index>=HALF]
            else: daily=result.daily_returns.loc[label]
            sd=daily.std(ddof=1)
            rows.append({'version':version,'cost':cost,'period':label,'trades':len(x),
                'gross_bps':float(x.gross_bps.mean()) if len(x) else None,
                'net_bps':float(x.net_bps.mean()) if len(x) else None,
                'net_R':float(x.net_R.mean()) if len(x) else None,
                'period_return':float(np.prod(1+daily)-1),
                'sharpe':float(daily.mean()/sd*np.sqrt(365)) if sd>0 else None})
    return pd.DataFrame(rows)


def event_exposure(results):
    """All hourly boundaries of the two predeclared dates, not selected instants."""
    rows=[]
    for version in VERSIONS:
        tr=results[(version,'main_17')].trades
        for date in ('2020-03-12','2021-05-19'):
            for now in pd.date_range(date,periods=24,freq='h',tz='UTC'):
                live=tr.loc[(tr.entry_time<=now)&(tr.exit_time>now)]
                rows.append(dict(version=version,time=now,position_open_after_boundary=bool(len(live)),
                    trade_id=int(live.iloc[0].trade_id) if len(live) else None))
    return pd.DataFrame(rows)


def figures(results,comparison,periods,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True,layout='constrained')
    for version in VERSIONS:
        eq=results[(version,'main_17')].daily_equity
        axes[0].plot(eq.index,eq/INITIAL_CAPITAL,label=version)
        peaks=np.maximum.accumulate(np.r_[INITIAL_CAPITAL,eq.to_numpy()])[1:]
        axes[1].plot(eq.index,eq/peaks-1,label=version)
    axes[0].set_ylabel('Net equity / initial');axes[1].set_ylabel('Drawdown');axes[0].legend()
    fig.suptitle('Main 17bp + funding');fig.savefig(out/'figure_1_equity_drawdown.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    years=[str(x) for x in range(2020,2024)];x=np.arange(4)
    for k,version in enumerate(VERSIONS):
        rows=periods.loc[(periods.version==version)&(periods.cost=='main_17')].set_index('period').loc[years]
        axes[0].bar(x+(k-.5)*.36,rows.net_bps,width=.36,label=version)
        axes[1].bar(x+(k-.5)*.36,rows.trades,width=.36,label=version)
    for ax in axes: ax.set_xticks(x,years)
    axes[0].axhline(0,color='black',lw=.7);axes[0].set_ylabel('Mean net bps / trade');axes[1].set_ylabel('Trades')
    axes[0].legend();fig.savefig(out/'figure_2_years.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    costs=list(COSTS);x=np.arange(3)
    for k,version in enumerate(VERSIONS):
        rows=comparison.loc[comparison.version==version].set_index('cost').loc[costs]
        axes[0].bar(x+(k-.5)*.36,rows.net_bps,width=.36,label=version)
        axes[1].bar(x+(k-.5)*.36,rows.total_return,width=.36,label=version)
    for ax in axes: ax.set_xticks(x,costs);ax.axhline(0,color='black',lw=.7)
    axes[0].set_ylabel('Mean net bps / trade');axes[1].set_ylabel('Total portfolio return');axes[0].legend()
    fig.savefig(out/'figure_3_costs.png',dpi=140);plt.close(fig)
