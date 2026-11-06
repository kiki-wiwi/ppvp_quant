"""Diagnostics and three figures consuming only the fixed batch outputs."""
from pathlib import Path
import numpy as np
import pandas as pd
from . import config


def sharpe(returns):
    values=np.asarray(returns,float)
    sd=values.std(ddof=1) if len(values)>1 else np.nan
    return float(np.sqrt(365)*values.mean()/sd) if sd>0 else np.nan


def portfolio_summary(result):
    t=result.trades;d=result.daily_returns;x=result.equity.to_numpy(float)
    peak=np.maximum.accumulate(np.r_[config.INITIAL_CAPITAL,x])[1:]
    mean=lambda c:float(t[c].mean()) if len(t) else np.nan
    count=lambda c:int(t[c].sum()) if len(t) else 0
    return dict(version=result.metadata['version'],cost=result.metadata['cost'],
        trade_count=len(t),total_return=float((1+d).prod()-1),daily_mean=float(d.mean()),
        sharpe=sharpe(d),max_drawdown=float(np.min(x/peak-1)) if len(x) else 0.,
        gross_bps=mean('gross_bps'),net_bps=mean('net_bps'),net_R=mean('net_R'),
        exposure_fraction=float(t.minutes_held.sum()/(len(d)*1440)),
        mean_holding_minutes=mean('minutes_held'),entry_month_count=int(t.entry_time.dt.strftime('%Y-%m').nunique()),
        fees=float(t.fees.sum()),funding_cashflow=float(t.funding_cashflow.sum()),
        reference_turnover=float(t.reference_notional.sum()/config.INITIAL_CAPITAL),
        actual_two_way_notional=float((t.q*(t.entry_fill+t.exit_fill)).sum()),
        reference_leverage_max=float(t.reference_leverage.max()) if len(t) else 0.,
        actual_leverage_max=float(t.actual_leverage.max()) if len(t) else 0.,
        risk_budget_exceeded=count('risk_budget_exceeded'),
        actual_leverage_exceeds_cap=count('actual_leverage_exceeds_cap'),
        archive_gap_trades=count('has_archive_gap'),halt_trades=count('halt_in_holding'),
        overrun_trades=count('overrun_72h'),
        cross_month_trades=int((t.entry_time.dt.strftime('%Y-%m')!=t.exit_time.dt.strftime('%Y-%m')).sum()))


def period_rows(result):
    t=result.trades;d=result.daily_returns
    windows=[(str(y),pd.Timestamp(f'{y}-01-01',tz='UTC'),pd.Timestamp(f'{y+1}-01-01',tz='UTC')) for y in range(2020,2024)]
    windows += [('first_half',config.MAIN_START,config.HALF),('second_half',config.HALF,config.IS_END)]
    windows += [('risk_2020-03',pd.Timestamp('2020-03-01',tz='UTC'),pd.Timestamp('2020-04-01',tz='UTC')),
                ('risk_2021-05',pd.Timestamp('2021-05-01',tz='UTC'),pd.Timestamp('2021-06-01',tz='UTC'))]
    rows=[]
    for label,start,end in windows:
        ret=d.loc[(d.index>=start)&(d.index<end)]
        trades=t.loc[(t.entry_time>=start)&(t.entry_time<end)]
        # Exposure intersects the calendar period, unlike entry-attributed trade means.
        durations=np.maximum(0,np.minimum(t.exit_time.astype('int64'),end.value)-
            np.maximum(t.entry_time.astype('int64'),max(start,config.MAIN_START).value))/60e9
        rows.append(dict(version=result.metadata['version'],cost=result.metadata['cost'],period=label,
            total_return=float((1+ret).prod()-1),daily_mean=float(ret.mean()),sharpe=sharpe(ret),
            trade_count=len(trades),net_bps=float(trades.net_bps.mean()),gross_bps=float(trades.gross_bps.mean()),
            net_R=float(trades.net_R.mean()),funding_cashflow=float(trades.funding_cashflow.sum()),
            exposure_fraction=float(np.sum(durations)/(len(ret)*1440)) if len(ret) else np.nan))
    return rows


def execution_balance(result):
    decisions=result.decisions
    rows=[]
    for decision,group in decisions.groupby('decision',sort=True):
        prevol=group.prevol.to_numpy(float) if 'prevol' in group else np.array([])
        good=prevol[np.isfinite(prevol)&(prevol>0)]
        rows.append(dict(version=result.metadata['version'],cost=result.metadata['cost'],decision=decision,
            count=len(group),prevol_valid_count=len(good),prevol_mean=float(good.mean()) if len(good) else np.nan,
            prevol_p10=float(np.quantile(good,.1)) if len(good) else np.nan,
            prevol_median=float(np.quantile(good,.5)) if len(good) else np.nan,
            prevol_p90=float(np.quantile(good,.9)) if len(good) else np.nan))
    return rows


def figures(s_results,random_equity,stats,comparison,periods,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(output)
    main=s_results['main_17'];daily=stats['daily']
    rbar_index=config.INITIAL_CAPITAL*(1+daily['Rbar']).cumprod()
    q=random_equity.quantile([.025,.5,.975],axis=1)
    fig,ax=plt.subplots(figsize=(12,5))
    ax.fill_between(random_equity.index,q.loc[.025],q.loc[.975],alpha=.2,color='gray',
        label='Random equity 2.5–97.5% replicate dispersion (not market CI)')
    ax.plot(main.daily_equity.index,main.daily_equity,label='S main_17')
    ax.plot(rbar_index.index,rbar_index,label='Rbar daily-mean index (statistical benchmark)')
    ax.plot(q.columns,q.loc[.5],lw=.8,label='Median actual random equity')
    ax.set(ylabel='Equity / index (USDT)',title='1. Unfiltered channel and fixed conditional random benchmark')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'figure_1_equity.png',dpi=150);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for axis,column,title in zip(axes,['total_return','trade_count'],['Annual net return','Annual entered trades']):
        distributions=[];svalues=[]
        for year in ('2020','2021','2022','2023'):
            p=periods.loc[(periods.period==year)&(periods.cost=='main_17')]
            distributions.append(p.loc[p.version.str.startswith('R_'),column].to_numpy())
            svalues.append(p.loc[p.version==config.S_VERSION,column].iloc[0])
        axis.boxplot(distributions,tick_labels=['2020*','2021','2022','2023'],showfliers=True)
        axis.plot(range(1,5),svalues,'o-',label='S main_17')
        axis.set_title(title);axis.legend()
    fig.suptitle('2. S and 200 random paths; boxes show replicate dispersion; * March onward')
    fig.tight_layout();fig.savefig(out/'figure_2_annual.png',dpi=150);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    s=comparison.loc[comparison.version==config.S_VERSION].set_index('cost').loc[list(config.COSTS)]
    axes[0].bar(range(3),s.total_return.to_numpy()*100)
    axes[0].set_xticks(range(3),list(config.COSTS));axes[0].set(ylabel='Net return (%)',title='S cost sensitivity; no random harsh run')
    monthly=daily['D'].resample('MS').sum()*10000
    axes[1].bar(range(len(monthly)),monthly,color=np.where(monthly>=0,'steelblue','indianred'))
    axes[1].set_xticks(range(0,len(monthly),6),monthly.index.strftime('%Y-%m')[::6],rotation=45)
    axes[1].set(ylabel='Sum of paired daily differences (bp)',title='Monthly S − Rbar contribution; additive, not compounded')
    for ax in axes:ax.axhline(0,color='black',lw=.6)
    fig.tight_layout();fig.savefig(out/'figure_3_cost_and_months.png',dpi=150);plt.close(fig)
