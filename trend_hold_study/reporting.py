"""Registered summaries and figures from the six existing account outputs only."""
from pathlib import Path
import numpy as np
import pandas as pd
from . import config


def sharpe(values):
    values=np.asarray(values,float)
    sd=values.std(ddof=1) if len(values)>1 else np.nan
    return float(np.sqrt(365)*values.mean()/sd) if sd>0 else np.nan



def position_counts(result,start=None,end=None):
    """Signal flips and actual position episodes; closed episodes belong to exit time."""
    minute=result.minute;orders=result.orders;decisions=result.decisions
    if start is not None:
        minute=minute.loc[(minute.index>=start)&(minute.index<end)]
        if len(orders):orders=orders.loc[(orders.time>=start)&(orders.time<end)]
        if len(decisions):decisions=decisions.loc[(decisions.time>=start)&(decisions.time<end)]
    held=minute['held_qty'] if 'held_qty' in minute else minute.qty
    held_days=int((held>0).groupby(minute.index.normalize()).any().sum())
    return dict(signal_switch_count=int(decisions.trend_flip.sum()) if len(decisions) else 0,
        position_entry_count=int(((orders.qty_before==0)&(orders.qty_target>0)).sum()) if len(orders) else 0,
        completed_holding_intervals=int(((orders.qty_before>0)&(orders.qty_target==0)).sum()) if len(orders) else 0,
        held_calendar_days=held_days)


def portfolio_summary(result):
    m=result.minute;d=result.daily_returns;meta=result.metadata
    x=m.equity.to_numpy(float)
    peak=np.maximum.accumulate(np.r_[config.INITIAL_CAPITAL,x])[1:]
    held=m['held_qty'] if 'held_qty' in m else m.qty
    return dict(version=meta['version'],cost=meta['cost'],
        total_return=float((1+d).prod()-1),daily_mean=float(d.mean()),sharpe=sharpe(d),
        max_drawdown=float(np.min(x/peak-1)),realized_volatility=float(d.std(ddof=1)*np.sqrt(365)),
        exposure_fraction=float((held>0).mean()),mean_actual_leverage=float(m.actual_leverage.mean()),
        fees=meta['total_fees'],slippage_cost=meta['total_slippage_cost'],funding_cashflow=meta['total_funding'],
        adjustment_count=meta['adjustment_count'],order_count=meta['order_count'],
        turnover_notional=meta['turnover_notional'],
        reference_turnover=meta.get('total_turnover_fraction',np.nan),
        actual_leverage_max=meta['max_actual_leverage'],
        minutes_above_leverage_cap=meta['minutes_above_leverage_cap'],
        maintenance_held_minutes=meta['maintenance_held_minutes'],gap_held_minutes=meta['gap_held_minutes'],
        terminal_preclose_actual_leverage=meta.get('terminal_preclose_actual_leverage',np.nan),
        **position_counts(result))


def period_rows(result):
    m=result.minute;d=result.daily_returns;o=result.orders;f=result.funding
    windows=[(str(y),pd.Timestamp(f'{y}-01-01',tz='UTC'),pd.Timestamp(f'{y+1}-01-01',tz='UTC')) for y in range(2020,2024)]
    windows += [('first_half',config.MAIN_START,config.HALF),('second_half',config.HALF,config.IS_END)]
    rows=[]
    for label,start,end in windows:
        ret=d.loc[(d.index>=start)&(d.index<end)]
        minute=m.loc[(m.index>=start)&(m.index<end)]
        orders=o.loc[(o.time>=start)&(o.time<end)] if len(o) else o
        funding=f.loc[(f.time>=start)&(f.time<end)] if len(f) else f
        if not len(ret):continue
        held=minute['held_qty'] if 'held_qty' in minute else minute.qty
        prior=result.daily_equity.loc[result.daily_equity.index<ret.index[0]]
        initial=float(prior.iloc[-1]) if len(prior) else config.INITIAL_CAPITAL
        equity=minute.equity.to_numpy(float)
        peaks=np.maximum.accumulate(np.r_[initial,equity])[1:]
        rows.append(dict(version=result.metadata['version'],cost=result.metadata['cost'],period=label,
            total_return=float((1+ret).prod()-1),daily_mean=float(ret.mean()),sharpe=sharpe(ret),
            max_drawdown=float(np.min(equity/peaks-1)),realized_volatility=float(ret.std(ddof=1)*np.sqrt(365)),
            exposure_fraction=float((held>0).mean()),mean_actual_leverage=float(minute.actual_leverage.mean()),
            order_count=len(orders),adjustment_count=int((~orders.terminal).sum()) if len(orders) else 0,
            terminal_order_count=int(orders.terminal.sum()) if len(orders) else 0,
            **position_counts(result,start,end),fees=float(orders.fee.sum()) if len(orders) else 0.,
            funding_cashflow=float(funding.cash_flow.sum()) if len(funding) else 0.,
            turnover_notional=float(orders.turnover_notional.sum()) if len(orders) else 0.,
            reference_turnover=float(orders.turnover_fraction.sum()) if len(orders) and 'turnover_fraction' in orders else 0.))
    return rows


def figures(accounts,features,stats,comparison,periods,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(output)
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    for version in config.VERSIONS:
        account=accounts[(version,'main_17')]
        axes[0].plot(account.daily_equity.index,account.daily_equity,label=version)
    axes[0].set(ylabel='Equity (USDT)',title='1. Daily trend vs continuous long — main costs')
    axes[0].legend()
    axes[1].step(features.index,features.budget,where='post',label='Shared weekly reference leverage budget')
    axes[1].step(features.index,features.g.astype(int),where='post',alpha=.6,label='Trend switch (1=on)')
    axes[1].set(ylabel='Budget / switch',ylim=(-.05,1.1));axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'figure_1_equity_and_budget.png',dpi=150);plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    years=['2020','2021','2022','2023'];x=np.arange(4)
    for i,version in enumerate(config.VERSIONS):
        p=periods.loc[(periods.version==version)&(periods.cost=='main_17')].set_index('period')
        for ax,key,mult,title in zip(axes,['total_return','exposure_fraction','reference_turnover'],
                [100,100,1],['Calendar net return (%)','Time in position (%)','One-way reference turnover (x)']):
            ax.bar(x+(i-.5)*.36,p.reindex(years)[key]*mult,width=.36,label=version)
            ax.set_xticks(x,['2020*','2021','2022','2023']);ax.set_title(title);ax.axhline(0,color='black',lw=.6)
    axes[0].legend(fontsize=8);fig.suptitle('2. Annual main-cost accounts; * March onward')
    fig.tight_layout();fig.savefig(out/'figure_2_annual.png',dpi=150);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    x=np.arange(3)
    for i,version in enumerate(config.VERSIONS):
        p=comparison.loc[comparison.version==version].set_index('cost').loc[list(config.COSTS)]
        axes[0].bar(x+(i-.5)*.36,p.total_return*100,width=.36,label=version)
    axes[0].set_xticks(x,list(config.COSTS));axes[0].set(ylabel='Net return (%)',title='Both accounts at all three costs');axes[0].legend(fontsize=8)
    monthly=stats['monthly'];values=monthly.D_sum.to_numpy()*10000
    axes[1].bar(range(len(monthly)),values,color=np.where(values>=0,'steelblue','indianred'))
    axes[1].set_xticks(range(0,len(monthly),6),list(monthly.index.astype(str))[::6],rotation=45)
    axes[1].set(ylabel='Sum of paired daily differences (bp)',title='T − P by month: additive, not compounded')
    for ax in axes:ax.axhline(0,color='black',lw=.6)
    fig.tight_layout();fig.savefig(out/'figure_3_cost_and_months.png',dpi=150);plt.close(fig)
