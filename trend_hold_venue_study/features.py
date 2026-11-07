"""Causal canonical daily signals, separated from venue execution references."""
import numpy as np
import pandas as pd
from . import config, scope
DAY=pd.Timedelta(days=1)
MINUTE=pd.Timedelta(minutes=1)

def _daily(ctx):
    scope.assert_scope(ctx)
    first=ctx.index[0].ceil('D')
    last=(ctx.index[-1]+MINUTE).floor('D')
    a,b=ctx.index.searchsorted(first),ctx.index.searchsorted(last)
    valid=~ctx.bad[a:b].reshape(-1,1440)
    counts=valid.sum(axis=1)
    if (counts==0).any(): raise ValueError('UTC day has no valid minutes')
    close=ctx.C[a:b].reshape(-1,1440)
    offsets=1439-valid[:,::-1].argmax(axis=1)
    values=close[np.arange(len(close)),offsets]
    gaps=ctx.archive_gap[a:b].reshape(-1,1440).any(axis=1)
    return first,values,counts,gaps

def build_features(ctx,signal_origin,end=None):
    origin=config.utc(signal_origin)
    if origin!=origin.normalize(): raise ValueError('signal origin must be midnight')
    first,closes,counts,gaps=_daily(ctx)
    final=ctx.index[-1].floor('D') if end is None else min(ctx.index[-1].floor('D'),config.utc(end)-DAY)
    days=pd.date_range(origin,final,freq='D',name='decision_time').as_unit('ns')
    prior=np.asarray((days-first).days,dtype=np.intp)-1
    if not len(days) or (prior<59).any() or (prior>=len(closes)).any(): raise ValueError('60 completed UTC days required')
    eref=closes[prior]
    sma=closes[prior[:,None]+np.arange(-59,1)].mean(axis=1)
    rdays=prior[:,None]+np.arange(-19,1)
    returns=closes[rdays]/closes[rdays-1]-1.
    sigma=np.sqrt(365.)*returns.std(axis=1,ddof=1)
    if not np.isfinite(sma).all() or not np.isfinite(sigma).all(): raise ValueError('nonfinite daily features')
    updates=np.asarray((days==origin)|(days.weekday==0))
    held=np.full(len(days),np.nan);held[updates]=config.LEVERAGE_CAP
    limited=updates&(sigma>config.TARGET_VOL/config.LEVERAGE_CAP)
    held[limited]=config.TARGET_VOL/sigma[limited]
    budget=pd.Series(held,index=days).ffill()
    if not np.isfinite(budget).all() or (budget<=0).any(): raise ValueError('invalid budget')
    return pd.DataFrame(dict(eref=eref,sma60=sma,sigma=sigma,budget=budget,g=eref>sma,
        budget_update=updates,n_valid_prior=counts[prior],has_archive_gap=gaps[prior]),index=days)

def references(ctx,days):
    """Last valid close of each preceding UTC day; no canonical-source price reuse."""
    scope.assert_scope(ctx)
    days=scope.utc_index(days,'reference days')
    rows=[]
    for day in days:
        a=ctx.index.searchsorted(day-DAY);b=ctx.index.searchsorted(day)
        valid=np.flatnonzero(~ctx.bad[a:b])
        if not len(valid): raise ValueError('prior UTC day has no valid close')
        rows.append((ctx.C[a+valid[-1]],len(valid),bool(ctx.archive_gap[a:b].any())))
    return pd.DataFrame(rows,index=days,columns=['eref','n_valid_prior','has_archive_gap'])
