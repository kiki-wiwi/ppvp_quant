"""CP3 statistics implementing frozen plan §§8–9 (no market loading).

Implementation choices registered before unlock:
* Undefined numbers are None in reports; every undefined gate p is 1.
* Cluster sums use pandas' sorted group aggregation, as the frozen helper does.
  Missing matched excess is excluded; nonfinite ordinary trade metrics raise.
* Three paired D metrics reuse one CI matrix and one Webb matrix per seed.
* Webb weights are drawn in one exact (99999,G) call; derived arithmetic is
  processed in batches of 1024 rows without additional random draws.
* Monthly Sharpe draws concatenate natural-month daily arrays in sampled order,
  exactly as §8.4 specifies. Daily indices must contain the full frozen window.
* Max drawdown is daily-close drawdown including the initial-capital anchor,
  consistent with the repository metrics convention; minute drawdown is an
  additional descriptive item when a minute equity series is available.
* Stable Holm ties keep the fixed input order; frozen gates use adjusted p.
* Stationary auxiliary diagnostics reuse the frozen attribution core helpers
  with exactly one draw matrix per seed; their results are checked against
  the public helpers on synthetic inputs. This also counts discarded draws
  without replaying a seed or changing the frozen public helpers.
* G7 tied month totals use ascending month order. F4/F5 point ties use <1e-12.
The CP4 auxiliary functions are defined here but are not invoked by CP3.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from ppvp_audit.nulls import cluster_bootstrap_mean
from ppvp_audit.experiments import holm
from . import versions as VS
from .scope import IS_START, IS_END, ScopeError

NAMES = list(VS.VERSIONS)
B_CI, B_WILD, B_SHARPE = 10000, 99999, 5000
WEBB = np.array([-np.sqrt(1.5), -1., -np.sqrt(.5), np.sqrt(.5), 1., np.sqrt(1.5)])
HALF = pd.Timestamp('2022-01-01', tz='UTC')
STOP_PAIRS = ((NAMES[0],NAMES[1]), (NAMES[0],NAMES[2]),
              (NAMES[1],NAMES[2]), (NAMES[5],NAMES[6]))
OFFSETS = {'match':100,'g2_ci':200,'g2_p':250,'g3_ci':300,'g3_p':350,
           'box_ci':400,'box_p':420,'box_sharpe':450,'inside_ci':460,'inside_p':470,
           'g4':500,'net_r_ci':550,'f4':600,'f5':700,
           'daily_g2_ci':800,'daily_g2_p':820,'daily_g3_ci':810,'daily_g3_p':830,
           'daily_box_ci':850,'daily_box_p':860,'daily_inside_ci':870,'daily_inside_p':880,
           'randomization':900,'onset_ci':1500,'appendix_ci':3500,'appendix_p':3550}


def seed_for(process, version=0, *, other=None, figure=None, sensitivity=None, auxiliary=False):
    v = NAMES.index(version) if isinstance(version,str) else int(version)
    if process == 'figure_ci':
        if figure not in (1,2,3,4):
            raise ValueError('figure must be 1..4')
        offset=1000+100*figure+v
    elif process == 'f5':
        u=NAMES.index(other) if isinstance(other,str) else int(other)
        a,b=sorted((v,u))
        if a==b:
            raise ValueError('F5 requires different versions')
        offset=700+10*a+b
    else:
        offset=OFFSETS[process]+v
    if sensitivity is not None:
        if sensitivity not in range(2,8):
            raise ValueError('sensitivity seeds use §11 items 2..7')
        offset+=10000*sensitivity
    return VS.SEED_BASE+offset+(5000 if auxiliary else 0)


def _utc_index(values, label):
    idx=pd.DatetimeIndex(values)
    if idx.tz is None or idx.hasnans:
        raise ScopeError(f'{label}: timezone-aware finite UTC timestamps required')
    return idx.tz_convert('UTC').as_unit('ns')


def assert_statistics_scope(trades=None, daily_returns=None, *, appendix=False):
    start=IS_START if appendix else VS.MAIN_START
    if trades is not None and len(trades):
        for col in ('trigger_time','entry_time','exit_time'):
            if col not in trades:
                raise ScopeError(f'trade scope requires {col}')
        trigger=_utc_index(trades.trigger_time,'trigger_time')
        entry=_utc_index(trades.entry_time,'entry_time')
        leave=_utc_index(trades.exit_time,'exit_time')
        if ((trigger<start)|(trigger>VS.LAST_TRIGGER)).any():
            raise ScopeError('trade trigger outside frozen comparison window')
        if ((entry<start)|(entry>IS_END-pd.Timedelta(minutes=30))).any():
            raise ScopeError('trade entry outside frozen comparison window')
        if ((leave<entry)|(leave>IS_END)).any():
            raise ScopeError('trade exit outside frozen IS window')
    if daily_returns is not None:
        idx=_utc_index(daily_returns.index,'daily returns')
        if len(idx) and ((idx<start)|(idx>=IS_END)).any():
            raise ScopeError('daily returns outside frozen comparison window')
        if not idx.equals(idx.floor('D')) or idx.has_duplicates or not idx.is_monotonic_increasing:
            raise ScopeError('daily returns must be unique ascending UTC dates')


def _require_full_days(series, appendix=False):
    assert_statistics_scope(daily_returns=series,appendix=appendix)
    start=IS_START if appendix else VS.MAIN_START
    want=pd.date_range(start,IS_END,freq='D',inclusive='left').as_unit('ns')
    if not pd.DatetimeIndex(series.index).as_unit('ns').equals(want):
        raise ValueError('daily series must cover the complete frozen window')
    if not np.isfinite(series.to_numpy(dtype=float)).all():
        raise ValueError('daily series contains nonfinite values')


def daily_returns_from_equity(daily_equity, *, appendix=False):
    _require_full_days(daily_equity,appendix)
    x=daily_equity.to_numpy(dtype=float)
    prior=np.r_[VS.INITIAL_CAPITAL,x[:-1]]
    with np.errstate(divide='ignore',invalid='ignore'):
        ret=x/prior-1
    if not np.isfinite(ret).all():
        raise ValueError('daily equity produces undefined daily returns')
    out=pd.Series(ret,index=daily_equity.index,name='daily_return')
    assert_statistics_scope(daily_returns=out,appendix=appendix)
    return out


def annualized_sharpe(x):
    x=np.asarray(x,float)
    if len(x)<2 or not np.isfinite(x).all():
        return None
    sd=float(x.std(ddof=1))
    if sd==0:
        return None
    val=float(x.mean()/sd*np.sqrt(365.))
    return val if np.isfinite(val) else None


def _clean_values(trades,value_col,appendix=False):
    assert_statistics_scope(trades=trades,appendix=appendix)
    x=trades[value_col].to_numpy(dtype=float)
    valid=np.isfinite(x)
    if (~valid).any() and value_col!='excess_bps':
        raise ValueError(f'{value_col} contains nonfinite trade values')
    return trades.loc[valid],x[valid]


def _groups(trades,value_col,cluster='month',appendix=False):
    if cluster not in ('month','day'):
        raise ValueError('clusters must be month or day')
    frame,x=_clean_values(trades,value_col,appendix)
    keys=_utc_index(frame.entry_time,'cluster entry times').strftime('%Y-%m' if cluster=='month' else '%Y-%m-%d').to_numpy()
    agg=pd.DataFrame({'x':x,'group':keys}).groupby('group',sort=True).x.agg(['sum','count'])
    return frame,x,keys,agg


def cluster_point_mean(trades,value_col='net_bps',*,appendix=False):
    _,_,_,agg=_groups(trades,value_col,'month',appendix)
    n=int(agg['count'].sum())
    return float(agg['sum'].sum()/n) if n else None


def _mcse(p,b):
    return float(np.sqrt(p*(1-p)/b)) if b else None


def _number(value):
    return float(value) if value is not None and np.isfinite(value) else None


def cluster_mean(trades,value_col='net_bps',*,ci_seed,p_seed,cluster='month',appendix=False):
    _,x,keys,agg=_groups(trades,value_col,cluster,appendix)
    sums,counts=agg['sum'].to_numpy(),agg['count'].to_numpy()
    n,g=len(x),len(sums)
    mean=float(sums.sum()/n) if n else None
    out={'mean':mean,'lo':None,'hi':None,'p_le_0':None,'p':1.,'p_positive':1.,'p_negative':1.,
         'se':None,'t':None,'n':n,'n_clusters':g,'cluster':cluster,'discarded':0,
         'ci_seed':int(ci_seed),'p_seed':None if p_seed is None else int(p_seed),
         'wild_B':B_WILD,'ci_B':B_CI,'mcse_positive':0.,'mcse_negative':0.,'reason':None}
    if g<12:
        out['reason']='fewer than 12 occupied clusters'
        return out
    ci=cluster_bootstrap_mean(x,keys,n_boot=B_CI,seed=ci_seed)
    out.update({k:ci[k] for k in ('lo','hi','p_le_0')})
    se=float(np.sqrt(g/(g-1)*np.square(sums-mean*counts).sum()/n**2))
    out['se']=se
    if se==0 or not np.isfinite(se):
        out['reason']='zero or undefined cluster standard error'
        return out
    out['t']=mean/se
    if p_seed is None:
        out['reason']='interval-only report; no wild test requested'
        return out
    weights=np.random.default_rng(p_seed).choice(WEBB,size=(B_WILD,g))
    ge,le=0,0
    for first in range(0,B_WILD,1024):
        star=weights[first:first+1024]*sums
        m=star.sum(axis=1)/n
        sd=np.sqrt(g/(g-1)*np.square(star-m[:,None]*counts).sum(axis=1)/n**2)
        t=np.divide(m,sd,out=np.zeros(len(m)),where=sd>0)
        ge+=int((t>=out['t']).sum())
        le+=int((t<=out['t']).sum())
    out['p']=out['p_positive']=ge/B_WILD
    out['p_negative']=le/B_WILD
    out['mcse_positive']=_mcse(out['p_positive'],B_WILD)
    out['mcse_negative']=_mcse(out['p_negative'],B_WILD)
    return out


def paired_cluster_mean(a,b,value_col='net_bps',*,ci_seed,p_seed,cluster='month',appendix=False,draw_bundle=None):
    _,xa,_,aa=_groups(a,value_col,cluster,appendix)
    _,xb,_,bb=_groups(b,value_col,cluster,appendix)
    groups=aa.index.union(bb.index).sort_values()
    aa,bb=aa.reindex(groups,fill_value=0),bb.reindex(groups,fill_value=0)
    sa,ka=aa['sum'].to_numpy(),aa['count'].to_numpy()
    sb,kb=bb['sum'].to_numpy(),bb['count'].to_numpy()
    na,nb,g=len(xa),len(xb),len(groups)
    ma,mb=(float(sa.sum()/na) if na else None),(float(sb.sum()/nb) if nb else None)
    d=ma-mb if na and nb else None
    out={'mean':d,'diff':d,'mean_a':ma,'mean_b':mb,'lo':None,'hi':None,'p_le_0':None,
         'p':1.,'p_positive':1.,'p_negative':1.,'se':None,'t':None,'n_a':na,'n_b':nb,
         'n_clusters':g,'cluster':cluster,'discarded':0,'ci_seed':int(ci_seed),'p_seed':int(p_seed),
         'wild_B':B_WILD,'ci_B':B_CI,'mcse_positive':0.,'mcse_negative':0.,'reason':None}
    if g<12:
        out['reason']='fewer than 12 occupied union clusters'
        return out
    if draw_bundle is None:
        draws=np.random.default_rng(ci_seed).integers(0,g,size=(B_CI,g))
    else:
        if draw_bundle['groups']!=tuple(groups) or draw_bundle['ci_seed']!=ci_seed or draw_bundle['p_seed']!=p_seed:
            raise ValueError('paired common draw bundle does not match clusters/seeds')
        draws=draw_bundle['ci_draws']
    nsa,nsb=ka[draws].sum(axis=1),kb[draws].sum(axis=1)
    keep=(nsa>0)&(nsb>0)
    out['discarded']=int((~keep).sum())
    if keep.any():
        boot=sa[draws].sum(axis=1)[keep]/nsa[keep]-sb[draws].sum(axis=1)[keep]/nsb[keep]
        out['lo'],out['hi']=[float(v) for v in np.percentile(boot,[2.5,97.5])]
        out['p_le_0']=float((boot<=0).mean())
    if not na or not nb:
        out['reason']='one paired sample is empty'
        return out
    phi=(sa-ma*ka)/na-(sb-mb*kb)/nb
    se=float(np.sqrt(g/(g-1)*np.square(phi).sum()))
    out['se']=se
    if se==0 or not np.isfinite(se) or not keep.any():
        out['reason']='zero/undefined paired standard error or all CI resamples discarded'
        return out
    out['t']=d/se
    pooled=(sa.sum()+sb.sum())/(na+nb)
    weights=(np.random.default_rng(p_seed).choice(WEBB,size=(B_WILD,g))
             if draw_bundle is None else draw_bundle['wild_weights'])
    ge,le=0,0
    for first in range(0,B_WILD,1024):
        w=weights[first:first+1024]
        astar=pooled*ka+w*(sa-pooled*ka)
        bstar=pooled*kb+w*(sb-pooled*kb)
        mastar,mbstar=astar.sum(axis=1)/na,bstar.sum(axis=1)/nb
        phi=(astar-mastar[:,None]*ka)/na-(bstar-mbstar[:,None]*kb)/nb
        sd=np.sqrt(g/(g-1)*np.square(phi).sum(axis=1))
        t=np.divide(mastar-mbstar,sd,out=np.zeros(len(sd)),where=sd>0)
        ge+=int((t>=out['t']).sum())
        le+=int((t<=out['t']).sum())
    out['p']=out['p_positive']=ge/B_WILD
    out['p_negative']=le/B_WILD
    out['mcse_positive']=_mcse(out['p_positive'],B_WILD)
    out['mcse_negative']=_mcse(out['p_negative'],B_WILD)
    return out



def paired_cluster_metrics(a,b,*,ci_seed,p_seed,cluster='month',appendix=False):
    """Three registered D metrics share one CI matrix and one Webb matrix."""
    _,_,_,aa=_groups(a,'net_bps',cluster,appendix)
    _,_,_,bb=_groups(b,'net_bps',cluster,appendix)
    groups=aa.index.union(bb.index).sort_values()
    g=len(groups)
    bundle=None
    if g>=12:
        bundle={'groups':tuple(groups),'ci_seed':ci_seed,'p_seed':p_seed,
                'ci_draws':np.random.default_rng(ci_seed).integers(0,g,size=(B_CI,g)),
                'wild_weights':np.random.default_rng(p_seed).choice(WEBB,size=(B_WILD,g))}
    return {metric:paired_cluster_mean(a,b,metric,ci_seed=ci_seed,p_seed=p_seed,
                cluster=cluster,appendix=appendix,draw_bundle=bundle)
            for metric in ('net_bps','gross_bps','net_R')}


def _month_sharpe_samples(returns,draws):
    groups=[group.to_numpy(dtype=float) for _,group in returns.groupby(returns.index.strftime('%Y-%m'),sort=True)]
    out=np.full(len(draws),np.nan)
    for b,row in enumerate(draws):
        value=annualized_sharpe(np.concatenate([groups[i] for i in row]))
        if value is not None:
            out[b]=value
    return out


def month_block_sharpe(returns,*,seed):
    _require_full_days(returns)
    point=annualized_sharpe(returns.to_numpy())
    draws=np.random.default_rng(seed).integers(0,46,size=(B_SHARPE,46))
    sample=_month_sharpe_samples(returns,draws)
    good=sample[np.isfinite(sample)]
    out={'sharpe':point,'lo':None,'hi':None,'p0':1.,'p':1.,'discarded':int(len(sample)-len(good)),
         'n_days':len(returns),'n_months':46,'n_boot':B_SHARPE,'seed':int(seed),
         'mcse':0.,'mc_boundary_sensitive':False,'reason':None}
    if point is None or not len(good):
        out['reason']='undefined point Sharpe or all monthly resamples discarded'
        return out
    out['lo'],out['hi']=[float(v) for v in np.percentile(good,[2.5,97.5])]
    out['p0']=out['p']=float((good<=0).mean())
    out['mcse']=_mcse(out['p0'],B_SHARPE)
    out['mc_boundary_sensitive']=bool(abs(out['p0']-.025)<=2*np.sqrt(.025*.975/B_SHARPE))
    return out


def paired_month_block_sharpe(a,b,*,seed):
    _require_full_days(a)
    _require_full_days(b)
    if not a.index.equals(b.index):
        raise ValueError('paired returns must share exactly the same daily index')
    pa,pb=annualized_sharpe(a.to_numpy()),annualized_sharpe(b.to_numpy())
    point=pa-pb if pa is not None and pb is not None else None
    draws=np.random.default_rng(seed).integers(0,46,size=(B_SHARPE,46))
    sa,sb=_month_sharpe_samples(a,draws),_month_sharpe_samples(b,draws)
    delta=sa-sb
    good=delta[np.isfinite(delta)]
    out={'diff':point,'mean':point,'sharpe_a':pa,'sharpe_b':pb,'lo':None,'hi':None,
         'p_positive':1.,'p_negative':1.,'p_two_sided':1.,'p':1.,
         'discarded':int(len(delta)-len(good)),'n_days':len(a),'n_months':46,'n_boot':B_SHARPE,
         'seed':int(seed),'mcse_positive':0.,'mcse_negative':0.,'mcse_two_sided':0.,'reason':None}
    if point is None or not len(good):
        out['reason']='undefined paired point Sharpe or all monthly resamples discarded'
        return out
    out['lo'],out['hi']=[float(v) for v in np.percentile(good,[2.5,97.5])]
    out['p_positive']=float((good<=0).mean())
    out['p_negative']=float((good>=0).mean())
    out['p']=out['p_two_sided']=min(1.,2*min(out['p_positive'],out['p_negative']))
    for suffix in ('positive','negative','two_sided'):
        out['mcse_'+suffix]=_mcse(out['p_'+suffix],B_SHARPE)
    return out


def holm_family(raw_p,level,*,n_boot=B_WILD):
    pvals={name:float(p) if p is not None and np.isfinite(p) else 1. for name,p in raw_p.items()}
    if any(p<0 or p>1 for p in pvals.values()):
        raise ValueError('p values must lie in [0,1]')
    adjusted=holm(pvals)
    ordered=sorted(pvals,key=pvals.get)
    tests={}
    for rank,name in enumerate(ordered):
        threshold=level/(len(ordered)-rank)
        p=pvals[name]
        se=_mcse(p,n_boot)
        tests[name]={'raw_p':p,'adjusted_p':float(adjusted[name]),'passed':bool(adjusted[name]<=level),
                     'rank':rank+1,'remaining':len(ordered)-rank,'raw_step_threshold':threshold,
                     'mcse':se,'mc_boundary_sensitive':bool(abs(p-threshold)<2*se)}
    return {'size':len(pvals),'level':level,'n_boot':n_boot,'tests':tests}


def _drawdown(x):
    x=np.r_[VS.INITIAL_CAPITAL,np.asarray(x,float)]
    if not len(x) or not np.isfinite(x).all():
        return None
    peak=np.maximum.accumulate(x)
    return float(np.min(x/peak-1))


def portfolio_metrics(result,*,appendix=False):
    tr=result.trades
    assert_statistics_scope(trades=tr,appendix=appendix)
    ret=daily_returns_from_equity(result.daily_equity,appendix=appendix)
    n=len(tr)
    minutes=2103840 if appendix else 2017440
    gross_R=tr.assign(gross_R=tr.gross_pnl/tr.risk_amount) if n else tr.assign(gross_R=pd.Series(dtype=float))
    reasons={str(k):int(v) for k,v in tr.reason_name.value_counts().items()}
    out={'n':n,'gross_bps':cluster_point_mean(tr,'gross_bps',appendix=appendix),
         'net_bps':cluster_point_mean(tr,'net_bps',appendix=appendix),
         'net_R':cluster_point_mean(tr,'net_R',appendix=appendix),
         'gross_R':cluster_point_mean(gross_R,'gross_R',appendix=appendix),
         'win_rate':float((tr.net_pnl>0).mean()) if n else None,
         'sharpe':annualized_sharpe(ret.to_numpy()),
         'total_return':float(result.daily_equity.iloc[-1]/VS.INITIAL_CAPITAL-1),
         'max_drawdown':_drawdown(result.daily_equity.to_numpy()),
         'holding_time_share':float(tr.minutes_held.sum()/minutes),
         'holding_minutes_denominator':minutes,'funding_total':float(tr.funding.sum()),
         'cap_bound_n':int(tr.cap_bound.sum()),'floor_bound_n':int(tr.floor_bound.sum()),
         'stop_exit_share':float(tr.reason.isin([1,2,3]).mean()) if n else None,
         'exit_reasons':reasons,'daily_return_count':len(ret)}
    if result.equity is not None and len(result.equity):
        series=result.equity.dropna()
        idx=_utc_index(series.index,'minute equity')
        start=IS_START if appendix else VS.MAIN_START
        if len(idx) and ((idx<start)|(idx>=IS_END)).any():
            raise ScopeError('minute equity outside comparison window')
        out['max_drawdown_minute']=_drawdown(series.to_numpy())
    return out


def _halves(trades):
    entry=pd.DatetimeIndex(trades.entry_time)
    return [trades.loc[entry<HALF],trades.loc[entry>=HALF]]


def point_gate_statistics(main,harsh,causal=None):
    for tr in (main,harsh,causal):
        if tr is not None:
            assert_statistics_scope(trades=tr)
    half_means=[cluster_point_mean(t) for t in _halves(main)]
    _,_,_,group=_groups(main,'net_bps')
    ranked=group.reset_index().sort_values(['sum','group'],ascending=[False,True],kind='stable')
    removed=list(ranked['group'].iloc[:3])
    keep=~pd.DatetimeIndex(main.entry_time).strftime('%Y-%m').isin(removed)
    remainder=cluster_point_mean(main.loc[keep])
    harsh_mean=cluster_point_mean(harsh)
    causal_mean=cluster_point_mean(causal) if causal is not None else None
    positive=lambda x:x is not None and x>0
    return {'G1':len(main)>=200,'G5':bool(len(main) and all(positive(m) for m in half_means)),
            'G6':bool(len(main) and positive(harsh_mean)),
            'G7':bool(len(main) and positive(remainder)),
            'G8z':bool(len(main) and positive(causal_mean)),
            'half_means':half_means,'harsh_mean':harsh_mean,'causal_mean':causal_mean,
            'removed_top_months':removed,'remaining_n':int(keep.sum()),'remaining_mean':remainder}


def box_decision(pairs):
    pos=[p for p in pairs if p['positive']]
    neg=[p for p in pairs if p['negative']]
    unmet=[]
    if any(a is not b for a in pos for b in neg):
        return {'rule':1,'conclusion':'两个配对方向相反，箱体效果依赖止损口径，不能认定为真实的正贡献或负贡献','unmet_conditions':[]}
    for p in pos:
        if p['a_passes_g2'] and all(d is not None and d>0 for d in p['half_differences']):
            return {'rule':2,'conclusion':'箱体有可交易的正贡献','unmet_conditions':[]}
    if pos:
        for p in pos:
            bad=[]
            for h,d in enumerate(p['half_differences']):
                if d is None or d<=0:
                    bad.append(f'半段 {h+1} 的 D '+('未定义' if d is None else '≤ 0'))
            if not p['a_passes_g2']:
                bad.append('箱体版本本身不能覆盖 17bp 摩擦')
            unmet.append({'a':p['a'],'conditions':bad})
        return {'rule':3,'conclusion':'箱体有统计上可信的相对改进，但未达到可交易正贡献的标准','unmet_conditions':unmet}
    if neg:
        return {'rule':4,'conclusion':'箱体有可信的负贡献','unmet_conditions':[]}
    return {'rule':5,'conclusion':'没有可信证据表明箱体有正或负贡献','unmet_conditions':[]}


def eligible_sets(gates):
    eligible,cost_sensitive=[],[]
    for name in NAMES:
        g=gates[name]
        z_ok=g.get('G8z',False) if VS.VERSIONS[name].trigger_rule=='z' else True
        if all(g[f'G{k}'] for k in range(1,8)) and z_ok:
            eligible.append(name)
        if all(g[f'G{k}'] for k in (1,2,3,4,5,7)) and not g['G6'] and z_ok:
            cost_sensitive.append(name)
    return eligible,cost_sensitive


def _unique_top(names,sharpes):
    finite=[n for n in names if sharpes[n] is not None and np.isfinite(sharpes[n])]
    if len(finite)!=len(names) or not finite:
        return None
    candidate=max(finite,key=sharpes.get)
    if sum(abs(sharpes[n]-sharpes[candidate])<1e-12 for n in finite)>1:
        return None
    return candidate


def _pair_key(a,b):
    return f'{a}--{b}'


def _subset_mean_diff(a,b):
    ma,mb=cluster_point_mean(a),cluster_point_mean(b)
    return ma-mb if ma is not None and mb is not None else None


def evaluate_study(main_results,harsh_results,excess_trades,causal_results,*,context):
    """All frozen §9 diagnostics; CP3 reports these provisionally, not as CP4 approval."""
    if set(main_results)!=set(NAMES) or set(harsh_results)!=set(NAMES) or set(excess_trades)!=set(NAMES):
        raise ValueError('F1/F2 require all ten frozen versions including empty versions')
    if set(causal_results)!={NAMES[4],NAMES[9]}:
        raise ValueError('G8z requires A5 and B5 causal results')
    detail,returns={},{}
    for v,name in enumerate(NAMES):
        result=main_results[name]
        tr=result.trades
        metric=portfolio_metrics(result)
        returns[name]=daily_returns_from_equity(result.daily_equity)
        g2=cluster_mean(tr,ci_seed=seed_for('g2_ci',v),p_seed=seed_for('g2_p',v))
        g3=cluster_mean(excess_trades[name],'excess_bps',ci_seed=seed_for('g3_ci',v),p_seed=seed_for('g3_p',v))
        g4=month_block_sharpe(returns[name],seed=seed_for('g4',v))
        nr=cluster_mean(tr,'net_R',ci_seed=seed_for('net_r_ci',v),p_seed=None)
        point=point_gate_statistics(tr,harsh_results[name].trades,
                                   causal_results[name].trades if name in causal_results else None)
        detail[name]={'metrics':metric,'G2':g2,'G3':g3,'G4':g4,'net_R_interval':nr,
                      'point_gates':point,'note':VS.VERSIONS[name].note}
    families={'F1':holm_family({n:detail[n]['G2']['p'] for n in NAMES},.025),
              'F2':holm_family({n:detail[n]['G3']['p'] for n in NAMES},.025)}
    gates={}
    disclosures=[]
    for name in NAMES:
        d=detail[name]
        nonempty=d['metrics']['n']>0
        g={k:d['point_gates'][k] for k in ('G1','G5','G6','G7')}
        g['G2']=bool(nonempty and families['F1']['tests'][name]['passed'] and d['G2']['t'] is not None)
        g['G3']=bool(nonempty and families['F2']['tests'][name]['passed'] and d['G3']['t'] is not None)
        g['G4']=bool(nonempty and d['G4']['sharpe'] is not None and d['G4']['lo'] is not None and d['G4']['lo']>0)
        if VS.VERSIONS[name].trigger_rule=='z':
            g['G8z']=d['point_gates']['G8z']
        gates[name]=d['gates']=g
        d['undefined_gate_reasons']={gate:d[gate]['reason'] for gate in ('G2','G3','G4') if d[gate]['reason']}
        if not nonempty:
            d['undefined_gate_reasons']['all']='no trades; every gate fails'
        for gate in ('G2','G3'):
            interval_positive=d[gate]['lo'] is not None and d[gate]['lo']>0
            if interval_positive!=g[gate]:
                disclosures.append({'type':'interval_gate_disagreement','version':name,'gate':gate,
                                    'ci_positive':interval_positive,'gate_passed':g[gate]})
        if d['G4']['mc_boundary_sensitive']:
            disclosures.append({'type':'g4_monte_carlo_boundary','version':name,'p0':d['G4']['p0']})
    boxes={}
    for p,(a,b) in enumerate(VS.BOX_SINGLE_VARIABLE_PAIRS):
        ta,tb=main_results[a].trades,main_results[b].trades
        key=_pair_key(a,b)
        metrics=paired_cluster_metrics(ta,tb,ci_seed=seed_for('box_ci',p),p_seed=seed_for('box_p',p))
        halves=[_subset_mean_diff(x,y) for x,y in zip(_halves(ta),_halves(tb))]
        years={str(year):_subset_mean_diff(ta.loc[pd.DatetimeIndex(ta.entry_time).year==year],
                                         tb.loc[pd.DatetimeIndex(tb.entry_time).year==year]) for year in range(2020,2024)}
        boxes[key]={'a':a,'b':b,'metrics':metrics,'half_differences':halves,'year_differences_net_bps':years,
                    'sharpe_difference':paired_month_block_sharpe(returns[a],returns[b],seed=seed_for('box_sharpe',p)),
                    'frequency_note':'A组交易频率远低于B组；同时比较逐笔net_bps、net_R与成交笔数'}
    families['F3_positive']=holm_family({k:d['metrics']['net_bps']['p_positive'] for k,d in boxes.items()},.025)
    families['F3_negative']=holm_family({k:d['metrics']['net_bps']['p_negative'] for k,d in boxes.items()},.025)
    box_rules=[]
    for key,d in boxes.items():
        d['positive']=families['F3_positive']['tests'][key]['passed']
        d['negative']=families['F3_negative']['tests'][key]['passed']
        d['a_passes_g2']=gates[d['a']]['G2']
        box_rules.append(d)
        ci=d['metrics']['net_bps']
        for direction,bound in [('positive',ci['lo'] is not None and ci['lo']>0),
                                ('negative',ci['hi'] is not None and ci['hi']<0)]:
            if bool(bound)!=d[direction]:
                disclosures.append({'type':'interval_gate_disagreement','pair':key,'gate':f'F3_{direction}',
                                    'ci_excludes_zero':bool(bound),'gate_passed':d[direction]})
    inside={}
    for j,name in enumerate((NAMES[5],NAMES[6])):
        tr=main_results[name].trades
        ids=tr.trigger_idx.to_numpy(dtype=int)
        mask=np.asarray(context.box_ok[ids],dtype=bool)
        inside[name]=paired_cluster_mean(tr.loc[mask],tr.loc[~mask],ci_seed=seed_for('inside_ci',j),p_seed=seed_for('inside_p',j))
    stops={}
    for c,(a,b) in enumerate(STOP_PAIRS):
        stops[_pair_key(a,b)]={'a':a,'b':b,**paired_month_block_sharpe(returns[a],returns[b],seed=seed_for('f4',c))}
    families['F4']=holm_family({k:d['p_two_sided'] for k,d in stops.items()},.05,n_boot=B_SHARPE)
    sharpes={n:detail[n]['metrics']['sharpe'] for n in NAMES}
    stop_decisions={}
    for family,members in VS.STOP_FAMILIES.items():
        top=_unique_top(members,sharpes)
        best=None
        if top is not None:
            passed=True
            for other in members:
                if other==top:
                    continue
                key=next(k for k,d in stops.items() if {d['a'],d['b']}=={top,other})
                d=stops[key]
                difference=d['diff'] if d['a']==top else (-d['diff'] if d['diff'] is not None else None)
                passed &= difference is not None and difference>0 and families['F4']['tests'][key]['passed']
            if passed:
                best=top
        stop_decisions[family]={'best_version':best,'best_stop':VS.VERSIONS[best].stop if best else None,
                               'conclusion':f'{best}的止损在该族内最优' if best else '该族内没有可区分的最优止损',
                               'all_sharpes_negative':all(sharpes[n] is not None and sharpes[n]<0 for n in members)}
        if stop_decisions[family]['all_sharpes_negative']:
            stop_decisions[family]['note']='相对比较只说明亏损程度的差别'
    eligible,cost_sensitive=eligible_sets(gates)
    candidate=_unique_top(eligible,sharpes) if len(eligible)>1 else (eligible[0] if eligible else None)
    winner=None
    winner_tests={}
    if len(eligible)==1:
        winner=eligible[0]
    elif len(eligible)>1 and candidate is not None:
        for other in eligible:
            if other==candidate:
                continue
            key=_pair_key(candidate,other)
            winner_tests[key]={'a':candidate,'b':other,**paired_month_block_sharpe(
                returns[candidate],returns[other],seed=seed_for('f5',candidate,other=other))}
    families['F5']=holm_family({k:d['p_two_sided'] for k,d in winner_tests.items()},.025,n_boot=B_SHARPE)
    if winner_tests and all(d['diff'] is not None and d['diff']>0 and families['F5']['tests'][k]['passed']
                            for k,d in winner_tests.items()):
        winner=candidate
    alpha=[n for n in NAMES if VS.VERSIONS[n].group=='B' and all(gates[n][f'G{k}'] for k in (1,2,3))
           and (VS.VERSIONS[n].trigger_rule!='z' or gates[n]['G8z'])]
    b5_calibration_dependence=all(gates[NAMES[9]][f'G{k}'] for k in (1,2,3)) and not gates[NAMES[9]]['G8z']
    for family,value in families.items():
        for test,d in value['tests'].items():
            if d['mc_boundary_sensitive']:
                disclosures.append({'type':'holm_monte_carlo_boundary','family':family,'test':test,**d})
    disclosures.extend([{'type':'cost_sensitive_candidate','version':n,'label':'成本敏感的候选，不具备申请资格'} for n in cost_sensitive])
    decisions={'box':box_decision(box_rules),
               'pure_alpha':{'versions':alpha,'conclusion':'纯时间价格脉冲存在超越匹配随机与17bp摩擦的净Alpha' if alpha else '没有可信证据表明纯时间价格脉冲有超越匹配随机与17bp摩擦的净Alpha',
                             'b5_depends_on_full_is_calibration':b5_calibration_dependence},
               'stops':stop_decisions,
               'qualification':{'eligible_versions':eligible,'candidate':candidate,'winner':winner,'cost_sensitive_versions':cost_sensitive,
                                'conclusion':'没有版本具备申请资格' if not eligible else
                                  (f'{winner}是唯一胜出版本' if winner else '存在多个达标版本，没有唯一胜出')}}
    bests=[x['best_stop'] for x in stop_decisions.values()]
    if bests and bests[0] is not None and all(x==bests[0] for x in bests):
        decisions['same_best_stop_both_families']=bests[0]
    return {'status':'CP3 results; CP4 final review not completed','versions':detail,'families':families,
            'box_comparisons':boxes,'inside_outside_comparisons':inside,'stop_comparisons':stops,
            'winner_comparisons':winner_tests,'decisions':decisions,'disclosures':disclosures,
            'method_notes':['A5/B5为全IS波动标定的稳健性版本，IS内非完全无前视',
                            'F1、F2的族大小均为10，A、B共同校正',
                            'G4按月块95%区间下限>0判定；F5为双侧0.025',
                            '按月块夏普检验在A组式月度集中度下略偏松，CP4须披露；CP3不执行VAL/OOS/ETH']}

def stationary_sharpe(returns, *, seed):
    """Frozen 10-day stationary bootstrap, auxiliary only; one seed/draw matrix."""
    from attribution import _stationary_indices, _ann_sharpe_rows
    _require_full_days(returns)
    x=returns.to_numpy(dtype=float)
    point=annualized_sharpe(x)
    indices=_stationary_indices(len(x),B_SHARPE,10.,seed)
    sample=_ann_sharpe_rows(x[indices])
    good=sample[np.isfinite(sample)]
    out={'sharpe':point,'lo':None,'hi':None,'p0':1.,'p':1.,'p_gt0':None,
         'discarded':int(B_SHARPE-len(good)),'n_days':len(x),'n_boot':B_SHARPE,
         'mean_block':10.,'seed':int(seed),'mcse':0.,'auxiliary_only':True,'reason':None}
    if point is None or not len(good):
        out['reason']='undefined point Sharpe or all stationary resamples discarded'
        return out
    out['lo'],out['hi']=[float(z) for z in np.percentile(good,[2.5,97.5])]
    out['p0']=out['p']=float((good<=0).mean())
    out['p_gt0']=float((good>0).mean())
    out['mcse']=_mcse(out['p0'],B_SHARPE)
    return out


def paired_stationary_sharpe(a,b,*,seed):
    """Same frozen stationary indices for both returns; benchmark_factor=1."""
    from attribution import _stationary_indices, _ann_sharpe_rows
    _require_full_days(a)
    _require_full_days(b)
    if not _utc_index(a.index,'a').equals(_utc_index(b.index,'b')):
        raise ValueError('paired returns must share exactly the same UTC daily index')
    xa,xb=a.to_numpy(dtype=float),b.to_numpy(dtype=float)
    pa,pb=annualized_sharpe(xa),annualized_sharpe(xb)
    point=pa-pb if pa is not None and pb is not None else None
    indices=_stationary_indices(len(xa),B_SHARPE,10.,seed)
    # max(S_ref,1*S_ref) is exactly S_ref, including negative reference Sharpe.
    with np.errstate(invalid='ignore'):
        delta=_ann_sharpe_rows(xa[indices])-_ann_sharpe_rows(xb[indices])
    good=delta[np.isfinite(delta)]
    out={'diff':point,'mean':point,'lo':None,'hi':None,'p':1.,'p_two_sided':1.,
         'p_positive':1.,'p_negative':1.,'p_gt0':None,'benchmark':pb,'benchmark_factor':1.,
         'sharpe_a':pa,'sharpe_b':pb,'discarded':int(B_SHARPE-len(good)),
         'n_days':len(a),'n_boot':B_SHARPE,'mean_block':10.,'seed':int(seed),
         'mcse_positive':0.,'mcse_negative':0.,'mcse_two_sided':0.,
         'auxiliary_only':True,'reason':None}
    if point is None or not len(good):
        out['reason']='undefined paired point Sharpe or all stationary resamples discarded'
        return out
    out['lo'],out['hi']=[float(z) for z in np.percentile(good,[2.5,97.5])]
    out['p_positive']=float((good<=0).mean())
    out['p_negative']=float((good>=0).mean())
    out['p_gt0']=float((good>0).mean())
    out['p']=out['p_two_sided']=min(1.,2*min(out['p_positive'],out['p_negative']))
    for suffix in ('positive','negative','two_sided'):
        out['mcse_'+suffix]=_mcse(out['p_'+suffix],B_SHARPE)
    return out


def _box_sensitivity_statistics(results,*,section=None,cluster='month'):
    ci='daily_box_ci' if cluster=='day' else 'box_ci'
    ps='daily_box_p' if cluster=='day' else 'box_p'
    boxes={}
    for p,(a,b) in enumerate(VS.BOX_SINGLE_VARIABLE_PAIRS):
        if a not in results or b not in results:
            continue
        boxes[_pair_key(a,b)]={'a':a,'b':b,'metrics':paired_cluster_metrics(
            results[a].trades,results[b].trades,ci_seed=seed_for(ci,p,sensitivity=section),
            p_seed=seed_for(ps,p,sensitivity=section),cluster=cluster)}
    families={direction:holm_family({k:d['metrics']['net_bps']['p_'+direction] for k,d in boxes.items()},.025)
              for direction in ('positive','negative')}
    return {'pairs':boxes,'F3':families}


def cp4_auxiliary_statistics(main_results,excess_trades,*,context,primary_summary=None):
    """§11(8): daily trade clusters and 10-day stationary daily-return blocks.

    All values are explicitly auxiliary; no primary gate is overwritten. F5
    uses exactly the primary qualifying set/candidate comparisons. If the
    caller omits the saved primary summary, F5 is reported as unavailable
    rather than re-selecting candidates using an auxiliary statistic.
    """
    if set(main_results)!=set(NAMES) or set(excess_trades)!=set(NAMES):
        raise ValueError('daily auxiliary F1/F2 require all ten versions')
    detail,returns={},{}
    for v,name in enumerate(NAMES):
        tr=main_results[name].trades
        returns[name]=daily_returns_from_equity(main_results[name].daily_equity)
        detail[name]={'G2':cluster_mean(tr,ci_seed=seed_for('daily_g2_ci',v),p_seed=seed_for('daily_g2_p',v),cluster='day'),
            'G3':cluster_mean(excess_trades[name],'excess_bps',ci_seed=seed_for('daily_g3_ci',v),p_seed=seed_for('daily_g3_p',v),cluster='day'),
            'G4':stationary_sharpe(returns[name],seed=seed_for('g4',v,auxiliary=True))}
    families={'F1':holm_family({n:d['G2']['p'] for n,d in detail.items()},.025),
              'F2':holm_family({n:d['G3']['p'] for n,d in detail.items()},.025)}
    box=_box_sensitivity_statistics(main_results,cluster='day')
    inside={}
    for j,name in enumerate((NAMES[5],NAMES[6])):
        tr=main_results[name].trades
        mask=np.asarray(context.box_ok[tr.trigger_idx.to_numpy(dtype=int)],bool)
        inside[name]=paired_cluster_mean(tr.loc[mask],tr.loc[~mask],
            ci_seed=seed_for('daily_inside_ci',j),p_seed=seed_for('daily_inside_p',j),cluster='day')
    box_sharpe={_pair_key(a,b):paired_stationary_sharpe(returns[a],returns[b],seed=seed_for('box_sharpe',p,auxiliary=True))
                 for p,(a,b) in enumerate(VS.BOX_SINGLE_VARIABLE_PAIRS)}
    stops={_pair_key(a,b):paired_stationary_sharpe(returns[a],returns[b],seed=seed_for('f4',c,auxiliary=True))
           for c,(a,b) in enumerate(STOP_PAIRS)}
    families['F4']=holm_family({k:d['p_two_sided'] for k,d in stops.items()},.05,n_boot=B_SHARPE)
    winner={}
    if primary_summary is not None:
        for key,pair in primary_summary.get('winner_comparisons',{}).items():
            a,b=pair['a'],pair['b']
            winner[key]=paired_stationary_sharpe(returns[a],returns[b],seed=seed_for('f5',a,other=b,auxiliary=True))
    families['F5']=holm_family({k:d['p_two_sided'] for k,d in winner.items()},.025,n_boot=B_SHARPE)
    families['F3_positive'],families['F3_negative']=box['F3']['positive'],box['F3']['negative']
    return {'auxiliary_only':True,'primary_gates_unchanged':True,'versions':detail,'families':families,
            'box_comparisons':box['pairs'],'box_sharpe_comparisons':box_sharpe,
            'inside_outside_comparisons':inside,'stop_comparisons':stops,'winner_comparisons':winner,
            'F5_primary_summary_supplied':primary_summary is not None}


def _sensitivity_descriptive_results(results,section):
    out={}
    for name,result in results.items():
        if isinstance(name,tuple):
            version,cost,sizing=name
            key='__'.join(name)
        else:
            version=name
            key=name
        v=NAMES.index(version)
        out[key]={'metrics':portfolio_metrics(result),
            'net_bps_month':cluster_mean(result.trades,ci_seed=seed_for('g2_ci',v,sensitivity=section),
                                         p_seed=seed_for('g2_p',v,sensitivity=section)),
            'net_R_interval':cluster_mean(result.trades,'net_R',
                ci_seed=seed_for('net_r_ci',v,sensitivity=section),p_seed=None)}
    return out


def _defect_sensitivity_statistics(results,harsh_results,excess_trades,retained_trigger_indices):
    """§11(3): remove the flagged ledger rows, retaining their original matched controls."""
    if set(results)!=set(NAMES) or set(harsh_results)!=set(NAMES) or set(excess_trades)!=set(NAMES):
        raise ValueError('defect sensitivity needs complete ten-version families')
    detail={}
    for v,name in enumerate(NAMES):
        excess=excess_trades[name]
        keep=excess.trigger_idx.isin(retained_trigger_indices[name])
        tr=results[name].trades
        g2=cluster_mean(tr,ci_seed=seed_for('g2_ci',v,sensitivity=3),p_seed=seed_for('g2_p',v,sensitivity=3))
        g3=cluster_mean(excess.loc[keep],'excess_bps',ci_seed=seed_for('g3_ci',v,sensitivity=3),
                        p_seed=seed_for('g3_p',v,sensitivity=3))
        g4=month_block_sharpe(daily_returns_from_equity(results[name].daily_equity),seed=seed_for('g4',v,sensitivity=3))
        detail[name]={'metrics':portfolio_metrics(results[name]),'G2':g2,'G3':g3,'G4':g4,
                      'point_gates':point_gate_statistics(tr,harsh_results[name].trades)}
    families={'F1':holm_family({n:d['G2']['p'] for n,d in detail.items()},.025),
              'F2':holm_family({n:d['G3']['p'] for n,d in detail.items()},.025)}
    for name,d in detail.items():
        nonempty=d['metrics']['n']>0
        gates={k:d['point_gates'][k] for k in ('G1','G5','G6','G7')}
        gates['G2']=bool(nonempty and d['G2']['t'] is not None and families['F1']['tests'][name]['passed'])
        gates['G3']=bool(nonempty and d['G3']['t'] is not None and families['F2']['tests'][name]['passed'])
        gates['G4']=bool(nonempty and d['G4']['sharpe'] is not None and d['G4']['lo'] is not None and d['G4']['lo']>0)
        d['sensitivity_gates']=gates
    box=_box_sensitivity_statistics(results,section=3)
    families['F3_positive'],families['F3_negative']=box['F3']['positive'],box['F3']['negative']
    return {'versions':detail,'families':families,'box_D':box['pairs'],
            'matched_control_policy':'original matched rows retained by surviving trigger indices',
            'primary_gates_unchanged':True}


def make_cp4_statistics_hook(excess_trades,*,primary_summary=None):
    """Bind saved CP3 matched rows/summary; return the §11 sensitivity callback.

    No market data is loaded here. The future caller supplies CP4 simulation
    results in the payload protocol of cp3_sensitivity.run_cp4_sensitivities.
    Existing outcomes and matched rows are not mutated. F5's auxiliary version
    preserves the primary comparison set; no sensitivity may confer eligibility.
    """
    def hook(payload):
        section=int(payload['section'])
        results=payload['results']
        if section==1:
            return {'reconciliation':payload['reconciliation'],'primary_gates_unchanged':True}
        if section==3:
            return _defect_sensitivity_statistics(results,payload['harsh_results'],excess_trades,
                                                  payload['retained_trigger_indices'])
        if section==8:
            return cp4_auxiliary_statistics(results,excess_trades,context=payload['context'],primary_summary=primary_summary)
        if section==9:
            out={}
            for name,result in results.items():
                v=NAMES.index(name)
                out[name]={'metrics':portfolio_metrics(result,appendix=True),
                    'net_bps_month':cluster_mean(result.trades,ci_seed=seed_for('appendix_ci',v),
                                                 p_seed=seed_for('appendix_p',v),appendix=True)}
            return {'versions':out,'matched_controls':False,'enters_primary_gates':False}
        if section not in (2,4,5,6,7):
            raise ValueError('CP4 statistics hook only implements frozen §11 items 1–9')
        out={'versions':_sensitivity_descriptive_results(results,section),'primary_gates_unchanged':True}
        if section==5:
            out['G8z_point_means']={name:cluster_point_mean(result.trades) for name,result in results.items()}
            out['pct_z_comparison']={name:{metric:cluster_point_mean(tr,metric) for metric in ('gross_bps','net_bps','net_R')}
                                    for name,tr in payload['pct_z_comparison_trades'].items()}
        return out
    return hook
