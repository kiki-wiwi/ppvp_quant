"""Fixed R5 paired inference on each complete venue/window calendar.

No file access, seed override, reduced-draw option, or cross-venue winner rule.
The two-member Holm family is descriptive within one venue and one window.
"""
from hashlib import sha256
import math
from numbers import Integral
import sys
import numpy as np
import pandas as pd
from breakout_study.statistics import webb_test
from trend_hold_study.statistics import _holm, _month_intervals, FAMILY, ALPHA, MIN_STATE_DAYS
from .config import INITIAL_CAPITAL, WEBB_SEED, WEBB_DRAWS, MONTH_SEED, MONTH_DRAWS, DECLARATION

HALF = pd.Timestamp('2022-01-01',tz='UTC')
IS_END = pd.Timestamp('2024-01-01',tz='UTC')


def _calendar(t_daily):
    if not isinstance(t_daily,pd.Series) or not isinstance(t_daily.index,pd.DatetimeIndex):
        raise ValueError('calendar: a UTC daily return Series is required')
    index = t_daily.index
    if not len(index) or str(index.tz) != 'UTC' or index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError('calendar: unique ordered UTC days required')
    index = index.as_unit('ns')
    expected = pd.date_range(index[0],index[-1],freq='D').as_unit('ns')
    if not index.equals(expected) or (index != index.normalize()).any():
        raise ValueError('calendar: complete consecutive UTC midnight days required')
    end = index[-1]+pd.Timedelta(days=1)
    if index[0].day != 1 or end.day != 1:
        raise ValueError('calendar: complete natural months required')
    if len(index.strftime('%Y-%m').unique()) < 2:
        raise ValueError('calendar: at least two complete natural months required')
    if index[0] < IS_END <= index[-1]:
        raise ValueError('calendar: IS and Post-IS must be separate accounts')
    return index


def _validate_calendar(index,expected,label):
    if not isinstance(index,pd.DatetimeIndex) or str(index.tz) != 'UTC' or not index.as_unit('ns').equals(expected):
        raise ValueError(f'{label}: exact paired UTC calendar required')


def _daily_returns(series,calendar,label):
    if not isinstance(series,pd.Series): raise ValueError(f'{label}: return Series required')
    _validate_calendar(series.index,calendar,label)
    if not (pd.api.types.is_float_dtype(series.dtype) or pd.api.types.is_integer_dtype(series.dtype)):
        raise ValueError(f'{label}: numeric daily returns required')
    values = series.to_numpy(dtype=float,na_value=np.nan)
    if not np.isfinite(values).all(): raise ValueError(f'{label}: finite daily returns required')
    if (values < -1).any(): raise ValueError(f'{label}: return below -1 is invalid')
    zero = np.flatnonzero(values == -1)
    if len(zero) and np.any(values[zero[0]+1:] != 0):
        raise ValueError(f'{label}: zero equity requires all subsequent returns zero')
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        equity = INITIAL_CAPITAL*np.cumprod(1.+values)
    positive_end = zero[0] if len(zero) else len(values)
    if not np.isfinite(equity).all() or np.any(equity[:positive_end] <= 0):
        raise ValueError(f'{label}: nonfinite equity or numerical underflow')
    return values,float(equity[-1]/INITIAL_CAPITAL-1.)


def _diagnostic_gates(daily,states,cumulative_returns,monthly,tests):
    """IS-only V1–V6 diagnostics with the inherited fixed half split."""
    months = daily.index.strftime('%Y-%m')
    masks = (daily.index < HALF,daily.index >= HALF)
    state_counts = [{'on':int(states[mask].sum()),'off':int((~states[mask]).sum())} for mask in masks]
    expected_days = pd.date_range(daily.index[0],daily.index[-1],freq='D').as_unit('ns')
    expected_months = expected_days.strftime('%Y-%m').unique()
    coverage = (daily.index.equals(expected_days) and len(monthly)==len(expected_months)
                and int(monthly['day_count'].sum())==len(daily)
                and all(count>=MIN_STATE_DAYS for half in state_counts for count in half.values()))
    t_mean,d_mean = (float(daily[n].mean()) for n in FAMILY)
    t_holm,d_holm = (float(tests.loc[n,'p_holm']) for n in FAMILY)
    half_means = [{n:float(daily.loc[mask,n].mean()) if mask.any() else None for n in FAMILY} for mask in masks]
    removed = monthly['D_sum'].sort_values(ascending=False,kind='stable').index[:3].tolist()
    remain = ~months.isin(removed)
    remaining_t,remaining_d = (float(daily.loc[remain,n].mean()) if remain.any() else None for n in FAMILY)
    harsh_t = cumulative_returns['harsh']['T']
    harsh_d = float(daily['D_harsh'].mean())
    t_cumulative = cumulative_returns['main']['T']
    gates = [
        (coverage,{'day_count':len(daily),'month_count':len(monthly),
                   'minimum_state_days':MIN_STATE_DAYS,'state_counts_by_half':state_counts,
                   'status':'通过' if coverage else '状态覆盖不足'}),
        (t_cumulative>0 and t_mean>0 and t_holm<=ALPHA,
         {'T_cumulative_return':t_cumulative,'T_mean':t_mean,'T_p_holm':t_holm}),
        (d_mean>0 and d_holm<=ALPHA,{'D_mean':d_mean,'D_p_holm':d_holm}),
        (all(x is not None and x>0 for half in half_means for x in half.values()),{'means_by_half':half_means}),
        (remaining_t is not None and remaining_d is not None and remaining_t>0 and remaining_d>0,
         {'removed_months':removed,'remaining_days':int(remain.sum()),
          'remaining_T_mean':remaining_t,'remaining_D_mean':remaining_d}),
        (harsh_t>0 and harsh_d>0,{'harsh_T_cumulative_return':harsh_t,'harsh_D_mean':harsh_d}),
    ]
    return pd.DataFrame(gates,columns=['passed','details'],index=pd.Index(['V1','V2','V3','V4','V5','V6'],name='gate'))


def analyze(t_daily,p_daily,harsh_t_daily,harsh_p_daily,features,t_order_count):
    """Apply unchanged fixed draws and paired procedures to the supplied calendar."""
    calendar = _calendar(t_daily)
    if not isinstance(features,pd.DataFrame): raise ValueError('features: DataFrame required')
    _validate_calendar(features.index,calendar,'features')
    if list(features.columns).count('g')!=1 or not pd.api.types.is_bool_dtype(features['g']) or features['g'].isna().any():
        raise ValueError('features: nonmissing boolean g required')
    if not isinstance(t_order_count,Integral) or isinstance(t_order_count,bool) or t_order_count<0:
        raise ValueError('t_order_count: nonnegative integer required')
    t,t_total = _daily_returns(t_daily,calendar,'main/T')
    p,p_total = _daily_returns(p_daily,calendar,'main/P')
    ht,ht_total = _daily_returns(harsh_t_daily,calendar,'harsh/T')
    hp,hp_total = _daily_returns(harsh_p_daily,calendar,'harsh/P')
    daily = pd.DataFrame({'T':t,'P':p,'D':t-p,'T_harsh':ht,'P_harsh':hp,'D_harsh':ht-hp},index=calendar)
    cumulative = {'main':{'T':t_total,'P':p_total},'harsh':{'T':ht_total,'P':hp_total}}
    monthly = daily.groupby(calendar.strftime('%Y-%m'),sort=True).agg(
        day_count=('T','size'),T_sum=('T','sum'),P_sum=('P','sum'),D_sum=('D','sum')).rename_axis('month')
    counts = monthly['day_count'].to_numpy()
    g = len(monthly)
    points = np.array([-math.sqrt(1.5),-1.,-math.sqrt(.5),math.sqrt(.5),1.,math.sqrt(1.5)])
    weights = np.random.Generator(np.random.PCG64(WEBB_SEED)).choice(points,size=(WEBB_DRAWS,g))
    draws = np.random.Generator(np.random.PCG64(MONTH_SEED)).integers(0,g,size=(MONTH_DRAWS,g))
    matrices = {'webb_weights':weights,'month_indices':draws}
    tests = {n:webb_test(monthly[f'{n}_sum'],counts,weights,two_sided=False,has_trades=t_order_count>0) for n in FAMILY}
    tests['T']['hypothesis']='mean(T)<=0'
    tests['D']['hypothesis']='mean(T-P)<=0'
    if np.array_equal(t,p): tests['D']['forced_p_one_reasons'].append('identical_main_daily_returns')
    holm = _holm({n:row['p_raw'] for n,row in tests.items()})
    for n in FAMILY: tests[n].update(holm[n])
    tests = pd.DataFrame.from_dict(tests,orient='index').rename_axis('test')
    intervals = _month_intervals(daily[['T','P','D']].to_numpy(),counts,draws)
    is_is = calendar[-1] < IS_END
    if is_is:
        gates = _diagnostic_gates(daily,features['g'],cumulative,monthly,tests)
        classification = 'IS同构诊断通过；不构成自动升级' if gates['passed'].all() else 'IS同构诊断未全部通过'
    else:
        gates = pd.DataFrame({'passed':pd.array([pd.NA]*6,dtype='boolean'),
            'details':[{'applicable':False,'reason':'Post-IS descriptive only; no invented 2022 halves'} for _ in range(6)]},
            index=pd.Index(['V1','V2','V3','V4','V5','V6'],name='gate'))
        classification = '用户授权的描述性样本外评估；无自动晋级'
    disagreement = []
    for name,metric in [('T','T_daily_mean'),('D','D_daily_mean')]:
        lower = intervals.loc[metric,'lower']
        supports = bool(np.isfinite(lower) and lower>0)
        disagreement.append({'hypothesis':name,'interval_supports_positive_mean':supports,
            'holm_rejects':bool(tests.loc[name,'reject']),'disagrees':supports!=bool(tests.loc[name,'reject'])})
    return {'tests':tests,'intervals':intervals,'gates':gates,'daily':daily,'monthly':monthly,
        'classification':classification,'matrices':matrices,'metadata':{
            'declaration':DECLARATION,'automatic_promotion':False,
            'calendar':{'start':calendar[0].isoformat(),
                'end_exclusive':(calendar[-1]+pd.Timedelta(days=1)).isoformat(),
                'half_split':HALF.isoformat() if is_is else None,'days':len(calendar),
                'months':g,'month_order':monthly.index.tolist()},
            'family':list(FAMILY),'family_size':len(FAMILY),'alpha':ALPHA,
            'cumulative_returns':cumulative,'t_order_count':int(t_order_count),
            'matrices':{name:{'shape':list(matrix.shape),'dtype':str(matrix.dtype),
                'order':'C','byte_order':sys.byteorder,'seed':seed,
                'sha256':sha256(matrix.tobytes(order='C')).hexdigest()}
                for (name,matrix),seed in zip(matrices.items(),(WEBB_SEED,MONTH_SEED))},
            'generator':'NumPy Generator(PCG64)',
            'mc_boundary_sensitive_tests':[n for n in FAMILY if holm[n]['mc_boundary_sensitive']],
            'interval_p_disagreements':disagreement,
            'zero_equity_rule':'positive-to-zero return -1; subsequent zero-to-zero return 0; no resurrection',
            'cluster_limitation':f'Month clustering permits within-month dependence; cross-month dependence and {g} months limit inference.',
            'multiplicity_limitation':'Holm is a two-test descriptive reference within one venue/window; it does not control a global cross-venue/window family. Prior IS selection remains.',
        }}
