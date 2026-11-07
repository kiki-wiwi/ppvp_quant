"""Frozen calendar-month inference and candidate gates for the trend study."""
from __future__ import annotations

from hashlib import sha256
import math
from numbers import Integral
import sys

import numpy as np
import pandas as pd

from breakout_study.statistics import webb_test
from .config import (
    HALF, INITIAL_CAPITAL, IS_END, MAIN_START, MONTH_DRAWS, MONTH_SEED,
    N_DAYS, N_MONTHS, P_VERSION, T_VERSION, WEBB_DRAWS, WEBB_SEED,
)

FAMILY = ('T', 'D')
ALPHA = .05
MIN_STATE_DAYS = 60
METRICS = ('T_daily_mean', 'P_daily_mean', 'D_daily_mean', 'T_sharpe', 'P_sharpe')


def _validate_calendar(index, expected, label):
    if (not isinstance(index, pd.DatetimeIndex) or str(index.tz) != 'UTC'
            or not index.as_unit('ns').equals(expected)):
        raise ValueError(f'{label}: exactly {N_DAYS} ordered UTC calendar days required')


def _daily_returns(series, calendar, label):
    if not isinstance(series, pd.Series):
        raise ValueError(f'{label}: a daily return Series is required')
    _validate_calendar(series.index, calendar, label)
    if not (pd.api.types.is_float_dtype(series.dtype) or pd.api.types.is_integer_dtype(series.dtype)):
        raise ValueError(f'{label}: real numeric daily returns required')
    values = series.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all():
        raise ValueError(f'{label}: finite daily returns required')
    if (values <= -1).any():
        raise ValueError(f'{label}: daily account equity must remain positive')
    with np.errstate(over='ignore', under='ignore'):
        equity = INITIAL_CAPITAL * np.cumprod(1. + values)
    if not np.isfinite(equity).all() or (equity <= 0).any():
        raise ValueError(f'{label}: daily account equity must remain finite and positive')
    return values, float(equity[-1] / INITIAL_CAPITAL - 1.)


def _webb(sums, counts, weights, *, has_trades):
    return webb_test(sums, counts, weights, two_sided=False, has_trades=has_trades)


def _holm(raw):
    if set(raw) != set(FAMILY) or any(not math.isfinite(p) or not 0. <= p <= 1. for p in raw.values()):
        raise ValueError('Holm requires finite T and D p values between zero and one')
    ranked = sorted(FAMILY, key=raw.__getitem__)
    corrected = {}
    running = 0.
    for rank, name in enumerate(ranked, start=1):
        multiplier = len(FAMILY) - rank + 1
        p = raw[name]
        threshold = ALPHA / multiplier
        running = min(1., max(running, p * multiplier))
        mcse = math.sqrt(p * (1. - p) / WEBB_DRAWS)
        corrected[name] = {
            'rank': rank, 'holm_threshold': threshold, 'p_holm': running,
            'reject': running <= ALPHA, 'mcse': mcse,
            'mc_boundary_sensitive': abs(p - threshold) < 2. * mcse,
        }
    return corrected


def _interval_metrics(sample):
    means = sample.mean(axis=0)
    both = sample[:, :2]
    sharpe = np.full(2, np.nan)
    if len(sample) > 1:
        # Detect exact constants before variance roundoff can turn them into a defined Sharpe.
        varying = both.min(axis=0) != both.max(axis=0)
        sd = both.std(axis=0, ddof=1)
        np.divide(math.sqrt(365.) * means[:2], sd, out=sharpe, where=varying & (sd > 0))
    return np.concatenate((means, sharpe))


def _month_intervals(daily, counts, draws):
    """Concatenate complete chronological months with the same draws for T, P and D."""
    months = np.split(daily, np.cumsum(counts)[:-1])
    resamples = np.array([
        _interval_metrics(np.concatenate([months[index] for index in draw]))
        for draw in draws
    ])
    estimates = _interval_metrics(daily)
    intervals = []
    for estimate, column in zip(estimates, resamples.T):
        valid = column[np.isfinite(column)]
        lower, upper = np.quantile(valid, [.025, .975], method='linear') if len(valid) else (np.nan, np.nan)
        intervals.append({
            'estimate': estimate, 'lower': lower, 'upper': upper,
            'resamples': len(column), 'valid_resamples': len(valid),
            'discarded': len(column) - len(valid),
        })
    return pd.DataFrame(intervals, index=pd.Index(METRICS, name='metric'))


def _gates(daily, states, cumulative_returns, monthly, tests):
    months = daily.index.strftime('%Y-%m')
    first_half = daily.index < HALF
    halves = (first_half, ~first_half)
    state_counts = [
        {'on': int(states[mask].sum()), 'off': int((~states[mask]).sum())}
        for mask in halves
    ]
    coverage = len(daily) == N_DAYS and len(monthly) == N_MONTHS and all(
        count >= MIN_STATE_DAYS for half in state_counts for count in half.values()
    )

    t_mean, d_mean = (float(daily[name].mean()) for name in FAMILY)
    t_holm, d_holm = (float(tests.loc[name, 'p_holm']) for name in FAMILY)
    t_cumulative = cumulative_returns['main']['T']

    half_means = [
        {name: float(daily.loc[mask, name].mean()) for name in FAMILY}
        for mask in halves
    ]

    # The monthly table is in chronological order. Stable sorting preserves it at exact ties.
    removed = monthly['D_sum'].sort_values(ascending=False, kind='stable').index[:3].tolist()
    remaining = ~months.isin(removed)
    remaining_t, remaining_d = (float(daily.loc[remaining, name].mean()) for name in FAMILY)
    harsh_t = cumulative_returns['harsh']['T']
    harsh_d = float(daily['D_harsh'].mean())

    gates = [
        (coverage, {
            'day_count': len(daily), 'month_count': len(monthly), 'minimum_state_days': MIN_STATE_DAYS,
            'state_counts_by_half': state_counts, 'status': '通过' if coverage else '状态覆盖不足',
        }),
        (t_cumulative > 0 and t_mean > 0 and t_holm <= ALPHA, {
            'T_cumulative_return': t_cumulative, 'T_mean': t_mean, 'T_p_holm': t_holm,
        }),
        (d_mean > 0 and d_holm <= ALPHA, {'D_mean': d_mean, 'D_p_holm': d_holm}),
        (all(mean > 0 for half in half_means for mean in half.values()), {'means_by_half': half_means}),
        (remaining_t > 0 and remaining_d > 0, {
            'removed_months': removed, 'remaining_days': int(remaining.sum()),
            'remaining_T_mean': remaining_t, 'remaining_D_mean': remaining_d,
        }),
        (harsh_t > 0 and harsh_d > 0, {'harsh_T_cumulative_return': harsh_t, 'harsh_D_mean': harsh_d}),
    ]
    return pd.DataFrame(gates, columns=['passed', 'details'],
                        index=pd.Index(['V1', 'V2', 'V3', 'V4', 'V5', 'V6'], name='gate'))


def _classification(gates):
    failed = gates.index[~gates['passed']].tolist()
    if not failed:
        return '在本规则下具有净增量的BTC IS探索性候选'
    if failed == ['V6']:
        return '成本敏感，仍无候选资格'
    return '本规则下无候选资格'


def analyze(t_daily, p_daily, harsh_t_daily, harsh_p_daily, features, t_order_count):
    """Run both one-sided tests, paired intervals and all six gates on the fixed UTC calendar."""
    calendar = pd.date_range(MAIN_START, IS_END, freq='D', inclusive='left').as_unit('ns')
    if not isinstance(features, pd.DataFrame):
        raise ValueError('features: a daily feature DataFrame is required')
    _validate_calendar(features.index, calendar, 'features')
    if (list(features.columns).count('g') != 1 or not pd.api.types.is_bool_dtype(features['g'].dtype)
            or features['g'].isna().any()):
        raise ValueError('features: a nonmissing boolean decision state g is required')
    if not isinstance(t_order_count, Integral) or isinstance(t_order_count, bool) or t_order_count < 0:
        raise ValueError('t_order_count: a nonnegative integer execution count is required')

    t, t_total = _daily_returns(t_daily, calendar, 'main/T')
    p, p_total = _daily_returns(p_daily, calendar, 'main/P')
    harsh_t, harsh_t_total = _daily_returns(harsh_t_daily, calendar, 'harsh/T')
    harsh_p, harsh_p_total = _daily_returns(harsh_p_daily, calendar, 'harsh/P')
    daily = pd.DataFrame({
        'T': t, 'P': p, 'D': t - p,
        'T_harsh': harsh_t, 'P_harsh': harsh_p, 'D_harsh': harsh_t - harsh_p,
    }, index=calendar)
    cumulative_returns = {'main': {'T': t_total, 'P': p_total}, 'harsh': {'T': harsh_t_total, 'P': harsh_p_total}}

    monthly = daily.groupby(daily.index.strftime('%Y-%m'), sort=True).agg(
        day_count=('T', 'size'), T_sum=('T', 'sum'), P_sum=('P', 'sum'), D_sum=('D', 'sum'),
    ).rename_axis('month')
    if len(calendar) != N_DAYS or len(monthly) != N_MONTHS:
        raise ValueError(f'expected the fixed {N_DAYS}-day, {N_MONTHS}-month study calendar')
    counts = monthly['day_count'].to_numpy()

    points = np.array([-math.sqrt(1.5), -1., -math.sqrt(.5), math.sqrt(.5), 1., math.sqrt(1.5)])
    weights = np.random.Generator(np.random.PCG64(WEBB_SEED)).choice(points, size=(WEBB_DRAWS, N_MONTHS))
    draws = np.random.Generator(np.random.PCG64(MONTH_SEED)).integers(0, N_MONTHS, size=(MONTH_DRAWS, N_MONTHS))
    matrices = {'webb_weights': weights, 'month_indices': draws}
    tests = {
        name: _webb(monthly[f'{name}_sum'], counts, weights, has_trades=t_order_count > 0)
        for name in FAMILY
    }
    tests['T']['hypothesis'] = f'mean({T_VERSION})<=0'
    tests['D']['hypothesis'] = f'mean({T_VERSION}-{P_VERSION})<=0'
    if np.array_equal(t, p):
        tests['D']['forced_p_one_reasons'].append('identical_main_daily_returns')
    holm = _holm({name: row['p_raw'] for name, row in tests.items()})
    for name in FAMILY:
        tests[name].update(holm[name])
    tests = pd.DataFrame.from_dict(tests, orient='index').rename_axis('test')
    intervals = _month_intervals(daily[['T', 'P', 'D']].to_numpy(), counts, draws)
    gates = _gates(daily, features['g'], cumulative_returns, monthly, tests)
    return {
        'tests': tests,
        'intervals': intervals,
        'gates': gates,
        'monthly': monthly,
        'classification': _classification(gates),
        'matrices': matrices,
        'metadata': {
            'calendar': {
                'start': MAIN_START.isoformat(), 'end_exclusive': IS_END.isoformat(),
                'half_split': HALF.isoformat(), 'days': N_DAYS, 'months': N_MONTHS,
                'month_order': monthly.index.tolist(),
            },
            'family': list(FAMILY), 'family_size': len(FAMILY), 'alpha': ALPHA,
            'cumulative_returns': cumulative_returns,
            't_order_count': int(t_order_count),
            'matrices': {
                name: {
                    'shape': list(matrix.shape), 'dtype': str(matrix.dtype), 'order': 'C',
                    'byte_order': sys.byteorder, 'seed': seed,
                    'sha256': sha256(matrix.tobytes(order='C')).hexdigest(),
                }
                for (name, matrix), seed in zip(matrices.items(), (WEBB_SEED, MONTH_SEED))
            },
            'generator': 'NumPy Generator(PCG64)',
            'mc_boundary_sensitive_tests': [
                name for name in FAMILY if holm[name]['mc_boundary_sensitive']
            ],
            'cluster_limitation': 'Month clustering permits within-month dependence. Cross-month '
                                  'dependence and the finite set of 46 months limit inference.',
            'multiplicity_limitation': 'Holm controls the two tests in this study. Prior IS studies '
                                       'can contribute selection bias.',
        },
    }
