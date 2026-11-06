"""Frozen conditional inference and U1–U7 gates (approved plan §§6–9).

``analyze`` accepts the complete registered calendar and has no seed, draw-count,
family, or significance override. Pure numerical helpers accept small arrays for
independent synthetic verification; none loads market data or draws randomness.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd

from .config import (HALF, INITIAL_CAPITAL, IS_END, LAST_ENTRY, MAIN_START,
                     MONTH_DRAWS, MONTH_SEED, N_DAYS, N_MONTHS, R_COUNT,
                     R_VERSIONS, WEBB_DRAWS, WEBB_SEED)
from .scope import ScopeError

FAMILY = ('S', 'D')
ALPHA = .05


def _finite(value):
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _calendar():
    return pd.date_range(MAIN_START, IS_END, freq='D', inclusive='left')


def _months():
    return pd.date_range(MAIN_START, IS_END, freq='MS', inclusive='left').strftime('%Y-%m')


def _trade_times(trades, column):
    try:
        idx = pd.DatetimeIndex(trades[column])
    except (TypeError, ValueError) as exc:
        raise ScopeError(f'{column}: timestamps required') from exc
    if not len(idx):
        return pd.DatetimeIndex([], tz='UTC')
    if idx.tz is None or idx.hasnans:
        raise ScopeError(f'{column}: finite timezone-aware timestamps required')
    return idx.tz_convert('UTC')


def _numbers(values, label):
    try:
        result = np.asarray(values, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ScopeError(f'{label}: numeric values required') from exc
    if not np.isfinite(result).all():
        raise ScopeError(f'{label}: nonfinite values are forbidden')
    return result


def _random_counts(frame):
    return frame['trade_count'] if isinstance(frame, pd.DataFrame) else frame


def validate_inputs(s_daily_returns, r_daily_returns, s_trades, harsh_daily_returns,
                    harsh_trades, strata, random_trade_counts):
    """Stop on incomplete pairing, invalid inputs, or failed account equity."""
    expected = _calendar().as_unit('ns')
    if not isinstance(r_daily_returns, pd.DataFrame) or list(r_daily_returns.columns) != list(R_VERSIONS):
        raise ScopeError('exactly ordered R_000 through R_199 daily columns required')
    for label, values in [('S', s_daily_returns), ('R', r_daily_returns), ('harsh', harsh_daily_returns)]:
        if label != 'R' and not isinstance(values, pd.Series):
            raise ScopeError(f'{label}: a daily Series is required')
        idx = values.index
        if not isinstance(idx, pd.DatetimeIndex) or str(idx.tz) != 'UTC' or not idx.as_unit('ns').equals(expected):
            raise ScopeError(f'{label}: exactly 1,401 ascending UTC calendar days in 46 months required')
        a = _numbers(values, f'{label} daily returns')
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            equity = INITIAL_CAPITAL*np.cumprod(1+a, axis=0)
        if (a <= -1).any() or (equity <= 0).any() or not np.isfinite(equity).all():
            raise ScopeError(f'{label}: nonpositive or nonfinite account equity; stop inference')
    for label, trades, daily in [('S', s_trades, s_daily_returns), ('harsh', harsh_trades, harsh_daily_returns)]:
        if not isinstance(trades, pd.DataFrame) or not {'entry_time', 'exit_time', 'net_bps'} <= set(trades.columns):
            raise ScopeError(f'{label}: entry_time, exit_time and net_bps ledger columns required')
        entries, exits = _trade_times(trades, 'entry_time'), _trade_times(trades, 'exit_time')
        _numbers(trades.net_bps, f'{label} net_bps')
        if len(entries):
            if entries.has_duplicates or not entries.is_monotonic_increasing:
                raise ScopeError(f'{label}: unique ascending entries required')
            if (entries < MAIN_START).any() or (entries > LAST_ENTRY).any() or (exits < entries).any() or (exits >= IS_END).any():
                raise ScopeError(f'{label}: invalid trade scope or time order')
            if (entries[1:] < exits[:-1]).any():
                raise ScopeError(f'{label}: overlapping positions are forbidden')
        elif np.any(daily.to_numpy(dtype=float) != 0):
            raise ScopeError(f'{label}: no-trade account must retain all zero daily returns')
    if not isinstance(strata, pd.DataFrame) or not {'month', 'utc_hour', 'vol_bin', 'N', 'n'} <= set(strata.columns):
        raise ScopeError('strata requires month, utc_hour, vol_bin, N, n')
    if strata.duplicated(['month', 'utc_hour', 'vol_bin']).any() or not strata.month.isin(_months()).all():
        raise ScopeError('strata must have unique keys in the registered months')
    for name, lower, upper in [('utc_hour', 0, 23), ('vol_bin', 0, 5), ('N', 0, None), ('n', 0, None)]:
        values = _numbers(strata[name], f'strata {name}')
        if (values != np.floor(values)).any() or (values < lower).any() or (upper is not None and (values > upper).any()):
            raise ScopeError(f'strata {name}: invalid integer counts or labels')
    if (strata.n > strata.N).any():
        raise ScopeError('stratum n exceeds N')
    if not isinstance(random_trade_counts, (pd.Series, pd.DataFrame)):
        raise ScopeError('random trade counts must be a Series or summary DataFrame')
    if isinstance(random_trade_counts, pd.DataFrame) and 'trade_count' not in random_trade_counts:
        raise ScopeError('random summary requires trade_count')
    if list(random_trade_counts.index) != list(R_VERSIONS):
        raise ScopeError('random trade counts must be ordered R_000 through R_199')
    counts = _numbers(_random_counts(random_trade_counts), 'random trade counts')
    if (counts < 0).any() or (counts != np.floor(counts)).any():
        raise ScopeError('random trade counts must be nonnegative integers')
    if isinstance(random_trade_counts, pd.DataFrame) and 'exposure_fraction' in random_trade_counts:
        exposures = _numbers(random_trade_counts.exposure_fraction, 'random exposure fraction')
        if (exposures < 0).any() or (exposures > 1).any():
            raise ScopeError('random exposure fraction outside [0,1]')
    parents = int(strata.n.sum())
    if len(s_trades) > parents or len(harsh_trades) > parents or (counts > parents).any():
        raise ScopeError('actual trades cannot exceed the frozen parent opportunities')
    if np.any(r_daily_returns.to_numpy(dtype=float)[:, counts == 0] != 0):
        raise ScopeError('zero-trade random accounts must retain zero daily returns')


def webb_test(sums, counts, weights, *, has_trades):
    """Positive single-tail restricted Webb test using a supplied shared matrix."""
    sums, counts, weights = [np.asarray(x, dtype=float) for x in (sums, counts, weights)]
    g = len(sums)
    if g < 2 or counts.shape != sums.shape or weights.ndim != 2 or weights.shape[1] != g or len(weights) == 0:
        raise ValueError('Webb requires matching month arrays and a nonempty weight matrix')
    if not all(np.isfinite(x).all() for x in (sums, counts, weights)) or (counts <= 0).any():
        raise ValueError('Webb requires finite inputs and positive month day counts')
    n = float(counts.sum())
    mean = float(sums.sum()/n)
    residual = sums-mean*counts
    se = math.sqrt(g/(g-1)*float(residual @ residual))/n
    observed = mean/se if se > 0 else None
    exceedances, zero_se = 0, 0
    for start in range(0, len(weights), 4096):
        null_sums = weights[start:start+4096]*sums
        null_means = null_sums.sum(axis=1)/n
        deviations = null_sums-null_means[:, None]*counts
        null_se = np.sqrt(g/(g-1)*np.einsum('ij,ij->i', deviations, deviations))/n
        zero_se += int(np.count_nonzero(null_se == 0))
        null_t = np.divide(null_means, null_se, out=np.zeros_like(null_means), where=null_se > 0)
        if observed is not None:
            exceedances += int(np.count_nonzero(null_t >= observed))
    reasons = []
    if not has_trades:
        reasons.append('no_trades_in_required_path')
    if se == 0:
        reasons.append('observed_zero_cluster_se')
    p = 1. if reasons else (1+exceedances)/(len(weights)+1)
    return {'mean': mean, 'se': se, 't': observed, 'p_raw': p,
            'mcse_p': math.sqrt(p*(1-p)/len(weights)), 'tail': 'one_sided_positive',
            'bootstrap_replicates': len(weights), 'finite_mc_denominator': len(weights)+1,
            'tail_exceedances': exceedances if not reasons else None,
            'zero_bootstrap_se_count': zero_se, 'forced_p_one_reasons': reasons}


def holm_two(raw_p):
    """Always retain S and D; exact ties place S first."""
    if set(raw_p) != set(FAMILY) or any(not math.isfinite(float(p)) or not 0 <= p <= 1 for p in raw_p.values()):
        raise ValueError('Holm requires exactly two finite S/D p values in [0,1]')
    order = sorted(FAMILY, key=lambda name: (raw_p[name], FAMILY.index(name)))
    result, running = {}, 0.
    for i, name in enumerate(order):
        multiplier = 2-i
        running = min(1., max(running, multiplier*float(raw_p[name])))
        result[name] = {'rank': i+1, 'p_holm': running, 'holm_threshold': ALPHA/multiplier,
                        'reject': running <= ALPHA}
    return result


def _interval(values, estimate):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return {'estimate': _finite(estimate),
            'lower': _finite(np.quantile(finite, .025, method='linear')) if len(finite) else None,
            'upper': _finite(np.quantile(finite, .975, method='linear')) if len(finite) else None,
            'resamples': len(values), 'valid_resamples': len(finite),
            'undefined_resamples': len(values)-len(finite),
            'fraction_below_zero': float(np.mean(finite < 0)) if len(finite) else None}


def _daily_sharpe(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.all(values == values[0]):
        return None
    sd = float(values.std(ddof=1))
    return math.sqrt(365)*float(values.mean())/sd if sd > 0 else None


def month_intervals(month_data, draws):
    """Shared whole-month means via sufficient sums; direct daily Sharpe.

    One frequency matrix applies to S and every random account. This is exactly
    concatenating the selected months, without a draws × days × paths allocation.
    S Sharpe alone uses each concatenated daily sample and direct ddof=1 std,
    preserving positive variances even when raw second moments would cancel.
    """
    draws = np.asarray(draws)
    counts = np.asarray(month_data['counts'], dtype=float)
    g = len(counts)
    if draws.ndim != 2 or draws.shape[1] != g or not np.issubdtype(draws.dtype, np.integer) or (draws < 0).any() or (draws >= g).any():
        raise ValueError('month draws must contain a complete in-range month resample per row')
    frequencies = np.zeros((len(draws), g), dtype=np.int64)
    np.add.at(frequencies, (np.arange(len(draws))[:, None], draws), 1)
    picked = lambda key: frequencies @ np.asarray(month_data[key], dtype=float)
    n, s_total = picked('counts'), picked('sums')
    s_mean = s_total/n
    # All 200 fixed paths use the same sampled months; average their daily means.
    r_means = picked('r_sums')/n[:, None]
    rbar_mean = r_means.mean(axis=1)
    trade_n, trade_total = picked('trade_counts'), picked('trade_sums')
    trade_mean = np.divide(trade_total, trade_n, out=np.full_like(trade_total, np.nan), where=trade_n > 0)
    original_n, original_s = float(counts.sum()), float(np.sum(month_data['sums']))
    original_r = float(np.asarray(month_data['r_sums']).sum(axis=0).mean()/original_n)
    original_trade_n = int(np.sum(month_data['trade_counts']))
    s_days = [np.asarray(values, dtype=float) for values in month_data['s_days']]
    if len(s_days) != g or any(len(values) != count for values, count in zip(s_days, counts)):
        raise ValueError('complete S daily sequences must match each month day count')
    sharpes = [_daily_sharpe(np.concatenate([s_days[index] for index in row])) for row in draws]
    rows = {
        'S_daily_mean': _interval(s_mean, original_s/original_n),
        'Rbar_daily_mean': _interval(rbar_mean, original_r),
        'D_daily_mean': _interval(s_mean-rbar_mean, original_s/original_n-original_r),
        'S_sharpe': _interval(sharpes, _daily_sharpe(np.concatenate(s_days))),
        'S_trade_net_bps': _interval(trade_mean, np.sum(month_data['trade_sums'])/original_trade_n if original_trade_n else None),
    }
    return pd.DataFrame.from_dict(rows, orient='index')


def trade_diagnostics(trades):
    entries, exits = _trade_times(trades, 'entry_time'), _trade_times(trades, 'exit_time')
    values = trades.net_bps.to_numpy(dtype=float)
    months = entries.strftime('%Y-%m')
    monthly = pd.Series(values, index=months).groupby(level=0).sum()
    removed = sorted(monthly.index, key=lambda month: (-float(monthly.loc[month]), month))[:3]
    remain = values[~np.isin(months, removed)]
    halves = [values[entries < HALF], values[entries >= HALF]]
    return {'trade_count': len(values), 'entry_month_count': len(monthly),
            'mean_net_bps': float(values.mean()) if len(values) else None,
            'half_trade_counts': [len(x) for x in halves],
            'half_trade_mean_bps': [float(x.mean()) if len(x) else None for x in halves],
            'removed_best_months': removed,
            'removed_best_month_net_bps_sums': [float(monthly.loc[x]) for x in removed],
            'remaining_trade_count': len(remain),
            'remaining_trade_mean_bps': float(remain.mean()) if len(remain) else None,
            'cross_month_trade_count': int(np.count_nonzero(months != exits.strftime('%Y-%m')))}


def _daily_metrics(values):
    a = np.asarray(values, dtype=float)
    terminal = float(INITIAL_CAPITAL*np.prod(1+a))
    return {'daily_mean': float(a.mean()),
            'annualized_sharpe': _finite(_daily_sharpe(a)),
            'total_return': terminal/INITIAL_CAPITAL-1, 'final_equity': terminal,
            'zero_return_days': int(np.count_nonzero(a == 0))}


def identifiability(strata, *, mean_difference, mcse_r):
    total = int(strata.n.sum())
    unsaturated = int(strata.loc[(strata.n > 0) & (strata.n < strata.N), 'n'].sum())
    positive_v = int(strata.loc[strata.vol_bin != 5, 'n'].sum())
    fraction = unsaturated/total if total else None
    finite_fraction = positive_v/total if total else None
    precision = bool(mean_difference > 2*mcse_r)
    return {'parent_count': total, 'unsaturated_parent_count': unsaturated,
            'saturated_parent_count': total-unsaturated,
            'missing_volatility_parent_count': total-positive_v,
            'unsaturated_parent_fraction': fraction,
            'finite_positive_v_parent_fraction': finite_fraction,
            'mean_difference': float(mean_difference), 'mcse_r': float(mcse_r),
            'strict_mc_precision_passed': precision,
            'passed': bool(total and fraction >= .9 and finite_fraction >= .9 and precision)}


def candidate_status(gates):
    """Input execution/equity failures raise before gates; sample and U7 lead."""
    if not gates['U1']['passed']:
        return 'insufficient_sample_evidence'
    if not gates['U7']['passed']:
        return 'insufficient_random_identifiability_or_precision'
    failed = [name for name in (f'U{i}' for i in range(1, 8)) if not gates[name]['passed']]
    if not failed:
        return 'is_exploratory_candidate'
    if failed == ['U6']:
        return 'cost_sensitive_not_candidate'
    if gates['U2']['passed'] and not gates['U3']['passed']:
        return 'no_evidence_beyond_random'
    return 'not_candidate'


def _matrix_metadata(array, seed):
    return {'shape': list(array.shape), 'dtype': str(array.dtype), 'dtype_string': array.dtype.str,
            'order': 'C', 'sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest(),
            'seed': seed, 'generator': 'Generator(PCG64)'}


def analyze(s_daily_returns, r_daily_returns, s_trades, harsh_daily_returns,
            harsh_trades, strata, random_trade_counts):
    """Run the sole registered two-test family and all seven candidate gates."""
    validate_inputs(s_daily_returns, r_daily_returns, s_trades, harsh_daily_returns,
                    harsh_trades, strata, random_trade_counts)
    rbar = r_daily_returns.mean(axis=1)
    daily = pd.DataFrame({'S': s_daily_returns, 'Rbar': rbar, 'D': s_daily_returns-rbar})
    labels = daily.index.strftime('%Y-%m')
    grouped = daily.groupby(labels)
    monthly = grouped.sum().reindex(_months()).rename(columns={'S': 'S_sum', 'Rbar': 'Rbar_sum', 'D': 'D_sum'})
    monthly['days'] = grouped.size().reindex(_months())
    monthly['S_squares'] = (s_daily_returns**2).groupby(labels).sum().reindex(_months())
    entry_labels = _trade_times(s_trades, 'entry_time').strftime('%Y-%m')
    trade_values = pd.Series(s_trades.net_bps.to_numpy(dtype=float), index=entry_labels)
    monthly['S_trade_count'] = trade_values.groupby(level=0).size().reindex(_months(), fill_value=0)
    monthly['S_trade_net_bps_sum'] = trade_values.groupby(level=0).sum().reindex(_months(), fill_value=0.)
    webb_values = np.array([-math.sqrt(1.5), -1., -math.sqrt(.5), math.sqrt(.5), 1., math.sqrt(1.5)])
    weights = np.random.Generator(np.random.PCG64(WEBB_SEED)).choice(webb_values, size=(WEBB_DRAWS, N_MONTHS))
    month_draws = np.random.Generator(np.random.PCG64(MONTH_SEED)).integers(0, N_MONTHS, size=(MONTH_DRAWS, N_MONTHS), dtype=np.int64)
    counts = _random_counts(random_trade_counts)
    tests = {name: webb_test(monthly[f'{name}_sum'], monthly.days, weights,
                             has_trades=bool(len(s_trades) and (name == 'S' or counts.sum() > 0)))
             for name in FAMILY}
    holm = holm_two({name: tests[name]['p_raw'] for name in FAMILY})
    for name in FAMILY:
        tests[name].update(holm[name])
        tests[name]['p_boundary_sensitive'] = abs(tests[name]['p_raw']-tests[name]['holm_threshold']) < 2*tests[name]['mcse_p']
        if name == 'D' and counts.sum() == 0:
            tests[name]['forced_p_one_reasons'].append('all_random_paths_have_no_trades')
        if len(s_trades) == 0:
            tests[name]['forced_p_one_reasons'].append('S_has_no_trades')
    month_data = {'counts': monthly.days.to_numpy(), 'sums': monthly.S_sum.to_numpy(),
                  'squares': monthly.S_squares.to_numpy(),
                  's_days': [s_daily_returns.loc[labels == month].to_numpy(dtype=float) for month in _months()],
                  'r_sums': r_daily_returns.groupby(labels).sum().reindex(_months()).to_numpy(),
                  'trade_counts': monthly.S_trade_count.to_numpy(),
                  'trade_sums': monthly.S_trade_net_bps_sum.to_numpy()}
    intervals = month_intervals(month_data, month_draws)
    main, harsh = _daily_metrics(s_daily_returns), _daily_metrics(harsh_daily_returns)
    trade_info, harsh_info = trade_diagnostics(s_trades), trade_diagnostics(harsh_trades)
    main.update(trade_info)
    harsh.update(harsh_info)
    path_means = r_daily_returns.mean(axis=0).to_numpy()
    mcse_r = float(path_means.std(ddof=1)/math.sqrt(R_COUNT))
    difference = float(daily.D.mean())
    half_differences = [float(daily.loc[daily.index < HALF, 'D'].mean()),
                        float(daily.loc[daily.index >= HALF, 'D'].mean())]
    positive = lambda value: value is not None and value > 0
    u7 = identifiability(strata, mean_difference=difference, mcse_r=mcse_r)
    gates = {
        'U1': {'passed': trade_info['trade_count'] >= 100 and trade_info['entry_month_count'] >= 24,
               'details': {'trade_count': trade_info['trade_count'], 'entry_month_count': trade_info['entry_month_count']}},
        'U2': {'passed': tests['S']['reject'] and main['total_return'] > 0 and positive(trade_info['mean_net_bps']),
               'details': {'p_holm': tests['S']['p_holm'], 'total_return': main['total_return'], 'mean_net_bps': trade_info['mean_net_bps']}},
        'U3': {'passed': tests['D']['reject'] and difference > 0,
               'details': {'p_holm': tests['D']['p_holm'], 'daily_mean_difference': difference}},
        'U4': {'passed': all(positive(x) for x in trade_info['half_trade_mean_bps']+half_differences),
               'details': {'S_half_trade_mean_bps': trade_info['half_trade_mean_bps'], 'D_half_daily_mean': half_differences}},
        'U5': {'passed': positive(trade_info['remaining_trade_mean_bps']),
               'details': {key: trade_info[key] for key in ('removed_best_months', 'removed_best_month_net_bps_sums', 'remaining_trade_count', 'remaining_trade_mean_bps')}},
        'U6': {'passed': harsh['total_return'] > 0 and positive(harsh_info['mean_net_bps']),
               'details': {'total_return': harsh['total_return'], 'mean_net_bps': harsh_info['mean_net_bps']}},
        'U7': {'passed': u7['passed'], 'details': u7},
    }
    for row in gates.values():
        row['passed'] = bool(row['passed'])
    random_summary = pd.DataFrame.from_dict({name: _daily_metrics(r_daily_returns[name]) for name in R_VERSIONS}, orient='index')
    random_summary['trade_count'] = counts.astype(np.int64)
    random_summary['exposure_fraction'] = random_trade_counts.exposure_fraction if isinstance(random_trade_counts, pd.DataFrame) and 'exposure_fraction' in random_trade_counts else np.nan
    metadata = {
        'calendar': {'days': N_DAYS, 'months': N_MONTHS, 'start': str(MAIN_START), 'end_exclusive': str(IS_END)},
        'family': list(FAMILY), 'alpha': ALPHA, 'main': main, 'harsh': harsh,
        'mcse_r': mcse_r, 'mcse_r_scope': 'finite_random_replicates_on_fixed_market_path_not_market_uncertainty',
        'inference_conditioning': 'fixed_200_random_schedules',
        'sharpe_method': 'direct_daily_std_ddof_1_per_shared_whole_month_draw_no_variance_epsilon',
        'random_fraction_daily_mean_above_S': float(np.mean(path_means > main['daily_mean'])),
        'random_fraction_daily_mean_below_S': float(np.mean(path_means < main['daily_mean'])),
        'random_fraction_total_return_above_S': float(np.mean(random_summary.total_return > main['total_return'])),
        'random_fraction_total_return_below_S': float(np.mean(random_summary.total_return < main['total_return'])),
        'random_comparison_is_exact_randomization_p': False,
        'Rbar_equity_is_statistical_index_not_actual_account': True,
        'D_months_descending': sorted(monthly.index, key=lambda month: (-float(monthly.loc[month, 'D_sum']), month)),
        'matrices': {'webb_weights': _matrix_metadata(weights, WEBB_SEED),
                     'month_indices': _matrix_metadata(month_draws, MONTH_SEED)},
    }
    return {'tests': pd.DataFrame.from_dict(tests, orient='index'), 'intervals': intervals,
            'gates': pd.DataFrame.from_dict(gates, orient='index'), 'verdict': candidate_status(gates),
            'daily': daily, 'monthly': monthly, 'random_summary': random_summary,
            'random_distribution': random_summary.quantile([0., .025, .25, .5, .75, .975, 1.]),
            'metadata': metadata, 'matrices': {'webb_weights': weights, 'month_indices': month_draws}}
