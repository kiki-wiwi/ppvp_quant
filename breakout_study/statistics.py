"""Registered calendar-month inference and candidate gates (approved plan §6).

Only ``analyze`` is the study entry point. It has no seed, sample-size, calendar,
family, or significance overrides. The smaller pure helpers permit hand-computed
synthetic checks of the exact procedure; no function reads market files.
"""
from __future__ import annotations

import hashlib
import math
import platform
import sys
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .config import HALF, INITIAL_CAPITAL, IS_END, LAST_ENTRY, MAIN_START, VERSIONS
from .scope import ScopeError

WEBB_SEED = 2026110701
MONTH_SEED = 2026110702
WEBB_REPLICATES = 99_999
MONTH_REPLICATES = 10_000
FAMILY = ('P', 'V', 'D')
ALPHA = .05


def _finite(value):
    """JSON-safe scalar for quantities undefined on a degenerate sample."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _calendar():
    return pd.date_range(MAIN_START, IS_END, freq='D', inclusive='left')


def _months():
    return pd.date_range(MAIN_START, IS_END, freq='MS', inclusive='left').strftime('%Y-%m').tolist()


def _trade_times(trades, column):
    try:
        values = pd.DatetimeIndex(trades[column])
    except (TypeError, ValueError) as exc:
        raise ScopeError(f'{column}: timestamps required') from exc
    if len(values) == 0:
        return pd.DatetimeIndex([], tz='UTC')
    if values.tz is None or values.hasnans:
        raise ScopeError(f'{column}: finite timezone-aware timestamps required')
    return values.tz_convert('UTC')


def validate_inputs(main_returns, main_trades, harsh_returns, harsh_trades, *, failed_versions=None):
    """Fail closed on a truncated/unaligned daily calendar or out-of-IS ledger."""
    if failed_versions is not None and (
            not isinstance(failed_versions, Mapping)
            or set(failed_versions) != set(VERSIONS)
            or any(not isinstance(value, bool) for value in failed_versions.values())):
        raise ScopeError('failed_versions must map exactly both registered versions to booleans')
    expected = _calendar().as_unit('ns')
    for label, frame, ledgers in [('main_17', main_returns, main_trades),
                                   ('harsh_34', harsh_returns, harsh_trades)]:
        if not isinstance(frame, pd.DataFrame) or set(frame.columns) != set(VERSIONS) or len(frame.columns) != 2:
            raise ScopeError(f'{label}: exactly P_channel and V_vah daily columns required')
        if not isinstance(frame.index, pd.DatetimeIndex) or str(frame.index.tz) != 'UTC' or not frame.index.as_unit('ns').equals(expected):
            raise ScopeError(f'{label}: exactly 1,401 ascending UTC calendar days in 46 months required')
        try:
            finite = np.isfinite(frame.loc[:, list(VERSIONS)].to_numpy(dtype=float)).all()
        except (TypeError, ValueError) as exc:
            raise ScopeError(f'{label}: numeric daily returns required') from exc
        if not finite:
            raise ScopeError(f'{label}: nonfinite daily returns are forbidden')
        if not isinstance(ledgers, Mapping) or set(ledgers) != set(VERSIONS):
            raise ScopeError(f'{label}: both version ledgers required, including empty ledgers')
        for version in VERSIONS:
            trades = ledgers[version]
            if not isinstance(trades, pd.DataFrame) or not {'entry_time', 'exit_time', 'net_bps'} <= set(trades.columns):
                raise ScopeError(f'{label}/{version}: entry_time, exit_time, net_bps required')
            entries, exits = _trade_times(trades, 'entry_time'), _trade_times(trades, 'exit_time')
            if len(entries):
                if not entries.is_monotonic_increasing or entries.has_duplicates:
                    raise ScopeError(f'{label}/{version}: entries must be unique and ascending')
                if (entries < MAIN_START).any() or (entries > LAST_ENTRY).any() or (exits >= IS_END).any() or (exits < entries).any():
                    raise ScopeError(f'{label}/{version}: invalid trade scope, time order, or last entry')
                if (entries[1:] < exits[:-1]).any():
                    raise ScopeError(f'{label}/{version}: overlapping positions are forbidden')
            try:
                finite = np.isfinite(trades['net_bps'].to_numpy(dtype=float)).all()
            except (TypeError, ValueError) as exc:
                raise ScopeError(f'{label}/{version}: numeric net_bps required') from exc
            if not finite:
                raise ScopeError(f'{label}/{version}: finite net_bps required')


def webb_test(sums, counts, weights, *, two_sided, has_trades):
    """Restricted Webb t with a supplied, shared matrix; +1 finite-MC correction."""
    sums, counts, weights = (np.asarray(a, dtype=float) for a in (sums, counts, weights))
    g, n = len(sums), float(counts.sum())
    if g < 2 or counts.shape != sums.shape or weights.ndim != 2 or weights.shape[1] != g or n <= 0 or len(weights) == 0:
        raise ValueError('Webb inputs require >=2 months, positive days, and a nonempty matching matrix')
    mean = float(sums.sum() / n)
    residual = sums - mean * counts
    se = float(math.sqrt(g / (g - 1) * float(residual @ residual)) / n)
    observed_t = mean / se if se > 0 else None
    exceedances = 0
    zero_se = 0
    for start in range(0, len(weights), 4096):
        null_sums = weights[start:start + 4096] * sums
        null_means = null_sums.sum(axis=1) / n
        deviations = null_sums - null_means[:, None] * counts
        null_se = np.sqrt(g / (g - 1) * np.einsum('ij,ij->i', deviations, deviations)) / n
        zero_se += int(np.count_nonzero(null_se == 0))
        null_t = np.divide(null_means, null_se, out=np.zeros_like(null_means), where=null_se > 0)
        if observed_t is not None:
            exceedances += int(np.count_nonzero(np.abs(null_t) >= abs(observed_t) if two_sided else null_t >= observed_t))
    forced = not has_trades or se == 0
    p = 1. if forced else (1 + exceedances) / (len(weights) + 1)
    reasons = []
    if not has_trades:
        reasons.append('no_trades_in_required_version')
    if se == 0:
        reasons.append('observed_zero_cluster_se')
    return {'mean': _finite(mean), 'se': _finite(se), 't': _finite(observed_t),
            'p_raw': p, 'mcse': math.sqrt(p * (1-p) / len(weights)),
            'tail': 'two_sided_absolute' if two_sided else 'one_sided_positive',
            'bootstrap_replicates': len(weights), 'finite_mc_denominator': len(weights)+1,
            'tail_exceedances': exceedances if not forced else None,
            'zero_bootstrap_se_count': zero_se, 'forced_p_one_reasons': reasons}


def holm_three(raw_p):
    """Fixed three-member family, with P/V/D order breaking exact ties."""
    if set(raw_p) != set(FAMILY) or any(not math.isfinite(float(x)) or not 0 <= float(x) <= 1 for x in raw_p.values()):
        raise ValueError('Holm requires exactly three finite P/V/D p values')
    order = sorted(FAMILY, key=lambda name: (raw_p[name], FAMILY.index(name)))
    result, running = {}, 0.
    for i, name in enumerate(order):
        multiplier = 3-i
        running = min(1., max(running, multiplier * float(raw_p[name])))
        result[name] = {'rank': i+1, 'p_holm': running, 'holm_threshold': ALPHA/multiplier,
                        'reject': running <= ALPHA}
    return result


def _interval(values, *, estimate=None):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return {'estimate': _finite(estimate),
            'lower': _finite(np.quantile(finite, .025, method='linear')) if len(finite) else None,
            'upper': _finite(np.quantile(finite, .975, method='linear')) if len(finite) else None,
            'resamples': len(values), 'valid_resamples': len(finite), 'discarded': len(values)-len(finite),
            'fraction_below_zero': float(np.mean(finite < 0)) if len(finite) else None}


def _moment_sharpe(n, total, squares):
    n, total, squares = (np.asarray(a, dtype=float) for a in (n, total, squares))
    centered = squares - total * total / n
    # Subtracting nearly equal moments can leave roundoff for a constant series.
    roundoff = 16*np.finfo(float).eps*np.maximum(np.abs(squares), np.abs(total * total / n))
    centered = np.where(np.abs(centered) <= roundoff, 0., centered)
    sd = np.sqrt(np.maximum(centered, 0.) / np.maximum(n-1, 1.))
    return np.divide(math.sqrt(365.)*total/n, sd, out=np.full_like(sd, np.nan), where=(sd > 0) & (n > 1))


def month_intervals(month_data, draws):
    """Whole-month ratios and moment-equivalent concatenated daily Sharpe.

    Empty entry months stay in ``draws``. Undefined statistics are discarded for
    their own interval only, with no replacement draws.
    """
    draws = np.asarray(draws, dtype=np.int64)
    def picked(name):
        return np.asarray(month_data[name], dtype=float)[draws].sum(axis=1)
    n, total, squares = picked('counts'), picked('sums'), picked('squares')
    trade_n, trade_total = picked('trade_counts'), picked('trade_sums')
    means = np.divide(total, n, out=np.full_like(total, np.nan), where=n > 0)
    trade_means = np.divide(trade_total, trade_n, out=np.full_like(trade_total, np.nan), where=trade_n > 0)
    original_n = float(np.sum(month_data['counts']))
    original_total = float(np.sum(month_data['sums']))
    original_trade_n = int(np.sum(month_data['trade_counts']))
    return {'daily_mean': _interval(means, estimate=original_total/original_n),
            'annualized_sharpe': _interval(_moment_sharpe(n, total, squares), estimate=_moment_sharpe(original_n, original_total, np.sum(month_data['squares']))),
            'trade_net_bps': _interval(trade_means, estimate=float(np.sum(month_data['trade_sums']))/original_trade_n if original_trade_n else None)}


def trade_diagnostics(trades):
    """Entry-month ledger arithmetic; dropping top months never rematches trades."""
    entries, exits = _trade_times(trades, 'entry_time'), _trade_times(trades, 'exit_time')
    values = trades['net_bps'].to_numpy(dtype=float)
    months = entries.strftime('%Y-%m')
    monthly = pd.Series(values, index=months).groupby(level=0).sum()
    removed = sorted(monthly.index, key=lambda month: (-float(monthly.loc[month]), month))[:3]
    remain = values[~np.isin(months, removed)]
    halves = [values[entries < HALF], values[entries >= HALF]]
    half_means = [float(x.mean()) if len(x) else None for x in halves]
    remainder = float(remain.mean()) if len(remain) else None
    return {'trade_count': len(values), 'entry_month_count': len(monthly),
            'mean_net_bps': float(values.mean()) if len(values) else None,
            'half_trade_counts': [len(x) for x in halves], 'half_trade_mean_bps': half_means,
            'removed_best_months': removed, 'remaining_trade_count': len(remain),
            'remaining_trade_mean_bps': remainder,
            'cross_month_trade_count': int(np.count_nonzero(months != exits.strftime('%Y-%m'))),
            'cross_month_trade_fraction': float(np.mean(months != exits.strftime('%Y-%m'))) if len(values) else None,
            'N3': all(x is not None and x > 0 for x in half_means),
            'N4': remainder is not None and remainder > 0}


def _daily_metrics(series):
    r = series.to_numpy(dtype=float)
    # Once equity becomes nonpositive, do not present post-bankruptcy compounding.
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        equity = INITIAL_CAPITAL*np.cumprod(1+r)
    failed = (r <= -1) | (equity <= 0)
    bankrupt = bool(failed.any())
    numeric_failure = bool((~np.isfinite(equity)).any())
    terminal = None if bankrupt or numeric_failure else float(equity[-1])
    sd = float(np.std(r, ddof=1))
    return {'daily_mean': float(r.mean()), 'annualized_sharpe': math.sqrt(365)*float(r.mean())/sd if sd > 0 else None,
            'total_return': terminal/INITIAL_CAPITAL-1 if terminal is not None else None,
            'final_equity': terminal, 'bankrupt': bankrupt, 'equity_numerically_undefined': numeric_failure,
            'first_nonpositive_equity_day': str(series.index[np.flatnonzero(failed)[0]].date()) if bankrupt else None,
            'zero_return_days': int(np.count_nonzero(r == 0))}


def _month_data(series, trades):
    months = _months()
    labels = series.index.strftime('%Y-%m')
    grouped = series.groupby(labels)
    entries = _trade_times(trades, 'entry_time').strftime('%Y-%m')
    trade_values = pd.Series(trades['net_bps'].to_numpy(dtype=float), index=entries)
    trade_grouped = trade_values.groupby(level=0)
    return {'counts': grouped.size().reindex(months, fill_value=0).to_numpy(dtype=float),
            'sums': grouped.sum().reindex(months, fill_value=0).to_numpy(dtype=float),
            'squares': (series*series).groupby(labels).sum().reindex(months, fill_value=0).to_numpy(dtype=float),
            'trade_counts': trade_grouped.size().reindex(months, fill_value=0).to_numpy(dtype=float),
            'trade_sums': trade_grouped.sum().reindex(months, fill_value=0).to_numpy(dtype=float)}


def _passed(gates):
    return all(gates[name]['passed'] for name in ('N1','N2','N3','N4','N5'))


def candidate_status(gates, *, bankrupt):
    if bankrupt:
        return 'disqualified_bankruptcy'
    if not gates['N1']['passed']:
        return 'insufficient_evidence'
    failed = [name for name in ('N1','N2','N3','N4','N5') if not gates[name]['passed']]
    if not failed:
        return 'is_exploratory_candidate'
    if failed == ['N5']:
        return 'cost_sensitive_candidate'
    return 'not_candidate'


def relative_conclusion(mean_difference, half_differences, adjusted_p, gates, bankrupt):
    direction = 1 if mean_difference > 0 else -1 if mean_difference < 0 else 0
    directional_version = 'V_vah' if direction > 0 else 'P_channel' if direction < 0 else None
    consistent = direction != 0 and all(x is not None and x*direction > 0 for x in half_differences)
    significant = adjusted_p <= ALPHA
    n6 = bool(significant and consistent)
    qualifying = {v: bool(_passed(gates[v]) and not bankrupt[v]) for v in VERSIONS}
    winner = directional_version if n6 and qualifying.get(directional_version, False) else None
    if winner:
        status = 'tradable_increment_supported'
    elif all(qualifying.values()):
        status = 'indistinguishable'
    elif significant:
        status = 'relative_difference_only'
    elif not any(qualifying.values()):
        status = 'stop_current_hypothesis'
    else:
        status = 'no_supported_relative_increment'
    return {'status': status, 'N6': n6, 'statistically_significant': significant,
            'directional_version': directional_version, 'qualified_winner': winner,
            'mean_paired_difference': _finite(mean_difference),
            'half_mean_paired_differences': [_finite(x) for x in half_differences],
            'half_directions_agree': bool(consistent),
            'directional_version_failed_gates': [n for n in ('N1','N2','N3','N4','N5') if not gates[directional_version][n]['passed']] if directional_version else [],
            'directional_version_bankrupt': bool(bankrupt[directional_version]) if directional_version else False,
            'stop_current_hypothesis': not any(qualifying.values())}


def analyze(main_returns, main_trades, harsh_returns, harsh_trades, *,
            failed_versions: Mapping[str, bool] | None = None):
    """Run the immutable P/V/D family and descriptive intervals on supplied ledgers.

    Inputs must have all 1,401 UTC calendar days, including idle zero-return days.
    No real prices are loaded, and gross/harsh results never create extra tests.
    ``failed_versions`` preserves each version's execution failure OR across costs,
    including intraminute nonpositive equity subsequently hidden by daily closes.
    """
    validate_inputs(main_returns, main_trades, harsh_returns, harsh_trades,
                    failed_versions=failed_versions)
    execution_failed_versions = {v: failed_versions[v] if failed_versions is not None else False
                                 for v in VERSIONS}
    webb_values = np.array([-math.sqrt(1.5), -1., -math.sqrt(.5), math.sqrt(.5), 1., math.sqrt(1.5)])
    weights = np.random.Generator(np.random.PCG64(WEBB_SEED)).choice(webb_values, size=(WEBB_REPLICATES, 46))
    draws = np.random.Generator(np.random.PCG64(MONTH_SEED)).integers(0, 46, size=(MONTH_REPLICATES, 46))
    month_data = {v: _month_data(main_returns[v], main_trades[v]) for v in VERSIONS}
    paired = main_returns['V_vah'] - main_returns['P_channel']
    difference = _month_data(paired, main_trades['P_channel'].iloc[:0])
    tests = {}
    for name, version in [('P','P_channel'), ('V','V_vah'), ('D',None)]:
        data = difference if name == 'D' else month_data[version]
        has_trades = all(len(main_trades[v]) > 0 for v in VERSIONS) if name == 'D' else len(main_trades[version]) > 0
        tests[name] = webb_test(data['sums'], data['counts'], weights, two_sided=name == 'D', has_trades=has_trades)
        tests[name]['hypothesis'] = 'mean(V_vah-P_channel)=0' if name == 'D' else f'mean({version})<=0'
    holm = holm_three({name: tests[name]['p_raw'] for name in FAMILY})
    for name in FAMILY:
        tests[name].update(holm[name])
        tests[name]['mc_boundary_sensitive'] = abs(tests[name]['p_raw']-tests[name]['holm_threshold']) < 2*tests[name]['mcse']
    intervals = {'daily_mean': {}, 'paired_difference': None, 'annualized_sharpe': {}, 'trade_net_bps': {}}
    metrics, harsh_metrics, gates, bankrupt = {}, {}, {}, {}
    for name, v in zip(('P','V'), VERSIONS):
        ci = month_intervals(month_data[v], draws)
        for metric in ('daily_mean','annualized_sharpe','trade_net_bps'):
            intervals[metric][v] = ci[metric]
        diag = trade_diagnostics(main_trades[v])
        metrics[v] = {**_daily_metrics(main_returns[v]), **diag}
        harsh_metrics[v] = {**_daily_metrics(harsh_returns[v]), **trade_diagnostics(harsh_trades[v])}
        m, h = metrics[v], harsh_metrics[v]
        bankrupt[v] = bool(m['bankrupt'] or h['bankrupt'] or execution_failed_versions[v])
        gates[v] = {
            'N1': {'passed': diag['trade_count'] >= 100 and diag['entry_month_count'] >= 24,
                   'trade_count': diag['trade_count'], 'entry_month_count': diag['entry_month_count'], 'minimum_trades': 100, 'minimum_entry_months': 24},
            'N2': {'passed': bool(tests[name]['reject'] and m['total_return'] is not None and m['total_return'] > 0 and diag['mean_net_bps'] is not None and diag['mean_net_bps'] > 0),
                   'p_holm': tests[name]['p_holm'], 'total_return': m['total_return'], 'mean_net_bps': diag['mean_net_bps']},
            'N3': {'passed': diag['N3'], 'half_trade_mean_bps': diag['half_trade_mean_bps'], 'half_trade_counts': diag['half_trade_counts']},
            'N4': {'passed': diag['N4'], 'removed_best_months': diag['removed_best_months'], 'remaining_trade_mean_bps': diag['remaining_trade_mean_bps'], 'remaining_trade_count': diag['remaining_trade_count']},
            'N5': {'passed': bool(h['total_return'] is not None and h['total_return'] > 0 and h['mean_net_bps'] is not None and h['mean_net_bps'] > 0),
                   'harsh_total_return': h['total_return'], 'harsh_mean_net_bps': h['mean_net_bps']}}
    intervals['paired_difference'] = month_intervals(difference, draws)['daily_mean']
    half_differences = [float(paired.loc[paired.index < HALF].mean()), float(paired.loc[paired.index >= HALF].mean())]
    relative = relative_conclusion(float(paired.mean()), half_differences, tests['D']['p_holm'], gates, bankrupt)
    for v in VERSIONS:
        gates[v]['N6'] = {'passed': bool(relative['N6'] and relative['directional_version'] == v),
                          'p_holm': tests['D']['p_holm'], 'half_mean_paired_differences': half_differences,
                          'directional_version': relative['directional_version']}
    disagreements = []
    for name, interval in [('P',intervals['daily_mean']['P_channel']), ('V',intervals['daily_mean']['V_vah']), ('D',intervals['paired_difference'])]:
        excludes = interval['lower'] is not None and (interval['lower'] > 0 or (name == 'D' and interval['upper'] < 0))
        disagreements.append({'hypothesis': name, 'interval_supports_test_direction': bool(excludes),
                              'holm_rejects': tests[name]['reject'], 'disagrees': bool(excludes != tests[name]['reject'])})
    return {
        'schema_version': 1, 'calendar': {'start': MAIN_START.isoformat(), 'end_exclusive': IS_END.isoformat(), 'days': 1401, 'months': 46, 'month_order': _months()},
        'estimand': 'full_calendar_portfolio_mean_daily_return', 'family': list(FAMILY), 'family_size': 3, 'family_alpha': ALPHA,
        'tests': tests, 'intervals': intervals, 'metrics': metrics, 'harsh_metrics': harsh_metrics, 'gates': gates,
        'execution_failed_versions': execution_failed_versions,
        'conclusions': {'versions': {v: candidate_status(gates[v], bankrupt=bankrupt[v]) for v in VERSIONS}, 'relative': relative,
                        'candidate_label': '值得后续独立验证的 IS 探索性候选', 'cost_sensitive_label': '成本敏感候选，不具备本轮成本稳健资格',
                        'no_sharpe_winner_selection': True},
        'seeds': {'webb': WEBB_SEED, 'month_indices': MONTH_SEED, 'generator': 'NumPy Generator(PCG64)'},
        'draw_hashes': {'webb': hashlib.sha256(weights.tobytes(order='C')).hexdigest(), 'month_indices': hashlib.sha256(draws.tobytes(order='C')).hexdigest()},
        'draw_metadata': {'webb_shape': list(weights.shape), 'webb_dtype': str(weights.dtype), 'month_indices_shape': list(draws.shape), 'month_indices_dtype': str(draws.dtype), 'byte_order': sys.byteorder, 'hash_serialization': 'ndarray.tobytes(order=C)'},
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'platform': platform.platform()},
        'disclosures': {'interval_p_disagreements': disagreements,
                        'mc_boundary_sensitive_hypotheses': [n for n in FAMILY if tests[n]['mc_boundary_sensitive']],
                        'mcse_formula': 'sqrt(p_raw*(1-p_raw)/99999)', 'mc_boundary_rule': 'abs(p_raw-step_threshold)<2*MCSE',
                        'no_resampling_refill': True, 'intervals_are_not_significance_gates': True,
                        'cross_month_trade_fraction': {v: metrics[v]['cross_month_trade_fraction'] for v in VERSIONS},
                        'cluster_limitation': 'Month clustering permits within-month dependence; cross-month holdings and market dependence remain. This is exploratory IS evidence, not unconditional market alpha.',
                        'sample_minimum_is_not_power_guarantee': True, 'independent_validation_required': True}}
