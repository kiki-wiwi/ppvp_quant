"""Pure registered metrics and lossless slices of existing account outputs.

No loaders, plotting, account simulation, order execution, or data access live
here. Period boundaries select an already continuous account history.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import INITIAL_CAPITAL, utc


def _number(value, label, *, nonnegative=False):
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0.):
        raise ValueError(f'invalid {label}')
    return value


def _index(index, label):
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError(f'{label} requires a timezone-aware DatetimeIndex')
    index = index.tz_convert('UTC').as_unit('ns')
    if len(index) == 0 or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError(f'{label} must be nonempty, unique and sorted')
    return index


def _daily(result, initial):
    equity, returns = result.daily_equity, result.daily_returns
    if not isinstance(equity, pd.Series) or not isinstance(returns, pd.Series):
        raise ValueError('daily equity and returns must be Series')
    index = _index(equity.index, 'daily equity')
    return_index = _index(returns.index, 'daily returns')
    if not index.equals(return_index) or not index.equals(index.normalize()):
        raise ValueError('daily series must have identical UTC-midnight indices')
    expected = pd.date_range(index[0], index[-1], freq='D').as_unit('ns')
    if not index.equals(expected):
        raise ValueError('missing calendar day in account output')
    e, r = equity.to_numpy(dtype=float), returns.to_numpy(dtype=float)
    if not np.isfinite(e).all() or np.any(e < 0.) or not np.isfinite(r).all() or np.any(r < -1.):
        raise ValueError('invalid daily equity or returns')
    previous = np.r_[initial, e[:-1]]
    if np.any((previous == 0.) & (e != 0.)):
        raise ValueError('zero-equity account cannot recover without registered cash flow')
    if np.any((previous == 0.) & (r != 0.)):
        raise ValueError('zero-to-zero daily return must be zero')
    if np.any((previous > 0.) & (e == 0.) & (r != -1.)):
        raise ValueError('positive-to-zero daily return must be minus one')
    # Validate supplied returns but never replace them. This preserves the actual
    # boundary denominator and the account's exact stored values when slicing.
    expected_returns = np.zeros(len(e), dtype=float)
    positive = previous > 0.
    expected_returns[positive] = e[positive] / previous[positive] - 1.
    if not np.allclose(r, expected_returns, rtol=1e-12, atol=1e-12):
        raise ValueError('daily returns do not reconcile to continuous daily equity')
    return equity, returns, index


def _minute(result):
    minute = result.minute
    if not isinstance(minute, pd.DataFrame) or not {'equity', 'bad'}.issubset(minute):
        raise ValueError('minute equity and bad flag are required')
    index = _index(minute.index, 'minute output')
    if not pd.api.types.is_bool_dtype(minute['bad']) or minute['bad'].isna().any():
        raise ValueError('minute bad flags must be complete booleans')
    values = minute['equity'].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0.):
        raise ValueError('invalid minute equity')
    return minute, index


def _ratio(final, initial):
    # The specified ratio is undefined if its actual preceding equity is zero.
    # Keep NA instead of silently resetting a ruined account's denominator.
    return final / initial - 1. if initial > 0. else math.nan


def metrics(result, start_equity=INITIAL_CAPITAL) -> dict:
    """Seven registered performance metrics of an existing full result.

Max drawdown is a positive loss fraction and includes the supplied initial
capital peak. Bad minute placeholders do not establish a drawdown peak/trough.
Coin growth is reported separately from the primary cash-comparable equity.
    """
    initial = _number(start_equity, 'start equity', nonnegative=True)
    daily_equity, daily_returns, _ = _daily(result, initial)
    minute, _ = _minute(result)
    final = float(daily_equity.iloc[-1])
    if float(minute['equity'].iloc[-1]) != final:
        raise ValueError('terminal minute and daily equity disagree')
    values = daily_returns.to_numpy(dtype=float)
    n = len(values)
    mean = float(values.mean())
    sd = (0. if np.all(values == values[0]) else float(values.std(ddof=1))) if n > 1 else math.nan
    annual_vol = math.sqrt(365.) * sd
    sharpe = math.sqrt(365.) * mean / sd if sd > 0. else math.nan
    downside = math.sqrt(float(np.mean(np.minimum(values, 0.) ** 2)))
    sortino = math.sqrt(365.) * mean / downside if downside > 0. else math.nan
    total = _ratio(final, initial)
    cagr = (final / initial) ** (365. / n) - 1. if final > 0. and initial > 0. else math.nan
    effective = minute.loc[~minute['bad'], 'equity'].to_numpy(dtype=float)
    if len(effective) == 0:
        raise ValueError('drawdown requires at least one effective minute')
    peaks = np.maximum.accumulate(np.r_[initial, effective])[1:]
    known = peaks > 0.
    max_drawdown = float(np.max(1. - effective[known] / peaks[known])) if known.any() else math.nan
    calmar = cagr / max_drawdown if math.isfinite(cagr) and max_drawdown > 0. else math.nan
    metadata = result.metadata
    summary = {
        'version': metadata.get('version'), 'cost': metadata.get('cost'),
        'asset': metadata.get('asset'), 'window': metadata.get('window'),
        'start_equity': initial, 'final_equity': final, 'n_days': n,
        'total_return': total, 'cagr': cagr, 'realized_volatility': annual_vol,
        'sharpe': sharpe, 'sortino': sortino,
        'max_drawdown': max_drawdown, 'calmar': calmar,
        'liquidated': bool(metadata.get('liquidated', False)),
    }
    for field in ('initial_coin', 'final_coin', 'coin_growth'):
        if field in metadata:
            summary[field] = _number(metadata[field], field, nonnegative=field != 'coin_growth')
    return summary


def period_slice(result, start, end) -> dict:
    """Copy [start,end) daily outputs with their actual preceding denominator.

No returns are recomputed and no synthetic endpoint trades, fees, funding, or
FX conversion are applied. A full-account first slice uses registered initial
capital/coin; subsequent slices inherit the immediately preceding close.
    """
    start, end = utc(start), utc(end)
    if start != start.normalize() or end != end.normalize() or start >= end:
        raise ValueError('slice boundaries must be increasing UTC midnights')
    initial = _number(result.metadata.get('initial_capital', INITIAL_CAPITAL),
                      'initial capital', nonnegative=True)
    equity, returns, index = _daily(result, initial)
    if start < index[0] or end > index[-1] + pd.Timedelta(days=1):
        raise ValueError('slice lies outside the existing account history')
    selected = (index >= start) & (index < end)
    if not selected.any():
        raise ValueError('empty account slice')
    prior = np.flatnonzero(index < start)
    preceding = float(equity.iloc[prior[-1]]) if len(prior) else initial
    positions = np.flatnonzero(selected)
    span = slice(int(positions[0]), int(positions[-1]) + 1)
    sliced_equity = equity.iloc[span].copy(deep=True)
    sliced_returns = returns.iloc[span].copy(deep=True)
    final = float(sliced_equity.iloc[-1])
    answer = {
        'start': start, 'end': end,
        'prior_day': index[prior[-1]] if len(prior) else None,
        'daily_equity': sliced_equity, 'daily_returns': sliced_returns,
        'start_equity': preceding, 'final_equity': final,
        'total_return': _ratio(final, preceding),
    }
    if 'coin_equity' in result.minute:
        minute, minute_index = _minute(result)
        coins = minute['coin_equity'].to_numpy(dtype=float)
        if not np.isfinite(coins).all() or np.any(coins < 0.):
            raise ValueError('invalid minute coin equity')
        daily_coin = pd.Series(coins, index=minute_index, name='coin_equity').groupby(
            minute_index.normalize()).last()
        if not daily_coin.index.equals(index):
            raise ValueError('coin and primary daily calendars differ')
        daily_coin.index = equity.index.copy()
        if len(prior):
            start_coin = float(daily_coin.iloc[prior[-1]])
        else:
            if 'initial_coin' not in result.metadata:
                raise ValueError('initial coin required for first-period growth')
            start_coin = _number(result.metadata['initial_coin'], 'initial coin', nonnegative=True)
        coin_slice = daily_coin.iloc[span].copy(deep=True)
        final_coin = float(coin_slice.iloc[-1])
        answer.update(daily_coin=coin_slice, start_coin=start_coin, final_coin=final_coin,
                      coin_growth=_ratio(final_coin, start_coin))
    return answer
