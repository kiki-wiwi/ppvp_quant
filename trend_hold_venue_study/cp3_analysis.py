"""Pure CP3 output slicing and descriptive, observed-ledger analysis.

No input loaders, feature construction, account engines, or counterfactuals.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from . import config, reporting
from . import cp3_windows as windows
from .models import Result

DAY = pd.Timedelta(days=1)
CHOP_START = pd.Timestamp('2024-03-01', tz='UTC')
CHOP_END = pd.Timestamp('2024-11-01', tz='UTC')
SPLIT = pd.Timestamp('2025-07-01', tz='UTC')


def _select_events(frame, start, end):
    if frame.empty:
        return frame.copy()
    if 'time' not in frame:
        raise ValueError('nonempty account event table requires time')
    times = pd.DatetimeIndex(pd.to_datetime(frame['time'], utc=True))
    return frame.loc[(times >= start) & (times < end)].copy()


def account_slice(result, start, end):
    """Lossless existing ledger slice; actual predecessor is never reinitialized."""
    selected = reporting.period_slice(result, start, end)
    start, end = selected['start'], selected['end']
    minute = result.minute.loc[(result.minute.index >= start) & (result.minute.index < end)].copy()
    metadata = dict(result.metadata)
    # Frozen metrics reads complete-account coin fields from metadata. Remove
    # those fields before computing the segment, then attach actual slice ratios.
    for key in ('initial_coin', 'final_coin', 'coin_growth'):
        metadata.pop(key, None)
    metadata['initial_capital'] = selected['start_equity']
    metadata['final_equity'] = selected['final_equity']
    if 'liquidated' in minute:
        metadata['liquidated'] = bool(minute['liquidated'].any())
    sliced = Result(minute, selected['daily_equity'], selected['daily_returns'],
        _select_events(result.orders, start, end), _select_events(result.decisions, start, end),
        _select_events(result.funding, start, end), metadata,
        _select_events(result.attribution, start, end), _select_events(result.pressure_events, start, end))
    summary = reporting.metrics(sliced, start_equity=selected['start_equity'])
    summary.update(start=start, end=end, prior_day=selected['prior_day'])
    if 'start_coin' in selected:
        summary.update(start_coin=selected['start_coin'], final_coin=selected['final_coin'],
                       coin_growth=selected['coin_growth'])
    # Returns after an already-ruined boundary remain the frozen zero convention,
    # but those zeros do not imply an investable segment wealth path or volatility.
    if selected['start_equity'] == 0.:
        for field in ('total_return', 'cagr', 'realized_volatility', 'sharpe', 'sortino',
                      'max_drawdown', 'calmar'):
            summary[field] = math.nan
    return {'result': sliced, 'metrics': summary}


def _daily_sum(frame, column, days):
    out = np.zeros(len(days), dtype=np.longdouble)
    if frame.empty or column not in frame:
        return out
    times = pd.DatetimeIndex(pd.to_datetime(frame['time'], utc=True)).normalize().as_unit('ns')
    positions = days.get_indexer(times)
    if np.any(positions < 0):
        raise ValueError('account event is outside daily ledger calendar')
    values = frame[column].to_numpy(dtype=np.longdouble)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite observed attribution: ' + column)
    np.add.at(out, positions, values)
    return out


def daily_accounting(result):
    """Aggregate observed ledgers with signed PnL and positive expense columns.

Linear price PnL is the observed cashflow bridge because its frozen engine does
not retain minute attribution. Coin terms retain the extended precision ledger.
"""
    days = result.daily_equity.index
    attr, md = result.attribution, result.metadata
    daily = pd.DataFrame(index=days)
    initial = md.get('initial_capital', config.INITIAL_CAPITAL)
    values = result.daily_equity.to_numpy(dtype=np.longdouble)
    daily['equity_change_usd'] = values - np.r_[np.longdouble(initial), values[:-1]]
    coin = md.get('version') in config.CM_VERSIONS
    if coin:
        mapping = {'funding_usd': 'funding_spot_usd', 'fee_usd': 'fee_spot_usd',
            'slippage_usd': 'slippage_spot_usd', 'holding_price_pnl_usd': 'holding_spot_usd',
            'contract_price_pnl_usd': 'contract_mtm_spot_usd',
            'zero_floor_adjustment_usd': 'zero_floor_spot_usd'}
        for target, origin in mapping.items():
            daily[target] = _daily_sum(attr, origin, days)
        daily['price_pnl_usd'] = daily['holding_price_pnl_usd'] + daily['contract_price_pnl_usd']
        daily['cash_roundoff_adjustment_usd'] = np.longdouble(0.)
    else:
        daily['funding_usd'] = _daily_sum(result.funding, 'cash_flow', days)
        daily['fee_usd'] = _daily_sum(result.orders, 'fee', days)
        daily['slippage_usd'] = -_daily_sum(result.orders, 'slippage_cost', days)
        daily['zero_floor_adjustment_usd'] = np.longdouble(0.)
        daily['cash_roundoff_adjustment_usd'] = _daily_sum(attr, 'cash_roundoff_adjustment', days)
        if 'price_pnl' in attr:
            daily['price_pnl_usd'] = _daily_sum(attr, 'price_pnl', days)
        else:
            daily['price_pnl_usd'] = (daily['equity_change_usd'] - daily['funding_usd']
                + daily['fee_usd'] - daily['slippage_usd'] - daily['cash_roundoff_adjustment_usd'])
        daily['holding_price_pnl_usd'] = daily['price_pnl_usd'] if md.get('version') in ('V2_SPOT', 'P_SPOT') else np.longdouble(0.)
        daily['contract_price_pnl_usd'] = daily['price_pnl_usd'] if md.get('version') in ('V1_USDM', 'P_USDM') else np.longdouble(0.)
    daily['conversion_usd'] = np.zeros(len(days), dtype=np.longdouble)
    daily.iloc[0, daily.columns.get_loc('conversion_usd')] += np.longdouble(md.get('initial_conversion_pnl', 0.))
    daily.iloc[-1, daily.columns.get_loc('conversion_usd')] += np.longdouble(md.get('terminal_conversion_pnl', 0.))
    if (daily['fee_usd'] < 0.).any() or (daily['slippage_usd'] > 1e-12).any() or (daily['conversion_usd'] > 1e-9).any():
        raise ValueError('expense signs disagree with frozen observed ledger')
    daily['friction_usd'] = daily['fee_usd'] - daily['slippage_usd'] - daily['conversion_usd']
    daily['reconciliation_error_usd'] = daily['equity_change_usd'] - (
        daily['price_pnl_usd'] + daily['funding_usd'] - daily['friction_usd']
        + daily['zero_floor_adjustment_usd'] + daily['cash_roundoff_adjustment_usd'])
    return daily


def period_metrics(result, job, accounting=None):
    accounting = daily_accounting(result) if accounting is None else accounting
    first, end = result.daily_equity.index[0], result.daily_equity.index[-1] + DAY
    if job['window'] == 'Post-IS':
        end = min(end, windows.window('Post-IS', job['asset']).end)
    periods = [(job['window'], first, end)]
    if job['window'] == 'Post-IS':
        periods += [('VAL', first, SPLIT), ('OOS', SPLIT, end)]
    rows = []
    for period, start, stop in periods:
        piece = account_slice(result, start, stop)
        summary = {**piece['metrics'], **job, 'period': period}
        span = accounting.loc[(accounting.index >= start) & (accounting.index < stop)]
        for column in accounting:
            summary[column] = float(np.sum(span[column].to_numpy(dtype=np.longdouble), dtype=np.longdouble))
        summary['order_count'] = len(piece['result'].orders)
        summary['pressure_event_count'] = len(piece['result'].pressure_events)
        rows.append(summary)
        del piece
    return pd.DataFrame(rows)


def chart_daily(result, accounting=None):
    """Daily envelopes retain exact effective-minute DD and its UTC trough."""
    accounting = daily_accounting(result) if accounting is None else accounting
    minute = result.minute
    equity = minute['equity'].to_numpy(dtype=float)
    effective = ~minute['bad'].to_numpy(dtype=bool)
    initial = float(result.metadata.get('initial_capital', config.INITIAL_CAPITAL))
    peaks = np.maximum.accumulate(np.maximum(initial, np.where(effective, equity, -np.inf)))
    drawdown = np.divide(equity, peaks, out=np.full(len(equity), np.nan), where=peaks > 0.) - 1.
    drawdown[~effective] = np.nan
    days = minute.index.normalize()
    dd = pd.Series(drawdown, index=minute.index)
    grouped = dd.groupby(days)
    daily = pd.DataFrame({'equity': result.daily_equity, 'return': result.daily_returns})
    daily['normalized_equity'] = daily['equity'] / initial if initial > 0. else np.nan
    daily['equity_drawdown_minute_min'] = grouped.min()
    daily['equity_drawdown_minute_max'] = grouped.max()
    daily['equity_drawdown_minute_min_time'] = grouped.idxmin()
    for field in ('coin_equity', 'equity_cm', 'equity_spot', 'liquidated'):
        if field in minute:
            daily[field] = minute[field].groupby(days).last()
    if 'coin_equity' in minute:
        troughs = pd.DatetimeIndex(daily['equity_drawdown_minute_min_time'])
        positions = minute.index.get_indexer(troughs)
        daily['coin_equity_at_minute_dd_min'] = np.where(positions >= 0,
            minute['coin_equity'].to_numpy(dtype=float)[np.maximum(positions, 0)], np.nan)
    if 'actual_leverage' in minute:
        exposures = minute['actual_leverage'].groupby(days)
        daily['actual_leverage_mean'] = exposures.mean()
        daily['actual_leverage_min'] = exposures.min()
        daily['actual_leverage_max'] = exposures.max()
    daily['funding_usd'] = accounting['funding_usd'].to_numpy(dtype=float)
    daily['funding_cumulative_usd'] = np.cumsum(accounting['funding_usd'].to_numpy(dtype=np.longdouble)).astype(float)
    return daily


def chop_diagnosis(result, features, job, accounting=None):
    """Observed canonical-trend-state episodes shared by T/P/HODL references."""
    accounting = daily_accounting(result) if accounting is None else accounting
    selected = accounting.loc[(accounting.index >= CHOP_START) & (accounting.index < CHOP_END)]
    expected = pd.date_range(CHOP_START, CHOP_END, inclusive='left', freq='D').as_unit('ns')
    if not selected.index.equals(expected):
        raise ValueError('chop diagnosis requires complete registered March–October calendar')
    states = features['g'].loc[selected.index].astype(bool)
    episode = states.ne(states.shift()).cumsum()
    all_states = features['g'].astype(bool)
    rows = []
    for ordinal, (_, frame) in enumerate(selected.groupby(episode), start=1):
        start, end = frame.index[0], frame.index[-1] + DAY
        state = bool(states.loc[start])
        previous = start - DAY
        prior_state = bool(all_states.loc[previous]) if previous in all_states.index else None
        next_state = bool(all_states.loc[end]) if end in all_states.index else None
        price = float(np.sum(frame['price_pnl_usd'].to_numpy(dtype=np.longdouble), dtype=np.longdouble))
        contract = float(np.sum(frame['contract_price_pnl_usd'].to_numpy(dtype=np.longdouble), dtype=np.longdouble))
        rows.append({**job, 'episode': ordinal, 'start': start, 'end': end, 'g': state,
            'episode_basis': 'canonical-trend-state episodes',
            'state': 'ON' if state else 'OFF', 'start_boundary_clipped': start == CHOP_START and prior_state == state,
            'end_boundary_clipped': end == CHOP_END and next_state == state,
            'price_pnl_usd': price, 'negative_episode_price_loss_usd': max(0., -price),
            'contract_price_pnl_usd': contract, 'negative_episode_contract_loss_usd': max(0., -contract),
            'funding_usd': float(frame['funding_usd'].sum()), 'friction_usd': float(frame['friction_usd'].sum())})
    episodes = pd.DataFrame(rows)
    loss = float(episodes['negative_episode_price_loss_usd'].sum())
    friction = float(selected['friction_usd'].sum())
    denominator = loss + friction
    funding = float(selected['funding_usd'].sum())
    preceding_states = all_states.shift().loc[selected.index]
    switches = states.ne(preceding_states) & preceding_states.notna()
    # These are canonical UM trend-switch boundaries for every account.
    # A passive/HODL account need not place an order on such a boundary.
    orders = _select_events(result.orders, CHOP_START, CHOP_END)
    switch_times = selected.index[switches]
    if orders.empty:
        switch_friction = 0.
    else:
        times = pd.DatetimeIndex(pd.to_datetime(orders['time'], utc=True))
        selected_orders = orders.loc[times.isin(switch_times)]
        if job['version'] in config.CM_VERSIONS:
            attr = _select_events(result.attribution, CHOP_START, CHOP_END)
            event_times = pd.DatetimeIndex(pd.to_datetime(attr['time'], utc=True))
            events = attr.loc[event_times.isin(switch_times) & attr['kind'].eq('trade').to_numpy()]
            switch_friction = float(events['fee_spot_usd'].sum() - events['slippage_spot_usd'].sum())
        else:
            switch_friction = float(selected_orders['fee'].sum() + selected_orders['slippage_cost'].sum())
    piece = account_slice(result, CHOP_START, CHOP_END)
    summary = {**piece['metrics'], **job, 'period': 'Chop', 'funding_usd': funding,
        'fee_usd': float(selected['fee_usd'].sum()), 'slippage_usd': float(selected['slippage_usd'].sum()),
        'conversion_usd': float(selected['conversion_usd'].sum()), 'friction_usd': friction,
        'negative_episode_price_loss_usd': loss,
        'negative_on_episode_price_loss_usd': float(episodes.loc[episodes['g'], 'negative_episode_price_loss_usd'].sum()),
        'negative_episode_contract_loss_usd': float(episodes['negative_episode_contract_loss_usd'].sum()),
        'funding_offset_denominator_usd': denominator,
        'funding_offset_ratio': funding / denominator if denominator > 0. else math.nan,
        'switch_only_friction_usd': switch_friction, 'switch_count': int(switches.sum()),
        'canonical_switch_count': int(switches.sum()), 'actual_order_count': len(orders),
        'episode_basis': 'canonical-trend-state episodes',
        'switch_count_definition': 'Changes of canonical UM g, not account-specific orders; P/HODL share the episode calendar.',
        'switch_friction_definition': 'Observed order friction at canonical UM g switch boundaries; not an incremental or causal switch cost.',
        'episode_count': len(episodes), 'interpretation': 'Observed negative price-attribution loss across maximal canonical-trend-state episodes clipped to March 1–November 1, plus actual trading/FX friction; signed funding offset is descriptive, not causal.'}
    return {'summary': summary, 'episodes': episodes}


def combined_reference(is_common, post):
    """Compound independently initialized accounts, never splice positions."""
    keys = ['asset', 'version', 'cost', 'signal_source', 'whole_contracts']
    def primary(frame, window):
        frame = frame.loc[(frame['window'] == window) & (frame['signal_source'] == 'UM') & ~frame['whole_contracts']]
        if frame.duplicated(keys).any():
            raise ValueError('duplicate independent-account compound reference key')
        return frame[keys + ['total_return']]
    left, right = primary(is_common, 'IS-Common'), primary(post, 'Post-IS')
    joined = left.merge(right, on=keys, suffixes=('_IS_Common', '_Post_IS'), validate='one_to_one')
    joined['combined_reference_return'] = (1. + joined['total_return_IS_Common']) * (1. + joined['total_return_Post_IS']) - 1.
    joined['continuous_positions'] = False
    joined['interpretation'] = 'Independent IS-Common and Post-IS account return product covering 2020-09 through 2026-05 (69 months); no continuous position history across IS/Post initialization.'
    return joined
