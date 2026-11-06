"""Minute execution for the frozen long-only channel/VAH comparison.

Only the separate Context gate can authorize real inputs. No I/O, parameter
search, forced EOF liquidation, or borrowing of old strategy state occurs here.
Minute equity is labelled by minute open; matching uses the active stop before
any close-derived update. Time-exit funding belongs to its ending minute.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config, scope


@dataclass
class Result:
    trades: pd.DataFrame
    equity: pd.Series
    daily_equity: pd.Series
    daily_returns: pd.Series
    decisions: pd.DataFrame
    stop_updates: pd.DataFrame
    metadata: dict


_TIME_COLUMNS = ('signal_time', 'entry_time', 'exit_time', 'cooldown_until')
_BOOL_COLUMNS = ('cap_bound', 'floor_bound', 'has_archive_gap',
                 'signal_lookback_has_archive_gap', 'halt_in_holding',
                 'overrun_72h', 'risk_budget_exceeded', 'actual_leverage_exceeds_cap')
_INT_COLUMNS = ('trade_id', 'signal_idx', 'entry_idx', 'exit_idx', 'direction',
                'n_funding', 'bars_held', 'funding_last_boundary')
_STR_COLUMNS = ('version', 'cost', 'exit_reason')
_FLOAT_COLUMNS = (
    'eref', 'sl', 'atr_entry', 'final_stop', 'q', 'qty', 'signal_equity',
    'risk_budget', 'risk_amount', 'reference_notional', 'reference_leverage',
    'actual_leverage', 'entry_open', 'entry_price', 'entry_fill', 'exit_ref',
    'exit_price', 'exit_fill', 'entry_fee', 'exit_fee', 'fees', 'funding_cashflow',
    'funding', 'gross_pnl', 'net_pnl', 'gross_bps', 'net_bps', 'net_R',
    'minutes_held', 'overrun_minutes', 'MAE', 'MFE',
)
TRADE_COLUMNS = _TIME_COLUMNS + _INT_COLUMNS + _STR_COLUMNS + _FLOAT_COLUMNS + _BOOL_COLUMNS
_STOP_COLUMNS = ('trade_id', 'hour_open', 'close_time', 'effective_time',
                 'hour_close', 'running_max_close', 'atr_entry', 'old_stop', 'new_stop')
_MINUTE = pd.Timedelta(minutes=1)
_HOUR = pd.Timedelta(hours=1)


def _ledger(records):
    out = pd.DataFrame(records, columns=TRADE_COLUMNS)
    for col in _TIME_COLUMNS:
        out[col] = pd.to_datetime(out[col], utc=True).astype('datetime64[ns, UTC]')
    for columns, dtype in [(_INT_COLUMNS, 'int64'), (_FLOAT_COLUMNS, 'float64'),
                           (_BOOL_COLUMNS, 'bool'), (_STR_COLUMNS, 'str')]:
        for col in columns:
            out[col] = out[col].astype(dtype)
    return out


def _checked_signals(ctx, signals):
    if not isinstance(signals, pd.DataFrame):
        raise ValueError('signals must be an hourly DataFrame')
    required = {'signal_time', 'entry_time', 'eref', 'atr', 'sl', 'eligible', 'skip_reason'}
    if not required.issubset(signals.columns):
        raise ValueError(f'missing signal columns: {sorted(required - set(signals.columns))}')
    out = signals.copy(deep=True)
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None or idx.hasnans or idx.has_duplicates or not idx.is_monotonic_increasing:
        raise ValueError('signals require unique ascending timezone-aware entry times')
    idx = idx.tz_convert('UTC').as_unit('ns')
    out.index = idx
    for col in ('signal_time', 'entry_time'):
        values = pd.DatetimeIndex(out[col])
        if values.tz is None or values.hasnans:
            raise ValueError(f'{col} requires finite timezone-aware timestamps')
        out[col] = values.tz_convert('UTC').as_unit('ns')
    if not (out.entry_time.array == idx).all():
        raise ValueError('signal index must equal entry_time')
    if np.any(idx.asi8 % (60 * scope.MINUTE_NS)):
        raise ValueError('entry times must be natural UTC hours')
    if not ((out.signal_time + _HOUR).array == idx).all():
        raise ValueError('entry must be the minute open after its complete signal hour')
    if len(out) and ((out.signal_time < config.IS_START).any() or (out.entry_time > config.IS_END).any()):
        raise ValueError('signals outside the approved IS scope')
    if not out.eligible.isin([True, False]).all():
        raise ValueError('eligible must be boolean')
    valid = out.eligible.to_numpy(dtype=bool)
    vals = out.loc[valid, ['eref', 'atr', 'sl']].to_numpy(dtype=float)
    if len(vals):
        if not np.isfinite(vals).all() or np.any(vals[:, :2] <= 0):
            raise ValueError('eligible signals require finite positive reference price and ATR')
        if not np.allclose(vals[:, 2], vals[:, 0] - 2 * vals[:, 1], rtol=1e-12, atol=1e-12):
            raise ValueError('initial stop must equal eref - 2 * atr_entry')
    # All executable candidate times must be represented, never shifted to a later bar.
    executable = valid & (out.signal_time >= config.MAIN_START) & (idx <= config.LAST_ENTRY)
    entry_indices = ctx.index.get_indexer(idx[executable])
    if np.any(entry_indices < 0):
        raise ValueError('eligible entry time is missing from the supplied minute grid')
    hour_indices = ctx.index.get_indexer(pd.DatetimeIndex(out.loc[executable, 'signal_time']))
    if np.any(hour_indices < 0) or np.any(entry_indices - hour_indices != 60):
        raise ValueError('eligible entry requires its complete signal hour in the context')
    return out


def run_portfolio(ctx, signals, version='P_channel', cost='main_17'):
    """Run one of the six approved combinations on a complete minute grid.

    Synthetic short grids are permitted, but any position still open at the end
    raises instead of creating a fictitious exit. Failed nonpositive accounts
    continue to be marked and recorded; they can never re-enter or qualify.
    """
    scope.require_synthetic(ctx)  # Must precede any matching or input inspection.
    if not isinstance(ctx, scope.Context):
        raise ValueError('use scope.make_context to construct validated immutable input')
    if version not in config.VERSIONS or cost not in config.COSTS:
        raise ValueError('only the two approved versions and three cost schedules are allowed')
    fee = config.COSTS[cost]
    sig = _checked_signals(ctx, signals)
    grid = ctx.index
    n = len(ctx)
    by_time = {t.value: row for t, row in sig.iterrows()}
    decisions = []
    trades = []
    stop_updates = []
    equity = np.full(n, config.INITIAL_CAPITAL)
    cash = config.INITIAL_CAPITAL
    cooldown_until = config.MAIN_START
    pos = None
    failed = False
    failure_time = None

    def mark_failure(value, when):
        nonlocal failed, failure_time
        if not np.isfinite(value):
            raise ValueError('nonfinite account equity')
        if value <= 0 and not failed:
            failed, failure_time = True, when

    def funding_at(i, price):
        nonlocal cash
        if i < n and np.isfinite(ctx.funding_rate[i]) and i > pos['entry_idx']:
            pos['n_funding'] += 1
            amount = -pos['q'] * price * ctx.funding_rate[i] if fee.include_funding else 0.0
            pos['funding_cashflow'] += amount
            cash += amount
            mark_failure(cash + pos['q'] * (price - pos['entry_price']), grid[i])

    def close_position(i, reference, reason, when, intrabar=False):
        nonlocal pos, cash, cooldown_until
        fill = reference * (1 - fee.exit_slip)
        exit_fee = pos['q'] * fill * fee.fee
        cash += pos['q'] * (fill - pos['entry_price']) - exit_fee
        net = pos['q'] * (fill - pos['entry_price']) - pos['entry_fee'] - exit_fee + pos['funding_cashflow']
        cooldown_until = when + pd.Timedelta(minutes=config.COOLDOWN_MINUTES)
        minutes = (when - pos['entry_time']).total_seconds() / 60
        row = {k: v for k, v in pos.items() if k in TRADE_COLUMNS}
        row.update(exit_idx=i, exit_time=when, exit_ref=reference, exit_price=fill,
                   exit_fill=fill, exit_reason=reason, exit_fee=exit_fee,
                   fees=pos['entry_fee'] + exit_fee, funding=pos['funding_cashflow'],
                   gross_pnl=pos['q'] * (reference - pos['entry_open']), net_pnl=net,
                   gross_bps=(reference - pos['entry_open']) / pos['eref'] * 10000,
                   net_bps=net / pos['reference_notional'] * 10000,
                   net_R=net / pos['risk_amount'], final_stop=pos['stop'],
                   bars_held=i - pos['entry_idx'] + 1, minutes_held=minutes,
                   cooldown_until=cooldown_until, overrun_72h=minutes > config.HOLD_MINUTES,
                   overrun_minutes=max(0.0, minutes - config.HOLD_MINUTES),
                   risk_budget_exceeded=-net > pos['risk_budget'],
                   funding_last_boundary=i if intrabar or when == grid[i] else i + 1)
        trades.append(row)
        pos = None
        mark_failure(cash, when)

    for i, now in enumerate(grid):
        # Existing positions own the opening funding boundary, even when about
        # to gap-stop or exit after maintenance. New entries do not own it.
        if pos is not None:
            funding_at(i, ctx.O[i])
        signal = by_time.pop(now.value, None)
        if signal is not None:
            record = signal.to_dict()
            record.update(version=version, cost=cost, decision='', trade_id=-1,
                          decision_equity=cash if pos is None else equity[i - 1], q=np.nan)
            if signal.signal_time < config.MAIN_START:
                decision = 'before_main'
            elif now > config.LAST_ENTRY:
                decision = 'too_late'
            elif not bool(signal.eligible):
                decision = 'not_eligible'
            elif failed:
                decision = 'bankrupt'
            elif pos is not None:
                decision = 'in_position'
            elif now < cooldown_until:
                decision = 'cooldown'
            elif ctx.bad[i]:
                decision = 'bad_entry'
            elif signal.sl <= 0:
                decision = 'nonpositive_stop'
            else:
                eref, atr, sl = float(signal.eref), float(signal.atr), float(signal.sl)
                distance = eref - sl
                denominator = max(distance, config.MIN_DIST_FRAC * eref) + fee.c_rt * eref
                risk_budget = config.RISK_FRACTION * cash
                risk_q, cap_q = risk_budget / denominator, config.LEVERAGE_CAP * cash / eref
                q = min(risk_q, cap_q)
                if not np.isfinite(q) or q <= 0:
                    raise ValueError('position quantity must be positive and finite')
                fill = ctx.O[i] * (1 + fee.entry_slip)
                entry_fee = q * fill * fee.fee
                pos = dict(trade_id=len(trades), version=version, cost=cost,
                           signal_time=signal.signal_time, signal_idx=grid.get_indexer([signal.signal_time])[0],
                           entry_time=now, entry_idx=i, direction=1, eref=eref, sl=sl,
                           atr_entry=atr, q=q, qty=q, signal_equity=cash,
                           risk_budget=risk_budget, risk_amount=q * distance,
                           reference_notional=q * eref, reference_leverage=q * eref / cash,
                           actual_leverage=q * fill / cash,
                           actual_leverage_exceeds_cap=q * fill / cash > config.LEVERAGE_CAP,
                           entry_open=ctx.O[i], entry_price=fill, entry_fill=fill, entry_fee=entry_fee,
                           cap_bound=cap_q < risk_q, floor_bound=distance < config.MIN_DIST_FRAC * eref,
                           funding_cashflow=0.0, n_funding=0, has_archive_gap=False,
                           signal_lookback_has_archive_gap=bool(signal.get('signal_lookback_has_archive_gap', False)
                               or signal.get('window_has_archive_gap', False)
                               or signal.get('width_history_has_archive_gap', False)),
                           halt_in_holding=False, stop=sl, max_close=eref, MAE=0.0, MFE=0.0)
                cash -= entry_fee
                mark_failure(cash + q * (ctx.O[i] - fill), now)
                decision = 'entered'
                record.update(trade_id=pos['trade_id'], q=q)
            record['decision'] = decision
            decisions.append(record)

        if pos is not None:
            pos['has_archive_gap'] |= bool(ctx.archive_gap[i])
            if ctx.bad[i]:
                pos['halt_in_holding'] = True
            else:
                reference = ctx.O[i]
                opening_exit = pos['halt_in_holding'] or ctx.O[i] <= pos['stop']
                if opening_exit:
                    pos['MAE'] = min(pos['MAE'], reference - pos['entry_open'])
                    pos['MFE'] = max(pos['MFE'], reference - pos['entry_open'])
                    close_position(i, reference, 'halt_exit' if pos['halt_in_holding'] else 'gap_stop', now)
                elif ctx.L[i] <= pos['stop']:
                    pos['MAE'] = min(pos['MAE'], pos['stop'] - pos['entry_open'])
                    pos['MFE'] = max(pos['MFE'], ctx.O[i] - pos['entry_open'])
                    close_position(i, pos['stop'], 'intrabar_stop', now + _MINUTE, intrabar=True)
                else:
                    pos['MAE'] = min(pos['MAE'], ctx.L[i] - pos['entry_open'])
                    pos['MFE'] = max(pos['MFE'], ctx.H[i] - pos['entry_open'])
                    if i - pos['entry_idx'] + 1 >= config.HOLD_MINUTES:
                        # Funding at the close is booked in this minute, using
                        # this minute's close (never the next minute's open).
                        funding_at(i + 1, ctx.C[i])
                        close_position(i, ctx.C[i], 'time_exit', now + _MINUTE)
                    elif (now + _MINUTE).minute == 0:
                        old = pos['stop']
                        pos['max_close'] = max(pos['max_close'], ctx.C[i])
                        pos['stop'] = max(old, pos['max_close'] - 3 * pos['atr_entry'])
                        stop_updates.append(dict(trade_id=pos['trade_id'], hour_open=now.floor('h'),
                            close_time=now + _MINUTE, effective_time=now + _MINUTE,
                            hour_close=ctx.C[i], running_max_close=pos['max_close'],
                            atr_entry=pos['atr_entry'], old_stop=old, new_stop=pos['stop']))
        equity[i] = cash if pos is None else cash + pos['q'] * (ctx.C[i] - pos['entry_price'])
        mark_failure(equity[i], now + _MINUTE)

    if pos is not None:
        raise ValueError(f"open position from {pos['entry_time']} cannot exit within supplied grid; "
                         'extend synthetic bars within IS or report the unclosed real-data position; never liquidate at EOF')
    # Out-of-grid rows are permitted only when they cannot be an order, e.g.
    # final IS hours after LAST_ENTRY and pre-main warmup diagnostics.
    for stamp, signal in by_time.items():
        record = signal.to_dict()
        decision = 'before_main' if signal.signal_time < config.MAIN_START else (
            'too_late' if pd.Timestamp(stamp, tz='UTC') > config.LAST_ENTRY else 'not_eligible')
        record.update(version=version, cost=cost, decision=decision, trade_id=-1,
                      decision_equity=np.nan, q=np.nan)
        decisions.append(record)
    selected = grid >= config.MAIN_START
    eq = pd.Series(equity[selected], index=grid[selected], name='equity')
    daily = eq.resample('D').last().rename('equity')
    previous = daily.shift(1, fill_value=config.INITIAL_CAPITAL)
    returns = (daily / previous - 1).rename('daily_return')
    decision_cols = list(sig.columns) + ['version', 'cost', 'decision', 'trade_id', 'decision_equity', 'q']
    decision_frame = pd.DataFrame(decisions, columns=list(dict.fromkeys(decision_cols)))
    if len(decision_frame):
        decision_frame = decision_frame.sort_values('entry_time', ignore_index=True)
    update_frame = pd.DataFrame(stop_updates, columns=_STOP_COLUMNS)
    for col in ('hour_open', 'close_time', 'effective_time'):
        update_frame[col] = pd.to_datetime(update_frame[col], utc=True).astype('datetime64[ns, UTC]')
    ledger = _ledger(trades)
    return Result(ledger, eq, daily, returns, decision_frame, update_frame,
                  dict(version=version, cost=cost, source=ctx.source, symbol=ctx.symbol,
                       initial_capital=config.INITIAL_CAPITAL, failed=failed,
                       failure_reason='nonpositive_equity' if failed else None,
                       failure_time=failure_time, candidate_eligible=not failed,
                       n_signals=len(sig), n_trades=len(ledger),
                       funding_boundary='open-owned; time-exit close belongs to ending minute',
                       archive_gap_minutes=int(np.count_nonzero(ctx.archive_gap))))
