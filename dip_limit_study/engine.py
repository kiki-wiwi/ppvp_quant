"""Independent, causal minute execution for the registered dip-limit experiment.

All calculations use this package's immutable, default-locked Context. Idle
calendar intervals are filled in one slice; timestamps are materialized only for
ledger events, never in the minute loop. Independent events share this same
execution primitive without allocating a full calendar for each parent.
"""
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext

import numpy as np
import pandas as pd

from .config import (MAIN_START, IS_END, LAST_DECISION, INITIAL_CAPITAL,
                     RISK_FRACTION, MIN_DIST_FRAC, LEVERAGE_CAP, PRICE_TICK,
                     QTY_STEP, HOLD_MINUTES, TTL_MINUTES, COOLDOWN_MINUTES,
                     STRESS_DELAY_MINUTES, STRESS_PENETRATION, VERSIONS, COSTS,
                     STOP_ATR, LIMIT_ATR)
from .scope import require_synthetic, MINUTE_NS
from .features import floor_price_offset


TRADE_COLUMNS = [
    'signal_time', 'decision_time', 'entry_time', 'exit_time', 'eref', 'sl',
    'target', 'limit', 'q', 'qty', 'signal_equity', 'risk_budget',
    'reference_notional', 'entry_price', 'exit_price', 'fees',
    'funding_cashflow', 'net_pnl', 'gross_bps', 'net_bps', 'net_R',
    'minutes_held', 'risk_budget_exceeded', 'actual_leverage_exceeds_cap',
    'has_archive_gap', 'halt_in_holding', 'overrun_24h', 'exit_reason',
    'entry_minute_stop', 'entry_event_kind', 'exit_minute', 'exit_event_kind',
    'exit_boundary', 'deadline', 'wait_minutes', 'fill_minute_volume',
    'quantity_to_volume', 'funding_count', 'actual_leverage',
    'pending_crossed_stop', 'entry_gap', 'signal_lookback_has_archive_gap',
    'entry_fee', 'exit_fee', 'gross_pnl', 'parent_id',
    'entry_ref', 'exit_ref', 'execution_pnl', 'slippage_cost',
]
ORDER_COLUMNS = ['parent_id', 'signal_time', 'decision_time', 'activation_time',
                 'expiry_time', 'deadline', 'eref', 'sl', 'target', 'limit',
                 'q', 'qty', 'signal_equity', 'risk_budget', 'reference_notional',
                 'status', 'terminal_time', 'terminal_minute', 'terminal_event_kind',
                 'fill_time', 'wait_minutes', 'penetration', 'pending_crossed_stop',
                 'has_archive_gap', 'post_only_open_proxy', 'entry_fee_rate',
                 'signal_lookback_has_archive_gap']
DECISION_COLUMNS = ['parent_id', 'signal_time', 'decision_time', 'eligible',
                    'status', 'skip_reason', 'signal_equity']
EVENT_COLUMNS = ['parent_id', 'signal_time', 'decision_time', 'M_net_bps',
                 'L_net_bps', 'difference_bps', 'L_filled', 'L_status',
                 'L_wait_minutes', 'M_status', 'M_entry_price', 'L_entry_price',
                 'entry_price_improvement_bps', 'q', 'reference_notional']


@dataclass
class Result:
    trades: pd.DataFrame
    equity: pd.Series
    daily_equity: pd.Series
    daily_returns: pd.Series
    decisions: pd.DataFrame
    orders: pd.DataFrame
    metadata: dict


def _ledger(rows, columns):
    frame = pd.DataFrame(rows, columns=columns)
    timestamps = {'signal_time', 'decision_time', 'entry_time', 'exit_time',
                  'exit_minute', 'exit_boundary', 'deadline', 'activation_time',
                  'expiry_time', 'terminal_time', 'terminal_minute', 'fill_time'}
    for column in timestamps.intersection(frame.columns):
        frame[column] = pd.to_datetime(frame[column], utc=True).astype('datetime64[ns, UTC]')
    return frame


def _stamp(ns):
    return pd.Timestamp(int(ns), unit='ns', tz='UTC')


def _prepare(ctx, signals):
    required = {'signal_time', 'decision_time', 'deadline', 'eref', 'atr', 'sl',
                'limit', 'target', 'eligible', 'skip_reason'}
    if not required.issubset(signals.columns):
        raise ValueError('signals missing required columns: ' + ', '.join(sorted(required - set(signals.columns))))
    if signals.empty:
        return []
    ds = pd.DatetimeIndex(signals['decision_time'])
    if ds.tz is None or ds.hasnans or ds.has_duplicates:
        raise ValueError('unique timezone-aware decision times required')
    ds = ds.tz_convert('UTC').as_unit('ns')
    if not ds.is_monotonic_increasing or np.any(ds.asi8 % (60 * MINUTE_NS)):
        raise ValueError('ascending natural-hour decision times required')
    signal_index = pd.DatetimeIndex(signals.index)
    if signal_index.tz is None or signal_index.hasnans or not np.array_equal(
            signal_index.tz_convert('UTC').as_unit('ns').asi8, ds.asi8):
        raise ValueError('signal index must equal decision_time')
    base, n = int(ctx.index.asi8[0]), len(ctx.index)
    prepared = []
    for row in signals.to_dict('records'):
        d = pd.Timestamp(row['decision_time']).tz_convert('UTC')
        h = pd.Timestamp(row['signal_time'])
        deadline = pd.Timestamp(row['deadline'])
        if h.tz is None or deadline.tz is None:
            raise ValueError('signal/deadline times require timezone')
        h, deadline = h.tz_convert('UTC'), deadline.tz_convert('UTC')
        if h != d - pd.Timedelta(hours=1):
            raise ValueError('signal_time must equal decision_time minus one natural hour')
        for field, multiplier in (('sl', STOP_ATR), ('limit', LIMIT_ATR)):
            expected = floor_price_offset(row['eref'], row['atr'], multiplier)
            actual = row[field]
            if not (actual == expected or (np.isnan(actual) and np.isnan(expected))):
                raise ValueError(f'{field}: frozen price differs from registered decimal-grid formula')
        if deadline != d + pd.Timedelta(minutes=HOLD_MINUTES):
            raise ValueError('deadline must remain anchored at decision plus 24 hours')
        row.update(decision_time=d, signal_time=h, deadline=deadline,
                   parent_id=str(row.get('parent_id', d.isoformat())))
        where = (d.value - base) // MINUTE_NS
        if 0 <= where < n:
            prepared.append((int(where), row))
        elif bool(row['eligible']):
            raise ValueError('eligible decision outside supplied context; refusing silent loss of parent')
    return prepared


def _is_parent(row):
    return bool(row['eligible']) and row['signal_time'] >= MAIN_START and row['decision_time'] <= LAST_DECISION and row['deadline'] <= IS_END - pd.Timedelta(minutes=1)


def _valid_prices(row):
    vals = [row[k] for k in ('eref', 'atr', 'sl', 'limit', 'target')]
    return np.isfinite(vals).all() and row['atr'] > 0 and 0 < row['sl'] < row['limit'] < row['eref'] < row['target']


def _quantity(equity, row, cost):
    distance = max(row['eref'] - row['sl'], MIN_DIST_FRAC * row['eref'])
    raw = min(RISK_FRACTION * equity / (distance + cost.budget * row['eref']),
              LEVERAGE_CAP * equity / row['eref'])
    # Preserve the registered floating raw-sizing model; only its grid
    # projection uses canonical decimal values, with no upward tolerance.
    with localcontext() as arithmetic:
        arithmetic.prec = 50
        step = Decimal(str(QTY_STEP))
        return float((Decimal(str(raw)) / step).to_integral_value(rounding=ROUND_FLOOR) * step)


def _stress_penetration(limit):
    with localcontext() as arithmetic:
        arithmetic.prec = 50
        tick = Decimal(str(PRICE_TICK))
        raw = max(tick, Decimal(str(STRESS_PENETRATION)) * Decimal(str(limit)))
        return float((raw / tick).to_integral_value(rounding=ROUND_CEILING) * tick)


def _run_core(ctx, prepared, version, cost_name, stress, *, start, calendar):
    cost = COSTS[cost_name]
    ns = ctx.index.asi8
    n = len(ns)
    cash = INITIAL_CAPITAL
    equity = np.empty(n - start, dtype=float) if calendar else None
    trades, orders, decisions = [], [], []
    pending = position = None
    cooldown = -1
    failed, failure_reason = False, ''
    k, i = 0, start
    while k < len(prepared) and prepared[k][0] < start:
        k += 1
    last_close = float(ctx.C[max(0, start - 1)])

    def assess(value):
        nonlocal failed, failure_reason
        if not np.isfinite(value) or value <= 0:
            failed = True
            failure_reason = 'nonpositive_equity' if np.isfinite(value) else 'nonfinite_equity'

    def mark(price):
        return cash if position is None else cash + position['q'] * (price - position['entry_price'])

    def charge_funding(idx, reference):
        nonlocal cash
        if cost.include_funding and idx < n and np.isfinite(ctx.funding_rate[idx]):
            flow = -position['q'] * float(reference) * float(ctx.funding_rate[idx])
            cash += flow
            position['funding_cashflow'] += flow
            position['funding_count'] += 1
            assess(mark(reference))

    def terminate_order(status, idx):
        nonlocal pending, cooldown
        pending.update(status=status, terminal_time=_stamp(ns[idx]),
                       terminal_minute=_stamp(ns[idx]), terminal_event_kind='open',
                       wait_minutes=(int(ns[idx]) - pending['decision_time'].value) / MINUTE_NS)
        pending = None
        cooldown = int(ns[idx]) + COOLDOWN_MINUTES * MINUTE_NS

    def fill(order, idx, entry_price, kind):
        nonlocal cash, position, pending
        fee_rate = cost.taker if version == 'M_market' or stress else cost.maker
        entry_fee = order['q'] * entry_price * fee_rate
        cash -= entry_fee
        order.update(status='filled', fill_time=_stamp(ns[idx]),
                     wait_minutes=(int(ns[idx]) - order['decision_time'].value) / MINUTE_NS,
                     terminal_time=_stamp(ns[idx]), terminal_minute=_stamp(ns[idx]),
                     terminal_event_kind=kind)
        position = dict(order)
        position.update(entry_i=idx, entry_time=_stamp(ns[idx]), entry_price=float(entry_price),
                        entry_ref=float(ctx.O[idx]) if version == 'M_market' else float(order['limit']),
                        entry_fee=entry_fee, funding_cashflow=0., funding_count=0,
                        mean_due=None, halt_in_holding=False, entry_event_kind=kind,
                        fill_minute_volume=float(ctx.V[idx]),
                        quantity_to_volume=order['q'] / ctx.V[idx] if ctx.V[idx] > 0 else np.nan,
                        actual_leverage=order['q'] * entry_price / order['signal_equity'],
                        entry_gap=bool(ctx.O[idx] <= order['sl']),
                        has_archive_gap=bool(order['has_archive_gap'] or ctx.archive_gap[idx]))
        pending = None
        assess(mark(ctx.O[idx] if kind == 'open' else entry_price))

    def close(idx, raw_price, reason, kind):
        nonlocal cash, position, cooldown
        p = position
        boundary = int(ns[idx]) + (MINUTE_NS if kind in ('intraminute', 'close') else 0)
        if kind == 'close':
            charge_funding(idx + 1, raw_price)
        price = raw_price * (1 - cost.exit_slip)
        exit_fee = p['q'] * price * cost.taker
        execution_pnl = p['q'] * (price - p['entry_price'])
        cash += execution_pnl - exit_fee
        fees = p['entry_fee'] + exit_fee
        net = execution_pnl - fees + p['funding_cashflow']
        # Reporting convention: gross removes modeled slippage as well as
        # fees/funding; actual execution PnL remains separately inspectable.
        gross = p['q'] * (raw_price - p['entry_ref'])
        t = {c: p.get(c, np.nan) for c in TRADE_COLUMNS}
        t.update(exit_time=_stamp(boundary if kind == 'close' else ns[idx]),
                 exit_minute=_stamp(ns[idx]), exit_boundary=_stamp(boundary),
                 exit_event_kind=kind, exit_price=float(price), exit_fee=exit_fee,
                 fees=fees, gross_pnl=gross, net_pnl=net, exit_ref=float(raw_price),
                 execution_pnl=execution_pnl, slippage_cost=gross - execution_pnl,
                 gross_bps=gross / p['reference_notional'] * 10000,
                 net_bps=net / p['reference_notional'] * 10000,
                 net_R=net / (p['q'] * (p['eref'] - p['sl'])),
                 minutes_held=(boundary - int(ns[p['entry_i']])) / MINUTE_NS,
                 risk_budget_exceeded=bool(-net > p['risk_budget'] + 1e-10),
                 actual_leverage_exceeds_cap=bool(p['actual_leverage'] > LEVERAGE_CAP),
                 overrun_24h=bool(boundary > p['deadline'].value), exit_reason=reason,
                 entry_minute_stop=bool(idx == p['entry_i'] and reason in ('stop', 'gap_stop')))
        trades.append(t)
        position = None
        cooldown = boundary + COOLDOWN_MINUTES * MINUTE_NS
        assess(cash)

    def activate(idx):
        nonlocal pending
        if ctx.bad[idx]:
            terminate_order('maintenance_rejected', idx)
        elif ctx.O[idx] <= pending['limit']:
            terminate_order('post_only_rejected', idx)
        else:
            pending['activated'] = True

    while i < n:
        # No minute loop is needed when no position/order can change account state.
        if position is None and pending is None:
            nxt = prepared[k][0] if k < len(prepared) else n
            if nxt > i:
                if calendar:
                    equity[i - start:nxt - start] = cash
                i = nxt
                if i >= n:
                    break
                last_close = float(ctx.C[max(0, i - 1)])
        now, bad = int(ns[i]), bool(ctx.bad[i])
        # Existing positions settle funding before any opening execution.
        if position is not None:
            position['has_archive_gap'] |= bool(ctx.archive_gap[i])
            charge_funding(i, last_close if bad else ctx.O[i])
            if bad:
                position['halt_in_holding'] = True
            elif ctx.O[i] <= position['sl']:
                close(i, float(ctx.O[i]), 'gap_stop', 'open')
            elif position['halt_in_holding']:
                close(i, float(ctx.O[i]), 'maintenance', 'open')
            elif position['mean_due'] is not None and i >= position['mean_due']:
                close(i, float(ctx.O[i]), 'mean_reversion', 'open')
        # Opening pending-order events precede new fathers at this boundary.
        if pending is not None:
            if now >= pending['expiry_time'].value:
                terminate_order('expired', i)
            elif bad and now < pending['activation_time'].value:
                terminate_order('maintenance_cancelled', i)
            elif now >= pending['activation_time'].value:
                pending['has_archive_gap'] |= bool(ctx.archive_gap[i])
                if not pending['activated']:
                    activate(i)
                elif bad:
                    terminate_order('maintenance_cancelled', i)
                if pending is not None and not bad:
                    pending['pending_crossed_stop'] |= bool(ctx.L[i] <= pending['sl'])
                    if ctx.O[i] <= pending['limit'] - pending['penetration']:
                        fill(pending, i, pending['limit'], 'open')
                        if ctx.O[i] <= position['sl']:
                            close(i, float(ctx.O[i]), 'gap_stop', 'open')
        while k < len(prepared) and prepared[k][0] == i:
            _, row = prepared[k]
            k += 1
            signal_equity = mark(last_close if bad else ctx.O[i])
            d = dict(parent_id=row['parent_id'], signal_time=row['signal_time'],
                     decision_time=row['decision_time'], eligible=bool(row['eligible']),
                     signal_equity=signal_equity, status='', skip_reason='')
            if not bool(row['eligible']):
                d.update(status='ineligible', skip_reason=str(row['skip_reason']))
            elif not _is_parent(row):
                d.update(status='outside_window', skip_reason='registered_signal_window')
            elif failed:
                d.update(status='failed', skip_reason=failure_reason)
            elif position is not None:
                d.update(status='position', skip_reason='already_holding')
            elif pending is not None:
                d.update(status='pending', skip_reason='already_pending')
            elif now < cooldown:
                d.update(status='cooldown', skip_reason='four_hour_cooldown')
            elif not _valid_prices(row):
                d.update(status='invalid_prices', skip_reason='invalid_frozen_order_prices')
            else:
                q = _quantity(signal_equity, row, cost)
                if q <= 0:
                    d.update(status='zero_quantity', skip_reason='rounded_quantity_nonpositive')
                else:
                    d['status'] = 'submitted'
                    penetration = PRICE_TICK
                    if stress:
                        penetration = _stress_penetration(row['limit'])
                    pending = {key: row[key] for key in ('parent_id', 'signal_time', 'decision_time', 'deadline', 'eref', 'sl', 'target', 'limit')}
                    pending.update(activation_time=_stamp(now + (STRESS_DELAY_MINUTES * MINUTE_NS if stress else 0)),
                                   expiry_time=_stamp(now + TTL_MINUTES * MINUTE_NS),
                                   q=q, qty=q, signal_equity=signal_equity,
                                   risk_budget=RISK_FRACTION * signal_equity,
                                   reference_notional=q * row['eref'], status='pending',
                                   terminal_time=pd.NaT, terminal_minute=pd.NaT,
                                   terminal_event_kind='', fill_time=pd.NaT, wait_minutes=np.nan,
                                   penetration=penetration, pending_crossed_stop=False,
                                   has_archive_gap=False, post_only_open_proxy=version == 'L_limit',
                                   entry_fee_rate=cost.taker if version == 'M_market' or stress else cost.maker,
                                   signal_lookback_has_archive_gap=bool(row.get('signal_lookback_has_archive_gap', False)),
                                   activated=False)
                    orders.append(pending)
                    if version == 'M_market':
                        if bad:
                            terminate_order('maintenance_rejected', i)
                        else:
                            fill(pending, i, float(ctx.O[i] * (1 + cost.entry_slip)), 'open')
                            if ctx.O[i] <= position['sl']:
                                close(i, float(ctx.O[i]), 'gap_stop', 'open')
                    elif not stress:
                        activate(i)
                    elif bad:
                        # The submitted delayed order cannot survive a known
                        # maintenance minute merely because activation is later.
                        terminate_order('maintenance_cancelled', i)
            decisions.append(d)
        # All intraminute executions follow boundary decisions.
        if pending is not None and pending['activated'] and not bad:
            pending['has_archive_gap'] |= bool(ctx.archive_gap[i])
            pending['pending_crossed_stop'] |= bool(ctx.L[i] <= pending['sl'])
            if ctx.L[i] <= pending['limit'] - pending['penetration']:
                fill(pending, i, pending['limit'], 'intraminute')
                if ctx.O[i] <= position['sl']:
                    # The approved adverse opening-price proxy still applies
                    # when deep stress penetration is only known intraminute.
                    close(i, float(ctx.O[i]), 'gap_stop', 'intraminute')
        if position is not None and not bad:
            if ctx.L[i] <= position['sl']:
                close(i, float(position['sl']), 'stop', 'intraminute')
            elif now + MINUTE_NS >= position['deadline'].value:
                close(i, float(ctx.C[i]), 'deadline', 'close')
            elif ctx.C[i] >= position['target']:
                position['mean_due'] = i + 1
        if not bad:
            last_close = float(ctx.C[i])
        value = mark(last_close)
        assess(value)
        if calendar:
            equity[i - start] = value
        i += 1
    if position is not None or pending is not None:
        raise RuntimeError('unresolved position/order at context end; no EOF liquidation or external prices permitted')
    return trades, orders, decisions, equity, failed, failure_reason


def run_portfolio(ctx, signals, version, cost, *, stress=False):
    """Run one registered portfolio, preserving every supplied calendar minute."""
    require_synthetic(ctx)
    if version not in VERSIONS or cost not in COSTS:
        raise ValueError('unregistered version or cost')
    if stress and (version != 'L_limit' or cost not in ('main', 'harsh')):
        raise ValueError('joint stress is registered only for L main/harsh')
    prepared = _prepare(ctx, signals)
    start = int(np.searchsorted(ctx.index.asi8, MAIN_START.value))
    ts, os, ds, values, failed, reason = _run_core(ctx, prepared, version, cost, stress, start=start, calendar=True)
    equity = pd.Series(values, index=ctx.index[start:], name='equity')
    daily_equity = equity.resample('D').last()
    prior = daily_equity.shift(1)
    if len(prior):
        prior.iloc[0] = INITIAL_CAPITAL
    with np.errstate(divide='ignore', invalid='ignore'):
        daily_returns = (daily_equity / prior - 1).rename('return')
    if not np.isfinite(daily_returns.to_numpy()).all():
        failed, reason = True, reason or 'undefined_daily_return'
    return Result(_ledger(ts, TRADE_COLUMNS), equity, daily_equity,
                  daily_returns, _ledger(ds, DECISION_COLUMNS),
                  _ledger(os, ORDER_COLUMNS),
                  dict(version=version, cost=cost, stress=bool(stress), source=ctx.source,
                       symbol=ctx.symbol, failed=failed, failure_reason=reason,
                       n_signals=sum(_is_parent(r) for _, r in prepared), n_trades=len(ts)))


def simulate_events(ctx, signals):
    """All eligible parents, independently sized from fresh 100,000 accounts.

    No portfolio mutual exclusion, cooldown between parents, or portfolio Sharpe
    is applied. Unfilled L events retain zero returns and the complete M result.
    """
    require_synthetic(ctx)
    prepared = _prepare(ctx, signals)
    rows = []
    for i, row in prepared:
        if not _is_parent(row):
            continue
        outcomes = {}
        for version in VERSIONS:
            ts, os, ds, _, failed, reason = _run_core(ctx, [(i, row)], version, 'main', False, start=i, calendar=False)
            if failed:
                raise RuntimeError('independent event execution failed: ' + reason)
            trade = ts[0] if ts else None
            order = os[0] if os else None
            outcomes[version] = (trade, order, ds[0])
        m, mo, md = outcomes['M_market']
        l, lo, ld = outcomes['L_limit']
        mb, lb = (m['net_bps'] if m else 0.), (l['net_bps'] if l else 0.)
        q = (mo or lo or {}).get('q', 0.)
        rows.append(dict(parent_id=row['parent_id'], signal_time=row['signal_time'],
                         decision_time=row['decision_time'], M_net_bps=mb, L_net_bps=lb,
                         difference_bps=lb - mb, L_filled=l is not None,
                         L_status=lo['status'] if lo else ld['status'],
                         L_wait_minutes=lo['wait_minutes'] if lo else np.nan,
                         M_status=mo['status'] if mo else md['status'],
                         M_entry_price=m['entry_price'] if m else np.nan,
                         L_entry_price=l['entry_price'] if l else np.nan,
                         entry_price_improvement_bps=(m['entry_price'] - l['entry_price']) / row['eref'] * 10000 if m and l else np.nan,
                         q=q, reference_notional=q * row['eref']))
    return _ledger(rows, EVENT_COLUMNS)
