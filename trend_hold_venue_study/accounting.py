"""Pure accounting for the approved cross-venue experiment.

Prices are positive quote-currency amounts and inverse-contract ``n`` is signed
USD face value. These functions perform no I/O, hold no account state, and do
not select parameters. Pressure liquidation is the registered simplified model,
not a reconstruction of an exchange's liquidation engine.
"""
from __future__ import annotations

import math

from .config import CASH_EPS, MMR


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def _positive(value, name):
    value = _finite(value, name)
    if value <= 0.:
        raise ValueError(f'{name} must be positive')
    return value


def _nonnegative(value, name):
    value = _finite(value, name)
    if value < 0.:
        raise ValueError(f'{name} must be nonnegative')
    return value


def _cost_terms(cost):
    fee = _nonnegative(cost.fee, 'fee')
    buy_slip = _nonnegative(cost.buy_slip, 'buy_slip')
    sell_slip = _nonnegative(cost.sell_slip, 'sell_slip')
    if fee >= 1. or sell_slip >= 1.:
        raise ValueError('fee and sell_slip must be below one')
    return fee, buy_slip, sell_slip


def inverse_pnl(n, p0, p1) -> float:
    """Signed inverse-contract mark-to-market change, denominated in coin."""
    n = _finite(n, 'n')
    p0 = _positive(p0, 'p0')
    p1 = _positive(p1, 'p1')
    return _finite(n * (1. / p0 - 1. / p1), 'inverse pnl')


def initial_coin(cash, spot_reference, cost) -> float:
    """Convert cash, or estimate conversion using a known previous close.

The caller must keep the prior-close estimate separate from the actual open
conversion: an observed opening price must never rewrite a frozen first order.
    """
    cash = _nonnegative(cash, 'cash')
    spot_reference = _positive(spot_reference, 'spot_reference')
    fee, buy_slip, _ = _cost_terms(cost)
    fill = _positive(spot_reference * (1. + buy_slip), 'initial fill')
    total_price = _positive(fill * (1. + fee), 'initial all-in price')
    return _nonnegative(cash / total_price, 'initial coin')


def trade_change(delta_n, reference, cost) -> dict:
    """Cost of the signed net adjustment after old inventory is marked to O.

Slippage is recognized once, through delta_n * (1/F - 1/O). The separate fee
is positive expense. A positive delta buys, including closing a short; a
negative delta sells, including closing a long.
    """
    delta_n = _finite(delta_n, 'delta_n')
    reference = _positive(reference, 'reference')
    fee, buy_slip, sell_slip = _cost_terms(cost)
    if delta_n == 0.:
        return {'fill_price': reference, 'slippage_coin': 0.,
                'fee_coin': 0., 'delta_coin': 0.}
    multiplier = 1. + buy_slip if delta_n > 0. else 1. - sell_slip
    fill = _positive(reference * multiplier, 'trade fill')
    slippage_coin = _finite(delta_n * (1. / fill - 1. / reference), 'trade slippage')
    fee_coin = _nonnegative(fee * abs(delta_n) / fill, 'trade fee')
    delta_coin = _finite(slippage_coin - fee_coin, 'trade coin change')
    return {'fill_price': fill, 'slippage_coin': slippage_coin,
            'fee_coin': fee_coin, 'delta_coin': delta_coin}


def funding_change(n, reference, rate, cost) -> float:
    """Funding on the old signed face at the settlement reference price."""
    n = _finite(n, 'n')
    reference = _positive(reference, 'reference')
    rate = _finite(rate, 'rate')
    _cost_terms(cost)
    if not cost.include_funding or n == 0.:
        return 0.
    return _finite(-n / reference * rate, 'funding coin change')


def _pressure_inputs(c, n, p, mmr):
    # Equity can already be negative after an adverse event. Preserve it so the
    # pressure record exposes raw insolvency and the approved floor credit.
    c = _finite(c, 'coin equity')
    n = _finite(n, 'n')
    p = _positive(p, 'reference')
    mmr = _nonnegative(mmr, 'mmr')
    if mmr >= 1.:
        raise ValueError('mmr must be below one')
    return c, n, p, mmr


def liquidation_price(c, n, p, mmr=MMR) -> float | None:
    """Positive finite pressure line, or None when no such line exists."""
    c, n, p, mmr = _pressure_inputs(c, n, p, mmr)
    if n == 0.:
        return None
    if n > 0.:
        denominator = _finite(c + n / p, 'long pressure denominator')
        numerator = _finite(n + mmr * n, 'long pressure numerator')
    else:
        face = abs(n)
        denominator = _finite(face / p - c, 'short pressure denominator')
        numerator = _finite(face - mmr * face, 'short pressure numerator')
    if denominator <= 0.:
        return None
    line = numerator / denominator
    return line if math.isfinite(line) and line > 0. else None


def pressure_event(c, n, p, open_, high, low, mmr=MMR) -> dict | None:
    """Check opening breach before the adverse intraminute extreme.

The inequality is multiplied by the positive candidate price and regrouped:
(c + n/p) * candidate <= n + mmr*abs(n). This is the same registered
inequality without a numerical trigger tolerance, and preserves exact equality
at the hand-computable pressure line. A gap exits at the actual open; a
continuous intraminute crossing exits at the theoretical line.
    """
    c, n, p, mmr = _pressure_inputs(c, n, p, mmr)
    open_ = _positive(open_, 'open')
    high = _positive(high, 'high')
    low = _positive(low, 'low')
    if not low <= open_ <= high:
        raise ValueError('open must lie within low and high')
    if n == 0.:
        return None
    coefficient = _finite(c + n / p, 'pressure coefficient')
    boundary = _finite(n + mmr * abs(n), 'pressure boundary')

    def breached(candidate):
        return _finite(coefficient * candidate, 'pressure equity comparison') <= boundary

    gap = breached(open_)
    if gap:
        trigger = open_
    else:
        extreme = low if n > 0. else high
        if not breached(extreme):
            return None
        trigger = liquidation_price(c, n, p, mmr)
        if trigger is None:
            raise ValueError('intraminute breach has no positive finite pressure line')
    raw = _finite(c + inverse_pnl(n, p, trigger), 'raw liquidation coin')
    residual = max(0., raw)
    floor_credit = max(0., -raw)
    return {
        'trigger_price': trigger,
        'gap': bool(gap),
        'raw_coin': raw,
        'residual_coin': residual,
        'zero_floor_coin': floor_credit,
        'trigger_usd': _finite(residual * trigger, 'residual trigger USD'),
    }


def spot_fill(qty, balance, target, open_, cost) -> dict:
    """Execute a nonnegative spot target with exact cash-budget clipping.

``cash_clip`` records quantity clipped by the cash budget. The separately
reported ``roundoff_adjustment`` is only a negative-cash clamp in [-CASH_EPS,0).
It must be included as an accounting adjustment, never hidden in trading PnL.
    """
    qty = _nonnegative(qty, 'qty')
    balance = _nonnegative(balance, 'balance')
    target = _nonnegative(target, 'target')
    open_ = _positive(open_, 'open')
    fee_rate, buy_slip, sell_slip = _cost_terms(cost)
    requested = _finite(target - qty, 'requested spot change')
    if requested == 0.:
        return {
            'qty_after': qty, 'cash_after': balance, 'delta_qty': 0.,
            'requested_delta': 0., 'fill_price': open_, 'fee': 0.,
            'slippage_cost': 0., 'cash_clip': False, 'roundoff_adjustment': 0.,
        }
    roundoff_adjustment = 0.
    if requested > 0.:
        fill = _positive(open_ * (1. + buy_slip), 'spot buy fill')
        all_in_price = _positive(fill * (1. + fee_rate), 'spot all-in buy price')
        affordable = _nonnegative(balance / all_in_price, 'affordable quantity')
        delta_qty = min(requested, affordable)
        cash_after = _finite(balance - delta_qty * fill * (1. + fee_rate), 'cash after buy')
        if cash_after < -CASH_EPS:
            raise ValueError('spot cash below -CASH_EPS after buy')
        if cash_after < 0.:
            roundoff_adjustment = -cash_after
            cash_after = 0.
        cash_clip = delta_qty < requested
    else:
        fill = _positive(open_ * (1. - sell_slip), 'spot sell fill')
        # target >= 0 precludes selling beyond inventory; retain the explicit
        # bound to make the execution contract independent of subtraction drift.
        delta_qty = max(requested, -qty)
        cash_after = _nonnegative(balance + abs(delta_qty) * fill * (1. - fee_rate),
                                  'cash after sell')
        cash_clip = False
    qty_after = _nonnegative(qty + delta_qty, 'qty after spot trade')
    fee = _nonnegative(abs(delta_qty) * fill * fee_rate, 'spot fee')
    slippage_cost = _nonnegative(delta_qty * (fill - open_), 'spot slippage')
    return {
        'qty_after': qty_after,
        'cash_after': cash_after,
        'delta_qty': delta_qty,
        'requested_delta': requested,
        'fill_price': fill,
        'fee': fee,
        'slippage_cost': slippage_cost,
        'cash_clip': bool(cash_clip),
        'roundoff_adjustment': roundoff_adjustment,
    }


def dynamic_attribution(previous_coin, coin, previous_price, price) -> tuple[float, float]:
    """Exact difference identity with the preceding accumulated coin balance.

The second component includes all current coin movements, including a separately
recorded zero-floor credit when the caller invokes the pressure rule.
    """
    previous_coin = _finite(previous_coin, 'previous coin')
    coin = _finite(coin, 'coin')
    previous_price = _positive(previous_price, 'previous price')
    price = _positive(price, 'price')
    holding_pnl = _finite(previous_coin * (price - previous_price), 'holding pnl')
    coin_change_value = _finite(price * (coin - previous_coin), 'coin change value')
    return holding_pnl, coin_change_value
