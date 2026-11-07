"""Causal minute marking and net execution for the registered daily policies."""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from .config import COSTS, INITIAL_CAPITAL, IS_END, IS_START, LEVERAGE_CAP, MAIN_START
from .config import P_VERSION, VERSIONS, Cost
from .scope import Context, ScopeError, require_real_authorization, utc_index

_MINUTES_PER_DAY = 1440
_MINUTE = pd.Timedelta(minutes=1)
_LAST_INSTANT = _MINUTE - pd.Timedelta(nanoseconds=1)
_ACCOUNT_RTOL = 1e-12
_ACCOUNT_ATOL = 1e-7

_ORDER_COLUMNS = (
    "time", "terminal", "equity_before", "qty_before", "qty_target", "delta_qty",
    "reference_price", "fill_price", "fee", "slippage_cost", "turnover_notional",
    "turnover_reference_notional", "turnover_fraction",
)
_FUNDING_COLUMNS = (
    "time", "rate", "qty_before", "reference_price", "cash_flow", "bad", "archive_gap",
)


@dataclass(frozen=True)
class Result:
    minute: pd.DataFrame
    daily_equity: pd.Series
    daily_returns: pd.Series
    orders: pd.DataFrame
    decisions: pd.DataFrame
    funding: pd.DataFrame
    metadata: dict


def _validate_schedule(ctx, features):
    end = ctx.index[-1]
    if end < MAIN_START or end + _MINUTE != (end + _MINUTE).normalize():
        raise ScopeError("portfolio requires complete main-window UTC days")
    start = ctx.index.searchsorted(MAIN_START)
    if ctx.index[start] != MAIN_START:
        raise ScopeError("portfolio requires the first main-window midnight")
    if ctx.source == "real" and (ctx.index[0] != IS_START or end != IS_END - _MINUTE):
        raise ScopeError("real portfolio requires the complete registered IS window")
    if ctx.bad[-1] or not np.isfinite(ctx.C[-1]) or ctx.C[-1] <= 0:
        raise ScopeError("terminal minute requires an effective close")

    days = utc_index(features.index, "decision schedule")
    expected = pd.date_range(MAIN_START, end.normalize(), freq="D").as_unit("ns")
    if not days.equals(expected):
        raise ScopeError("daily decision schedule must match every supplied main-window day")
    fields = ("eref", "sma60", "sigma", "budget", "g", "budget_update",
              "n_valid_prior", "has_archive_gap")
    if features.columns.has_duplicates or not set(fields).issubset(features.columns):
        raise ScopeError("complete daily feature columns required")
    if any(not pd.api.types.is_numeric_dtype(features[field]) for field in (*fields[:4], "n_valid_prior")):
        raise ScopeError("numeric daily feature columns required")
    numeric = features[list(fields[:4])].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or np.any(numeric[:, :2] <= 0):
        raise ScopeError("finite daily features and positive reference prices required")
    eref, _, sigma, budget = numeric.T
    if np.any(sigma < 0) or np.any(budget <= 0) or np.any(budget > LEVERAGE_CAP):
        raise ScopeError("invalid daily volatility or risk budget")
    for field in ("g", "budget_update", "has_archive_gap"):
        if not pd.api.types.is_bool_dtype(features[field]) or features[field].isna().any():
            raise ScopeError(f"{field} requires complete boolean values")
    counts = features["n_valid_prior"].to_numpy(dtype=np.float64)
    if not np.isfinite(counts).all() or np.any(
        (counts < 1) | (counts > _MINUTES_PER_DAY) | (counts != np.floor(counts))
    ):
        raise ScopeError("invalid prior-day valid minute count")
    expected_updates = (days.dayofweek == 0) | (days == MAIN_START)
    if not np.array_equal(features["budget_update"], expected_updates):
        raise ScopeError("daily budget update schedule must match the first day and Mondays")
    if np.any((budget[1:] != budget[:-1]) & ~expected_updates[1:]):
        raise ScopeError("daily budget changed outside the registered update schedule")
    if np.any(ctx.bad[start:].reshape(-1, _MINUTES_PER_DAY).all(axis=1)):
        raise ScopeError("every main-window day requires an effective close")

    settlements = np.flatnonzero(np.isfinite(ctx.funding_rate))
    if ctx.source == "real":
        expected_settlements = pd.date_range(IS_START, IS_END, inclusive="left", freq="8h")
        if not ctx.index[settlements].equals(expected_settlements.as_unit("ns")):
            raise ScopeError("real funding must match the complete IS settlement schedule")
    elif len(settlements) and np.any(
        (ctx.index[settlements].asi8 - IS_START.as_unit("ns").value)
        % pd.Timedelta(hours=8).value
    ):
        raise ScopeError("synthetic funding must fall on an eight-hour UTC settlement boundary")
    return start, days, settlements[settlements >= start] - start, eref


def _mark_prices(ctx, start, eref):
    closes = ctx.C[start:]
    bad = ctx.bad[start:]
    previous_valid = np.flatnonzero(~ctx.bad[:start])
    # Short synthetic grids may begin at the first decision with its supplied reference.
    initial_mark = ctx.C[previous_valid[-1]] if len(previous_valid) else eref[0]
    last_valid = np.maximum.accumulate(np.where(bad, -1, np.arange(len(closes))))
    marks = closes[last_valid]
    marks[last_valid < 0] = initial_mark
    previous_marks = np.r_[initial_mark, marks[_MINUTES_PER_DAY - 1:-1:_MINUTES_PER_DAY]]
    if not np.array_equal(eref, previous_marks):
        raise ScopeError("decision reference must equal the previous effective close")
    return marks, initial_mark


def _fill(time, terminal, equity_before, qty_before, target, reference, cost):
    delta = target - qty_before
    slip = cost.buy_slip if delta > 0 else -cost.sell_slip
    fill = reference * (1 + slip)
    notional = abs(delta) * fill
    ref_notional = abs(delta) * reference
    return {
        "time": time, "terminal": terminal, "equity_before": equity_before,
        "qty_before": qty_before, "qty_target": target, "delta_qty": delta,
        "reference_price": reference, "fill_price": fill, "fee": notional * cost.fee,
        "slippage_cost": delta * (fill - reference), "turnover_notional": notional,
        "turnover_reference_notional": ref_notional,
        "turnover_fraction": ref_notional / equity_before,
    }


def run_portfolio(ctx: Context, features: pd.DataFrame, version: str, cost: Cost) -> Result:
    """Run one registered combination on an independent, initially flat account."""
    if ctx.source == "real":
        require_real_authorization()
    if not isinstance(ctx, Context):
        raise ScopeError("portfolio requires a validated context")
    if version not in VERSIONS or COSTS.get(cost.name) != cost:
        raise ScopeError("portfolio requires a registered version and cost")

    start, days, settlements, eref = _validate_schedule(ctx, features)
    marks, prior_mark = _mark_prices(ctx, start, eref)
    index = ctx.index[start:]
    opens, bad = ctx.O[start:], ctx.bad[start:]
    gaps, rates = ctx.archive_gap[start:], ctx.funding_rate[start:]
    terminal = len(index) - 1
    midnight = np.arange(0, len(index), _MINUTES_PER_DAY)
    events = np.union1d(np.union1d(midnight, settlements), [terminal])
    inputs = features.to_dict(orient="records")

    equity = np.empty(len(index), dtype=np.float64)
    quantities = np.empty_like(equity)
    balance = previous_equity = INITIAL_CAPITAL
    qty = 0.0
    pending = previous_g = False
    orders, decisions, funding, price_pnl = [], [], [], []
    next_minute = 0
    terminal_qty = terminal_leverage = 0.0

    def fail(reason, at, observed_equity=None):
        error = ScopeError(f"{reason} at {index[at]}")
        error.partial = {
            "version": version, "cost": cost.name, "failure_time": index[at],
            "equity_at_failure": observed_equity, "qty_at_failure": qty,
            "balance_at_failure": balance,
            "last_confirmed_time": index[next_minute - 1] if next_minute else None,
            "last_confirmed_equity": previous_equity, "last_confirmed_mark": prior_mark,
            "orders": pd.DataFrame.from_records(orders, columns=_ORDER_COLUMNS),
            "decisions": pd.DataFrame.from_records(decisions),
            "funding": pd.DataFrame.from_records(funding, columns=_FUNDING_COLUMNS),
        }
        raise error

    def check_equity(value, at):
        if not math.isfinite(value) or value <= 0:
            fail("nonfinite or nonpositive equity", at, value)

    # Balance plus marked position is checked independently against event and
    # between-event mark-to-market P&L. Within each span the quantity is fixed.
    for minute in events:
        if next_minute < minute:
            span = slice(next_minute, minute)
            span_marks = marks[span]
            marked = balance + qty * span_marks
            invalid = ~np.isfinite(marked) | (marked <= 0)
            if invalid.any():
                offset = invalid.argmax()
                fail("nonfinite or nonpositive equity", next_minute + offset, marked[offset])
            expected = previous_equity + qty * (span_marks - prior_mark)
            if not np.allclose(marked, expected, rtol=_ACCOUNT_RTOL, atol=_ACCOUNT_ATOL):
                fail("minute mark-to-market reconciliation failed", next_minute)
            equity[span], quantities[span] = marked, qty
            price_pnl.append(qty * (span_marks[-1] - prior_mark))
            previous_equity, prior_mark = marked[-1], span_marks[-1]
            next_minute = minute

        time, mark = index[minute], marks[minute]
        qty_before = qty
        is_terminal = minute == terminal
        if minute % _MINUTES_PER_DAY == 0:
            row = inputs[minute // _MINUTES_PER_DAY]
            g = version == P_VERSION or row["g"]
            initial = minute == 0
            trend_flip = not initial and g != previous_g
            triggered = initial or row["budget_update"] or trend_flip or pending
            target = g * row["budget"] * previous_equity / row["eref"] if triggered else None
            target_leverage = target * row["eref"] / previous_equity if triggered else None
            if triggered and (
                not math.isfinite(target) or target < 0 or
                not math.isfinite(target_leverage) or target_leverage > LEVERAGE_CAP + 1e-12
            ):
                fail("invalid frozen target quantity or leverage", minute, previous_equity)
            decision = {
                **row, "time": time, "g": g, "initial": initial, "trend_flip": trend_flip,
                "retry": pending, "triggered": triggered, "equity_before": previous_equity,
                "qty_before": qty_before, "qty_target": target,
                "reference_leverage": target_leverage,
            }
            previous_g = g
        else:
            decision = None

        reference = mark if is_terminal else (prior_mark if bad[minute] else opens[minute])
        price_change = qty_before * (reference - prior_mark)
        event_equity = previous_equity + price_change
        check_equity(event_equity, minute)
        if np.isfinite(rates[minute]):
            flow = -qty_before * reference * rates[minute] if cost.include_funding else 0.0
            balance += flow
            event_equity += flow
            funding.append({
                "time": time, "rate": rates[minute], "qty_before": qty_before,
                "reference_price": reference, "cash_flow": flow, "bad": bad[minute],
                "archive_gap": gaps[minute],
            })
            check_equity(event_equity, minute)

        order = None
        if decision is not None:
            status = "hold"
            if triggered:
                pending = False
                if target == qty_before:
                    status = "zero_delta"
                elif bad[minute]:
                    status, pending = "canceled_bad", True
                else:
                    status = "executed"
                    order = _fill(time, False, previous_equity, qty_before, target, reference, cost)
            decision.update(status=status, pending_after=pending, qty_after=target if order else qty)
            decisions.append(decision)
        elif is_terminal and qty_before:
            terminal_qty = qty_before
            terminal_leverage = terminal_qty * mark / event_equity
            if not math.isfinite(terminal_leverage):
                fail("nonfinite terminal pre-close actual leverage", minute, event_equity)
            order = _fill(time + _LAST_INSTANT, True, event_equity, qty_before, 0.0, reference, cost)

        if order is not None:
            qty = order["qty_target"]
            balance -= order["delta_qty"] * order["fill_price"] + order["fee"]
            event_equity -= order["fee"] + order["slippage_cost"]
            orders.append(order)
            check_equity(event_equity, minute)

        price_to_close = qty * (mark - reference)
        price_pnl.append(price_change + price_to_close)
        marked = balance + qty * mark
        check_equity(marked, minute)
        if not math.isclose(marked, event_equity + price_to_close,
                            rel_tol=_ACCOUNT_RTOL, abs_tol=_ACCOUNT_ATOL):
            fail("event accounting reconciliation failed", minute, marked)
        if qty < 0 or not math.isfinite(qty):
            fail("invalid position quantity", minute, marked)
        equity[minute], quantities[minute] = marked, qty
        next_minute = minute + 1
        previous_equity, prior_mark = marked, mark

    total_fees = math.fsum(order["fee"] for order in orders)
    slippage = math.fsum(order["slippage_cost"] for order in orders)
    total_funding = math.fsum(receipt["cash_flow"] for receipt in funding)
    total_price_pnl = math.fsum(price_pnl)
    expected_final = INITIAL_CAPITAL + total_price_pnl - slippage - total_fees + total_funding
    if quantities[-1] != 0 or not math.isclose(equity[-1], expected_final,
                                              rel_tol=_ACCOUNT_RTOL, abs_tol=_ACCOUNT_ATOL):
        fail("terminal account reconciliation failed", terminal, equity[-1])

    notional = quantities * marks
    leverage = notional / equity
    invalid_leverage = ~np.isfinite(leverage)
    if invalid_leverage.any():
        invalid_minute = invalid_leverage.argmax()
        fail("nonfinite actual leverage", invalid_minute, equity[invalid_minute])
    held_quantities = quantities.copy()
    held_quantities[-1] = terminal_qty
    held = held_quantities > 0
    daily_values = equity[_MINUTES_PER_DAY - 1::_MINUTES_PER_DAY]
    day_open_equity = np.r_[INITIAL_CAPITAL, daily_values[:-1]]
    daily_returns = daily_values / day_open_equity - 1
    invalid_returns = ~np.isfinite(daily_returns)
    if invalid_returns.any():
        invalid_minute = (invalid_returns.argmax() + 1) * _MINUTES_PER_DAY - 1
        fail("nonfinite daily account return", invalid_minute, equity[invalid_minute])
    orders_frame = pd.DataFrame.from_records(orders, columns=_ORDER_COLUMNS)
    funding_frame = pd.DataFrame.from_records(funding, columns=_FUNDING_COLUMNS)
    buy_count = sum(order["delta_qty"] > 0 for order in orders)
    terminal_count = sum(order["terminal"] for order in orders)
    metadata = {
        "version": version, "cost": cost.name, "initial_capital": INITIAL_CAPITAL,
        "final_equity": float(equity[-1]), "price_pnl": total_price_pnl,
        "total_fees": total_fees, "total_slippage_cost": slippage,
        "total_funding": total_funding, "account_reconciliation_error": float(equity[-1] - expected_final),
        "turnover_notional": math.fsum(order["turnover_notional"] for order in orders),
        "turnover_reference_notional": math.fsum(order["turnover_reference_notional"] for order in orders),
        "total_turnover_fraction": math.fsum(order["turnover_fraction"] for order in orders),
        "order_count": len(orders), "adjustment_count": len(orders) - terminal_count,
        "terminal_order_count": terminal_count, "buy_count": int(buy_count),
        "sell_count": int(len(orders) - buy_count),
        "canceled_count": sum(row["status"] == "canceled_bad" for row in decisions),
        "funding_count": len(funding),
        "funding_held_count": int(sum(receipt["qty_before"] > 0 for receipt in funding)),
        "max_actual_leverage": float(leverage.max()),
        "terminal_preclose_actual_leverage": float(terminal_leverage),
        "minutes_above_leverage_cap": int(np.count_nonzero(leverage > LEVERAGE_CAP)),
        "held_minutes": int(np.count_nonzero(held)),
        "maintenance_minutes": int(np.count_nonzero(bad)),
        "gap_minutes": int(np.count_nonzero(gaps)),
        "maintenance_held_minutes": int(np.count_nonzero(held & bad)),
        "gap_held_minutes": int(np.count_nonzero(held & gaps)),
        "maintenance_notional_minutes": float(notional[bad].sum()),
        "gap_notional_minutes": float(notional[gaps].sum()),
    }
    minute = pd.DataFrame({
        "equity": equity, "qty": quantities, "mark_price": marks, "actual_leverage": leverage,
        "held_qty": held_quantities, "bad": bad, "archive_gap": gaps,
    }, index=index, copy=False)
    return Result(
        minute, pd.Series(daily_values, index=days, name="equity"),
        pd.Series(daily_returns, index=days, name="return"), orders_frame,
        pd.DataFrame.from_records(decisions), funding_frame, metadata,
    )
