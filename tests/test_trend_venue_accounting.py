"""Hand-computable CP1 accounting fixtures; no market data or filesystem I/O."""
import math

import pytest

from trend_hold_venue_study.accounting import (
    dynamic_attribution,
    funding_change,
    initial_coin,
    inverse_pnl,
    liquidation_price,
    pressure_event,
    spot_fill,
    trade_change,
)
from trend_hold_venue_study.config import CASH_EPS, COSTS


@pytest.mark.parametrize("price, expected_coin", [(25_000., 2.), (100_000., .5)])
def test_a_one_times_short_hedge_locks_usd_without_pressure(price, expected_coin):
    coin = 1. + inverse_pnl(-50_000., 50_000., price)
    assert coin == expected_coin
    assert abs(coin * price - 50_000.) < 1e-12
    assert liquidation_price(1., -50_000., 50_000.) is None
    assert pressure_event(1., -50_000., 50_000., 50_000.,
                          max(50_000., price), min(50_000., price)) is None


def test_inverse_long_short_pnl_and_multistep_telescope():
    assert inverse_pnl(50_000., 50_000., 25_000.) == -1.
    assert inverse_pnl(-50_000., 50_000., 25_000.) == 1.
    steps = [inverse_pnl(50_000., 50_000., 40_000.),
             inverse_pnl(50_000., 40_000., 25_000.)]
    assert math.fsum(steps) == pytest.approx(-1., rel=0., abs=1e-15)
    assert inverse_pnl(0., 50_000., 25_000.) == 0.


def test_b_intrabar_pressure_preserves_250_usd_at_25125():
    assert liquidation_price(1., 50_000., 50_000.) == 25_125.
    event = pressure_event(1., 50_000., 50_000., 50_000., 51_000., 25_000.)
    assert event is not None
    assert event['trigger_price'] == 25_125.
    assert event['gap'] is False
    assert event['raw_coin'] == pytest.approx(250. / 25_125., rel=0., abs=1e-15)
    assert event['residual_coin'] == event['raw_coin']
    assert abs(event['residual_coin'] * 25_125. - 250.) < 1e-9
    assert event['zero_floor_coin'] == 0.


def test_pressure_at_exact_opening_line_counts_as_open_breach():
    event = pressure_event(1., 50_000., 50_000., 25_125., 26_000., 25_125.)
    assert event is not None
    assert event['gap'] is True
    assert event['trigger_price'] == 25_125.


def test_b_opening_gap_uses_actual_price_and_records_half_coin_floor_credit():
    event = pressure_event(1., 50_000., 50_000., 20_000., 22_000., 18_000.)
    assert event is not None
    assert event['gap'] is True
    assert event['trigger_price'] == 20_000.
    assert event['raw_coin'] == pytest.approx(-.5, rel=0., abs=1e-15)
    assert event['residual_coin'] == 0.
    assert event['zero_floor_coin'] == pytest.approx(.5, rel=0., abs=1e-15)
    assert event['raw_coin'] + event['zero_floor_coin'] == 0.
    # Once the caller clears N, neither future price moves nor funding debit coin.
    assert pressure_event(0., 0., 20_000., 10_000., 15_000., 5_000.) is None
    assert inverse_pnl(0., 20_000., 10_000.) == 0.
    assert funding_change(0., 10_000., .001, COSTS['main_17']) == 0.


def test_short_pressure_exists_only_for_positive_denominator_and_uses_high():
    assert liquidation_price(2., -50_000., 50_000.) is None
    assert liquidation_price(1., -50_000., 50_000.) is None
    assert liquidation_price(1., -100_000., 50_000.) == 99_500.
    event = pressure_event(1., -100_000., 50_000., 50_000., 100_000., 10_000.)
    assert event is not None
    assert event['trigger_price'] == 99_500.
    assert event['gap'] is False
    assert abs(event['residual_coin'] * 99_500. - 500.) < 1e-9
    assert pressure_event(1., -100_000., 50_000., 50_000., 60_000., 10_000.) is None


def test_c_conversion_separates_causal_estimate_from_actual_open():
    cost = COSTS['main_17']
    estimated = initial_coin(100_000., 50_000., cost)
    actual = initial_coin(100_000., 55_000., cost)
    assert estimated == 100_000. / (50_025. * 1.0005)
    assert actual == pytest.approx(100_000. / (55_027.5 * 1.0005), rel=0., abs=1e-15)
    assert actual < estimated
    assert initial_coin(100_000., 50_000., COSTS['gross_0']) == 2.
    # The order is frozen from the estimate; the actual conversion cannot rewrite it.
    frozen_target = -.5 * estimated * 50_000.
    assert frozen_target != -.5 * actual * 50_000.


@pytest.mark.parametrize(
    'delta_n, expected_fill, expected_slip, expected_fee',
    [(1_000., 100.05, -5. / 1000.5, 5. / 1000.5),
     (-1_000., 99.98, -2. / 999.8, 5. / 999.8)],
)
def test_c_signed_net_change_charges_one_adverse_slippage_and_one_fee(
        delta_n, expected_fill, expected_slip, expected_fee):
    trade = trade_change(delta_n, 100., COSTS['main_17'])
    assert trade['fill_price'] == pytest.approx(expected_fill, rel=0., abs=1e-13)
    assert trade['slippage_coin'] == pytest.approx(expected_slip, rel=0., abs=2e-15)
    assert trade['fee_coin'] == pytest.approx(expected_fee, rel=0., abs=1e-15)
    assert trade['delta_coin'] == pytest.approx(expected_slip - expected_fee, rel=0., abs=2e-15)
    assert trade['slippage_coin'] < 0.
    assert trade['fee_coin'] > 0.


def test_zero_trade_and_gross_trade_have_no_costs():
    for delta_n, cost in [(0., COSTS['main_17']), (1_000., COSTS['gross_0']),
                          (-1_000., COSTS['gross_0'])]:
        trade = trade_change(delta_n, 100., cost)
        assert trade['fill_price'] == 100.
        assert trade['slippage_coin'] == 0.
        assert trade['fee_coin'] == 0.
        assert trade['delta_coin'] == 0.


@pytest.mark.parametrize('n, rate, expected', [
    (50_000., .001, -.001), (-50_000., .001, .001),
    (50_000., -.001, .001), (-50_000., -.001, -.001),
    (0., .001, 0.),
])
def test_funding_direction_uses_old_signed_face_and_gross_omits_it(n, rate, expected):
    assert funding_change(n, 50_000., rate, COSTS['main_17']) == expected
    assert funding_change(n, 50_000., rate, COSTS['gross_0']) == 0.


def test_d_spot_full_budget_clips_fees_without_negative_cash():
    trade = spot_fill(0., 100_000., 1_000., 100., COSTS['main_17'])
    assert trade['requested_delta'] == 1_000.
    assert trade['cash_clip'] is True
    assert trade['qty_after'] == pytest.approx(100_000. / (100.05 * 1.0005), rel=0., abs=1e-12)
    assert trade['delta_qty'] == trade['qty_after']
    assert 0. <= trade['cash_after'] <= CASH_EPS
    assert trade['roundoff_adjustment'] >= 0.
    assert trade['fee'] > 0.
    assert trade['slippage_cost'] > 0.
    # Reference-price equity loses exactly friction, except the disclosed float clamp.
    change = trade['cash_after'] + trade['qty_after'] * 100. - 100_000.
    assert abs(change + trade['fee'] + trade['slippage_cost']
               - trade['roundoff_adjustment']) < 1e-9


def test_spot_partial_buy_full_sale_and_no_order_accounting():
    buy = spot_fill(0., 1_000., 2., 100., COSTS['main_17'])
    assert buy['qty_after'] == 2.
    assert buy['cash_clip'] is False
    assert buy['cash_after'] == pytest.approx(799.79995, rel=0., abs=1e-12)
    sold = spot_fill(2., 10., 0., 100., COSTS['main_17'])
    assert sold['qty_after'] == 0.
    assert sold['delta_qty'] == -2.
    assert sold['cash_after'] == pytest.approx(209.86002, rel=0., abs=1e-12)
    assert sold['fee'] == pytest.approx(.09998, rel=0., abs=1e-15)
    assert sold['slippage_cost'] == pytest.approx(.04, rel=0., abs=1e-13)
    assert sold['cash_clip'] is False
    idle = spot_fill(2., 10., 2., 100., COSTS['main_17'])
    assert idle['qty_after'] == 2.
    assert idle['cash_after'] == 10.
    assert idle['delta_qty'] == idle['fee'] == idle['slippage_cost'] == 0.
    assert idle['roundoff_adjustment'] == 0.


def test_dynamic_attribution_uses_previous_accumulated_coin_and_current_conversion():
    holding, coin_change = dynamic_attribution(2., 3., 100., 110.)
    assert holding == 20.
    assert coin_change == 110.
    assert holding + coin_change == 130.
    assert holding + coin_change == 3. * 110. - 2. * 100.


def test_zero_floor_credit_closes_gap_insolvency_attribution():
    event = pressure_event(1., 50_000., 50_000., 20_000., 20_000., 20_000.)
    holding, coin_change = dynamic_attribution(1., event['residual_coin'], 50_000., 20_000.)
    assert holding == -30_000.
    assert coin_change == -20_000.
    contract_loss = inverse_pnl(50_000., 50_000., 20_000.)
    assert contract_loss == pytest.approx(-1.5, rel=0., abs=1e-15)
    assert abs(coin_change - 20_000. * (contract_loss + event['zero_floor_coin'])) < 1e-9
    assert holding + coin_change == -50_000.


@pytest.mark.parametrize('bad', [0., -1., math.inf, math.nan])
def test_prices_must_be_positive_and_finite(bad):
    with pytest.raises(ValueError):
        inverse_pnl(1., bad, 100.)
    with pytest.raises(ValueError):
        initial_coin(100., bad, COSTS['main_17'])
    with pytest.raises(ValueError):
        spot_fill(0., 100., 1., bad, COSTS['main_17'])


def test_invalid_balances_faces_or_ohlc_are_rejected():
    with pytest.raises(ValueError):
        spot_fill(1., -2e-9, 2., 100., COSTS['main_17'])
    with pytest.raises(ValueError):
        spot_fill(1., 100., -1., 100., COSTS['main_17'])
    with pytest.raises(ValueError):
        trade_change(math.nan, 100., COSTS['main_17'])
    with pytest.raises(ValueError):
        funding_change(1., 100., math.inf, COSTS['main_17'])
    with pytest.raises(ValueError):
        pressure_event(1., 50_000., 50_000., 50_000., 40_000., 30_000.)
    with pytest.raises(ValueError):
        dynamic_attribution(math.nan, 1., 100., 100.)
