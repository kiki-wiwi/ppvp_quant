"""Hand-computable portfolio fixtures; never open market files."""
import math
import numpy as np
import pandas as pd
import pytest

from trend_hold_venue_study.config import COSTS, Window
from trend_hold_venue_study.scope import make_context
from trend_hold_venue_study.portfolios import run_spot, run_coin


def fixture(days=2, venue='CM', price=50_000., changes=None, bad=(), rates=None):
    start = pd.Timestamp('2020-09-06', tz='UTC')  # Sunday, then scheduled Monday.
    index = pd.date_range(start-pd.Timedelta(days=1), periods=(days+1)*1440, freq='min').as_unit('ns')
    frame = pd.DataFrame({key: np.full(len(index), price) for key in ('open','high','low','close')}, index=index)
    frame['volume'] = 1.
    frame['bad'] = False
    frame['archive_gap'] = False
    for offset, values in (changes or {}).items():
        for key, value in values.items():
            frame.loc[index[1440+offset], key] = value
    for offset in bad:
        frame.loc[index[1440+offset], 'bad'] = True
    frame.attrs['provenance'] = 'synthetic'
    funding = pd.Series({start+pd.Timedelta(minutes=k): value for k,value in (rates or {}).items()}, dtype=float)
    if len(funding) == 0:
        funding = None
    context = make_context(frame, funding, venue=venue)
    decision_days = pd.date_range(start, periods=days, freq='D').as_unit('ns')
    features = pd.DataFrame({'g': True, 'budget': 1., 'budget_update': decision_days.dayofweek == 0}, index=decision_days)
    return context, features, Window('synthetic', start, start+pd.Timedelta(days=days), start-pd.Timedelta(days=6))


def test_spot_cash_clip_uses_actual_open_and_complete_roundtrip():
    ctx, features, window = fixture(days=1, venue='SPOT', price=100., changes={0: {'open': 200., 'high': 200.}})
    result = run_spot(ctx, features, 'V2_SPOT', COSTS['main_17'], window)
    first = result.orders.iloc[0]
    expected_qty = 100_000. / (200.*1.0005*1.0005)
    assert first['qty_after'] == expected_qty
    assert 0. <= first['cash_after'] <= 1e-9
    assert bool(first['cash_clip'])
    assert (result.minute['cash'] >= 0).all()
    assert result.orders.iloc[-1]['terminal']
    assert result.minute['qty'].iloc[-1] == 0.
    assert result.daily_equity.iloc[-1] == pytest.approx(expected_qty*(100.*.9998)*.9995,abs=1e-9,rel=0)
    assert abs(result.metadata['account_reconciliation_error']) < 1e-9


def test_spot_bad_midnight_retries_tomorrow_and_uses_own_reference():
    ctx, features, window = fixture(venue='SPOT', price=100., bad=(0,))
    features['eref'] = 50_000.  # Canonical UM close must never become Spot reference.
    result = run_spot(ctx, features, 'V2_SPOT', COSTS['gross_0'], window)
    assert result.decisions.iloc[0]['status'] == 'canceled_bad'
    assert not (result.minute['qty'].iloc[:1440] > 0).any()
    assert result.decisions.iloc[1]['retry']
    assert result.orders.iloc[0]['qty_after'] == 1000.


def test_coin_first_target_uses_estimated_not_actual_initial_coin():
    cm, features, window = fixture(days=1, rates={0:.001})
    spot, _, _ = fixture(days=1, venue='SPOT', changes={0:{'open':40_000.,'low':40_000.}})
    features['g'] = False
    results = [run_coin(cm, spot, features, version, COSTS['main_17'], window) for version in
               ('V3_CM_DeltaMatched','P_CM_DeltaMatched','V4_CM_CoinAccum','HODL_1x_Coin')]
    actual = 100_000./(40_000.*1.0005*1.0005)
    estimate = 100_000./(50_000.*1.0005*1.0005)
    assert all(result.metadata['initial_coin'] == actual for result in results)
    first = results[0].decisions.iloc[0]
    assert first['coin_before'] == estimate
    assert first['face_target'] == -estimate*50_000.
    assert results[0].funding.iloc[0]['face_before'] == 0.
    assert results[0].funding.iloc[0]['coin_flow'] == 0.
    assert results[0].orders.iloc[0]['face_target'] == first['face_target']
    assert abs(results[0].metadata['account_reconciliation_error']) < 1e-9


def test_coin_old_funding_after_frozen_target_and_wallet_reconciles():
    cm, features, window = fixture(rates={0:.001,480:.001,960:.001,1440:.001})
    spot, _, _ = fixture(venue='SPOT')
    result = run_coin(cm, spot, features, 'V4_CM_CoinAccum', COSTS['main_17'], window)
    second = result.decisions.iloc[1]
    assert second['face_target'] == second['coin_before']*50_000.
    receipt = result.funding[result.funding['time']==window.start+pd.Timedelta(days=1)].iloc[0]
    old_face = result.minute['face'].iloc[1439]
    assert receipt['face_before'] == old_face
    assert receipt['coin_flow'] == -old_face/50_000.*.001
    assert second['coin_before'] == result.minute['coin_equity'].iloc[1439]
    assert np.max(np.abs(result.minute['coin_equity']-result.minute['wallet_coin']-result.minute['unrealized_pnl_coin'])) <= 1e-12
    assert abs(result.metadata['account_reconciliation_error']) < 1e-9
    assert abs(result.metadata['cm_account_reconciliation_error']) < 1e-9


def test_intrabar_pressure_keeps_coins_and_marks_residue_without_reentry():
    cm, features, window = fixture(changes={1:{'low':25_125.,'close':40_000.}}, rates={0:.001,480:.001,1440:.001})
    spot, _, _ = fixture(venue='SPOT', changes={2:{'close':60_000.,'high':60_000.}})
    result = run_coin(cm, spot, features, 'V4_CM_CoinAccum', COSTS['gross_0'], window)
    event = result.pressure_events.iloc[0]
    assert event['trigger_price'] == pytest.approx(25_125., abs=1e-9)
    assert not bool(event['gap'])
    residual = 500./25_125.
    assert event['residual_coin'] == pytest.approx(residual, abs=1e-14)
    assert event['residual_cm_usd'] == pytest.approx(500., abs=1e-9)
    assert np.allclose(result.minute['coin_equity'].iloc[1:], residual, rtol=0, atol=1e-14)
    assert (result.minute['face'].iloc[1:] == 0).all()
    assert result.minute['equity_spot'].iloc[2] == pytest.approx(residual*60_000., abs=1e-9)
    assert (result.funding['coin_flow'].iloc[1:] == 0).all()
    assert len(result.orders) == 1  # Initial contract only; pressure closure is a separate event.
    assert abs(result.metadata['account_reconciliation_error']) < 1e-9


def test_gap_pressure_floor_is_explicit_and_zero_daily_returns_continue():
    cm, features, window = fixture(changes={1:{'open':20_000.,'low':20_000.,'close':20_000.}}, rates={480:.001,1440:.001})
    spot, _, _ = fixture(venue='SPOT')
    result = run_coin(cm, spot, features, 'V4_CM_CoinAccum', COSTS['gross_0'], window)
    event = result.pressure_events.iloc[0]
    assert event['trigger_price'] == 20_000.
    assert bool(event['gap'])
    assert event['raw_coin'] == pytest.approx(-1., abs=1e-14)
    assert event['zero_floor_coin'] == pytest.approx(1., abs=1e-14)
    assert event['residual_coin'] == 0.
    assert result.daily_returns.tolist() == [-1.,0.]
    assert result.metadata['final_coin'] == 0.
    assert result.attribution['zero_floor_coin'].sum() == pytest.approx(1., abs=1e-14)
    assert abs(result.metadata['account_reconciliation_error']) < 1e-9


def test_zero_cost_delta_hedge_same_price_feed_locks_dollars():
    changes = {10:{'open':25_000.,'low':25_000.,'close':25_000.},
               11:{'open':100_000.,'high':100_000.,'close':100_000.}}
    cm, features, window = fixture(days=1, changes=changes)
    spot, _, _ = fixture(days=1, venue='SPOT', changes=changes)
    features['g'] = False
    result = run_coin(cm, spot, features, 'V3_CM_DeltaMatched', COSTS['gross_0'], window)
    assert np.max(np.abs(result.minute['equity']-100_000.)) < 1e-9
    assert not result.metadata['liquidated']
    assert abs(result.metadata['account_reconciliation_error']) < 1e-9


def test_spot_full_budget_gross_clips_exactly_to_zero_cash():
    ctx, features, window = fixture(days=1, venue='SPOT', price=100., changes={0:{'open':200.,'high':200.}})
    result = run_spot(ctx,features,'V2_SPOT',COSTS['gross_0'],window)
    assert result.orders.iloc[0]['qty_after'] == 500.
    assert result.orders.iloc[0]['cash_after'] == 0.


def test_coin_bad_midnight_cancels_and_next_midnight_recomputes():
    cm, features, window = fixture(bad=(0,))
    spot, _, _ = fixture(venue='SPOT')
    features['g'] = False
    result = run_coin(cm,spot,features,'V3_CM_DeltaMatched',COSTS['gross_0'],window)
    assert result.decisions.iloc[0]['status'] == 'canceled_bad'
    assert (result.minute['face'].iloc[:1440] == 0).all()
    assert result.decisions.iloc[1]['retry']
    assert result.orders.iloc[0]['time'] == window.start+pd.Timedelta(days=1)
    assert result.orders.iloc[0]['face_target'] == -100_000.


def test_first_spot_conversion_cannot_be_deferred_across_bad_minute():
    cm, features, window = fixture(days=1)
    spot, _, _ = fixture(days=1,venue='SPOT',bad=(0,))
    with pytest.raises(ValueError,match='initial Spot conversion minute is bad'):
        run_coin(cm,spot,features,'HODL_1x_Coin',COSTS['gross_0'],window)
