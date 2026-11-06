"""Synthetic-only execution contract for the preregistered breakout study."""
import importlib
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd


START = pd.Timestamp('2020-03-01', tz='UTC')


def frame(n=180, start=START, price=100.0):
    idx = pd.date_range(start, periods=n, freq='min', tz='UTC')
    out = pd.DataFrame(dict(open=price, high=price + 0.5, low=price - 0.5,
                            close=price, volume=1.0, bad=False,
                            archive_gap=False), index=idx)
    out.attrs['provenance'] = 'synthetic'
    return out


def signals(times=None, eref=100.0, atr=1.0, eligible=True):
    times = [START + pd.Timedelta(hours=1)] if times is None else times
    idx = pd.DatetimeIndex(times, tz='UTC', name='entry_time')
    return pd.DataFrame(dict(signal_time=idx - pd.Timedelta(hours=1),
                             entry_time=idx, eref=eref, atr=atr,
                             sl=eref - 2 * atr, eligible=eligible,
                             skip_reason=''), index=idx)


class EngineTests(unittest.TestCase):
    def setUp(self):
        try:
            self.engine = importlib.import_module('breakout_study.engine')
        except ModuleNotFoundError:
            self.fail('The synthetic execution engine has not been implemented')
        self.scope = importlib.import_module('breakout_study.scope')

    def run_case(self, data, sig=None, cost='gross_0', funding=None):
        return self.engine.run_portfolio(self.scope.make_context(data, funding=funding),
                                         signals() if sig is None else sig, cost=cost)

    def test_quantity_is_frozen_before_gap_and_both_actual_fill_fees_remain(self):
        data = frame()
        data.loc[data.index[60], ['open', 'high', 'low', 'close']] = [90, 91, 89, 90]
        result = self.run_case(data, cost='main_17')
        t = result.trades.iloc[0]
        self.assertAlmostEqual(t.q, 500 / 2.17)
        self.assertAlmostEqual(t.entry_open, 90)
        self.assertAlmostEqual(t.entry_price, 90.045)
        self.assertAlmostEqual(t.exit_price, 89.982)
        self.assertAlmostEqual(t.entry_fee, t.q * 90.045 * .0005)
        self.assertAlmostEqual(t.exit_fee, t.q * 89.982 * .0005)
        self.assertEqual(t.exit_reason, 'gap_stop')
        self.assertEqual(t.minutes_held, 0)
        self.assertAlmostEqual(t.gross_bps, 0)
        self.assertAlmostEqual(t.net_pnl, t.q * (89.982 - 90.045) - t.entry_fee - t.exit_fee)
        self.assertAlmostEqual(result.equity.iloc[-1], 100000 + t.net_pnl)

    def test_entry_uses_next_minute_open_not_signal_close(self):
        data = frame()
        data.iloc[60, data.columns.get_indexer(['open', 'high', 'low', 'close'])] = [104, 105, 97, 100]
        t = self.run_case(data).trades.iloc[0]
        self.assertEqual(t.entry_time, START + pd.Timedelta(hours=1))
        self.assertEqual(t.entry_open, 104)
        self.assertEqual(t.q, 250)
        self.assertEqual(t.exit_ref, 98)
        self.assertEqual(t.exit_reason, 'intrabar_stop')
        self.assertTrue(t.risk_budget_exceeded)
        self.assertEqual(t.net_R, -3)

    def test_new_hourly_stop_only_applies_from_next_minute(self):
        data = frame(181)
        data.iloc[119, data.columns.get_indexer(['open', 'high', 'low', 'close'])] = [100, 106, 99, 105]
        data.iloc[120, data.columns.get_indexer(['open', 'high', 'low', 'close'])] = [101, 102, 100, 101]
        result = self.run_case(data)
        t = result.trades.iloc[0]
        self.assertEqual(t.exit_time, START + pd.Timedelta(hours=2))
        self.assertEqual(t.exit_ref, 101)
        update = result.stop_updates.iloc[0]
        self.assertEqual(update.old_stop, 98)
        self.assertEqual(update.new_stop, 102)
        self.assertEqual(update.effective_time, START + pd.Timedelta(hours=2))
        self.assertEqual(update.atr_entry, 1)

    def test_hourly_high_does_not_raise_trailing_stop(self):
        data = frame(181)
        data.iloc[90, data.columns.get_indexer(['high'])] = [150]
        data.iloc[120, data.columns.get_indexer(['low'])] = [97]
        result = self.run_case(data)
        self.assertEqual(result.trades.iloc[0].exit_ref, 98)
        self.assertEqual(result.stop_updates.iloc[0].new_stop, 98)

    def test_72_hour_exit_uses_preceding_minute_close(self):
        data = frame(4381)
        data.iloc[4379, data.columns.get_indexer(['close', 'high'])] = [101, 102]
        t = self.run_case(data).trades.iloc[0]
        self.assertEqual(t.exit_time, START + pd.Timedelta(hours=73))
        self.assertEqual(t.exit_reason, 'time_exit')
        self.assertEqual(t.exit_ref, 101)
        self.assertEqual(t.minutes_held, 4320)
        self.assertEqual(t.bars_held, 4320)

    def test_stop_wins_over_same_minute_time_exit(self):
        data = frame(4381)
        data.iloc[4379, data.columns.get_indexer(['low', 'close'])] = [97, 100]
        t = self.run_case(data).trades.iloc[0]
        self.assertEqual(t.exit_reason, 'intrabar_stop')
        self.assertEqual(t.exit_ref, 98)

    def test_bad_entry_is_cancelled_not_delayed(self):
        data = frame()
        data.loc[data.index[60], 'bad'] = True
        result = self.run_case(data)
        self.assertTrue(result.trades.empty)
        self.assertEqual(result.decisions.iloc[0].decision, 'bad_entry')
        self.assertTrue((result.equity == 100000).all())

    def test_halt_exits_first_valid_open_and_reports_overrun(self):
        data = frame(4386)
        data.loc[data.index[4378:4383], 'bad'] = True
        data.loc[data.index[4383], ['open', 'high', 'low', 'close']] = [90, 91, 89, 90]
        t = self.run_case(data).trades.iloc[0]
        self.assertEqual(t.exit_reason, 'halt_exit')
        self.assertEqual(t.exit_time, data.index[4383])
        self.assertEqual(t.exit_ref, 90)
        self.assertTrue(t.halt_in_holding)
        self.assertTrue(t.overrun_72h)
        self.assertEqual(t.overrun_minutes, 3)

    def test_open_position_at_end_raises_instead_of_silent_liquidation(self):
        with self.assertRaisesRegex(ValueError, 'open position|unclosed|cannot exit'):
            self.run_case(frame())

    def test_one_position_and_intrabar_cooldown_starts_at_minute_end(self):
        data = frame(481)
        data.loc[data.index[120], 'low'] = 97
        data.loc[data.index[420], 'low'] = 97
        sig = signals([START + pd.Timedelta(hours=h) for h in [1, 2, 6, 7]])
        result = self.run_case(data, sig)
        self.assertEqual(result.decisions.decision.tolist(), ['entered', 'in_position', 'cooldown', 'entered'])
        self.assertEqual(len(result.trades), 2)
        self.assertEqual(result.trades.iloc[0].cooldown_until, START + pd.Timedelta(hours=6, minutes=1))

    def test_funding_at_entry_is_excluded_and_at_open_exit_is_included(self):
        data = frame(181)
        data.loc[data.index[120], ['open', 'high', 'low', 'close']] = [97, 98, 96, 97]
        funding = pd.Series([.01, .01], index=data.index[[60, 120]])
        t = self.run_case(data, cost='main_17', funding=funding).trades.iloc[0]
        self.assertEqual(t.n_funding, 1)
        self.assertAlmostEqual(t.funding_cashflow, -t.q * 97 * .01)

    def test_previous_minute_intrabar_stop_excludes_next_funding(self):
        data = frame()
        data.loc[data.index[119], 'low'] = 97
        funding = pd.Series([.01], index=data.index[[120]])
        t = self.run_case(data, cost='main_17', funding=funding).trades.iloc[0]
        self.assertEqual(t.n_funding, 0)
        self.assertEqual(t.funding_cashflow, 0)

    def test_time_exit_funding_uses_closing_price_and_is_not_counted_twice(self):
        data = frame(4382)
        data.loc[data.index[4379], ['close', 'high']] = [101, 102]
        data.loc[data.index[4380], ['open', 'high', 'low', 'close']] = [110, 111, 109, 110]
        funding = pd.Series([.01], index=data.index[[4380]])
        result = self.run_case(data, cost='main_17', funding=funding)
        t = result.trades.iloc[0]
        self.assertEqual(t.n_funding, 1)
        self.assertAlmostEqual(t.funding_cashflow, -t.q * 101 * .01)
        self.assertEqual(result.equity.iloc[4379], result.equity.iloc[4380])

    def test_funding_has_no_count_cap_and_gross_keeps_count(self):
        data = frame(4381)
        funding = pd.Series(.001, index=data.index[120:4381:60])
        for cost in ['main_17', 'gross_0']:
            with self.subTest(cost=cost):
                t = self.run_case(data, cost=cost, funding=funding).trades.iloc[0]
                self.assertEqual(t.n_funding, 72)
                self.assertAlmostEqual(t.funding_cashflow, -t.q * 100 * .072 if cost == 'main_17' else 0)

    def test_equity_includes_unrealized_fees_and_calendar_flat_days(self):
        data = frame(4320)
        data.loc[data.index[60], ['close', 'high']] = [101, 102]
        data.loc[data.index[61], 'low'] = 97
        result = self.run_case(data, cost='main_17')
        t = result.trades.iloc[0]
        self.assertAlmostEqual(result.equity.iloc[60], 100000 - t.entry_fee + t.q * (101 - t.entry_price))
        self.assertEqual(len(result.equity), 4320)
        self.assertEqual(len(result.daily_returns), 3)
        self.assertEqual(result.daily_returns.iloc[1:].tolist(), [0, 0])
        self.assertAlmostEqual(result.daily_returns.iloc[0], t.net_pnl / 100000)

    def test_floor_cap_and_archive_gap_flags_are_recorded(self):
        data = frame()
        data.loc[data.index[60], 'low'] = 99.9
        data.loc[data.index[[59, 60]], 'archive_gap'] = True
        sig = signals(atr=.01)
        sig['signal_lookback_has_archive_gap'] = True
        t = self.run_case(data, sig).trades.iloc[0]
        self.assertEqual(t.q, 2000)
        self.assertTrue(t.floor_bound)
        self.assertTrue(t.cap_bound)
        self.assertTrue(t.has_archive_gap)
        self.assertTrue(t.signal_lookback_has_archive_gap)
        self.assertAlmostEqual(t.risk_amount, 40)

    def test_nonpositive_equity_disables_further_entries_and_marks_failure(self):
        data = frame(481)
        data['low'] = 99.99
        data.loc[data.index[61], ['open', 'high', 'low', 'close']] = [1, 2, .5, 1]
        sig = signals([START + pd.Timedelta(hours=h) for h in [1, 7]], atr=.01)
        result = self.run_case(data, sig)
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.decisions.iloc[1].decision, 'bankrupt')
        self.assertTrue(result.metadata['failed'])
        self.assertFalse(result.metadata['candidate_eligible'])

    def test_zero_or_negative_initial_stop_is_rejected_with_reason(self):
        result = self.run_case(frame(), signals(atr=60))
        self.assertTrue(result.trades.empty)
        self.assertEqual(result.decisions.iloc[0].decision, 'nonpositive_stop')

    def test_rejects_invalid_stops_nonfinite_signals_and_unknown_configuration(self):
        for key, value in [('sl', 99), ('eref', np.nan), ('atr', -1)]:
            sig = signals()
            sig.loc[sig.index[0], key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.run_case(frame(), sig)
        for kwargs in [dict(version='unknown'), dict(cost='unknown')]:
            with self.assertRaises(ValueError):
                self.engine.run_portfolio(self.scope.make_context(frame()), signals(), **kwargs)

    def test_last_entry_cutoff_and_no_warmup_entries(self):
        for start, decision in [('2023-12-29', 'too_late'), ('2020-02-29', 'before_main')]:
            data = frame(start=pd.Timestamp(start, tz='UTC'))
            result = self.run_case(data, signals([data.index[60]]))
            self.assertTrue(result.trades.empty)
            self.assertEqual(result.decisions.iloc[0].decision, decision)

    def test_empty_ledger_has_stable_columns_and_utc_time_types(self):
        result = self.run_case(frame(), signals(times=[]))
        for col in ['q', 'risk_amount', 'entry_fee', 'exit_fee', 'net_R', 'net_bps',
                    'gross_bps', 'n_funding', 'has_archive_gap', 'overrun_72h']:
            self.assertIn(col, result.trades.columns)
        self.assertEqual(str(result.trades.entry_time.dtype), 'datetime64[ns, UTC]')

    def test_midnight_funding_on_held_position_belongs_to_new_day(self):
        data = frame(2880)
        data.loc[data.index[1441], 'low'] = 97
        funding = pd.Series([.01], index=data.index[[1440]])
        result = self.run_case(data, signals([data.index[1380]]), 'main_17', funding)
        t = result.trades.iloc[0]
        self.assertAlmostEqual(result.equity.iloc[1440] - result.equity.iloc[1439], -t.q)
        self.assertAlmostEqual(result.daily_equity.iloc[0], 100000 - t.entry_fee + t.q * (100 - t.entry_price))
        self.assertAlmostEqual(result.daily_equity.iloc[1], 100000 + t.net_pnl)

    def test_midnight_time_exit_funding_belongs_to_ending_day(self):
        data = frame(5761)
        data.loc[data.index[5759], ['close', 'high']] = [101, 102]
        funding = pd.Series([.01], index=data.index[[5760]])
        result = self.run_case(data, signals([data.index[1440]]), 'main_17', funding)
        t = result.trades.iloc[0]
        self.assertEqual(t.exit_time, pd.Timestamp('2020-03-05', tz='UTC'))
        self.assertAlmostEqual(t.funding_cashflow, -t.q * 1.01)
        self.assertAlmostEqual(result.daily_equity.iloc[3], 100000 + t.net_pnl)
        self.assertEqual(result.daily_returns.iloc[4], 0)

    def test_funding_during_halt_and_recovery_includes_negative_rates(self):
        data = frame(181)
        data.loc[data.index[119:121], 'bad'] = True
        data.loc[data.index[121], ['open', 'high', 'low', 'close']] = [97, 98, 96, 97]
        funding = pd.Series([.01, -.02], index=data.index[[120, 121]])
        t = self.run_case(data, cost='main_17', funding=funding).trades.iloc[0]
        self.assertEqual(t.exit_reason, 'halt_exit')
        self.assertEqual(t.n_funding, 2)
        self.assertAlmostEqual(t.funding_cashflow, t.q * .94)

    def test_trailing_stop_never_loosens_and_keeps_entry_atr(self):
        data = frame(241)
        data.loc[data.index[119], ['close', 'high']] = [105, 106]
        data.loc[data.index[120:181], ['open', 'high', 'low', 'close']] = [104, 106, 103, 104]
        data.loc[data.index[179], 'close'] = 103
        data.loc[data.index[180], 'low'] = 101
        sig = signals([data.index[60], data.index[120]])
        sig.loc[data.index[120], ['eligible', 'atr', 'sl']] = [False, 100, -100]
        result = self.run_case(data, sig)
        self.assertEqual(result.stop_updates.new_stop.tolist(), [102, 102])
        self.assertEqual(result.trades.iloc[0].exit_ref, 102)
        self.assertEqual(result.trades.iloc[0].atr_entry, 1)

    def test_all_registered_costs_use_their_own_sizing_and_actual_fees(self):
        data = frame()
        data.loc[data.index[60], 'low'] = 97
        expected = [('gross_0', 250, 0, 0),
                    ('main_17', 500 / 2.17, .0005, .0002),
                    ('harsh_34', 500 / 2.34, .001, .0004)]
        for version in ['P_channel', 'V_vah']:
            for cost, qty, entry_slip, exit_slip in expected:
                with self.subTest(version=version, cost=cost):
                    result = self.engine.run_portfolio(self.scope.make_context(data), signals(), version, cost)
                    t = result.trades.iloc[0]
                    self.assertAlmostEqual(t.q, qty)
                    self.assertAlmostEqual(t.entry_price, 100 * (1 + entry_slip))
                    self.assertAlmostEqual(t.exit_price, 98 * (1 - exit_slip))
                    self.assertAlmostEqual(t.entry_fee, t.q * t.entry_price * entry_slip)
                    self.assertAlmostEqual(t.exit_fee, t.q * t.exit_price * entry_slip)

    def test_invalid_signal_timing_cannot_shift_an_order(self):
        for kind in ['mismatch', 'not_hour', 'duplicate', 'missing_bar']:
            sig = signals()
            if kind == 'mismatch':
                sig['signal_time'] += pd.Timedelta(minutes=1)
            elif kind == 'not_hour':
                sig.index += pd.Timedelta(minutes=1)
                sig['entry_time'] += pd.Timedelta(minutes=1)
                sig['signal_time'] += pd.Timedelta(minutes=1)
            elif kind == 'duplicate':
                sig = pd.concat([sig, sig])
            else:
                sig = signals([START + pd.Timedelta(hours=4)])
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.run_case(frame(), sig)

    def test_entry_requires_its_complete_signal_hour_in_context(self):
        data = frame(start=START + pd.Timedelta(minutes=1))
        data.loc[START + pd.Timedelta(hours=1), 'low'] = 97
        with self.assertRaisesRegex(ValueError, 'complete signal hour'):
            self.run_case(data)

    def test_real_source_guard_is_before_matching(self):
        ctx = SimpleNamespace(source='real', symbol='BTCUSDT')
        with self.assertRaises((ValueError, PermissionError)):
            self.engine.run_portfolio(ctx, signals())


if __name__ == '__main__':
    unittest.main()
