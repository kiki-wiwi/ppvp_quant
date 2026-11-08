"""CP3 matrix, authorization and one-shot control; no real input reads."""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from trend_hold_venue_study import config, cp3_runtime as runtime, scope


def test_exact_87_jobs_asset_signal_identity_and_registered_order():
    rows = runtime.jobs()
    assert len(rows) == len({row['job_id'] for row in rows}) == 87
    assert [(row['asset'], row['window']) for row in rows[:30]] == [('BTC', 'Post-IS')] * 30
    assert [(row['asset'], row['window']) for row in rows[30:54]] == [('ETH', 'IS-Common')] * 24
    assert [(row['asset'], row['window']) for row in rows[54:78]] == [('ETH', 'Post-IS')] * 24
    assert [row['window'] for row in rows[78:81]] == ['IS-CM-SelfSignal'] * 3
    assert all(row['whole_contracts'] for row in rows[24:27] + rows[84:87])
    assert all(row['signal_source'] == 'CM' for row in rows[27:30] + rows[78:84])
    assert [row['version'] for row in rows[:24:3]] == list(config.VERSIONS)
    assert [row['cost'] for row in rows[:3]] == ['gross_0', 'main_17', 'harsh_34']
    for row in rows:
        assert row['job_id'] == runtime.job_id(row['asset'], row['window'], row['signal_source'],
            row['version'], row['cost'], row['whole_contracts'])


def test_exact_21_paired_families_with_post_segments_and_eth_IS():
    families = runtime.PAIRED_FAMILIES
    assert len(families) == len(set(families)) == 21
    assert sum(asset == 'BTC' for asset, *_ in families) == 9
    assert sum(asset == 'ETH' for asset, *_ in families) == 12
    assert set(period for asset, period, *_ in families if asset == 'BTC') == {'Post-IS', 'VAL', 'OOS'}
    assert set(period for asset, period, *_ in families if asset == 'ETH') == {'IS-Common', 'Post-IS', 'VAL', 'OOS'}


def test_actual_baseline_and_real_post_guard_precede_index_or_feature_access(monkeypatch):
    assert scope.BTC_IS_UNLOCKED is True
    assert scope.POST_IS_AND_ETH_UNLOCKED is False
    contexts = {asset: {venue: SimpleNamespace(source='real', asset=asset, venue=venue)
        for venue in config.VENUES} for asset in config.ASSETS}
    with pytest.raises(scope.RealDataLockedError, match='POST'):
        runtime._validate_inputs(contexts)
    monkeypatch.setattr(scope, 'POST_IS_AND_ETH_UNLOCKED', True)
    with pytest.raises(ValueError, match='Context'):
        runtime._validate_inputs(contexts)


def test_synthetic_provenance_does_not_require_real_unlock(monkeypatch):
    contexts = {asset: {venue: SimpleNamespace(source='synthetic', asset=asset, venue=venue)
        for venue in config.VENUES} for asset in config.ASSETS}
    with pytest.raises(ValueError, match='Context'):
        runtime._validate_inputs(contexts)
    contexts['ETH']['CM'].source = 'real'
    with pytest.raises(ValueError, match='provenance'):
        runtime._validate_inputs(contexts)


def test_first_failure_is_recorded_without_retry_or_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, '_validate_inputs', lambda contexts: 'synthetic')
    days = pd.date_range('2024-01-01', periods=2, tz='UTC').as_unit('ns')
    frame = pd.DataFrame({'g': [True, False], 'budget': [.5, .5]}, index=days)
    monkeypatch.setattr(runtime, '_prepare_features', lambda contexts: {'BTC': {'Post-IS': {'UM': frame}}})
    attempted = []
    def failure(contexts, schedules, job):
        attempted.append(job['job_id'])
        raise ValueError('synthetic first error')
    monkeypatch.setattr(runtime, '_run_account', failure)
    out = tmp_path / 'one_attempt'
    with pytest.raises(ValueError, match='synthetic first error'):
        runtime.execute({}, out)
    stopped = json.loads((out / 'STOPPED.json').read_text())
    assert attempted == [runtime.jobs()[0]['job_id']]
    assert stopped['completed_jobs'] == []
    assert stopped['automatic_retry'] is False
    with pytest.raises(FileExistsError):
        runtime.execute({}, out)
    assert len(attempted) == 1


def test_archived_cp2_read_is_guarded_before_filesystem_access(tmp_path, monkeypatch):
    monkeypatch.setattr(scope, 'POST_IS_AND_ETH_UNLOCKED', False)
    with pytest.raises(scope.RealDataLockedError):
        runtime.load_cp2_comparison(tmp_path / 'does_not_exist', 'a' * 64)
    with pytest.raises(scope.RealDataLockedError):
        runtime.load_cp2_chart_accounts(tmp_path / 'does_not_exist', {}, pd.DataFrame())


def test_registered_archive_hash_verifies_exact_bytes_and_known_path(tmp_path):
    from hashlib import sha256
    folder = tmp_path / 'reports/trend_hold_venue_round/cp2_btc_is'
    folder.mkdir(parents=True)
    data = b'field\n1\n'
    (folder / 'comparison.csv').write_bytes(data)
    relative = 'reports/trend_hold_venue_round/cp2_btc_is/comparison.csv'
    assert runtime._verified_cp2_path(tmp_path, 'comparison.csv', {relative: sha256(data).hexdigest()}) == folder / 'comparison.csv'
    with pytest.raises(ValueError, match='hash mismatch'):
        runtime._verified_cp2_path(tmp_path, 'comparison.csv', {relative: '0' * 64})
    with pytest.raises(ValueError, match='registered'):
        runtime._verified_cp2_path(tmp_path, 'comparison.csv', {})


def test_archived_chart_adapter_uses_verified_observed_ledgers_only(tmp_path, monkeypatch):
    from hashlib import sha256
    import numpy as np
    from trend_hold_venue_study.runtime import _write_table
    monkeypatch.setattr(scope, 'POST_IS_AND_ETH_UNLOCKED', True)
    days = pd.date_range('2020-09-01', '2024-01-01', inclusive='left', tz='UTC').as_unit('ns')
    versions = ('V1_USDM', 'V3_CM_DeltaMatched', 'V4_CM_CoinAccum', 'HODL_1x_Coin')
    base = tmp_path / 'reports/trend_hold_venue_round/cp2_btc_is'
    hashes, rows = {}, []
    for version in versions:
        folder = base / 'accounts' / ('IS-Common__' + version + '__main_17')
        folder.mkdir(parents=True)
        daily = pd.DataFrame({'equity': 100000., 'return': 0.}, index=days)
        minute = pd.DataFrame({'equity': 100000., 'bad': False, 'actual_leverage': 1.},
            index=days + pd.Timedelta(hours=23, minutes=59))
        metadata = {'version': version, 'initial_capital': 100000., 'final_equity': 100000.}
        if version in config.CM_VERSIONS:
            daily['coin_equity'], minute['coin_equity'] = 2., 2.
            metadata['initial_coin'] = 2.
            attr = pd.DataFrame({'time': [days[0], days[1]],
                'funding_spot_usd': np.array([np.longdouble('1.125'), np.longdouble('-.125')])})
            _write_table(attr, folder / 'attribution.csv.gz')
        else:
            pd.DataFrame({'time': [days[0], days[1]], 'cash_flow': [1.125, -.125]}).to_csv(
                folder / 'funding.csv.gz', index=False)
        daily.to_csv(folder / 'daily.csv')
        minute.to_parquet(folder / 'minute.parquet')
        (folder / 'metadata.json').write_text(json.dumps(metadata))
        pd.DataFrame().to_csv(folder / 'pressure_events.csv.gz', index=False)
        for path in folder.iterdir():
            hashes[path.relative_to(tmp_path).as_posix()] = sha256(path.read_bytes()).hexdigest()
        rows.append({'asset': 'BTC', 'window': 'IS-Common', 'signal_source': 'UM', 'version': version,
            'cost': 'main_17', 'whole_contracts': False, 'funding_usd': 1.})
    accounts, comparison, evidence = runtime.load_cp2_chart_accounts(tmp_path, hashes, pd.DataFrame(rows))
    assert len(accounts) == len(comparison) == len(evidence) == 4
    assert all(item['no_account_or_feature_recomputation'] for item in evidence)
    for version in versions:
        identity = runtime.job_id('BTC', 'IS-Common', 'UM', version, 'main_17')
        account = accounts[identity]
        assert account['job']['job_id'] == identity
        assert 'minute' not in account
        assert account['chart_daily']['funding_usd'].iloc[:2].tolist() == [1.125, -.125]
        assert account['chart_daily']['funding_cumulative_usd'].iloc[-1] == 1.
        assert account['chart_daily']['equity_drawdown_minute_min'].min() == 0.
        assert account['pressure_events'].empty


def test_synthetic_chart_stage_skips_real_cp2_requirement_without_rendering(tmp_path, monkeypatch):
    import sys
    calls = []
    def render(*args):
        calls.append(args)
        return ['one.png', 'two.png', 'three.png', 'four.png']
    expected = ('one.png', 'two.png', 'three.png', 'four.png')
    monkeypatch.setitem(sys.modules, 'trend_hold_venue_study.cp3_plotting',
        SimpleNamespace(FIGURES=expected, render_outputs=render))
    figures, reason = runtime._render_outputs('synthetic', {}, {}, {}, pd.DataFrame(), {}, tmp_path)
    assert figures == []
    assert reason == 'synthetic-only execution has no real CP2 archive'
    assert calls == []
    figures, reason = runtime._render_outputs('real', {}, {}, {}, pd.DataFrame(), {}, tmp_path)
    assert figures == list(expected)
    assert reason is None
    assert len(calls) == 1


def test_all_post_jobs_pass_revised_window_and_exact_days_to_frozen_engines(monkeypatch):
    from trend_hold_venue_study import linear, portfolios
    days = pd.date_range('2024-01-01', '2026-06-02', inclusive='left', tz='UTC').as_unit('ns')
    frame = pd.DataFrame({'g': True, 'budget': .5}, index=days)
    schedules = {asset: {'Post-IS': {'UM': frame, 'CM': frame}} for asset in config.ASSETS}
    contexts = {asset: {venue: SimpleNamespace(asset=asset, venue=venue)
        for venue in config.VENUES} for asset in config.ASSETS}
    slices, received = [], []
    def context_slice(ctx, start, end):
        slices.append((ctx.asset, ctx.venue, start, end))
        return ctx
    def linear_or_spot(ctx, schedule, version, cost, window):
        received.append((schedule, window))
        return window
    def coin(ctx, spot, schedule, version, cost, window, *, whole_contracts):
        received.append((schedule, window))
        return window
    monkeypatch.setattr(runtime, '_context_slice', context_slice)
    monkeypatch.setattr(linear, 'run_linear', linear_or_spot)
    monkeypatch.setattr(portfolios, 'run_spot', linear_or_spot)
    monkeypatch.setattr(portfolios, 'run_coin', coin)
    post_jobs = [job for job in runtime.jobs() if job['window'] == 'Post-IS']
    for job in post_jobs:
        result = runtime._run_account(contexts, schedules, job)
        assert result.start == pd.Timestamp('2024-01-01T00:00Z')
        assert result.end == pd.Timestamp('2026-06-01T00:00Z')
        assert result.signal_origin == result.start
    assert len(received) == len(post_jobs) == 60
    for schedule, window in received:
        assert len(schedule) == 882
        assert schedule.index[-1] == pd.Timestamp('2026-05-31T00:00Z')
        assert len(schedule.loc[:'2025-06-30']) == 547
        assert len(schedule.loc['2025-07-01':]) == 335
    assert all(start == pd.Timestamp('2023-12-31T00:00Z')
        and end == pd.Timestamp('2026-06-01T00:00Z') for _, _, start, end in slices)
    assert config.window('Post-IS').end == pd.Timestamp('2026-11-01T00:00Z')


def test_post_feature_requests_end_at_amended_boundary_and_keep_native_eth_start(monkeypatch):
    from trend_hold_venue_study import features
    contexts = {asset: {venue: SimpleNamespace(asset=asset, venue=venue,
        index=pd.DatetimeIndex([pd.Timestamp('2020-08-18T07:01Z')
            if asset == 'ETH' and venue == 'CM' else pd.Timestamp('2020-07-01T00:00Z')]))
        for venue in config.VENUES} for asset in config.ASSETS}
    requests, slices = [], []
    def context_slice(ctx, start, end):
        slices.append((ctx.asset, ctx.venue, start, end))
        return ctx
    def build_features(ctx, origin, *, end):
        requests.append((ctx.asset, ctx.venue, origin, end))
        return pd.DataFrame(index=pd.date_range(origin, end, inclusive='left', freq='D'))
    monkeypatch.setattr(runtime, '_context_slice', context_slice)
    monkeypatch.setattr(features, 'build_features', build_features)
    schedules = runtime._prepare_features(contexts)
    post = [request for request in requests if request[2] == pd.Timestamp('2024-01-01T00:00Z')]
    assert len(post) == 4
    assert all(end == pd.Timestamp('2026-06-01T00:00Z') for _, _, _, end in post)
    for asset in config.ASSETS:
        for source in ('UM', 'CM'):
            assert len(schedules[asset]['Post-IS'][source]) == 882
    assert ('ETH', 'CM', pd.Timestamp('2020-08-18T07:01Z'), pd.Timestamp('2024-01-01T00:00Z')) in slices


def test_paired_statistics_clip_post_and_oos_to_amended_end(tmp_path, monkeypatch):
    import numpy as np
    from trend_hold_venue_study import statistics
    days = pd.date_range('2024-01-01', '2026-11-01', inclusive='left', tz='UTC').as_unit('ns')
    daily = pd.DataFrame({'return': 0.}, index=days)
    metrics = pd.DataFrame([
        {'period': 'Post-IS', 'start_equity': 100000., 'final_equity': 132000., 'total_return': .32},
        {'period': 'VAL', 'start_equity': 100000., 'final_equity': 120000., 'total_return': .2},
        {'period': 'OOS', 'start_equity': 120000., 'final_equity': 132000., 'total_return': .1},
    ])
    times = pd.DatetimeIndex(['2025-06-30T23:59Z', '2025-07-01T00:00Z',
        '2026-05-31T23:59Z', '2026-06-01T00:00Z'])
    accounts = {runtime.job_id('BTC', 'Post-IS', 'UM', version, cost): {
        'daily': daily, 'order_times': times, 'period_metrics': metrics}
        for cost in ('main_17', 'harsh_34') for version in ('V1_USDM', 'P_USDM')}
    schedule = pd.DataFrame({'g': True, 'budget': .5}, index=days)
    observed = []
    def analyze(main_t, main_p, harsh_t, harsh_p, features, order_count):
        observed.append((main_t.index, order_count))
        assert all(series.index.equals(main_t.index) for series in (main_p, harsh_t, harsh_p))
        assert features.index.equals(main_t.index)
        return {**{name: pd.DataFrame() for name in ('tests', 'intervals', 'gates', 'daily', 'monthly')},
            'metadata': {}, 'matrices': {'synthetic': np.empty((0, 0))}}
    monkeypatch.setattr(statistics, 'analyze', analyze)
    monkeypatch.setattr(runtime, 'PAIRED_FAMILIES', tuple(
        ('BTC', period, 'UM', 'V1_USDM', 'P_USDM') for period in ('Post-IS', 'VAL', 'OOS')))
    result = runtime._statistics(accounts, {'BTC': {'Post-IS': {'UM': schedule}}}, tmp_path, lambda message: None)
    assert [len(index) for index, _ in observed] == [882, 547, 335]
    assert [count for _, count in observed] == [3, 1, 2]
    assert observed[0][0][-1] == observed[2][0][-1] == pd.Timestamp('2026-05-31T00:00Z')
    assert observed[2][0][0] == pd.Timestamp('2025-07-01T00:00Z')
    actual = result['BTC__OOS__UM']['metadata']['actual_continuous_wealth_ratios']['main_T']
    assert actual['start_equity'] == 120000.
    assert actual['final_equity'] == 132000.
