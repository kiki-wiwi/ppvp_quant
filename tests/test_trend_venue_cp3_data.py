"""CP3 parser and lock tests never open real market directories."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from trend_hold_venue_study import cp3_data as data, scope


def kline_bytes(times, *, header=False, micro=False, override=None, close_offset=None):
    rows = ['open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote,ignore'] if header else []
    for i, time in enumerate(times):
        stamp = pd.Timestamp(time).value // (1000 if micro else 1_000_000)
        offset = (60_000_000 - 1 if micro else 60_000 - 1) if close_offset is None else close_offset
        values = (override or {}).get(i, (100, 101, 99, 100, 1))
        rows.append(','.join(map(str, (stamp, *values, stamp + offset, 0, 1, 0, 0, 0))))
    return ('\n'.join(rows) + '\n').encode()


@pytest.mark.parametrize('micro,header', [(False, False), (False, True), (True, False), (True, True)])
def test_native_units_and_headers_retain_ohlcv_and_decision_clock(micro, header):
    times = ['2025-01-01T00:00Z', '2025-01-01T00:01Z']
    result = data.parse_klines(kline_bytes(times, micro=micro, header=header), '2025-01', 'ETH', 'SPOT')
    assert result.index.equals(pd.DatetimeIndex(times).as_unit('ns'))
    assert result.close.tolist() == [100., 100.]
    assert result.attrs['raw_close_time_exceptions'] == []


def test_all_abnormal_close_metadata_is_inventoried_without_changing_ohlcv():
    times = ['2024-01-01T00:00Z', '2024-01-01T00:01Z']
    result = data.parse_klines(kline_bytes(times, close_offset=1), '2024-01', 'BTC', 'UM')
    records = result.attrs['raw_close_time_exceptions']
    assert len(records) == 2
    assert all(r['decision_clock_close_ns'] == r['open_time_ns'] + scope.MINUTE_NS for r in records)
    assert all(r['raw_ohlcv_unchanged'] for r in records)
    assert result.close.tolist() == [100., 100.]


def test_invalid_native_prices_order_and_scope_stop():
    time = '2024-01-01T00:00Z'
    for values in [(0, 1, 0, 1, 1), (100, 99, 98, 100, 1), (100, 101, 99, 100, -1),
                   (100, 101, 99, 100, np.nan)]:
        with pytest.raises(ValueError):
            data.parse_klines(kline_bytes([time], override={0: values}), '2024-01', 'BTC', 'UM')
    with pytest.raises(ValueError, match='unique'):
        data.parse_klines(kline_bytes([time, time]), '2024-01', 'BTC', 'UM')
    with pytest.raises(ValueError):
        data.parse_klines(kline_bytes([time]), '2023-12', 'BTC', 'UM')


def test_grid_preserves_bad_and_archive_gap_distinction_and_rejects_zero_effective_day():
    start = pd.Timestamp('2024-01-01T00:00Z')
    raw = data.parse_klines(kline_bytes([start, start + pd.Timedelta(minutes=2)],
                            override={1: (120, 120, 120, 120, 0)}), '2024-01', 'BTC', 'UM')
    grid = data.build_grid(raw, start, start + pd.Timedelta(minutes=3))
    assert grid.bad.tolist() == [False, True, True]
    assert grid.archive_gap.tolist() == [False, True, False]
    assert grid.close.tolist() == [100., 100., 100.]
    with pytest.raises(ValueError, match='complete UTC day'):
        data.build_grid(raw, start, start + pd.Timedelta(days=2))


def test_eth_native_start_requires_exact_approved_0701_minute():
    raw = data.parse_klines(kline_bytes(['2020-08-18T07:00Z', '2020-08-18T07:01Z'],
                            override={0: (100, 100, 100, 100, 0)}), '2020-08', 'ETH', 'CM')
    assert data.native_start(raw, 'ETH', 'CM') == pd.Timestamp('2020-08-18T07:01Z')
    for wrong in ('2020-08-17T23:59Z', '2020-08-18T07:00Z', '2020-08-18T07:02Z'):
        early = data.parse_klines(kline_bytes([wrong]), '2020-08', 'ETH', 'CM')
        with pytest.raises(ValueError, match='native'):
            data.native_start(early, 'ETH', 'CM')


@pytest.mark.parametrize('btc,post', [(False, False), (False, True), (True, False)])
def test_cp3_load_gate_precedes_any_filesystem_io(monkeypatch, btc, post):
    monkeypatch.setattr(scope, 'BTC_IS_UNLOCKED', btc)
    monkeypatch.setattr(scope, 'POST_IS_AND_ETH_UNLOCKED', post)
    def forbidden(*args, **kwargs):
        raise AssertionError('I/O before scope gate')
    monkeypatch.setattr(Path, 'read_bytes', forbidden)
    monkeypatch.setattr(Path, 'read_text', forbidden)
    with pytest.raises(scope.RealDataLockedError):
        data.load_inputs('/synthetic/no-market-data')


def test_actual_baseline_allows_raw_preflight_only_and_transition_requires_both(monkeypatch):
    monkeypatch.setattr(scope, 'BTC_IS_UNLOCKED', True)
    monkeypatch.setattr(scope, 'POST_IS_AND_ETH_UNLOCKED', False)
    data.require_preflight()
    with pytest.raises(scope.RealDataLockedError):
        data.require_cp3()
    monkeypatch.setattr(scope, 'POST_IS_AND_ETH_UNLOCKED', True)
    data.require_cp3()
    with pytest.raises(scope.RealDataLockedError):
        data.require_preflight()


def test_preparation_failure_is_logged_once_and_cannot_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(scope, 'BTC_IS_UNLOCKED', True)
    monkeypatch.setattr(scope, 'POST_IS_AND_ETH_UNLOCKED', False)
    def failure(*args, **kwargs):
        raise ValueError('synthetic first failure')
    monkeypatch.setattr(data, '_prepare', failure)
    with pytest.raises(ValueError, match='synthetic first failure'):
        data.prepare_inputs(tmp_path)
    import json
    stopped = json.loads((tmp_path / data.STOPPED).read_text())
    assert stopped['status'] == 'STOPPED'
    assert stopped['error'] == 'synthetic first failure'
    with pytest.raises(FileExistsError):
        data.prepare_inputs(tmp_path)


def test_gap_acceptance_requires_exact_fixed_eth_spot_whitelist():
    approved = data.approved_gap_ranges('ETH', 'SPOT')
    assert len(approved) == 10
    assert sum(row['minutes'] for row in approved) == 1424
    times = []
    for row in approved:
        times.extend(pd.date_range(row['start'], row['end_exclusive'], freq='min', inclusive='left'))
    grid = pd.DataFrame({'archive_gap': True}, index=pd.DatetimeIndex(times).as_unit('ns'))
    result = data.accept_grid(grid, approved, asset='ETH', venue='SPOT')
    assert result['approved_archive_gap_minutes'] == 1424
    assert result['unknown_archive_gap_minutes'] == 0
    with pytest.raises(ValueError, match='unknown archive gap'):
        data.accept_grid(grid.iloc[:-1], approved, asset='ETH', venue='SPOT')
    with pytest.raises(ValueError, match='approved ETH Spot'):
        data.accept_grid(grid, approved, asset='BTC', venue='SPOT')
    with pytest.raises(ValueError, match='approved ETH Spot'):
        data.accept_grid(grid, [{**approved[0], 'evidence_sha256': 'b' * 64}], asset='ETH', venue='SPOT')
    arbitrary = [{'start': '2024-01-01T00:01:00+00:00', 'end_exclusive': '2024-01-01T00:02:00+00:00',
                  'minutes': 1, 'reason': 'invented', 'evidence_path': 'invented.json', 'evidence_sha256': 'a' * 64}]
    with pytest.raises(ValueError, match='approved ETH Spot'):
        data.accept_grid(grid, arbitrary, asset='ETH', venue='SPOT')
    assert data.approved_gap_ranges('ETH', 'CM') == []


def test_bound_btc_warmup_uses_only_november_december_predicates(monkeypatch, tmp_path):
    bars_index = pd.date_range(data.WARM_START, data.POST_START, freq='min', inclusive='left').as_unit('ns')
    fund_index = pd.date_range(data.WARM_START, data.POST_START, freq='8h', inclusive='left').as_unit('ns')
    bars = pd.DataFrame({'open': 100., 'high': 101., 'low': 99., 'close': 100., 'volume': 1.,
                         'missing': False, 'placeholder': False, 'bad': False, 'archive_gap': False}, index=bars_index)
    bars.index.name = 'open_time'
    funding_frame = pd.DataFrame({'funding_rate': 0.0001}, index=fund_index)
    funding_frame.index.name = 'funding_time'
    def entry(path):
        return {'path': path, 'parquet_sha256': 'a' * 64, 'content_sha256': 'b' * 64}
    meta = {'caches': {'SPOT': entry('data/processed_1m/venue_cp2_BTC_SPOT_1m_IS.parquet'),
                       'CM': entry('data/processed_1m/venue_cp2_BTC_CM_1m_IS.parquet'),
                       'CM_funding': entry('data/processed_1m/venue_cp2_BTC_CM_funding_IS.parquet')}}
    binding = {'UM': {'path': 'data/processed_1m/BTCUSDT_perp_1m_IS.parquet',
                       'current_parquet_sha256': 'a' * 64, 'content_sha256': 'b' * 64}}
    reused = {'funding_cache': {'path': 'data/processed/BTCUSDT_funding.parquet', 'actual_sha256': 'a' * 64}}
    documents = {data.CP2_META: meta, data.CP2_BINDING: binding, data.CP2_REUSED: reused}
    monkeypatch.setattr(data, '_read_bound_json', lambda root, relative, record: documents[relative])
    monkeypatch.setattr(data.archives, 'file_hash', lambda path: 'a' * 64)
    predicates = []
    def read_fake(path, *, filters):
        predicates.append(filters)
        return funding_frame.copy() if 'funding' in str(path) else bars.copy()
    monkeypatch.setattr(pd, 'read_parquet', read_fake)
    record = {'warmup_sources': []}
    result = data._bounded_warmup(tmp_path, record)
    assert set(result) == {'UM', 'SPOT', 'CM', 'UM_funding', 'CM_funding'}
    assert len(predicates) == 5
    for filters in predicates:
        assert filters[0][1:] == ('>=', data.WARM_START)
        assert filters[1][1:] == ('<', data.POST_START)
    assert all(source['bounded_rows'] in (87840, 183) for source in record['warmup_sources'])


def test_stopped_attempt_blocks_loader_and_hash_verifier_before_manifest_read(tmp_path, monkeypatch):
    monkeypatch.setattr(scope, 'BTC_IS_UNLOCKED', True)
    monkeypatch.setattr(scope, 'POST_IS_AND_ETH_UNLOCKED', True)
    marker = tmp_path / data.STOPPED
    marker.parent.mkdir(parents=True)
    marker.write_text('{"status":"STOPPED"}')
    def forbidden(*args, **kwargs):
        raise AssertionError('STOPPED attempt reached manifest read')
    monkeypatch.setattr(Path, 'read_bytes', forbidden)
    with pytest.raises(ValueError, match='STOPPED'):
        data.load_inputs(tmp_path)
    with pytest.raises(ValueError, match='STOPPED'):
        data.verify_prepared_inputs(tmp_path)


@pytest.mark.parametrize('close_offset', [-1, 0, 60_000, 60_001])
def test_nonplaceholder_close_outside_authorized_native_clock_is_rejected(close_offset):
    raw = kline_bytes(['2024-01-01T00:00Z'], close_offset=close_offset)
    with pytest.raises(ValueError, match='close time'):
        data.parse_klines(raw, '2024-01', 'BTC', 'UM')


def test_negative_close_is_allowed_only_for_flat_zero_volume_placeholder():
    raw = kline_bytes(['2024-01-01T00:00Z'], override={0: (100, 100, 100, 100, 0)}, close_offset=-1)
    result = data.parse_klines(raw, '2024-01', 'BTC', 'UM')
    assert result.iloc[0].tolist() == [100., 100., 100., 100., 0.]
    records = result.attrs['raw_close_time_exceptions']
    assert len(records) == 1
    assert records[0]['raw_close_time_ns'] < records[0]['open_time_ns']
    assert records[0]['existing_placeholder_rule_marks_bad'] is True
    assert records[0]['decision_clock_close_ns'] == records[0]['open_time_ns'] + scope.MINUTE_NS
    with pytest.raises(ValueError, match='close time'):
        data.parse_klines(kline_bytes(['2024-01-01T00:00Z'],
                          override={0: (100, 100, 100, 100, 0)}, close_offset=60_000),
                          '2024-01', 'BTC', 'UM')


@pytest.mark.parametrize('micro,close_offset', [(False, 1), (False, 59_998),
                                               (True, 1), (True, 59_999_998)])
def test_authorized_early_close_keeps_native_prices_and_availability(micro, close_offset):
    result = data.parse_klines(kline_bytes(['2025-01-01T00:00Z'], micro=micro,
                              close_offset=close_offset), '2025-01', 'ETH', 'SPOT')
    record = result.attrs['raw_close_time_exceptions'][0]
    assert record['open_time_ns'] < record['raw_close_time_ns'] < record['expected_native_close_ns']
    assert record['decision_clock_close_ns'] == record['open_time_ns'] + scope.MINUTE_NS
    assert record['raw_ohlcv_unchanged'] is True
    assert result.iloc[0].tolist() == [100., 101., 99., 100., 1.]


def test_revised_native_cutoff_is_june_first_exclusive():
    assert data.END == pd.Timestamp('2026-06-01', tz='UTC')
    with pytest.raises(ValueError, match='scope'):
        data.parse_klines(b'never parsed', '2026-06', 'ETH', 'SPOT')


def test_daily_supplement_preserves_monthly_bars_and_four_exact_absent_dates():
    monthly = data.parse_klines(kline_bytes(['2023-08-27T23:59Z']), '2023-08', 'ETH', 'CM')
    original = monthly.copy()
    for day in ('2023-08-28', '2023-08-29', '2023-08-30', '2023-08-31'):
        index = pd.date_range(day, periods=1440, freq='min', tz='UTC')
        daily = data.parse_klines(kline_bytes(index), '2023-08', 'ETH', 'CM')
        monthly = data._merge_daily(monthly, daily, day)
    assert len(monthly) == 1 + 4 * 1440
    pd.testing.assert_frame_equal(monthly.loc[original.index], original)


def test_eth_spot_monthly_native_minutes_before_gap_are_not_discarded():
    index = pd.date_range('2020-12-21T13:48Z', periods=21, freq='min')
    native = data.parse_klines(kline_bytes(index), '2020-12', 'ETH', 'SPOT')
    assert len(native) == 21
    assert native.index[-1] == pd.Timestamp('2020-12-21T14:08Z')
    assert data.approved_gap_ranges('ETH', 'SPOT')[1]['start'] == '2020-12-21T14:09:00+00:00'
