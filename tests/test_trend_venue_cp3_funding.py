"""Decimal reconciliation uses small invented records and supplied raw bytes."""
import json
from decimal import Decimal
import pandas as pd
import pytest
from trend_hold_venue_study import cp3_funding as funding


def row(time, rate='0.0001', source='synthetic'):
    ms = int(pd.Timestamp(time).value // 1_000_000)
    return {'time': ms, 'rate': rate, 'raw_time': ms, 'offset_ms': 0, 'source_path': source}


def test_decimal_overlap_includes_exact_tolerance_and_rejects_more():
    api = [row('2024-01-01T00:00Z', '0.00010001')]
    vision = [row('2024-01-01T00:00Z', '0.0001')]
    observed = funding.compare_overlap(api, vision)
    assert Decimal(observed['max_abs_decimal_diff']) == Decimal('1e-8')
    assert observed['comparisons'][0]['decimal_diff'] == '1E-8'
    with pytest.raises(ValueError, match='1e-8'):
        funding.compare_overlap([row('2024-01-01T00:00Z', '0.00010001001')], vision)
    with pytest.raises(ValueError, match='overlap'):
        funding.compare_overlap(api, [])


@pytest.mark.parametrize('micro,header', [(False, False), (False, True), (True, False), (True, True)])
def test_vision_funding_keeps_decimal_and_native_time_units(micro, header):
    ms = funding.ms('2024-01-01')
    raw_time = (ms + 2) * (1000 if micro else 1)
    raw = (('calc_time,funding_interval_hours,last_funding_rate\n' if header else '')
           + f'{raw_time},8,0.000100000001\n').encode()
    rows = funding.parse_vision(raw, 'BTC', 'UM', '2024-01')
    assert rows[0]['time'] == ms
    assert rows[0]['rate'] == '0.000100000001'
    assert rows[0]['offset_ms'] == 2


def test_api_request_is_bounded_and_symbol_is_checked_before_rows():
    start = funding.ms('2020-08-18')
    end = funding.ms('2020-09-01') - 1
    params = {'symbol': 'ETHUSD_PERP', 'startTime': start, 'endTime': end, 'limit': 1000}
    assert funding.validate_request('ETH', params) == (start, end)
    raw = json.dumps([{'symbol': 'ETHUSD_PERP', 'fundingTime': start + funding.STEP,
                       'fundingRate': '-0.0001'}]).encode()
    assert funding.parse_api(raw, 'ETH', start, end)[0]['rate'] == '-0.0001'
    with pytest.raises(ValueError):
        funding.validate_request('ETH', {**params, 'startTime': start - 1})
    with pytest.raises(ValueError):
        funding.parse_api(raw, 'BTC', start, end)


def test_scheme_a_only_fills_month_end_three_and_preserves_vision_authority():
    times = pd.date_range('2024-01-01', '2024-02-01', freq='8h', inclusive='left', tz='UTC')
    api = [row(t, '0.000100001') for t in times]
    vision = [row(t) for t in times if t.day != 31]
    result = funding.assemble(api, vision, 'BTC', start=funding.ms('2024-01-01'),
                              end=funding.ms('2024-02-01'))
    assert result['month_end_supplement_count'] == 3
    assert result['canonical_records'][0]['funding_rate'] == '0.0001'
    assert result['canonical_records'][-1]['source'] == 'dapi_month_end_supplement'
    with pytest.raises(ValueError, match='gap'):
        funding.assemble(api, vision[1:], 'BTC', start=funding.ms('2024-01-01'),
                         end=funding.ms('2024-02-01'))


def test_eth_launch_starts_at_actual_0800_and_has_no_context_exclusion():
    times = pd.date_range('2020-08-18T08:00Z', '2020-09-01T00:00Z', freq='8h', inclusive='left')
    rows = [row(t) for t in times]
    result = funding.assemble(rows, [], 'ETH', end=funding.ms('2020-09-01'))
    original = list(result['canonical_records'])
    series, excluded = funding.for_context(original, pd.Timestamp('2020-08-18T07:01Z'),
                                           pd.Timestamp('2020-09-01T00:00Z'), 'ETH')
    assert excluded == []
    assert len(series) == len(original)
    assert original[0]['funding_time_ms'] == funding.ms('2020-08-18T08:00')
    assert all(item['funding_time_ms'] != funding.ms('2020-08-18') for item in original)
    assert result['launch']['absent_prelaunch_slot_ms'] == funding.ms('2020-08-18')
    assert result['launch']['context_exclusion_count'] == 0
    assert result['canonical_records'] == original
    with pytest.raises(ValueError, match='first actual'):
        funding.assemble([row('2020-08-18T00:00Z')] + rows, [], 'ETH', end=funding.ms('2020-09-01'))
    with pytest.raises(ValueError, match='first actual'):
        funding.assemble(rows[1:], [], 'ETH', end=funding.ms('2020-09-01'))


def test_um_rejects_missing_settlement_and_nonfinite_rates():
    rows = [row('2024-01-01T00:00Z'), row('2024-01-01T16:00Z')]
    with pytest.raises(ValueError, match='grid'):
        funding.canonical_um(rows, funding.ms('2024-01-01'), funding.ms('2024-01-02'))
    bad = b'1704067200000,8,NaN\n'
    with pytest.raises(ValueError, match='finite'):
        funding.parse_vision(bad, 'BTC', 'UM', '2024-01')


def test_revised_funding_cutoff_rejects_june_before_parsing():
    assert funding.END == funding.ms('2026-06-01')
    with pytest.raises(ValueError, match='scope'):
        funding.parse_vision(b'unparsed', 'ETH', 'CM', '2026-06')
    with pytest.raises(ValueError, match='history'):
        funding.validate_request('BTC', {'symbol': 'BTCUSD_PERP', 'startTime': funding.ms('2026-06-01'),
                                          'endTime': funding.ms('2026-07-01') - 1, 'limit': 1000})


def test_exact_saved_api_url_is_reused_with_new_name_and_no_request(tmp_path, monkeypatch):
    import urllib.parse
    from trend_hold_venue_study import cp3_archives as archives
    first = funding.ms('2020-08-18')
    last = funding.ms('2020-09-01') - 1
    params = dict(symbol='ETHUSD_PERP', startTime=first, endTime=last, limit=1000)
    url = funding.API + '?' + urllib.parse.urlencode(params)
    raw = json.dumps([{'symbol': 'ETHUSD_PERP', 'fundingTime': first + funding.STEP, 'fundingRate': '0.0001'}]).encode()
    source = tmp_path / 'saved-api.json'
    source.write_bytes(raw)
    path = tmp_path / archives.REG / 'reuse_sources.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'archives': {}, 'listings': {}, 'api_pages': {
        url: {'path': str(source), 'sha256': archives.digest(raw), 'url': url, 'parameters': params}}}))
    monkeypatch.setattr(funding, 'months', lambda first: ['2020-08'])
    monkeypatch.setattr(funding.urllib.request, 'urlopen', lambda *args, **kwargs: pytest.fail('exact saved API URL requested again'))
    record = {'api_pages': [], 'source_registrations': []}
    rows = funding.acquire_api(tmp_path, 'ETH', record, lambda: None)
    assert rows[0]['time'] == first + funding.STEP
    saved = record['api_pages'][0]
    assert saved['reused_from']['path'] == str(source)
    assert 'venue_cp3_revision_1_dapi_' in saved['path']
    assert (tmp_path / saved['path']).stat().st_ino == source.stat().st_ino
