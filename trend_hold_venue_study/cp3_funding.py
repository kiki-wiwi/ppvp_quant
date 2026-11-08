"""CP3 funding-only Scheme A: exact Decimal overlap and bounded official dapi.

Canonical records retain original timestamps and source paths. ETH CM begins at
the actual 08:00 settlement; the absent prelaunch 00:00 slot is never invented.
"""
from __future__ import annotations
import calendar
import csv
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_EVEN
import io
import json
import urllib.parse
import urllib.request

from .cp3_archives import REG, digest, encoded, month_bounds, months, stamp, write_new
from .cp3_windows import END_DATE, LAST_MONTH
from .cp3_reuse import ReuseInventory, hardlink_new, require_download_space

API = 'https://dapi.binance.com/dapi/v1/fundingRate'
STEP = 8 * 60 * 60 * 1000
MINUTE = 60_000
TOLERANCE = Decimal('1e-8')


def ms(value):
    return int(datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp() * 1000)


END = ms(END_DATE)
AUTHORITY = ms('2022-07-01')
STARTS = {'BTC': ms('2024-01-01'), 'ETH': ms('2020-08-18')}
ETH_NATIVE_START = ms('2020-08-18T07:01')
ETH_FIRST_SETTLEMENT = ms('2020-08-18T08:00')


def integer(value):
    number = Decimal(str(value))
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError('finite integer timestamp required')
    return int(number)


def rate(value):
    number = Decimal(str(value))
    if not number.is_finite():
        raise ValueError('finite Decimal funding rate required')
    return str(value)


def align_time(value):
    original = integer(value)
    micro = original > 10**14
    native_ms = Decimal(original) / (1000 if micro else 1)
    nearest = int((native_ms / STEP).to_integral_value(rounding=ROUND_HALF_EVEN)) * STEP
    rounded = int((native_ms / MINUTE).to_integral_value(rounding=ROUND_HALF_EVEN)) * MINUTE
    if abs(native_ms - nearest) >= MINUTE or rounded != nearest:
        raise ValueError('funding timestamp does not round to UTC 8h grid within one minute')
    offset = native_ms - nearest
    return nearest, int(offset) if offset == offset.to_integral_value() else float(offset), ('us' if micro else 'ms')


def unique(rows, label, allow_empty=False):
    times = [row['time'] for row in rows]
    if (not times and not allow_empty) or times != sorted(set(times)):
        raise ValueError(label + ': ordered unique settlement times required')
    return times


def validate_request(asset, parameters):
    if (asset not in STARTS or set(parameters) != {'symbol', 'startTime', 'endTime', 'limit'}
            or parameters['symbol'] != asset + 'USD_PERP' or parameters['limit'] != 1000):
        raise ValueError('only fixed asset-specific funding request allowed')
    start, end = integer(parameters['startTime']), integer(parameters['endTime'])
    if not STARTS[asset] <= start <= end < END:
        raise ValueError('funding request outside registered CP3 native history')
    # A request is confined to one registered month, never an unbounded history.
    if datetime.fromtimestamp(start / 1000, timezone.utc).strftime('%Y-%m') != datetime.fromtimestamp(end / 1000, timezone.utc).strftime('%Y-%m'):
        raise ValueError('funding request must stay in one month')
    return start, end


def parse_api(raw, asset, start, end):
    validate_request(asset, dict(symbol=asset + 'USD_PERP', startTime=start, endTime=end, limit=1000))
    payload = json.loads(raw)
    if not isinstance(payload, list) or len(payload) > 1000:
        raise ValueError('unexpected funding API payload')
    rows = []
    for item in payload:
        if not isinstance(item, dict) or item.get('symbol') != asset + 'USD_PERP':
            raise ValueError('unexpected funding API symbol')
        original = integer(item['fundingTime'])
        if not start <= original <= end:
            raise ValueError('API settlement outside exact requested time bounds')
        time, offset, unit = align_time(original)
        if not (ETH_FIRST_SETTLEMENT if asset == 'ETH' else STARTS[asset]) <= time < END:
            raise ValueError('API funding outside registered scope after rounding')
        rows.append(dict(time=time, rate=rate(item['fundingRate']), raw_time=original,
                         offset_ms=offset, timestamp_unit=unit))
    unique(rows, 'API page', allow_empty=True)
    return rows


def parse_vision(raw, asset, venue, month):
    if asset not in STARTS or venue not in ('UM', 'CM'):
        raise ValueError('registered funding asset/venue required')
    lower = '2024-01' if asset == 'BTC' else ('2022-07' if venue == 'CM' else '2020-07')
    if not lower <= month <= LAST_MONTH:
        raise ValueError('funding archive outside registered scope')
    begin, finish = month_bounds(month)
    start, end = ms(begin.isoformat()), ms(finish.isoformat())
    if not isinstance(raw, bytes) or not raw.strip():
        raise ValueError('nonempty supplied funding CSV bytes required')
    table = list(csv.reader(io.StringIO(raw.decode('utf-8-sig'))))
    if table and table[0] == ['calc_time', 'funding_interval_hours', 'last_funding_rate']:
        table = table[1:]
    rows = []
    units = set()
    for item in table:
        if len(item) != 3 or Decimal(item[1]) != 8:
            raise ValueError('official funding must contain three columns and 8h intervals')
        original = integer(item[0])
        time, offset, unit = align_time(original)
        units.add(unit)
        if not start <= time < end:
            raise ValueError('funding outside registered archive month')
        rows.append(dict(time=time, rate=rate(item[2]), raw_time=original,
                         offset_ms=offset, timestamp_unit=unit))
    if len(units) != 1:
        raise ValueError('funding archive must use one native timestamp unit')
    unique(rows, 'Vision funding')
    return rows


def compare_overlap(api, vision):
    a_times = unique(api, 'API overlap')
    v_times = unique(vision, 'Vision overlap')
    if a_times != v_times:
        raise ValueError('funding overlap has missing or extra settlement')
    differences = []
    for left, right in zip(api, vision):
        delta = Decimal(left['rate']) - Decimal(right['rate'])
        if abs(delta) > TOLERANCE:
            raise ValueError('CM funding overlap absolute Decimal difference exceeds 1e-8')
        differences.append(dict(funding_time_ms=left['time'], dapi_rate=left['rate'],
                                vision_rate=right['rate'], decimal_diff=str(delta),
                                abs_decimal_diff=str(abs(delta))))
    exact = sum(Decimal(row['decimal_diff']) == 0 for row in differences)
    return dict(settlements=len(differences), exact_decimal_match_count=exact,
                max_abs_decimal_diff=str(max(abs(Decimal(row['decimal_diff'])) for row in differences)),
                tolerance='1e-8', comparisons=differences)


def month_end_times(start, end):
    allowed = set()
    first = datetime.fromtimestamp(start / 1000, timezone.utc).strftime('%Y-%m')
    last = datetime.fromtimestamp((end - 1) / 1000, timezone.utc).strftime('%Y-%m')
    for month in months(first, last):
        year, number = map(int, month.split('-'))
        midnight = ms(f'{month}-{calendar.monthrange(year, number)[1]:02d}')
        allowed.update(t for t in (midnight + n * STEP for n in range(3)) if start <= t < end)
    return allowed


def _canonical(row, source):
    return dict(funding_time_ms=row['time'], funding_rate=row['rate'], source=source,
                raw_funding_time=row['raw_time'], timestamp_unit=row.get('timestamp_unit', 'ms'),
                source_path=row.get('source_path'), offset_ms=row['offset_ms'])


def canonical_hash(records):
    return digest(json.dumps(records, separators=(',', ':'), allow_nan=False).encode())


def assemble(api, vision, asset, *, start=None, end=END):
    start = STARTS[asset] if start is None else start
    if not STARTS[asset] <= start < end <= END:
        raise ValueError('canonical CM scope differs from registered bounds')
    api_times = unique(api, 'all dapi funding')
    vision_times = unique(vision, 'all Vision funding', allow_empty=True)
    expected = list(range(start, end, STEP))
    if asset == 'ETH' and start == STARTS['ETH']:
        if api_times[0] != ETH_FIRST_SETTLEMENT:
            raise ValueError('ETH first actual funding settlement must be 2020-08-18 08:00 UTC')
        expected = list(range(ETH_FIRST_SETTLEMENT, end, STEP))
    if api_times != expected:
        raise ValueError('CM dapi funding grid missing, duplicate or outside scope')
    overlap_expected = {time for time in expected if time >= AUTHORITY}
    if not set(vision_times).issubset(overlap_expected):
        raise ValueError('Vision funding outside registered authority interval')
    missing = overlap_expected - set(vision_times)
    allowed = month_end_times(max(start, AUTHORITY), end) if end > AUTHORITY else set()
    if not missing.issubset(allowed):
        raise ValueError('Vision gap exceeds approved month-end three settlements')
    # Entire later months can never be replaced by dapi, even if unusually short.
    for month in months(datetime.fromtimestamp(max(start, AUTHORITY) / 1000, timezone.utc).strftime('%Y-%m'),
                        datetime.fromtimestamp((end - 1) / 1000, timezone.utc).strftime('%Y-%m')):
        begin, finish = month_bounds(month)
        lo, hi = ms(begin.isoformat()), ms(finish.isoformat())
        if any(lo <= t < hi for t in overlap_expected) and not any(lo <= t < hi for t in vision_times):
            raise ValueError('entire later Vision funding month is absent')
    api_map = {row['time']: row for row in api}
    vision_map = {row['time']: row for row in vision}
    overlap = (compare_overlap([api_map[t] for t in vision_times], vision) if vision else
               dict(settlements=0, exact_decimal_match_count=0, max_abs_decimal_diff='0', tolerance='1e-8', comparisons=[]))
    canonical = []
    for time in api_times:
        if time < AUTHORITY:
            row, source = api_map[time], 'dapi_early_native'
        elif time in missing:
            row, source = api_map[time], 'dapi_month_end_supplement'
        else:
            row, source = vision_map[time], 'vision'
        canonical.append(_canonical(row, source))
    launch = (dict(first_native_minute_ms=ETH_NATIVE_START, first_actual_settlement_ms=ETH_FIRST_SETTLEMENT,
                   absent_prelaunch_slot_ms=STARTS['ETH'], absent_prelaunch_slot_is_not_record=True,
                   context_exclusion_count=0) if asset == 'ETH' else None)
    return dict(canonical_records=canonical, canonical_sha256=canonical_hash(canonical),
                launch=launch, overlap=overlap, month_end_supplement_count=len(missing),
                month_end_supplements=[r for r in canonical if r['source'] == 'dapi_month_end_supplement'],
                authority_split_ms=AUTHORITY, first_settlement_ms=api_times[0],
                last_settlement_ms=api_times[-1], maximum_abs_timestamp_offset_ms=max(abs(r['offset_ms']) for r in api + vision))


def canonical_um(rows, start, end):
    if unique(rows, 'UM funding') != list(range(start, end, STEP)):
        raise ValueError('UM funding must match complete native 8h grid')
    return [_canonical(row, 'vision') for row in rows]


def for_context(records, native_start, end, asset):
    import pandas as pd
    start = pd.Timestamp(native_start)
    finish = pd.Timestamp(end)
    if start.tz is None or finish.tz is None:
        raise ValueError('aware funding context bounds required')
    selected, excluded = [], []
    times = [r['funding_time_ms'] for r in records]
    if not times or times != sorted(set(times)):
        raise ValueError('canonical funding must remain ordered and unique')
    for row in records:
        time = pd.Timestamp(row['funding_time_ms'], unit='ms', tz='UTC')
        if time < start:
            raise ValueError('funding before exact native Context start is not an approved record')
        elif time >= finish:
            raise ValueError('canonical funding beyond context end')
        else:
            selected.append(row)
    index = pd.DatetimeIndex(pd.to_datetime([row['funding_time_ms'] for row in selected], unit='ms', utc=True)).as_unit('ns')
    index.name = 'funding_time'
    return pd.Series([float(Decimal(row['funding_rate'])) for row in selected], index=index, name='funding_rate'), excluded


def acquire_api(root, asset, record, persist, reuse=None):
    from pathlib import Path
    result = []
    reuse = reuse if reuse is not None else ReuseInventory(root, record, persist)
    first = '2024-01' if asset == 'BTC' else '2020-08'
    for month in months(first):
        begin, finish = month_bounds(month)
        start, end = max(STARTS[asset], ms(begin.isoformat())), ms(finish.isoformat()) - 1
        for page in range(1, 3):
            parameters = dict(symbol=asset + 'USD_PERP', startTime=start, endTime=end, limit=1000)
            validate_request(asset, parameters)
            url = API + '?' + urllib.parse.urlencode(parameters)
            requested = stamp()
            path = f'{REG}/venue_cp3_revision_1_dapi_{asset}_{month}_{page:02d}.json'
            if (Path(root) / path).exists():
                raise FileExistsError('CP3 revision API source already exists; never overwrite or retry')
            saved = reuse.api_source(url, parameters)
            if saved is None:
                require_download_space(root)
                request = urllib.request.Request(url, headers={'User-Agent': 'ppvp-quant-preregistered-research/CP3'})
                with urllib.request.urlopen(request, timeout=60) as response:
                    raw = response.read()
                write_new(Path(root) / path, raw)
            else:
                source, raw = saved
                hardlink_new(source['path'], Path(root) / path)
            entry = dict(path=path, url=url, parameters=parameters, sha256=digest(raw), bytes=len(raw),
                         requested_at=requested if saved is None else None, received_at=stamp(), asset=asset)
            if saved is not None:
                entry.update(reused_from=source, provenance='authenticated saved exact funding URL; no request')
            else:
                entry['provenance'] = 'new bounded official funding request; inventory contained no source'
            record['api_pages'].append(entry)
            persist()
            rows = parse_api(raw, asset, start, end)
            for row in rows:
                row['source_path'] = path
            entry['rows'] = len(rows)
            result.extend(rows)
            persist()
            if len(rows) < 1000:
                break
            start = rows[-1]['raw_time'] + 1
        else:
            raise ValueError('bounded monthly funding pagination exceeded two pages')
    return result
