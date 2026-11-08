"""CP3 input-only preflight and independently gated immutable Context loading.

Raw preflight never imports features, signals, budgets, accounting or engines.
Every source/cache uses a new path and bound bytes. The authorized revision
reuses verified saved sources; a failed revision attempt never retries itself. Frozen CP2 parsers are reused only where they are pure and scope-neutral.
"""
from __future__ import annotations
import re
import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import scope
from . import cp3_archives as archives
from . import cp3_funding as funding
from .cp3_windows import END_DATE, LAST_MONTH
from .cp3_reuse import ReuseInventory
from .data import _table, _timestamps, _ordered, _bounds, build_grid, frame_content_sha256

ROOT = Path(__file__).resolve().parents[1]
REG = archives.REG
MANIFEST = REG + '/venue_cp3_inputs.json'
PROGRESS = REG + '/venue_cp3_input_progress.json'
STOPPED = REG + '/STOPPED_venue_cp3_inputs.json'
OHLCV = ('open', 'high', 'low', 'close', 'volume')
GRID_COLUMNS = (*OHLCV, 'missing', 'placeholder', 'bad', 'archive_gap')
END = pd.Timestamp(END_DATE, tz='UTC')
ETH_CM_NATIVE_START = pd.Timestamp('2020-08-18T07:01:00+00:00')
POST_START = pd.Timestamp('2024-01-01', tz='UTC')
WARM_START = pd.Timestamp('2023-11-01', tz='UTC')
CP2_META = 'reports/trend_hold_venue_round/cp2_btc_is/input_verification.json'
CP2_BINDING = 'reports/trend_hold_venue_round/cp2_registration/input_revision_2/input_binding.json'
CP2_REUSED = 'reports/trend_hold_venue_round/cp2_registration/reused_um_inputs.json'
GAP_EVIDENCE = REG + '/approved_eth_spot_gap_ranges.json'
GAP_EVIDENCE_SHA256 = '8af0e5eafba5261d126762380fe7cce7ce12ada35ebfd51691e652882040d1f3'
# Frozen exact ranges from the approved diagnostic; monthly native bars survive.
ETH_SPOT_APPROVED_GAPS = (
    ('2020-11-30T06:00:00+00:00', '2020-11-30T07:00:00+00:00', 60),
    ('2020-12-21T14:09:00+00:00', '2020-12-21T18:00:00+00:00', 231),
    ('2020-12-25T02:00:00+00:00', '2020-12-25T03:00:00+00:00', 60),
    ('2021-02-11T03:41:00+00:00', '2021-02-11T05:00:00+00:00', 79),
    ('2021-03-06T02:00:00+00:00', '2021-03-06T03:30:00+00:00', 90),
    ('2021-04-20T02:00:00+00:00', '2021-04-20T04:30:00+00:00', 150),
    ('2021-04-25T04:01:00+00:00', '2021-04-25T08:45:00+00:00', 284),
    ('2021-08-13T02:00:00+00:00', '2021-08-13T06:30:00+00:00', 270),
    ('2021-09-29T07:00:00+00:00', '2021-09-29T09:00:00+00:00', 120),
    ('2023-03-24T12:40:00+00:00', '2023-03-24T14:00:00+00:00', 80),
)



def require_preflight():
    if not scope.BTC_IS_UNLOCKED or scope.POST_IS_AND_ETH_UNLOCKED:
        raise scope.RealDataLockedError('raw CP3 preflight requires BTC=True and POST=False')


def require_cp3():
    if not scope.BTC_IS_UNLOCKED or not scope.POST_IS_AND_ETH_UNLOCKED:
        raise scope.RealDataLockedError('CP3 requires BTC and POST-IS/ETH independently unlocked')


def cache_path(asset, venue, kind='1m'):
    if asset not in ('BTC', 'ETH') or venue not in ('UM', 'SPOT', 'CM') or kind not in ('1m', 'funding'):
        raise ValueError('unregistered CP3 cache identity')
    if venue == 'SPOT' and kind == 'funding':
        raise ValueError('Spot cannot have funding cache')
    return f'data/processed_1m/venue_cp3_{asset}_{venue}_{kind}.parquet'


def parse_klines(raw, month, asset, venue):
    """Pure supplied bytes parser; raw close metadata never controls availability."""
    if asset not in ('BTC', 'ETH') or venue not in ('UM', 'SPOT', 'CM'):
        raise ValueError('registered CP3 asset and venue required')
    lower = '2024-01' if asset == 'BTC' else ('2020-08' if venue == 'CM' else '2020-07')
    if not lower <= month <= LAST_MONTH:
        raise ValueError('archive month outside CP3 native source scope')
    begin, finish = archives.month_bounds(month)
    start, end = pd.Timestamp(begin, tz='UTC'), pd.Timestamp(finish, tz='UTC')
    table = _table(raw, 12)
    index = _timestamps(table.iloc[:, 0], 'kline open')
    _ordered(index, 'klines')
    _bounds(index, start, end, 'klines')
    if np.any(index.asi8 % scope.MINUTE_NS):
        raise ValueError('native open must lie on minute boundary')
    values = table.iloc[:, 1:6].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('finite native OHLCV required')
    if np.any(values[:, :4] <= 0) or np.any(values[:, 4] < 0):
        raise ValueError('positive native prices and nonnegative volume required')
    op, high, low, close, _ = values.T
    if np.any(high < np.maximum(op, close)) or np.any(low > np.minimum(op, close)) or np.any(high < low):
        raise ValueError('invalid native OHLC bounds')
    closes = _timestamps(table.iloc[:, 6], 'kline close')
    raw_open = pd.to_numeric(table.iloc[:, 0], errors='raise').to_numpy()
    raw_close = pd.to_numeric(table.iloc[:, 6], errors='raise').to_numpy()
    micro = bool((raw_open > 10**14).all())
    if bool((raw_close > 10**14).all()) != micro:
        raise ValueError('open and close timestamps use different native units')
    native_tick = 1000 if micro else 1_000_000
    expected_close = index.asi8 + scope.MINUTE_NS - native_tick
    records = []
    for position in np.flatnonzero(closes.asi8 != expected_close):
        opening = int(index.asi8[position])
        ending = int(closes.asi8[position])
        price = values[position]
        placeholder = bool(price[4] == 0 and price[0] == price[1] == price[2] == price[3])
        early_close = opening < ending < int(expected_close[position])
        negative_placeholder = ending < opening and placeholder
        if not (early_close or negative_placeholder):
            raise ValueError('native close time outside authorized early-close or negative-placeholder exception')
        records.append(dict(asset=asset, venue=venue, month=month,
                            open_time_ns=opening, raw_close_time_ns=ending,
                            open_time_utc=index[position].isoformat(),
                            raw_close_time_utc=closes[position].isoformat(),
                            raw_close_minus_open_ns=ending - opening,
                            expected_native_close_ns=int(expected_close[position]),
                            decision_clock_close_ns=opening + scope.MINUTE_NS,
                            timestamp_unit='us' if micro else 'ms', raw_ohlcv_unchanged=True,
                            existing_placeholder_rule_marks_bad=placeholder,
                            exception_kind='negative_close_placeholder' if negative_placeholder else 'early_close',
                            handling='authorized raw close metadata exception; available at open_time + 1min'))
    result = pd.DataFrame(values, index=index, columns=OHLCV)
    result.index.name = 'open_time'
    result.attrs = dict(raw_close_time_exceptions=records, timestamp_unit='us' if micro else 'ms')
    return result


def native_start(frame, asset, venue):
    index = scope.utc_index(frame.index, 'native history')
    if not len(index):
        raise ValueError('native history is empty')
    flat = (frame.open == frame.high) & (frame.high == frame.low) & (frame.low == frame.close)
    valid = ~((frame.volume == 0) & flat)
    if not valid.any():
        raise ValueError('no effective native minute')
    first = index[np.flatnonzero(valid.to_numpy())[0]]
    lower = (POST_START if asset == 'BTC' else
             pd.Timestamp('2020-08-18' if venue == 'CM' else '2020-07-01', tz='UTC'))
    if venue == 'CM' and asset == 'ETH':
        if first != ETH_CM_NATIVE_START:
            raise ValueError('ETH CM first native minute must be exactly 2020-08-18 07:01 UTC')
    elif first != lower:
        raise ValueError('native history must begin at registered first minute')
    return first


def _ranges(index):
    """Lossless compact inventory of selected minute ranges; no inferred cause."""
    index = pd.DatetimeIndex(index).as_unit('ns')
    if not len(index):
        return []
    splits = np.flatnonzero(np.diff(index.asi8) != scope.MINUTE_NS) + 1
    boundaries = np.r_[0, splits, len(index)]
    return [dict(start=index[a].isoformat(), end_exclusive=(index[b - 1] + pd.Timedelta(minutes=1)).isoformat(),
                 minutes=int(b - a)) for a, b in zip(boundaries[:-1], boundaries[1:])]



def approved_gap_ranges(asset, venue):
    if (asset, venue) != ('ETH', 'SPOT'):
        return []
    return [dict(start=start, end_exclusive=end, minutes=minutes,
                 reason='user-approved exact ETH Spot source gap; no inferred maintenance cause',
                 evidence_path=GAP_EVIDENCE, evidence_sha256=GAP_EVIDENCE_SHA256)
            for start, end, minutes in ETH_SPOT_APPROVED_GAPS]


def accept_grid(grid, approved_gaps=(), *, asset=None, venue=None):
    """Accept only the exact ten approved ETH Spot ranges, totaling 1,424 minutes."""
    observed = _ranges(grid.index[grid.archive_gap])
    expected = approved_gap_ranges(asset, venue)
    if list(approved_gaps) != expected:
        raise ValueError('gap approval differs from fixed approved ETH Spot whitelist or asset identity')
    approved = [{key: row[key] for key in ('start', 'end_exclusive', 'minutes')} for row in expected]
    if observed != approved:
        raise ValueError('unknown archive gap requires approval; raw structure retained, no successful input registration')
    return dict(approved_archive_gap_minutes=sum(row['minutes'] for row in observed),
                unknown_archive_gap_minutes=0, approved_ranges=expected)


def _merge_daily(monthly, daily, day):
    lower = pd.Timestamp(day, tz='UTC')
    upper = lower + pd.Timedelta(days=1)
    if daily.empty or daily.index[0] < lower or daily.index[-1] >= upper:
        raise ValueError('daily archive outside exact registered UTC date')
    _ordered(daily.index, 'daily supplement')
    overlap = monthly.index.intersection(daily.index)
    if len(overlap) and not np.array_equal(monthly.loc[overlap, list(OHLCV)].to_numpy(), daily.loc[overlap, list(OHLCV)].to_numpy()):
        raise ValueError('monthly/daily overlap OHLCV differs')
    # The approved reason is a wholly absent date, never replacement of native bars.
    if len(overlap) or ((monthly.index >= lower) & (monthly.index < upper)).any():
        raise ValueError('daily supplementation is limited to wholly absent UTC dates')
    return pd.concat([monthly, daily]).sort_index()


def _month_frame(acquisition, spec, record, persist):
    exists = acquisition.available(spec)
    if exists:
        entry, raw = acquisition.obtain(spec)
        frame = parse_klines(raw, spec['month'], spec['asset'], spec['venue'])
        _register_parsed(entry, frame, record)
        persist()
    else:
        frame = pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], tz='UTC', name='open_time')).astype(float)
        record['missing_monthly_archives'].append({**spec, 'reason': 'absent in bounded official listing'})
        persist()
    begin, finish = archives.month_bounds(spec['month'])
    lower = pd.Timestamp(begin, tz='UTC')
    upper = pd.Timestamp(finish, tz='UTC')
    if (spec['asset'], spec['venue'], spec['month']) == ('ETH', 'CM', '2020-08'):
        lower = pd.Timestamp('2020-08-18', tz='UTC')
        # Days preceding the observed first native bar are never synthesized.
        if len(frame):
            lower = max(lower, frame.index[0].floor('D'))
    present = set(frame.index.floor('D'))
    missing_days = [day for day in pd.date_range(lower, upper, freq='D', inclusive='left') if day not in present]
    for day in missing_days:
        reason = 'absent_whole_utc_date' if exists else 'missing_month'
        daily_spec = archives.daily_spec(spec, day.strftime('%Y-%m-%d'), reason)
        entry, raw = acquisition.obtain(daily_spec)
        daily = parse_klines(raw, spec['month'], spec['asset'], spec['venue'])
        expected = pd.date_range(day, day + pd.Timedelta(days=1), freq='min', inclusive='left').as_unit('ns')
        is_launch = spec['asset'] == 'ETH' and spec['venue'] == 'CM' and day == pd.Timestamp('2020-08-18', tz='UTC')
        if not is_launch and not daily.index.equals(expected):
            raise ValueError('approved daily supplement must contain complete 1440 native UTC minutes')
        frame = _merge_daily(frame, daily, day.strftime('%Y-%m-%d'))
        _register_parsed(entry, daily, record)
        entry['added_minutes'] = len(daily)
        persist()
    if frame.empty:
        raise ValueError('no official native rows in registered month')
    return frame


def _register_parsed(entry, frame, record):
    exceptions = [{**row, 'source_path': entry['path'], 'archive_relative_path': entry['relative_path']}
                  for row in frame.attrs.get('raw_close_time_exceptions', [])]
    entry.update(parsed_rows=len(frame), first_time=frame.index[0].isoformat(),
                 last_time=frame.index[-1].isoformat(), content_sha256=frame_content_sha256(frame),
                 timestamp_unit=frame.attrs.get('timestamp_unit'), raw_close_time_exception_count=len(exceptions))
    record['raw_close_time_exceptions'].extend(exceptions)


def _read_bound_json(root, relative, record):
    raw = (root / relative).read_bytes()
    record['source_registrations'].append(dict(path=relative, sha256=archives.digest(raw), bytes=len(raw)))
    return json.loads(raw)


def _bounded_warmup(root, record):
    """Only Nov-Dec2023 predicates; hash all bytes without parsing other dates."""
    metadata = _read_bound_json(root, CP2_META, record)
    binding = _read_bound_json(root, CP2_BINDING, record)
    reused = _read_bound_json(root, CP2_REUSED, record)
    sources = {'UM': dict(path=binding['UM']['path'], parquet_sha256=binding['UM']['current_parquet_sha256'],
                          content_sha256=binding['UM']['content_sha256']),
               'SPOT': metadata['caches']['SPOT'], 'CM': metadata['caches']['CM'],
               'UM_funding': dict(path=reused['funding_cache']['path'], parquet_sha256=reused['funding_cache']['actual_sha256']),
               'CM_funding': metadata['caches']['CM_funding']}
    expected_paths = {'UM': 'data/processed_1m/BTCUSDT_perp_1m_IS.parquet',
                      'SPOT': 'data/processed_1m/venue_cp2_BTC_SPOT_1m_IS.parquet',
                      'CM': 'data/processed_1m/venue_cp2_BTC_CM_1m_IS.parquet',
                      'UM_funding': 'data/processed/BTCUSDT_funding.parquet',
                      'CM_funding': 'data/processed_1m/venue_cp2_BTC_CM_funding_IS.parquet'}
    result = {}
    for key, source in sources.items():
        if source['path'] != expected_paths[key]:
            raise ValueError('warmup source path differs from CP2 registered input')
        path = root / source['path']
        actual = archives.file_hash(path)
        if actual != source['parquet_sha256']:
            raise ValueError('warmup cache byte hash differs from existing CP2 binding: ' + key)
        is_funding = key.endswith('_funding')
        index_name = 'funding_time' if is_funding else 'open_time'
        filters = [(index_name, '>=', WARM_START), (index_name, '<', POST_START)]
        frame = pd.read_parquet(path, filters=filters)
        frame.index = scope.utc_index(frame.index, key + ' bounded warmup')
        if len(frame) == 0 or frame.index[0] < WARM_START or frame.index[-1] >= POST_START:
            raise ValueError('bounded warmup predicate returned out-of-range data')
        if is_funding:
            frame.index = scope.utc_index(frame.index.round('min'), key + ' warmup rounded')
            frame.index.name = 'funding_time'
            series = frame['funding_rate'].astype(float)
            expected = pd.date_range(WARM_START, POST_START, freq='8h', inclusive='left').as_unit('ns')
            if not frame.index.equals(expected) or not np.isfinite(series.to_numpy()).all():
                raise ValueError('warmup funding requires exact continuous 8h settlements')
            result[key] = series
        else:
            if key == 'UM':
                frame = build_grid(frame, WARM_START, POST_START)
            frame = frame.loc[:, list(GRID_COLUMNS)].copy()
            expected = pd.date_range(WARM_START, POST_START, freq='min', inclusive='left').as_unit('ns')
            if not frame.index.equals(expected):
                raise ValueError('warmup requires exact Nov-Dec2023 minute grid')
            result[key] = frame
        record['warmup_sources'].append({**source, 'sha256': actual,
                                         'predicate': [['index', '>=', WARM_START.isoformat()], ['index', '<', POST_START.isoformat()]],
                                         'bounded_rows': len(frame), 'bounded_content_sha256': frame_content_sha256(frame)})
    return result


def _write_cache(root, asset, venue, kind, frame):
    relative = cache_path(asset, venue, kind)
    destination = root / relative
    if destination.exists():
        raise FileExistsError('new CP3 cache already exists; no overwrite: ' + relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive descriptor prevents a competing writer from being overwritten.
    with destination.open('xb') as stream:
        frame.to_parquet(stream, index=True)
    return dict(path=relative, asset=asset, venue=venue, kind=kind, rows=len(frame),
                columns=list(frame.columns), first_time=frame.index[0].isoformat(),
                last_time=frame.index[-1].isoformat(), parquet_sha256=archives.file_hash(destination),
                content_sha256=frame_content_sha256(frame))


def _prepare(root, record, persist):
    reuse = ReuseInventory(root, record, persist, required=True)
    evidence = (root / GAP_EVIDENCE).read_bytes()
    if archives.digest(evidence) != GAP_EVIDENCE_SHA256:
        raise ValueError('approved ETH Spot gap evidence changed')
    record['source_registrations'].append(dict(path=GAP_EVIDENCE, sha256=GAP_EVIDENCE_SHA256, bytes=len(evidence)))
    record['approved_gaps'] = {'ETH_SPOT': approved_gap_ranges('ETH', 'SPOT')}
    warmup = _bounded_warmup(root, record)
    persist()
    acquisition = archives.Acquisition(root, record, persist, reuse=reuse)
    catalog = archives.archive_specs()
    for asset in ('BTC', 'ETH'):
        record['caches'][asset] = {}
        for venue in ('UM', 'SPOT', 'CM'):
            record['current_input'] = f'{asset}/{venue}'
            persist()
            parts = [_month_frame(acquisition, spec, record, persist) for spec in catalog
                     if (spec['asset'], spec['venue'], spec['kind']) == (asset, venue, 'klines')]
            native = pd.concat(parts)
            del parts
            _ordered(native.index, 'joined native monthly archives')
            start = POST_START if asset == 'BTC' else native_start(native, asset, venue)
            if asset == 'BTC':
                # Existing closed warmup supplies the prior valid valuation if the
                # first Post minute is a registered zero-volume placeholder.
                combined = pd.concat([warmup[venue].loc[:, list(OHLCV)], native])
                completed = build_grid(combined, WARM_START, END)
                grid = pd.concat([warmup[venue], completed.loc[POST_START:]])
                grid.index.name = 'open_time'
                del combined, completed
            else:
                grid = build_grid(native.loc[start:], start, END)
            del native
            grid = grid.loc[:, list(GRID_COLUMNS)]
            grid.attrs = {}
            new_grid = grid.loc[start:]
            native_gap_ranges = _ranges(new_grid.index[new_grid.archive_gap])
            native_gap_minutes = int(new_grid.archive_gap.sum())
            # Persist unknown source loss before acceptance can stop the attempt.
            record['pending_gap_acceptance'] = dict(asset=asset, venue=venue,
                                                    ranges=native_gap_ranges, minutes=native_gap_minutes,
                                                    classification='unknown; no maintenance assumption')
            persist()
            acceptance = accept_grid(new_grid, approved_gap_ranges(asset, venue), asset=asset, venue=venue)
            del new_grid
            record.pop('pending_gap_acceptance', None)
            structure = dict(asset=asset, venue=venue, first_native_minute=start.isoformat(),
                             context_start=grid.index[0].isoformat(), context_end=END.isoformat(),
                             rows=len(grid), bad_minutes=int(grid.bad.sum()),
                             archive_gap_minutes=int(grid.archive_gap.sum()), placeholder_minutes=int(grid.placeholder.sum()),
                             missing_minute_ranges=_ranges(grid.index[grid.archive_gap]),
                             placeholder_ranges=_ranges(grid.index[grid.placeholder]),
                             missing_classification='new native gaps require evidence; BTC warmup retains previously bound CP2 flags',
                             native_gap_acceptance=acceptance,
                             native_sources=[{key: row[key] for key in ('path', 'relative_path', 'url', 'sha256', 'month', 'day', 'added_minutes') if key in row}
                                             for row in record['archives'] if (row['asset'], row['venue'], row['kind']) == (asset, venue, 'klines')],
                             rule='bad = missing OR flat zero-volume; archive_gap = missing; prior-only valuation',
                             decision_availability='open_time + 1min')
            record['structure'].append(structure)
            rates = None
            if venue != 'SPOT':
                rows = []
                for spec in catalog:
                    if (spec['asset'], spec['venue'], spec['kind']) != (asset, venue, 'fundingRate'):
                        continue
                    entry, raw = acquisition.obtain(spec)
                    parsed = funding.parse_vision(raw, asset, venue, spec['month'])
                    for row in parsed:
                        row['source_path'] = entry['path']
                    entry['parsed_rows'] = len(parsed)
                    rows.extend(parsed)
                    persist()
                if venue == 'CM':
                    api = funding.acquire_api(root, asset, record, persist, reuse=reuse)
                    source = funding.assemble(api, rows, asset)
                    canonical = source['canonical_records']
                    rates, excluded = funding.for_context(canonical, start, END, asset)
                    source['context_exclusions'] = excluded
                else:
                    first_ms = int(start.value // 1_000_000)
                    canonical = funding.canonical_um(rows, first_ms, funding.END)
                    rates, excluded = funding.for_context(canonical, start, END, asset)
                    source = dict(canonical_records=canonical, canonical_sha256=funding.canonical_hash(canonical),
                                  context_exclusions=excluded)
                record['funding'][f'{asset}_{venue}'] = source
                if asset == 'BTC':
                    rates = pd.concat([warmup[venue + '_funding'], rates])
                    rates.index.name = 'funding_time'
                if not rates.index.isin(grid.index).all():
                    raise ValueError('funding outside completed native minute grid')
            record['caches'][asset][venue] = {'bars': _write_cache(root, asset, venue, '1m', grid)}
            if rates is not None:
                record['caches'][asset][venue]['funding'] = _write_cache(root, asset, venue, 'funding', rates.to_frame())
            persist()
            print(f'CP3 input preflight {asset}/{venue}: {len(grid)} minutes; raw structure only', flush=True)
            del grid, rates
    record.pop('current_input', None)
    record['close_time_exception_summary'] = dict(count=len(record['raw_close_time_exceptions']),
                                                  ohlcv_changes=0, additional_bad_flags=0,
                                                  decision_clock='open_time + 1min')


def prepare_inputs(root=ROOT):
    """Authorized locked-stage raw/structural work only; first error is terminal."""
    require_preflight()
    root = Path(root)
    folder = root / REG
    folder.mkdir(parents=True, exist_ok=True)
    for name in (MANIFEST, PROGRESS, STOPPED):
        if (root / name).exists():
            raise FileExistsError('CP3 input attempt exists; no overwrite, resume or retry')
    record = dict(schema='venue_cp3_inputs_v1', status='preparing_raw_preflight', started_at=archives.stamp(),
                  input_revision='native_end_2026_06_01_v1', native_end_exclusive=END.isoformat(), last_native_month=LAST_MONTH,
                  reuse_policy='explicit authorized saved-source revision; authenticate hashes and official CHECKSUM; immutable hardlinks; no retries',
                  scope='BTC Post and ETH native raw preflight only; no features/signals/budgets/accounts',
                  locks=dict(BTC_IS_UNLOCKED=True, POST_IS_AND_ETH_UNLOCKED=False), automatic_retry=False,
                  archives=[], availability_listings=[], api_pages=[], source_registrations=[],
                  warmup_sources=[], missing_monthly_archives=[], raw_close_time_exceptions=[],
                  structure=[], funding={}, caches={})
    archives.write_new(root / PROGRESS, archives.encoded(record))
    def persist():
        # This is the sole mutable record belonging to this new attempt.
        (root / PROGRESS).write_bytes(archives.encoded(record))
    try:
        for asset in ('BTC', 'ETH'):
            for venue in ('UM', 'SPOT', 'CM'):
                for kind in ('1m', 'funding') if venue != 'SPOT' else ('1m',):
                    if (root / cache_path(asset, venue, kind)).exists():
                        raise FileExistsError('new CP3 cache exists before input attempt')
        _prepare(root, record, persist)
        record.update(status='validated', ended_at=archives.stamp())
        persist()
        # The success manifest is the final publication, after all other writes.
        archives.write_new(root / MANIFEST, archives.encoded(record))
        return record
    except Exception as exc:
        record.update(status='STOPPED', ended_at=archives.stamp(), error_type=type(exc).__name__, error=str(exc))
        persist()
        archives.write_new(root / STOPPED, archives.encoded(record))
        raise


def _verify_sources(root, document):
    catalog = {(s['asset'], s['venue'], s['kind'], s['month']) for s in archives.archive_specs()}
    monthly, absent, source_paths = set(), set(), set()
    for entry in document['archives']:
        archives.validate_spec(entry)
        if entry['path'] in source_paths or entry['checksum_path'] != entry['path'] + '.CHECKSUM':
            raise ValueError('duplicate source path or noncanonical CHECKSUM path')
        source_paths.add(entry['path'])
        key = tuple(entry[k] for k in ('asset', 'venue', 'kind', 'month'))
        if 'day' not in entry:
            if key in monthly:
                raise ValueError('duplicate registered monthly source')
            monthly.add(key)
        if archives.file_hash(root / entry['path']) != entry['sha256']:
            raise ValueError('raw source archive bytes changed')
        checksum = (root / entry['checksum_path']).read_bytes()
        if archives.digest(checksum) != entry['checksum_sha256'] or archives.checksum_hash(checksum, Path(entry['relative_path']).name) != entry['sha256']:
            raise ValueError('registered official checksum changed')
    for entry in document['missing_monthly_archives']:
        archives.validate_spec(entry)
        if entry['kind'] != 'klines':
            raise ValueError('funding month cannot be absent')
        absent.add(tuple(entry[k] for k in ('asset', 'venue', 'kind', 'month')))
    if monthly & absent or monthly | absent != catalog:
        raise ValueError('CP3 source catalog differs from fixed registration')
    for group in ('availability_listings', 'api_pages', 'source_registrations', 'warmup_sources'):
        for entry in document[group]:
            relative = Path(entry['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('registered source path escapes root')
            if archives.file_hash(root / relative) != entry['sha256']:
                raise ValueError('registered source bytes changed: ' + entry['path'])
    for source in document['funding'].values():
        if funding.canonical_hash(source['canonical_records']) != source['canonical_sha256']:
            raise ValueError('canonical funding content changed')


def _load_cache(root, entry, asset, venue, kind):
    if (entry['path'] != cache_path(asset, venue, kind) or entry['asset'] != asset
            or entry['venue'] != venue or entry['kind'] != kind):
        raise ValueError('cache identity differs from registered CP3 asset/venue')
    path = root / entry['path']
    if archives.file_hash(path) != entry['parquet_sha256']:
        raise ValueError('CP3 cache byte hash mismatch')
    frame = pd.read_parquet(path)
    frame.index = scope.utc_index(frame.index, 'CP3 cache')
    if (len(frame) != entry['rows'] or list(frame.columns) != entry['columns']
            or frame.index[0].isoformat() != entry['first_time'] or frame.index[-1].isoformat() != entry['last_time']
            or frame_content_sha256(frame) != entry['content_sha256']):
        raise ValueError('CP3 cache canonical content hash or shape mismatch')
    return frame


def load_inputs(root=ROOT):
    """Require both physical locks before I/O, then load all six real contexts."""
    require_cp3()
    root = Path(root)
    if (root / STOPPED).exists():
        raise ValueError('STOPPED CP3 input attempt cannot be loaded')
    document = json.loads((root / MANIFEST).read_bytes())
    if document.get('schema') != 'venue_cp3_inputs_v1' or document.get('status') != 'validated':
        raise ValueError('validated CP3 input registration required')
    _verify_sources(root, document)
    result = {}
    for asset in ('BTC', 'ETH'):
        result[asset] = {}
        for venue in ('UM', 'SPOT', 'CM'):
            entry = document['caches'][asset][venue]
            bars = _load_cache(root, entry['bars'], asset, venue, '1m')
            expected_start = WARM_START if asset == 'BTC' else pd.Timestamp('2020-07-01', tz='UTC')
            if asset == 'ETH' and venue == 'CM':
                if bars.index[0] != ETH_CM_NATIVE_START:
                    raise ValueError('ETH CM native context start differs from exact approved 07:01 launch minute')
            elif bars.index[0] != expected_start:
                raise ValueError('CP3 context start differs from registered warmup')
            if bars.index[-1] + pd.Timedelta(minutes=1) != END:
                raise ValueError('CP3 context end differs from fixed scope')
            rates = None
            if venue != 'SPOT':
                rates = _load_cache(root, entry['funding'], asset, venue, 'funding')['funding_rate']
            bars.attrs['source'] = 'real'
            result[asset][venue] = scope.make_context(bars, rates, source='real', asset=asset, venue=venue)
    return result


def verify_prepared_inputs(root=ROOT):
    """Metadata and byte-hash verification only, allowed before or after unlock.

    No Parquet/CSV is parsed and no Context is constructed. The controller binds
    the returned manifest SHA256 in its implementation registration.
    """
    if not scope.BTC_IS_UNLOCKED:
        raise scope.RealDataLockedError('CP3 input verification requires the existing BTC authorization')
    root = Path(root)
    if (root / STOPPED).exists():
        raise ValueError('STOPPED CP3 input attempt cannot be verified as successful')
    raw = (root / MANIFEST).read_bytes()
    document = json.loads(raw)
    if document.get('schema') != 'venue_cp3_inputs_v1' or document.get('status') != 'validated':
        raise ValueError('validated CP3 input registration required')
    _verify_sources(root, document)
    paths = {}
    for group in ('archives', 'availability_listings', 'api_pages', 'source_registrations', 'warmup_sources'):
        for entry in document[group]:
            paths[entry['path']] = entry['sha256']
            if group == 'archives':
                paths[entry['checksum_path']] = entry['checksum_sha256']
    for asset in ('BTC', 'ETH'):
        for venue in ('UM', 'SPOT', 'CM'):
            for kind, entry in document['caches'][asset][venue].items():
                cache_kind = '1m' if kind == 'bars' else 'funding'
                if entry['path'] != cache_path(asset, venue, cache_kind):
                    raise ValueError('registered cache identity differs from fixed catalog')
                if archives.file_hash(root / entry['path']) != entry['parquet_sha256']:
                    raise ValueError('prepared CP3 cache byte hash mismatch')
                paths[entry['path']] = entry['parquet_sha256']
    return dict(schema=document['schema'], status='validated',
                manifest=dict(path=MANIFEST, sha256=archives.digest(raw)),
                files=[dict(path=path, sha256=expected) for path, expected in sorted(paths.items())],
                cache_content_hashes={asset: {venue: {kind: entry['content_sha256']
                    for kind, entry in document['caches'][asset][venue].items()}
                    for venue in ('UM', 'SPOT', 'CM')} for asset in ('BTC', 'ETH')})
