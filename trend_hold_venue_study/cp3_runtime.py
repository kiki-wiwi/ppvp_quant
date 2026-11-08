"""One-shot CP3 orchestration around byte-frozen account and inference engines.

The entry point verifies registration, storage and the clean unlock commit.
This module additionally enforces provenance and authorization before features.
"""
from __future__ import annotations

from hashlib import sha256
import gc
import io
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

from . import config, scope
from . import cp3_analysis as analysis
from . import cp3_windows as windows
from .runtime import (_feature_hash, _save_partial, _stamp, _write_table,
                      comparison_row, crossvenue_differences, export_account, write_json)

ROOT = Path(__file__).resolve().parents[1]
END = pd.Timestamp(windows.END_DATE, tz='UTC')
IS_END = pd.Timestamp('2024-01-01', tz='UTC')
WARMUP = pd.Timestamp('2023-11-01', tz='UTC')
MINUTE = pd.Timedelta(minutes=1)
DAY = pd.Timedelta(days=1)
COST_ORDER = ('gross_0', 'main_17', 'harsh_34')
DIAGNOSTIC_VERSIONS = ('V3_CM_DeltaMatched', 'P_CM_DeltaMatched', 'V4_CM_CoinAccum')
PAIRS = (('UM', 'V1_USDM', 'P_USDM'), ('SPOT', 'V2_SPOT', 'P_SPOT'),
         ('CM_DeltaMatched', 'V3_CM_DeltaMatched', 'P_CM_DeltaMatched'))
PAIRED_FAMILIES = tuple((asset, period, venue, trend, passive)
    for asset, periods in (('BTC', ('Post-IS', 'VAL', 'OOS')),
                          ('ETH', ('IS-Common', 'Post-IS', 'VAL', 'OOS')))
    for period in periods for venue, trend, passive in PAIRS)


def job_id(asset, window, signal_source, version, cost, whole_contracts=False):
    return '__'.join((asset, window, signal_source, version, cost)) + ('__whole_contracts' if whole_contracts else '')


def jobs():
    rows = []
    def add(asset, window, versions, costs=COST_ORDER, signal='UM', whole=False):
        for version in versions:
            for cost in costs:
                rows.append({'asset': asset, 'window': window, 'signal_source': signal,
                    'version': version, 'cost': cost, 'whole_contracts': whole,
                    'job_id': job_id(asset, window, signal, version, cost, whole)})
    add('BTC', 'Post-IS', config.VERSIONS)
    add('BTC', 'Post-IS', DIAGNOSTIC_VERSIONS, ('main_17',), whole=True)
    add('BTC', 'Post-IS', DIAGNOSTIC_VERSIONS, ('main_17',), signal='CM')
    add('ETH', 'IS-Common', config.VERSIONS)
    add('ETH', 'Post-IS', config.VERSIONS)
    add('ETH', 'IS-CM-SelfSignal', DIAGNOSTIC_VERSIONS, ('main_17',), signal='CM')
    add('ETH', 'Post-IS', DIAGNOSTIC_VERSIONS, ('main_17',), signal='CM')
    add('ETH', 'Post-IS', DIAGNOSTIC_VERSIONS, ('main_17',), whole=True)
    if len(rows) != 87 or len({row['job_id'] for row in rows}) != 87:
        raise ValueError('registered 87-account matrix altered')
    return rows


def _validate_inputs(contexts):
    if set(contexts) != set(config.ASSETS) or any(set(contexts[asset]) != set(config.VENUES) for asset in contexts):
        raise ValueError('exact BTC and ETH UM/SPOT/CM contexts required')
    sources = {ctx.source for venues in contexts.values() for ctx in venues.values()}
    if len(sources) != 1 or next(iter(sources)) not in ('real', 'synthetic'):
        raise ValueError('input provenance must be explicit and uniform')
    source = next(iter(sources))
    # No calendar/price access is permitted until the independent POST gate.
    if source == 'real' and not scope.POST_IS_AND_ETH_UNLOCKED:
        raise scope.RealDataLockedError('POST-IS / ETH remains independently locked')
    if source == 'real' and not scope.BTC_IS_UNLOCKED:
        raise scope.RealDataLockedError('CP3 requires preserved BTC IS authorization')
    for asset, venues in contexts.items():
        for venue, ctx in venues.items():
            if not isinstance(ctx, scope.Context):
                raise ValueError('validated immutable Context required')
            if ctx.asset != asset or ctx.venue != venue:
                raise ValueError('asset/venue does not match its context key')
            scope.assert_scope(ctx)
            if ctx.index[-1] != END - MINUTE:
                raise ValueError('CP3 contexts must end at the amended ' + windows.END_DATE + ' exclusive boundary')
            if asset == 'BTC':
                if ctx.index[0] != WARMUP:
                    raise ValueError('BTC requires the exact 61-day native Post warmup')
            elif venue in ('UM', 'SPOT'):
                if ctx.index[0] != pd.Timestamp('2020-07-01', tz='UTC'):
                    raise ValueError('ETH UM/Spot require full native July 2020 history')
            elif ctx.index[0] != pd.Timestamp('2020-08-18T07:01Z'):
                raise ValueError('ETH CM must start at its registered first native minute')
    return source


def _context_slice(ctx, start, end):
    """Fresh immutable bounded snapshot; no state is shared between accounts."""
    start, end = config.utc(start), config.utc(end)
    first, last = ctx.index.searchsorted(start), ctx.index.searchsorted(end)
    if first >= last or ctx.index[first] != start or ctx.index[last - 1] != end - MINUTE:
        raise ValueError('context does not cover the exact requested minute slice')
    span = slice(first, last)
    return scope.Context(ctx.index[span], *(getattr(ctx, name)[span] for name in
        ('O', 'H', 'L', 'C', 'V', 'bad', 'archive_gap', 'funding_rate')),
        source=ctx.source, asset=ctx.asset, venue=ctx.venue)


def _prepare_features(contexts):
    from .features import build_features
    schedules = {}
    for asset in config.ASSETS:
        schedules[asset] = {'Post-IS': {}}
        for source in ('UM', 'CM'):
            post = _context_slice(contexts[asset][source], WARMUP, END)
            schedules[asset]['Post-IS'][source] = build_features(post, IS_END, end=END)
            del post
        if asset == 'ETH':
            for window, source in (('IS-Common', 'UM'), ('IS-CM-SelfSignal', 'CM')):
                ctx = contexts[asset][source]
                independent = _context_slice(ctx, ctx.index[0], IS_END)
                schedules[asset][window] = {source: build_features(independent,
                    windows.window(window, asset).signal_origin, end=IS_END)}
                del independent
    return schedules


def _schedule_items(schedules):
    for asset, windows in schedules.items():
        for window, sources in windows.items():
            for source, frame in sources.items():
                yield '__'.join((asset, window, source)), frame


def _run_account(contexts, schedules, job):
    from .linear import run_linear
    from .portfolios import run_coin, run_spot
    asset, window_name = job['asset'], job['window']
    window = windows.window(window_name, asset)
    days = pd.date_range(window.start, window.end, inclusive='left', freq='D').as_unit('ns')
    schedule = schedules[asset][window_name][job['signal_source']].loc[days].copy(deep=True)
    cost = config.COSTS[job['cost']]
    venues = contexts[asset]
    # Engines need their own previous-day references, not full feature history.
    def fresh(venue):
        return _context_slice(venues[venue], window.start - DAY, window.end)
    if job['version'] in ('V1_USDM', 'P_USDM'):
        return run_linear(fresh('UM'), schedule, job['version'], cost, window)
    if job['version'] in ('V2_SPOT', 'P_SPOT'):
        return run_spot(fresh('SPOT'), schedule, job['version'], cost, window)
    return run_coin(fresh('CM'), fresh('SPOT'), schedule, job['version'], cost, window,
                    whole_contracts=job['whole_contracts'])


def _statistics(accounts, schedules, out, log):
    from .statistics import analyze
    results = {}
    for asset, period, venue, trend, passive in PAIRED_FAMILIES:
        window = 'IS-Common' if period == 'IS-Common' else 'Post-IS'
        family = '__'.join((asset, period, venue))
        log('paired inference ' + family + ': fixed 99999 Webb / 10000 paired month draws')
        selected = [accounts[job_id(asset, window, 'UM', version, cost)]
                    for cost in ('main_17', 'harsh_34') for version in (trend, passive)]
        start = windows.window(window, asset).start if period != 'OOS' else analysis.SPLIT
        end = analysis.SPLIT if period == 'VAL' else windows.window(window, asset).end
        frames = [account['daily'].loc[(account['daily'].index >= start) & (account['daily'].index < end)]
                  for account in selected]
        features = schedules[asset][window]['UM'].loc[frames[0].index].copy(deep=True)
        times = selected[0]['order_times']
        order_count = int(((times >= start) & (times < end)).sum())
        stats = analyze(*(frame['return'] for frame in frames), features, order_count)
        actual = {}
        for label, account in zip(('main_T', 'main_P', 'harsh_T', 'harsh_P'), selected):
            row = account['period_metrics'].loc[account['period_metrics']['period'] == period].iloc[0]
            actual[label] = {'start_equity': row['start_equity'], 'final_equity': row['final_equity'],
                             'total_return': row['total_return']}
        stats['metadata'].update(asset=asset, period=period, venue=venue,
            actual_continuous_wealth_ratios=actual,
            inference_wealth_convention='Frozen analyze compounds stored returns from its conventional unit initial wealth. Actual slice ratios above stay NA when actual preceding equity is zero; they are not replaced by inference compounding.')
        folder = out / 'statistics' / family
        folder.mkdir(parents=True, exist_ok=False)
        for name in ('tests', 'intervals', 'gates', 'daily', 'monthly'):
            stats[name].to_csv(folder / (name + '.csv'))
        write_json(folder / 'metadata.json', stats['metadata'])
        matrices = stats.pop('matrices')
        for name, matrix in matrices.items():
            np.save(folder / (name + '.npy'), matrix, allow_pickle=False)
        del matrices, matrix
        results[family] = stats
    return results


def load_cp2_comparison(root, expected_sha256):
    """Only the registered immutable CP2 CSV is opened, after formal unlock."""
    if not scope.POST_IS_AND_ETH_UNLOCKED or not scope.BTC_IS_UNLOCKED:
        raise scope.RealDataLockedError('POST-IS / ETH lock must be formally opened before CP2 result reuse')
    path = Path(root) / 'reports/trend_hold_venue_round/cp2_btc_is/comparison.csv'
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError('registered CP2 comparison hash required')
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('registered CP2 comparison hash mismatch')
    table = pd.read_csv(io.BytesIO(raw), float_precision='round_trip')
    table['asset'] = 'BTC'
    return table



def _verified_cp2_path(root, relative, hashes):
    prefix = Path('reports/trend_hold_venue_round/cp2_btc_is')
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('unregistered CP2 archive path')
    registered = (prefix / relative).as_posix()
    expected = hashes.get(registered, hashes.get(relative.as_posix()))
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError('registered CP2 artifact hash required: ' + registered)
    path = Path(root) / prefix / relative
    digest = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError('registered CP2 artifact hash mismatch: ' + registered)
    return path


def _archived_table(path):
    # Frozen exporters create a one-newline CSV for empty event frames.
    import gzip
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as stream:
        header = stream.readline()
    if not header.strip():
        return pd.DataFrame()
    frame = pd.read_csv(path, float_precision='round_trip')
    if 'time' in frame:
        frame['time'] = pd.to_datetime(frame['time'], utc=True, format='ISO8601')
    return frame


def load_cp2_chart_accounts(root, hashes, comparison):
    """Read four registered completed BTC IS account ledgers, never resimulate.

Only compact chart outputs survive each iteration. CM signed funding uses the
original spot-valued attribution, not coin flow repriced at a later price.
"""
    if not scope.POST_IS_AND_ETH_UNLOCKED or not scope.BTC_IS_UNLOCKED:
        raise scope.RealDataLockedError('POST-IS / ETH unlock required before archived chart reads')
    from .models import Result
    versions = ('V1_USDM', 'V3_CM_DeltaMatched', 'V4_CM_CoinAccum', 'HODL_1x_Coin')
    accounts, rows, evidence = {}, [], []
    for version in versions:
        selected = comparison.loc[(comparison['window'] == 'IS-Common') & (comparison['version'] == version)
            & (comparison['cost'] == 'main_17') & (comparison['signal_source'] == 'UM') & ~comparison['whole_contracts']]
        if len(selected) != 1:
            raise ValueError('archived CP2 account must be unique: ' + version)
        row = selected.iloc[0].to_dict()
        old_id = '__'.join(('IS-Common', version, 'main_17'))
        folder = Path('accounts') / old_id
        verified = {}
        def file(name):
            path = _verified_cp2_path(root, folder / name, hashes)
            registered = path.relative_to(Path(root)).as_posix()
            expected = hashes.get(registered, hashes.get((folder / name).as_posix()))
            verified[name] = {'path': registered, 'sha256': expected}
            return path
        daily = pd.read_csv(file('daily.csv'), index_col=0, float_precision='round_trip')
        daily.index = pd.DatetimeIndex(pd.to_datetime(daily.index, utc=True, format='ISO8601')).as_unit('ns')
        expected_days = pd.date_range(windows.window('IS-Common').start, IS_END, inclusive='left', freq='D').as_unit('ns')
        if not daily.index.equals(expected_days):
            raise ValueError('archived CP2 daily calendar mismatch')
        metadata = json.loads(file('metadata.json').read_text())
        columns = ['equity', 'bad', 'actual_leverage']
        if version in config.CM_VERSIONS:
            columns += ['coin_equity']
        minute = pd.read_parquet(file('minute.parquet'), columns=columns)
        if version in config.CM_VERSIONS:
            funding = np.zeros(len(daily), dtype=np.longdouble)
            attribution_path = file('attribution.csv.gz')
            for chunk in pd.read_csv(attribution_path, usecols=['time', 'funding_spot_usd'],
                    dtype={'funding_spot_usd': str}, chunksize=100_000):
                chunk['time'] = pd.to_datetime(chunk['time'], utc=True, format='ISO8601')
                funding += analysis._daily_sum(chunk, 'funding_spot_usd', daily.index)
            del chunk
        else:
            receipts = _archived_table(file('funding.csv.gz'))
            funding = analysis._daily_sum(receipts, 'cash_flow', daily.index)
            del receipts
        if abs(float(np.sum(funding, dtype=np.longdouble)) - float(row['funding_usd'])) >= 1e-9:
            raise ValueError('archived CP2 funding curve does not reconcile to frozen comparison')
        accounting = pd.DataFrame({'funding_usd': funding}, index=daily.index)
        empty = pd.DataFrame()
        result = Result(minute, daily['equity'], daily['return'], empty, empty, empty, metadata)
        chart = analysis.chart_daily(result, accounting)
        pressure = _archived_table(file('pressure_events.csv.gz'))
        job = {'asset': 'BTC', 'window': 'IS-Common', 'signal_source': 'UM', 'version': version,
            'cost': 'main_17', 'whole_contracts': False,
            'job_id': job_id('BTC', 'IS-Common', 'UM', version, 'main_17')}
        row.update(job, source_stage='frozen_CP2')
        accounts[job['job_id']] = {'job': job, 'metadata': metadata, 'daily': daily,
            'chart_daily': chart, 'pressure_events': pressure,
            'folder': str(Path(root) / 'reports/trend_hold_venue_round/cp2_btc_is' / folder),
            'source_stage': 'frozen_CP2'}
        rows.append(row)
        evidence.append({'job_id': job['job_id'], 'source_stage': 'frozen_CP2', 'verified_sources': verified,
                         'no_account_or_feature_recomputation': True})
        del result, minute, accounting, funding
        gc.collect()
    return accounts, pd.DataFrame(rows), evidence

def _diagnostics(schedules, comparison):
    bridges, rounding, signals = [], [], {}
    for asset in config.ASSETS:
        for window in ('IS-Common', 'Post-IS'):
            selected = comparison.loc[(comparison['asset'] == asset) & (comparison['window'] == window)]
            if selected.empty:
                continue
            # The frozen endpoint bridge is calendar agnostic internally; map its
            # dispatch label then restore the actual CP3 account window.
            bridge = crossvenue_differences(selected.assign(window='IS-Common'))
            if not bridge.empty:
                bridge['asset'], bridge['window'] = asset, window
                bridges.append(bridge)
        for version in DIAGNOSTIC_VERSIONS:
            selected = comparison.loc[(comparison['asset'] == asset) & (comparison['window'] == 'Post-IS')
                & (comparison['version'] == version) & (comparison['cost'] == 'main_17')
                & (comparison['signal_source'] == 'UM')]
            continuous = selected.loc[~selected['whole_contracts']].iloc[0]
            whole = selected.loc[selected['whole_contracts']].iloc[0]
            rounding.append({'asset': asset, 'window': 'Post-IS', 'version': version,
                'contract_size_usd': config.CONTRACT_SIZE[asset],
                'continuous_final_equity': continuous['final_equity'], 'whole_final_equity': whole['final_equity'],
                'equity_difference_usd': whole['final_equity'] - continuous['final_equity'],
                'continuous_coin_growth': continuous['coin_growth'], 'whole_coin_growth': whole['coin_growth'],
                'coin_growth_difference': whole['coin_growth'] - continuous['coin_growth']})
        for window in ('Post-IS', 'IS-CM-SelfSignal'):
            if window not in schedules[asset]:
                continue
            cm = schedules[asset][window]['CM']
            um_window = 'IS-Common' if window == 'IS-CM-SelfSignal' else window
            um = schedules[asset][um_window]['UM'].loc[cm.index]
            frame = pd.DataFrame({'g_UM': um['g'], 'g_CM': cm['g'],
                'budget_UM': um['budget'], 'budget_CM': cm['budget']}, index=cm.index)
            frame['budget_difference'] = frame['budget_CM'] - frame['budget_UM']
            differences = []
            for version in DIAGNOSTIC_VERSIONS:
                native = comparison.loc[(comparison['asset'] == asset) & (comparison['window'] == window)
                    & (comparison['version'] == version) & (comparison['signal_source'] == 'CM')].iloc[0]
                if window == 'Post-IS':
                    primary = comparison.loc[(comparison['asset'] == asset) & (comparison['window'] == window)
                        & (comparison['version'] == version) & (comparison['signal_source'] == 'UM')
                        & (comparison['cost'] == 'main_17') & ~comparison['whole_contracts']].iloc[0]
                    differences.append({'version': version, 'equity_difference_usd': native['final_equity'] - primary['final_equity'],
                        'coin_growth_difference': native['coin_growth'] - primary['coin_growth'], 'equal_initial_state': True})
                else:
                    differences.append({'version': version, 'native_final_equity': native['final_equity'],
                        'equity_difference_usd': None, 'coin_growth_difference': None, 'equal_initial_state': False})
            signals[asset + '__' + window] = {'daily': frame, 'summary': {
                'asset': asset, 'window': window, 'days': len(frame),
                'concordance_pct': 100. * float((frame['g_UM'] == frame['g_CM']).mean()),
                'budget_difference_mean': float(frame['budget_difference'].mean()),
                'budget_difference_max_abs': float(frame['budget_difference'].abs().max()),
                'account_differences': differences,
                'limitation': 'IS native-CM account begins November 1 independently; no equal-initial-state UM reference account is added. Post accounts share January 1 start.'}}
    return {'crossvenue': pd.concat(bridges, ignore_index=True) if bridges else pd.DataFrame(),
            'rounding': pd.DataFrame(rounding), 'self_signal': signals}


def _export_slice_descriptors(account, accounting, folder):
    """Persist exact slice row ranges without duplicating large minute ledgers."""
    for row in account['period_metrics'].to_dict('records'):
        period, start, end = row['period'], row['start'], row['end']
        target = folder / 'slices' / period
        target.mkdir(parents=True, exist_ok=False)
        daily = account['daily']
        daily.loc[(daily.index >= start) & (daily.index < end)].to_csv(target / 'daily.csv')
        _write_table(accounting.loc[(accounting.index >= start) & (accounting.index < end)],
                     target / 'accounting_daily.csv.gz', index=True)
        offset = int((start - daily.index[0]) / MINUTE)
        count = int((end - start) / MINUTE)
        write_json(target / 'boundary.json', {**row, 'minute_source': '../../minute.parquet',
            'minute_start_row': offset, 'minute_row_count': count, 'interval': '[start,end)',
            'ledger_sources': {name: '../../' + name + '.csv.gz' for name in
                ('orders', 'decisions', 'funding', 'attribution', 'pressure_events')},
            'ledger_filter': 'start <= time < end; preserve all source columns and values',
            'synthetic_boundary_trades': False, 'boundary_cash_conversion': False})



def _render_outputs(source, accounts, schedules, statistics, comparison, diagnostics, out):
    if source == 'synthetic':
        return [], 'synthetic-only execution has no real CP2 archive'
    if source != 'real':
        raise ValueError('explicit real or synthetic chart provenance required')
    from .cp3_plotting import FIGURES, render_outputs
    figures = render_outputs(accounts, schedules, statistics, comparison, diagnostics, out)
    if figures != list(FIGURES) or len(figures) != 4:
        raise ValueError('formal CP3 requires all four registered figures')
    return figures, None

def execute(contexts, out, input_metadata=None):
    """Execute fixed 87 accounts once; retain only compact daily/chart outputs."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started, completed, current = _stamp(), [], None
    input_metadata = {} if input_metadata is None else input_metadata
    def log(message):
        with (out / 'run.log').open('a') as stream:
            stream.write(_stamp() + ' ' + message + '\n')
        print(message, flush=True)
    try:
        source = _validate_inputs(contexts)
        if source == 'real' and out.resolve() != (ROOT / 'reports/trend_hold_venue_round/cp3_post_is_and_eth').resolve():
            raise ValueError('real CP3 output must use its registered fresh directory')
        write_json(out / 'input_verification.json', input_metadata)
        schedules = _prepare_features(contexts)
        hashes = {key: _feature_hash(frame) for key, frame in _schedule_items(schedules)}
        for key, frame in _schedule_items(schedules):
            frame.to_csv(out / ('features_' + key + '.csv'))
        write_json(out / 'feature_registration.json', {'source': source, 'sha256': hashes,
            'generated_before_first_account': True, 'job_order': jobs(), 'frozen_at': _stamp(),
            'post_warmup': '[2023-11-01,2024-01-01), fresh immutable source contexts for each asset',
            'post_split': 'continuous through July 1 2025; output slices only',
            'post_end_exclusive': windows.END_DATE, 'post_last_month': windows.LAST_MONTH,
            'post_engine_window_source': 'cp3_windows.window; frozen config.window remains unchanged'})
        accounts, comparisons, periods, chops, initial_coins = {}, [], [], [], {}
        for ordinal, job in enumerate(jobs(), start=1):
            current = job['job_id']
            log(f'account {ordinal}/87: ' + current)
            result = _run_account(contexts, schedules, job)
            folder = out / 'accounts' / current
            account = export_account(result, folder, job)
            if 'initial_coin' in result.metadata:
                key = (job['asset'], windows.window(job['window'], job['asset']).start, job['cost'])
                coin = result.metadata['initial_coin']
                if key in initial_coins and coin != initial_coins[key]:
                    raise ValueError('same-start same-cost coin initial conversion mismatch')
                initial_coins[key] = coin
            accounting = analysis.daily_accounting(result)
            _write_table(accounting, folder / 'accounting_daily.csv.gz', index=True)
            account.update(folder=str(folder), chart_daily=analysis.chart_daily(result, accounting),
                period_metrics=analysis.period_metrics(result, job, accounting),
                pressure_events=result.pressure_events.copy(),
                order_times=pd.DatetimeIndex(pd.to_datetime(result.orders['time'], utc=True)) if 'time' in result.orders
                    else pd.DatetimeIndex([], tz='UTC'))
            account['chart_daily'].to_csv(folder / 'chart_daily.csv')
            account['period_metrics'].to_csv(folder / 'period_metrics.csv', index=False)
            _export_slice_descriptors(account, accounting, folder)
            if job['window'] == 'Post-IS' and job['signal_source'] == 'UM' and not job['whole_contracts'] and job['cost'] == 'main_17':
                chop = analysis.chop_diagnosis(result, schedules[job['asset']]['Post-IS']['UM'], job, accounting)
                chops.append(chop['summary'])
                chop['episodes'].to_csv(folder / 'chop_episodes.csv', index=False)
                write_json(folder / 'chop_summary.json', chop['summary'])
            comparisons.append(comparison_row(result, job))
            periods.append(account['period_metrics'])
            accounts[current] = account
            if any(_feature_hash(frame) != hashes[key] for key, frame in _schedule_items(schedules)):
                raise ValueError('frozen canonical schedule mutated by account')
            completed.append(current)
            write_json(out / 'progress.json', {'status': 'running', 'completed_jobs': completed,
                'current_job': current, 'automatic_retry': False})
            del result, accounting
            gc.collect()
        table = pd.DataFrame(comparisons)
        table.to_csv(out / 'comparison.csv', index=False)
        reconciliations = ['job_id', 'account_reconciliation_error', 'cm_account_reconciliation_error',
            'coin_reconciliation_error', 'max_wallet_reconciliation_error', 'zero_floor_adjustment_usd',
            'cash_roundoff_adjustment_usd', 'liquidated']
        table[reconciliations].to_csv(out / 'reconciliation.csv', index=False)
        diagnostics = _diagnostics(schedules, table)
        all_periods = pd.concat(periods, ignore_index=True)
        is_reference = table.loc[table['window'] == 'IS-Common'].copy()
        plot_comparison = table
        if source == 'real':
            old = load_cp2_comparison(ROOT, input_metadata.get('cp2_comparison_sha256'))
            old_common = old.loc[(old['window'] == 'IS-Common') & ~old['whole_contracts'] & (old['signal_source'] == 'UM')].copy()
            is_reference = pd.concat([is_reference, old_common], ignore_index=True)
            old_common['job_id'] = [job_id('BTC', 'IS-Common', 'UM', row['version'], row['cost'])
                for row in old_common.to_dict('records')]
            old_common['period'] = 'IS-Common'
            old_common['start'], old_common['end'] = windows.window('IS-Common').start, IS_END
            old_common['source_stage'] = 'frozen_CP2'
            all_periods = pd.concat([all_periods, old_common], ignore_index=True)
            archived_accounts, archived_rows, evidence = load_cp2_chart_accounts(
                ROOT, input_metadata.get('cp2_delivery_sha256', {}), old)
            accounts.update(archived_accounts)
            plot_comparison = pd.concat([table, archived_rows], ignore_index=True)
            write_json(out / 'archived_cp2_reuse.json', evidence)
            reused = out / 'archived_cp2_chart_data'
            reused.mkdir(exist_ok=False)
            for archived_id, archived_account in archived_accounts.items():
                archived_account['chart_daily'].to_csv(reused / (archived_id + '.csv'))
        diagnostics.update(period_metrics=all_periods, chop=pd.DataFrame(chops),
            combined_reference=analysis.combined_reference(is_reference, table))
        for name in ('crossvenue', 'rounding', 'period_metrics', 'chop', 'combined_reference'):
            diagnostics[name].to_csv(out / (name + '.csv'), index=False)
        for key, item in diagnostics['self_signal'].items():
            item['daily'].to_csv(out / ('self_signal_' + key + '.csv'))
            write_json(out / ('self_signal_' + key + '.json'), item['summary'])
        statistics = _statistics(accounts, schedules, out, log)
        figures, charts_skipped_reason = _render_outputs(
            source, accounts, schedules, statistics, plot_comparison, diagnostics, out)
        summary = {'status': 'completed', 'stage': 'CP3', 'source': source,
            'declaration': config.DECLARATION, 'started_at': started, 'ended_at': _stamp(),
            'combination_count': len(completed), 'completed_jobs': completed,
            'paired_families': list(statistics), 'figures': figures, 'feature_sha256': hashes,
            'charts_skipped_reason': charts_skipped_reason,
            'evaluation_window': {'post_start': '2024-01-01', 'post_end_exclusive': windows.END_DATE,
                'last_month': windows.LAST_MONTH, 'post_days': 882, 'post_months': 29,
                'split': '2025-07-01', 'val_days': 547, 'val_months': 18,
                'oos_days': 335, 'oos_months': 11,
                'engine_window_source': 'cp3_windows.window',
                'frozen_config_upper_bound': '2026-11-01'},
            'automatic_promotion': False, 'automatic_retry': False,
            'locks': {'BTC_IS_UNLOCKED': scope.BTC_IS_UNLOCKED,
                      'POST_IS_AND_ETH_UNLOCKED': scope.POST_IS_AND_ETH_UNLOCKED},
            'limitations': ['Post accounts are independent of IS and continuous through July 2025.',
                'Frozen two-hypothesis Holm applies within each of 21 descriptive families, not globally.',
                'Zero preceding equity or coin gives undefined segment wealth ratios.',
                'Combined IS/Post return is an independent-account descriptive product, not continuous positions.',
                'V4 includes extra collateral coin exposure and uses HODL as its coin-growth comparator.',
                'Fixed 0.5% MMR and contract OHLC are simplified pressure and marked-price proxies.',
                'Funding offset uses observed ledger episodes and is not a causal counterfactual.',
                '1 USDT = 1 USD; cross-venue endpoint bridges contain inseparable price and path interactions.']}
        write_json(out / 'summary.json', summary)
        write_json(out / 'progress.json', {'status': 'completed', 'completed_jobs': completed, 'automatic_retry': False})
        log('completed 87 accounts and 21 paired inference families; STOP at CP3')
        return summary
    except Exception as exc:
        write_json(out / 'STOPPED.json', {'stage': 'CP3', 'started_at': started, 'stopped_at': _stamp(),
            'error_type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc(),
            'completed_jobs': completed, 'current_job': current, 'automatic_retry': False})
        _save_partial(out, exc)
        log('STOPPED: ' + str(exc))
        raise
