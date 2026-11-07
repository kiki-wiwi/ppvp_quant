"""One fixed 42-account BTC IS batch with no retry, resume, or parameter overrides.

The caller must complete source/unlock/input verification before calling execute.
All engines, features, statistics, and accounting remain the frozen CP1 functions.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import gc
import json
import math
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

from . import config, scope
from .models import Result

ROOT = Path(__file__).resolve().parents[1]
END = pd.Timestamp('2024-01-01', tz='UTC')
COST_ORDER = ('gross_0', 'main_17', 'harsh_34')
PAIRED_FAMILIES = (
    ('IS-Full', 'UM', 'V1_USDM', 'P_USDM'),
    ('IS-Full', 'SPOT', 'V2_SPOT', 'P_SPOT'),
    ('IS-Common', 'UM', 'V1_USDM', 'P_USDM'),
    ('IS-Common', 'SPOT', 'V2_SPOT', 'P_SPOT'),
    ('IS-Common', 'CM_DeltaMatched', 'V3_CM_DeltaMatched', 'P_CM_DeltaMatched'),
)


def _stamp():
    return datetime.now(timezone.utc).isoformat()


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, (pd.Timestamp, datetime, Path)):
        return str(value)
    if value is pd.NA:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def _job_id(window, version, cost, whole_contracts=False):
    return '__'.join((window, version, cost)) + ('__whole_contracts' if whole_contracts else '')


def jobs():
    """Registered order: 12 Full, 24 Common, 3 rounding, 3 native-CM signal."""
    rows = []
    for window, versions in (('IS-Full', config.VERSIONS[:4]), ('IS-Common', config.VERSIONS)):
        for version in versions:
            for cost in COST_ORDER:
                rows.append({'job_id': _job_id(window, version, cost), 'window': window,
                             'version': version, 'cost': cost, 'signal_source': 'UM',
                             'whole_contracts': False})
    for version in ('V3_CM_DeltaMatched', 'P_CM_DeltaMatched', 'V4_CM_CoinAccum'):
        rows.append({'job_id': _job_id('IS-Common', version, 'main_17', True),
                     'window': 'IS-Common', 'version': version, 'cost': 'main_17',
                     'signal_source': 'UM', 'whole_contracts': True})
    for version in ('V3_CM_DeltaMatched', 'P_CM_DeltaMatched', 'V4_CM_CoinAccum'):
        rows.append({'job_id': _job_id('IS-CM-SelfSignal', version, 'main_17'),
                     'window': 'IS-CM-SelfSignal', 'version': version, 'cost': 'main_17',
                     'signal_source': 'CM', 'whole_contracts': False})
    if len(rows) != 42 or len({r['job_id'] for r in rows}) != 42:
        raise ValueError('registered 42-account matrix was altered')
    return rows


def _validate_inputs(contexts):
    # The second physical gate must stay False throughout CP2, even in test calls.
    if scope.POST_IS_AND_ETH_UNLOCKED:
        raise ValueError('CP2 requires POST_IS_AND_ETH_UNLOCKED=False')
    if set(contexts) != {'UM', 'SPOT', 'CM'}:
        raise ValueError('exactly the three BTC input contexts are required')
    sources = {ctx.source for ctx in contexts.values()}
    if len(sources) != 1 or next(iter(sources)) not in ('real', 'synthetic'):
        raise ValueError('input provenance must be explicit and uniform')
    source = next(iter(sources))
    if source == 'real' and not scope.BTC_IS_UNLOCKED:
        raise scope.RealDataLockedError('BTC IS remains locked')
    for venue, ctx in contexts.items():
        if ctx.asset != 'BTC' or ctx.venue != venue:
            raise ValueError('CP2 only accepts BTC with its own venue context')
        if not isinstance(ctx, scope.Context):
            raise ValueError('validated immutable Context required')
        scope.assert_scope(ctx)
        if ctx.index[-1] != END - pd.Timedelta(minutes=1):
            raise ValueError('CP2 input must end at 2023-12-31 23:59 UTC')
        if venue in ('UM', 'SPOT'):
            if ctx.index[0] != pd.Timestamp('2020-01-01', tz='UTC') or len(ctx) != 2_103_840:
                raise ValueError('UM/Spot require the complete fixed BTC IS minute grid')
        elif not (pd.Timestamp('2020-08-11', tz='UTC') <= ctx.index[0]
                  <= pd.Timestamp('2020-08-12', tz='UTC')):
            raise ValueError('CM native first minute does not satisfy the approved launch boundary')
    return source


def _clone(ctx):
    return scope.Context(ctx.index, *(getattr(ctx, name) for name in
        ('O', 'H', 'L', 'C', 'V', 'bad', 'archive_gap', 'funding_rate')),
        source=ctx.source, asset=ctx.asset, venue=ctx.venue)


def _prepare_features(contexts):
    from .features import build_features
    return {
        'UM': build_features(contexts['UM'], config.window('IS-Full').signal_origin, end=END),
        'CM': build_features(contexts['CM'], config.window('IS-CM-SelfSignal').signal_origin, end=END),
    }


def _feature_hash(frame):
    return sha256(frame.to_csv().encode()).hexdigest()


def _run_account(contexts, schedules, job):
    from .linear import run_linear
    from .portfolios import run_coin, run_spot
    window = config.window(job['window'])
    days = pd.date_range(window.start, window.end, inclusive='left', freq='D').as_unit('ns')
    schedule = schedules[job['signal_source']].loc[days].copy(deep=True)
    cost = config.COSTS[job['cost']]
    if job['version'] in ('V1_USDM', 'P_USDM'):
        return run_linear(_clone(contexts['UM']), schedule, job['version'], cost, window)
    if job['version'] in ('V2_SPOT', 'P_SPOT'):
        return run_spot(_clone(contexts['SPOT']), schedule, job['version'], cost, window)
    return run_coin(_clone(contexts['CM']), _clone(contexts['SPOT']), schedule,
                    job['version'], cost, window, whole_contracts=job['whole_contracts'])


def assert_r5_equal(result, archived_daily, archived_metadata):
    """No tolerance: all daily equity/returns and the terminal scalar must agree."""
    current = pd.DataFrame({'equity': result.daily_equity, 'return': result.daily_returns})
    if not isinstance(archived_daily.index, pd.DatetimeIndex) or archived_daily.index.tz is None:
        raise ValueError('R5 daily calendar is not timezone aware')
    old_index = archived_daily.index.tz_convert('UTC').as_unit('ns')
    if not current.index.as_unit('ns').equals(old_index):
        raise ValueError('R5 daily calendar mismatch')
    if not {'equity', 'return'}.issubset(archived_daily):
        raise ValueError('R5 daily columns missing')
    old = archived_daily[['equity', 'return']].to_numpy(dtype=np.float64)
    new = current[['equity', 'return']].to_numpy(dtype=np.float64)
    if not np.array_equal(new, old):
        where = np.argwhere(new != old)[0]
        raise ValueError(f'R5 daily mismatch at {current.index[where[0]]}, column {current.columns[where[1]]}')
    terminal = archived_metadata['final_equity']
    if result.metadata['final_equity'] != terminal or float(current['equity'].iloc[-1]) != terminal:
        raise ValueError('R5 terminal equity mismatch')
    return {'daily_exact': True, 'terminal_exact': True, 'day_count': len(current),
            'archived_final_equity': terminal}


def _r5_check(result, job, source):
    if job['window'] != 'IS-Full' or job['version'] not in ('V1_USDM', 'P_USDM'):
        return None
    if source != 'real':
        return {'status': 'synthetic_only_not_compared_to_real_archive'}
    if not scope.BTC_IS_UNLOCKED or scope.POST_IS_AND_ETH_UNLOCKED:
        raise scope.RealDataLockedError('R5 archive comparison requires BTC IS only')
    folder = ROOT / 'reports/trend_hold_round/results'
    manifest = json.loads((folder / 'output_hashes.json').read_text())['sha256']
    old_version = 'T_daily_trend' if job['version'] == 'V1_USDM' else 'P_continuous_long'
    prefix = old_version + '__' + job['cost']
    for suffix in ('daily.csv', 'metadata.json'):
        relative = prefix + '/' + suffix
        if sha256((folder / relative).read_bytes()).hexdigest() != manifest[relative]:
            raise ValueError('R5 archived output hash mismatch: ' + relative)
    daily = pd.read_csv(folder / prefix / 'daily.csv', index_col=0, parse_dates=[0],
                        float_precision='round_trip')
    daily.index = pd.DatetimeIndex(daily.index).tz_convert('UTC').as_unit('ns')
    metadata = json.loads((folder / prefix / 'metadata.json').read_text())
    return {**assert_r5_equal(result, daily, metadata), 'archived_version': old_version,
            'archived_result_commit': config.BASELINE}


def _float_text(value):
    # float128 cannot be represented by Parquet. CSV preserves the ledger's
    # extended-precision scalar rather than silently downcasting it to float64.
    return np.format_float_scientific(value, precision=21, unique=False, trim='k')


def _write_table(frame, path, *, index=False):
    frame.to_csv(path, index=index, compression='gzip' if str(path).endswith('.gz') else None,
                 float_format=_float_text)


def _finite_result(result):
    for name in ('minute', 'attribution'):
        frame = getattr(result, name)
        numeric = frame.select_dtypes(include=[np.number])
        if len(numeric.columns) and not np.isfinite(numeric.to_numpy()).all():
            raise ValueError(f'nonfinite {name} account ledger')
    if not np.isfinite(result.daily_equity).all() or not np.isfinite(result.daily_returns).all():
        raise ValueError('nonfinite daily account values')
    for key in ('account_reconciliation_error', 'cm_account_reconciliation_error',
                'coin_reconciliation_error', 'max_wallet_reconciliation_error'):
        if key in result.metadata and not math.isfinite(float(result.metadata[key])):
            raise ValueError('nonfinite reconciliation evidence: ' + key)


def export_account(result, folder, job):
    """Export one account, then retain only daily curves and small metadata."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    _finite_result(result)
    result.minute.to_parquet(folder / 'minute.parquet', index=True)
    for name in ('orders', 'decisions', 'funding', 'attribution', 'pressure_events'):
        _write_table(getattr(result, name), folder / (name + '.csv.gz'))
    daily = pd.DataFrame({'equity': result.daily_equity, 'return': result.daily_returns})
    for field in ('coin_equity', 'equity_cm', 'equity_spot', 'wallet_coin',
                  'unrealized_pnl_coin', 'face', 'liquidated'):
        if field in result.minute:
            values = result.minute[field].groupby(result.minute.index.normalize()).last()
            if not values.index.equals(daily.index):
                raise ValueError('account daily calendar differs from minute ledger: ' + field)
            daily[field] = values.to_numpy()
    # Default float64 repr is round-trip safe and matches the archived R5 format.
    daily.to_csv(folder / 'daily.csv')
    write_json(folder / 'metadata.json', result.metadata)
    return {'daily': daily, 'metadata': dict(result.metadata), 'job': dict(job)}


def _sum(frame, column):
    if column not in frame:
        return 0.
    return float(np.sum(frame[column].to_numpy(dtype=np.longdouble), dtype=np.longdouble))


def comparison_row(result, job):
    from .reporting import metrics
    row = {**metrics(result), **job}
    md, attr = result.metadata, result.attribution
    is_coin = job['version'] in config.CM_VERSIONS
    funding = _sum(attr, 'funding_spot_usd') if is_coin else float(md.get('total_funding', 0.))
    fee = _sum(attr, 'fee_spot_usd') if is_coin else float(md.get('total_fees', 0.))
    slip = _sum(attr, 'slippage_spot_usd') if is_coin else -float(md.get('total_slippage_cost', 0.))
    conversion = float(md.get('initial_conversion_pnl', 0.)) + float(md.get('terminal_conversion_pnl', 0.))
    basis = float(md.get('final_spot_equity_before_fx', md['final_equity'])) - float(md.get('final_cm_equity', md['final_equity']))
    row.update(funding_usd=funding, fee_usd=fee, slippage_usd=slip,
               conversion_usd=conversion, basis_final_usd=basis,
               cash_clip_count=int(md.get('cash_clip_count', 0)),
               zero_floor_adjustment_usd=_sum(attr, 'zero_floor_spot_usd'),
               cash_roundoff_adjustment_usd=_sum(attr, 'cash_roundoff_adjustment'),
               order_count=int(md['order_count']),
               final_spot_equity_before_fx=md.get('final_spot_equity_before_fx'),
               final_cm_equity=md.get('final_cm_equity'),
               contract_mtm_spot_usd=_sum(attr, 'contract_mtm_spot_usd'),
               holding_spot_usd=_sum(attr, 'holding_spot_usd'),
               price_pnl_usd=md.get('price_pnl', _sum(attr, 'price_pnl')),
               account_reconciliation_error=md['account_reconciliation_error'],
               cm_account_reconciliation_error=md.get('cm_account_reconciliation_error'),
               coin_reconciliation_error=md.get('coin_reconciliation_error'),
               max_wallet_reconciliation_error=md.get('max_wallet_reconciliation_error'),
               bad_minutes=int(result.minute['bad'].sum()),
               archive_gap_minutes=int(result.minute['archive_gap'].sum()) if 'archive_gap' in result.minute else 0,
               scope='extra_coin_collateral_exposure' if job['version'] in ('V4_CM_CoinAccum', 'HODL_1x_Coin') else 'same_target_exposure')
    buys = result.orders[result.orders['requested_delta'] > 0] if 'requested_delta' in result.orders else pd.DataFrame()
    requested = _sum(buys, 'requested_delta')
    filled = _sum(buys, 'delta_qty')
    row['cash_clip_quantity_fraction'] = max(0., (requested - filled) / requested) if requested else 0.
    return row


def crossvenue_differences(comparison):
    """Exact endpoint bridge; the price/path residual is not a causal estimate."""
    rows = []
    for window in ('IS-Full', 'IS-Common'):
        for cost in COST_ORDER:
            subset = comparison[(comparison['window'] == window) & (comparison['cost'] == cost)
                                & ~comparison['whole_contracts'] & (comparison['signal_source'] == 'UM')]
            for roles in (('V1_USDM', 'V2_SPOT', 'V3_CM_DeltaMatched'),
                          ('P_USDM', 'P_SPOT', 'P_CM_DeltaMatched')):
                current = {r['version']: r for r in subset.to_dict('records')}
                if roles[0] not in current:
                    continue
                base = current[roles[0]]
                for version in roles[1:]:
                    if version not in current:
                        continue
                    other = current[version]
                    diff = other['final_equity'] - base['final_equity']
                    components = {
                        'funding_difference_usd': other['funding_usd'] - base['funding_usd'],
                        'fee_pnl_difference_usd': -other['fee_usd'] + base['fee_usd'],
                        'slippage_difference_usd': other['slippage_usd'] - base['slippage_usd'],
                        'conversion_difference_usd': other['conversion_usd'] - base['conversion_usd'],
                        'basis_endpoint_difference_usd': other['basis_final_usd'] - base['basis_final_usd'],
                        'zero_floor_difference_usd': other['zero_floor_adjustment_usd'] - base['zero_floor_adjustment_usd'],
                        'cash_roundoff_difference_usd': other['cash_roundoff_adjustment_usd'] - base['cash_roundoff_adjustment_usd'],
                    }
                    residual = diff - math.fsum(components.values())
                    error = diff - math.fsum([*components.values(), residual])
                    if abs(error) >= 1e-9:
                        raise ValueError('cross-venue endpoint bridge failed')
                    rows.append({'window': window, 'cost': cost, 'baseline': roles[0], 'comparison': version,
                        'equity_difference_usd': diff, **components,
                        'price_and_path_residual_usd': residual, 'bridge_error_usd': error,
                        'cash_clip_count': other['cash_clip_count'],
                        'interpretation': 'Zero-floor debt forgiveness and disclosed Spot cash-roundoff adjustments are separate nonmarket components. Endpoint basis is a valuation bridge; remaining price, weekly rehedging, cash clipping and compounding effects are inseparable in this descriptive comparison.'})
    return pd.DataFrame(rows)


def _diagnostics(schedules, comparison):
    days = schedules['CM'].index
    um = schedules['UM'].loc[days]
    cm = schedules['CM']
    signals = pd.DataFrame({'g_UM': um['g'], 'g_CM': cm['g'],
                            'budget_UM': um['budget'], 'budget_CM': cm['budget']}, index=days)
    difference = signals['budget_CM'] - signals['budget_UM']
    weekly = difference[(days.weekday == 0) | (days == config.window('IS-CM-SelfSignal').start)]
    summary = {'days': len(days), 'concordance_pct': 100. * float((signals['g_UM'] == signals['g_CM']).mean()),
               'budget_difference_mean': float(difference.mean()),
               'budget_difference_max_abs': float(difference.abs().max()),
               'weekly_budget_difference_mean': float(weekly.mean()),
               'weekly_budget_difference_max_abs': float(weekly.abs().max()),
               'account_comparison_limitation': 'Self-signal accounts start independently on 2020-11-01. No extra UM-reference accounts were added to the fixed 42 jobs; signal concordance is not an equal-initial-state PnL comparison.'}
    rounding = []
    for version in ('V3_CM_DeltaMatched', 'P_CM_DeltaMatched', 'V4_CM_CoinAccum'):
        selected = comparison[(comparison['window'] == 'IS-Common') & (comparison['version'] == version)
                              & (comparison['cost'] == 'main_17')]
        continuous = selected[~selected['whole_contracts']].iloc[0]
        whole = selected[selected['whole_contracts']].iloc[0]
        rounding.append({'version': version, 'continuous_final_equity': continuous['final_equity'],
            'whole_final_equity': whole['final_equity'], 'continuous_coin_growth': continuous['coin_growth'],
            'whole_coin_growth': whole['coin_growth'],
            'equity_difference_usd': whole['final_equity'] - continuous['final_equity'],
            'coin_growth_difference': whole['coin_growth'] - continuous['coin_growth']})
    return {'self_signal': signals, 'self_signal_summary': summary, 'rounding': pd.DataFrame(rounding),
            'crossvenue': crossvenue_differences(comparison)}


def _statistics(accounts, schedules, out, log):
    from .statistics import analyze
    results = {}
    for window, venue, t_version, p_version in PAIRED_FAMILIES:
        family = window + '__' + venue
        log('paired inference ' + family + ': 99999 Webb / 10000 paired whole-month draws')
        t = accounts[_job_id(window, t_version, 'main_17')]
        p = accounts[_job_id(window, p_version, 'main_17')]
        ht = accounts[_job_id(window, t_version, 'harsh_34')]
        hp = accounts[_job_id(window, p_version, 'harsh_34')]
        features = schedules['UM'].loc[t['daily'].index].copy(deep=True)
        stats = analyze(t['daily']['return'], p['daily']['return'], ht['daily']['return'], hp['daily']['return'],
                        features, t['metadata']['order_count'])
        folder = out / 'statistics' / family
        folder.mkdir(parents=True, exist_ok=False)
        for name in ('tests', 'intervals', 'gates', 'daily', 'monthly'):
            stats[name].to_csv(folder / (name + '.csv'))
        write_json(folder / 'metadata.json', stats['metadata'])
        matrices = stats.pop('matrices')
        for name, matrix in matrices.items():
            np.save(folder / (name + '.npy'), matrix, allow_pickle=False)
        del matrices
        results[family] = stats
    return results


def _save_partial(out, exc):
    partial = getattr(exc, 'partial', None)
    if partial is None:
        return
    small = {}
    for key, value in partial.items():
        if isinstance(value, (pd.Series, pd.DataFrame)):
            _write_table(value, out / ('STOPPED_partial_' + key + '.csv.gz'), index=True)
        elif isinstance(value, list) and (not value or isinstance(value[0], dict)):
            _write_table(pd.DataFrame(value), out / ('STOPPED_partial_' + key + '.csv.gz'))
        else:
            small[key] = value
    write_json(out / 'STOPPED_partial_state.json', small)


def execute(contexts, out, input_metadata):
    """Execute the frozen matrix exactly once and preserve the first failure."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started, completed, current = _stamp(), [], None
    def log(message):
        with (out / 'run.log').open('a') as stream:
            stream.write(_stamp() + ' ' + message + '\n')
        print(message, flush=True)
    try:
        source = _validate_inputs(contexts)
        if source == 'real' and out.resolve() != (ROOT / 'reports/trend_hold_venue_round/cp2_btc_is').resolve():
            raise ValueError('real output must use the registered CP2 BTC IS directory')
        write_json(out / 'input_verification.json', input_metadata)
        log('all inputs supplied; preparing canonical UM and native CM daily features')
        schedules = _prepare_features(contexts)
        hashes = {key: _feature_hash(frame) for key, frame in schedules.items()}
        for key, frame in schedules.items():
            frame.to_csv(out / ('features_' + key + '.csv'))
        write_json(out / 'feature_registration.json', {'source': source, 'sha256': hashes,
            'generated_before_first_account': True, 'job_order': jobs(), 'frozen_at': _stamp()})
        accounts, comparison, r5_checks, initial_coins = {}, [], [], {}
        for ordinal, job in enumerate(jobs(), start=1):
            current = job['job_id']
            log(f'account {ordinal}/42: ' + current)
            result = _run_account(contexts, schedules, job)
            # Export first, so an equality failure never discards a finished account.
            account = export_account(result, out / 'accounts' / current, job)
            r5 = _r5_check(result, job, source)
            if r5 is not None:
                r5_checks.append({'job_id': current, **r5})
                write_json(out / 'r5_exact_reconciliation.json', r5_checks)
            if 'initial_coin' in result.metadata:
                key = (job['window'], job['cost'])
                coin = result.metadata['initial_coin']
                if key in initial_coins and coin != initial_coins[key]:
                    raise ValueError('same-cost coin accounts do not share the exact initial conversion')
                initial_coins[key] = coin
            comparison.append(comparison_row(result, job))
            accounts[current] = account
            if any(_feature_hash(frame) != hashes[key] for key, frame in schedules.items()):
                raise ValueError('a frozen feature schedule was mutated by an account')
            completed.append(current)
            write_json(out / 'progress.json', {'status': 'running', 'completed_jobs': completed,
                                               'current_job': current, 'automatic_retry': False})
            del result
            gc.collect()
        table = pd.DataFrame(comparison)
        table.to_csv(out / 'comparison.csv', index=False)
        reconciliation = table[['job_id', 'account_reconciliation_error', 'cm_account_reconciliation_error',
                                'coin_reconciliation_error', 'max_wallet_reconciliation_error',
                                'zero_floor_adjustment_usd', 'cash_roundoff_adjustment_usd', 'liquidated']]
        reconciliation.to_csv(out / 'reconciliation.csv', index=False)
        stats = _statistics(accounts, schedules, out, log)
        diagnostics = _diagnostics(schedules, table)
        for name in ('self_signal', 'rounding', 'crossvenue'):
            diagnostics[name].to_csv(out / (name + '.csv'), index=name == 'self_signal')
        write_json(out / 'self_signal_summary.json', diagnostics['self_signal_summary'])
        from .plotting import render_outputs
        log('rendering registered descriptive comparison figures')
        figures = render_outputs(accounts, schedules, stats, table, diagnostics, out)
        summary = {'status': 'completed', 'stage': 'CP2 BTC IS only', 'source': source,
            'started_at': started, 'ended_at': _stamp(), 'combination_count': len(completed),
            'completed_jobs': completed, 'input_metadata': input_metadata, 'figures': figures,
            'r5_exact_reconciliation': r5_checks, 'feature_sha256': hashes,
            'locks': {'BTC_IS_UNLOCKED': scope.BTC_IS_UNLOCKED,
                      'POST_IS_AND_ETH_UNLOCKED': scope.POST_IS_AND_ETH_UNLOCKED},
            'paired_families': list(stats), 'automatic_promotion': False,
            'limitations': ['Only five complete-cost primary families receive V1–V6 diagnostics.',
                'Rounding and native-CM self-signal jobs are diagnostic, not additional gate families.',
                'V4 includes collateral-coin exposure and is compared to HODL for coin accumulation.',
                'Fixed 0.5% MMR uses contract minute OHLC, not an exchange historical liquidation model.',
                '1 USD = 1 USDT is assumed. Cross-venue endpoint bridges are descriptive, not causal attribution.',
                'No Post-IS/ETH data or accounts are authorized in CP2.']}
        write_json(out / 'summary.json', summary)
        write_json(out / 'progress.json', {'status': 'completed', 'completed_jobs': completed,
                                          'automatic_retry': False})
        log('completed 42 accounts and five paired inference families; STOP at CP2')
        return summary
    except Exception as exc:
        write_json(out / 'STOPPED.json', {'stage': 'CP2', 'started_at': started, 'stopped_at': _stamp(),
            'error_type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc(),
            'completed_jobs': completed, 'current_job': current, 'automatic_retry': False})
        _save_partial(out, exc)
        log('STOPPED: ' + str(exc))
        raise
