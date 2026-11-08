"""Four descriptive CP3 figures from already materialized account outputs.

No market files, engines, features, inference or performance estimators are
called here. The runtime supplies daily extrema of its original effective 1m
DD series, including exact trough times and corresponding coin balances. The
plot shows that envelope without substituting drawdowns of daily closes.

Public API follows the runtime's frozen chart-call shape. ``build_figures``
returns caller-owned matplotlib figures; ``render_outputs`` writes four PNGs
and their captions, closes the figures, and never overwrites an artifact.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGURES = ('fig_1.png', 'fig_2.png', 'fig_3.png', 'fig_4.png')
PRIMARY = ('V1_USDM', 'V2_SPOT', 'V3_CM_DeltaMatched')
PAIRED = ('V1_USDM', 'P_USDM', 'V2_SPOT', 'P_SPOT',
          'V3_CM_DeltaMatched', 'P_CM_DeltaMatched')
COIN = ('V4_CM_CoinAccum', 'HODL_1x_Coin')
FUNDING = ('V1_USDM', 'V3_CM_DeltaMatched', 'V4_CM_CoinAccum')
WINDOWS = ('IS-Common', 'Post-IS')
SPLIT = pd.Timestamp('2025-07-01', tz='UTC')
CHOP_START = pd.Timestamp('2024-03-01', tz='UTC')
CHOP_END = pd.Timestamp('2024-11-01', tz='UTC')
LABELS = {
    'V1_USDM': 'V1 UM', 'P_USDM': 'P UM',
    'V2_SPOT': 'V2 Spot', 'P_SPOT': 'P Spot',
    'V3_CM_DeltaMatched': 'V3 CM hedge', 'P_CM_DeltaMatched': 'P CM hedge',
    'V4_CM_CoinAccum': 'V4 coin accumulation', 'HODL_1x_Coin': 'HODL coin',
}
COLORS = {
    'V1_USDM': '#2879AE', 'P_USDM': '#2879AE',
    'V2_SPOT': '#657C37', 'P_SPOT': '#657C37',
    'V3_CM_DeltaMatched': '#BA7729', 'P_CM_DeltaMatched': '#BA7729',
    'V4_CM_CoinAccum': '#AB537B', 'HODL_1x_Coin': '#556878',
}
STYLE = {
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 12, 'axes.labelsize': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#AEB8C1', 'axes.labelcolor': '#233544',
    'text.color': '#233544', 'xtick.color': '#425665', 'ytick.color': '#425665',
    'grid.color': '#DCE2E7', 'grid.linewidth': .6,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
}
CAPTIONS = '''# CP3 descriptive figure guide

All figures use main_17, canonical UM signals and continuous contract sizes.
This is the user-authorized one-shot descriptive evaluation with no automatic
promotion and no parameter selection. USD = USDT at the registered fixed
parity. These figures use completed account outputs, not new simulations.

1. **BTC Post-IS equity and effective 1m drawdown.** USD curves show daily
   closing cash-comparable equity. The shaded drawdown envelope is each day's
   minimum and maximum of the original effective 1m drawdown series; dots mark
   the exact original minute and magnitude of each account's maximum drawdown.
   Bad minute placeholders do not establish a peak or trough. This is not a
   drawdown computed from daily closes. V4 has additional collateral exposure
   (1 + gL), so its USD curve is not an equal-risk carrier ranking. The dashed
   2025-07-01 line divides VAL/OOS without resetting or liquidating the account.
   The shaded chop interval is [2024-03-01, 2024-11-01); its diagnosis inherits
   the February 29 valuation and does not invent a slice-end liquidation.
2. **Signed funding.** Curves are the exact daily accumulation of account
   funding ledger flows; positive means received. CM funding uses the existing
   event-time Spot conversion. IS-Common and Post-IS have independent initial
   accounts, so they are displayed separately with no connecting wealth path.
   Post funding remains cumulative through the VAL/OOS boundary. Three bars
   per carrier show the supplied IS, VAL and OOS funding totals, not differences
   between independently rescaled cumulative curves.
3. **V4 versus HODL coin accumulation.** C / C0 uses each independent window's
   actual initial conversion; C is total coin equity, including unrealized PnL.
   A ratio of 1 is zero coin growth. Diamonds locate the original maximum USD
   drawdown minute at its actual coin balance; crosses locate modeled pressure
   liquidation at its residual coin balance. The fixed 0.5% MMR and contract
   OHLC extreme are a simplified marked-price proxy, not the historical exchange
   liquidation engine. V4's additional collateral exposure is shown against
   HODL, not promoted as a same-risk substitute for V1/V2/V3.
4. **ETH same-target-exposure comparison.** V1/V2/V3 and their corresponding
   P controls are aligned within each independently initialized IS-Common or
   Post-IS panel. Solid lines are trend accounts and dashed lines are passive
   controls. Actual daily mean exposure may drift between rebalances. The
   Sharpe/MaxDD overview uses supplied registered daily Sharpe and effective
   1m MaxDD; undefined Sharpe remains NA, not zero. No 69-month continuous
   account is implied. Any separately reported (1+R_IS)*(1+R_Post)-1 is an
   independent-account compound reference, not continuous positions.

Partial-period valuations use the existing continuous account and preceding
close; no hypothetical fees or FX are added at VAL/OOS or chop boundaries.
Only the actual full-account endpoint includes terminal closing/conversion.
Coin endpoints retain final coin before final FX. Statistical references, if
reported elsewhere, are descriptive within one venue/window and do not create
a global multi-family conclusion or an automatic promotion decision.
'''


def _required(frame, columns, label):
    if not isinstance(frame, pd.DataFrame) or not set(columns).issubset(frame):
        raise ValueError(f'{label}: missing required columns')
    if frame.columns.has_duplicates:
        raise ValueError(f'{label}: duplicate columns')


def _finite(frame, column, label, *, nonnegative=False):
    values = frame[column].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (nonnegative and np.any(values < 0.)):
        raise ValueError(f'{label}: invalid {column}')
    return values


def _dates(values, label):
    try:
        index = pd.DatetimeIndex(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{label}: invalid timestamps') from error
    if index.tz is None or index.hasnans:
        raise ValueError(f'{label}: complete timezone-aware timestamps required')
    return index.tz_convert('UTC').as_unit('ns')


def _main(frame):
    if not pd.api.types.is_bool_dtype(frame['whole_contracts']) or frame['whole_contracts'].isna().any():
        raise ValueError('whole_contracts requires complete booleans')
    return frame.loc[(frame['cost'] == 'main_17') &
                     (frame['signal_source'] == 'UM') & ~frame['whole_contracts']]


def _required_accounts():
    for asset in ('BTC', 'ETH'):
        for window in WINDOWS:
            versions = (PAIRED + COIN if asset == 'ETH' else
                        FUNDING + ('HODL_1x_Coin',) if window == 'IS-Common' else
                        PRIMARY + COIN)
            for version in versions:
                yield asset, window, version


def _prepare(accounts, comparison, diagnostics):
    identity = ('job_id', 'asset', 'window', 'version', 'cost',
                'signal_source', 'whole_contracts')
    _required(comparison, identity, 'comparison')
    if comparison['job_id'].duplicated().any():
        raise ValueError('comparison: duplicate job_id')
    primary = _main(comparison)
    selected = {}
    columns = ('equity', 'equity_drawdown_minute_min',
               'equity_drawdown_minute_max', 'equity_drawdown_minute_min_time',
               'funding_usd', 'funding_cumulative_usd', 'actual_leverage_mean')
    for key in _required_accounts():
        asset, window, version = key
        rows = primary.loc[(primary['asset'] == asset) & (primary['window'] == window) &
                           (primary['version'] == version)]
        if len(rows) != 1:
            raise ValueError(f'plotted account must be unique: {key}')
        row = rows.iloc[0]
        job_id = row['job_id']
        if job_id not in accounts:
            raise ValueError(f'missing plotted account: {job_id}')
        account = accounts[job_id]
        job = account.get('job', {})
        if any(job.get(field) != row[field] for field in identity if field != 'job_id'):
            raise ValueError(f'account identity disagrees with comparison: {job_id}')
        chart = account.get('chart_daily')
        _required(chart, columns, f'account {job_id}')
        index = _dates(chart.index, 'chart_daily index')
        if not len(index) or index.has_duplicates or not index.is_monotonic_increasing or not index.equals(index.normalize()):
            raise ValueError('chart_daily requires unique ordered UTC days')
        _finite(chart, 'equity', job_id, nonnegative=True)
        low = _finite(chart, 'equity_drawdown_minute_min', job_id)
        high = _finite(chart, 'equity_drawdown_minute_max', job_id)
        if np.any(low < -1.) or np.any(high > 0.) or np.any(low > high):
            raise ValueError('effective minute DD envelope must lie in [-1, 0]')
        troughs = _dates(chart['equity_drawdown_minute_min_time'], 'minute DD extrema')
        if not troughs.normalize().equals(index):
            raise ValueError('minute DD extrema must belong to their exact UTC chart day')
        for column in ('funding_usd', 'funding_cumulative_usd', 'actual_leverage_mean'):
            _finite(chart, column, job_id)
        pressure = account.get('pressure_events', pd.DataFrame())
        if not isinstance(pressure, pd.DataFrame):
            raise ValueError('pressure events must be a DataFrame')
        if len(pressure):
            _required(pressure, ('time', 'residual_coin'), 'pressure events')
            event_times = _dates(pressure['time'], 'pressure event times')
            if (event_times < index[0]).any() or (event_times >= index[-1] + pd.Timedelta(days=1)).any():
                raise ValueError('pressure event lies outside its independent account')
            _finite(pressure, 'residual_coin', job_id, nonnegative=True)
        if version in COIN:
            _required(chart, ('coin_equity', 'coin_equity_at_minute_dd_min'), job_id)
            for column in ('coin_equity', 'coin_equity_at_minute_dd_min'):
                _finite(chart, column, job_id, nonnegative=True)
            initial = float(account.get('metadata', {}).get('initial_coin', np.nan))
            if not np.isfinite(initial) or initial <= 0.:
                raise ValueError('coin chart requires positive finite initial coin')
            if account.get('metadata', {}).get('liquidated', False) and pressure.empty:
                raise ValueError('liquidated coin account is missing original pressure events')
        selected[key] = account

    period_metrics = diagnostics.get('period_metrics')
    _required(period_metrics, identity + ('period', 'funding_usd', 'sharpe', 'max_drawdown'),
              'period metrics')
    periods = _main(period_metrics)
    if periods.duplicated(['job_id', 'period']).any():
        raise ValueError('period metrics: duplicate account period')
    selected_periods = {}
    for asset in ('BTC', 'ETH'):
        for period in ('IS-Common', 'VAL', 'OOS', 'Post-IS'):
            window = 'IS-Common' if period == 'IS-Common' else 'Post-IS'
            versions = set(FUNDING)
            if asset == 'ETH' and period in WINDOWS:
                versions.update(PAIRED)
            for version in versions:
                rows = periods.loc[(periods['asset'] == asset) & (periods['window'] == window) &
                                   (periods['period'] == period) & (periods['version'] == version)]
                if len(rows) != 1:
                    raise ValueError(f'missing or duplicate period metric: {asset}/{period}/{version}')
                _finite(rows, 'funding_usd', 'period funding')
                dd = rows['max_drawdown'].to_numpy(dtype=float)
                undefined = np.isnan(dd)
                if undefined.any():
                    if 'start_equity' not in rows or np.any(
                            rows['start_equity'].to_numpy(dtype=float)[undefined] != 0.):
                        raise ValueError('undefined period DD requires zero actual preceding equity')
                if (np.isinf(dd).any() or np.any(dd < 0.) or np.any(dd > 1.) or
                        np.isinf(rows['sharpe'].to_numpy(dtype=float)).any()):
                    raise ValueError('invalid period risk overview')
                selected_periods[asset, period, version] = rows.iloc[0]
    return selected, selected_periods


def _axis(ax, *, dates=False):
    ax.grid(axis='y', alpha=.85)
    ax.set_axisbelow(True)
    if dates:
        locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _post_boundaries(ax, *, chop=False):
    ax.axvline(SPLIT, color='#667480', lw=1., ls='--', label='VAL / OOS boundary')
    if chop:
        ax.axvspan(CHOP_START, CHOP_END, color='#E2CC96', alpha=.28,
                   label='2024-03 to 2024-10 chop window')


def _finish(fig, title, footer):
    fig.suptitle(title, x=.055, y=.98, ha='left', fontsize=18, fontweight='bold')
    fig.text(.055, .018, footer, fontsize=9, color='#526473', va='bottom')
    fig.tight_layout(rect=(.025, .075, .99, .945), h_pad=3., w_pad=3.)
    return fig


def _extreme(account):
    chart = account['chart_daily']
    position = int(np.argmin(chart['equity_drawdown_minute_min'].to_numpy(dtype=float)))
    row = chart.iloc[position]
    return pd.Timestamp(row['equity_drawdown_minute_min_time']), row


def _fig_equity(accounts):
    fig, (equity, dd) = plt.subplots(2, 1, figsize=(16., 10.), sharex=True,
                                   gridspec_kw={'height_ratios': [1.25, 1.]})
    for version in PRIMARY + ('V4_CM_CoinAccum',):
        account = accounts['BTC', 'Post-IS', version]
        chart = account['chart_daily']
        color, label = COLORS[version], LABELS[version]
        equity.plot(chart.index, chart['equity'].to_numpy() / 1000., color=color,
                    lw=1.6, label=label, ls='-.' if version.startswith('V4') else '-')
        dd.fill_between(chart.index, chart['equity_drawdown_minute_min'].to_numpy() * 100.,
                        chart['equity_drawdown_minute_max'].to_numpy() * 100., color=color, alpha=.13)
        dd.plot(chart.index, chart['equity_drawdown_minute_min'].to_numpy() * 100.,
                color=color, lw=1., label=label)
        time, row = _extreme(account)
        loss = float(row['equity_drawdown_minute_min']) * 100.
        dd.scatter([time], [loss], color=color, marker='D', s=38, zorder=5,
                   label=f'{label} maximum drawdown')
        dd.annotate(f'{label}: {loss:.2f}% | {time:%Y-%m-%d %H:%M} UTC',
                    xy=(.02, .08 + .065 * (PRIMARY + ('V4_CM_CoinAccum',)).index(version)),
                    xycoords='axes fraction', color=color, fontsize=9)
    equity.axhline(100., color='#A1ACB5', lw=.8, ls=':')
    equity.set_title('One continuous Post-IS account per version | fresh 100,000 USD start', loc='left')
    equity.set_ylabel('Cash-comparable equity (USD thousands)')
    equity.set_ylim(bottom=0.)
    dd.set_title('Original effective 1m drawdown | daily min/max envelope and exact extrema', loc='left')
    dd.set_ylabel('Drawdown from account peak (%)')
    dd.set_ylim(top=2.)
    for ax in (equity, dd):
        _post_boundaries(ax, chop=True)
        _axis(ax, dates=True)
    equity.legend(frameon=False, ncol=3, fontsize=9, loc='upper left')
    return _finish(fig, 'BTC Post-IS | Equity and minute drawdown',
                   'Main cost. V4 includes additional collateral exposure (1 + gL). VAL / OOS split: 2025-07-01, with no reset.\n'
                   'Chop shading: [2024-03-01, 2024-11-01). Partial periods use marked equity; only the real terminal close includes FX.')


def _fig_funding(accounts, periods):
    fig, axes = plt.subplots(2, 3, figsize=(18., 10.))
    for row, asset in enumerate(('BTC', 'ETH')):
        for column, window in enumerate(WINDOWS):
            ax = axes[row, column]
            for version in FUNDING:
                chart = accounts[asset, window, version]['chart_daily']
                ax.plot(chart.index, chart['funding_cumulative_usd'].to_numpy() / 1000.,
                        color=COLORS[version], lw=1.6, label=LABELS[version],
                        ls='-.' if version.startswith('V4') else '-')
            title = ('IS-Common | independent initial account' if window == 'IS-Common'
                     else 'Post-IS | continuous through VAL / OOS')
            ax.set_title(f'{asset} {title}', loc='left', fontsize=11)
            ax.axhline(0., color='#8897A4', lw=.8)
            ax.set_ylabel('Cumulative signed funding (USD thousands)')
            if window == 'Post-IS':
                _post_boundaries(ax)
            ax.legend(frameon=False, fontsize=8, loc='best')
            _axis(ax, dates=True)
        ax = axes[row, 2]
        x, width = np.arange(3), .24
        for offset, version in enumerate(FUNDING):
            values = [float(periods[asset, period, version]['funding_usd']) / 1000.
                      for period in ('IS-Common', 'VAL', 'OOS')]
            ax.bar(x + (offset - 1) * width, values, width,
                   color=COLORS[version], label=LABELS[version])
        ax.set_xticks(x, ['IS-Common', 'VAL', 'OOS'])
        ax.set_title(f'{asset} | Funding within each segment', loc='left')
        ax.set_ylabel('Signed funding (USD thousands)')
        ax.axhline(0., color='#8897A4', lw=.8)
        ax.legend(frameon=False, fontsize=8)
        _axis(ax)
    return _finish(fig, 'BTC and ETH | Funding by carrier and period',
                   'Positive = received. IS-Common and Post-IS use independent initial accounts; no curve joins them.\n'
                   'VAL / OOS bars use the existing continuous account ledger. CM funding is valued at each event\'s Spot reference.')


def _fig_coin(accounts):
    fig, axes = plt.subplots(2, 2, figsize=(16., 10.))
    for row, asset in enumerate(('BTC', 'ETH')):
        for column, window in enumerate(WINDOWS):
            ax = axes[row, column]
            for version in COIN:
                account = accounts[asset, window, version]
                chart = account['chart_daily']
                initial = float(account['metadata']['initial_coin'])
                color = COLORS[version]
                short = 'V4' if version.startswith('V4') else 'HODL'
                ax.plot(chart.index, chart['coin_equity'].to_numpy() / initial,
                        color=color, lw=1.7, label=LABELS[version],
                        ls='-' if short == 'V4' else '--')
                time, extreme = _extreme(account)
                value = float(extreme['coin_equity_at_minute_dd_min']) / initial
                ax.scatter([time], [value], color=color, s=45, marker='D', zorder=5,
                           label=f'{short} maximum USD drawdown')
                ax.annotate(f'{short} USD DD {100. * float(extreme["equity_drawdown_minute_min"]):.2f}%',
                            (time, value), textcoords='offset points',
                            xytext=(5, 10 if short == 'V4' else -16), color=color, fontsize=8)
                events = account.get('pressure_events', pd.DataFrame())
                if len(events):
                    ax.scatter(_dates(events['time'], 'pressure times'),
                               events['residual_coin'].to_numpy(dtype=float) / initial,
                               marker='x', s=80, linewidths=1.8, color='#9D5C12', zorder=6,
                               label=f'{short} pressure liquidation')
            ax.axhline(1., color='#A1ACB5', lw=.8, ls=':')
            ax.set_title(f'{asset} {window} | independent initial C0', loc='left')
            ax.set_ylabel('Total coin equity / initial coin (C / C0)')
            ax.set_ylim(bottom=0.)
            if window == 'Post-IS':
                _post_boundaries(ax)
            ax.legend(frameon=False, fontsize=8, ncol=2, loc='best')
            _axis(ax, dates=True)
    return _finish(fig, 'V4 versus HODL | Coin accumulation and pressure events',
                   'C / C0 = 1 means zero coin growth. Each window starts independently; no 69-month position is implied.\n'
                   'V4 adds collateral exposure. Diamonds: original USD-DD extrema; crosses: simplified 0.5% MMR pressure liquidation.')


def _risk_overview(ax, periods, window):
    missing = []
    for position, version in enumerate(PAIRED):
        row = periods['ETH', window, version]
        sharpe, loss = float(row['sharpe']), float(row['max_drawdown']) * 100.
        if np.isnan(sharpe) or np.isnan(loss):
            sharpe_text = 'NA' if np.isnan(sharpe) else f'{sharpe:.2f}'
            dd_text = 'NA' if np.isnan(loss) else f'{loss:.2f}%'
            missing.append(f'{LABELS[version]}: Sharpe {sharpe_text}, DD {dd_text}')
            continue
        color = COLORS[version]
        passive = version.startswith('P_')
        ax.scatter([loss], [sharpe], marker='o' if passive else 's', s=60,
                   facecolors='white' if passive else color, edgecolors=color,
                   linewidths=1.4, label=LABELS[version], zorder=3)
        ax.annotate(LABELS[version], (loss, sharpe), textcoords='offset points',
                    xytext=(7 if not passive else -7, 5 if position % 2 == 0 else -12),
                    ha='left' if not passive else 'right', color=color, fontsize=8)
    if missing:
        ax.text(.02, .97, '\n'.join(missing), transform=ax.transAxes, va='top', fontsize=8)
    ax.set_title(f'{window} | Supplied Sharpe and effective 1m MaxDD', loc='left', fontsize=11)
    ax.set_xlabel('Maximum drawdown (%)')
    ax.set_ylabel('Annualized daily Sharpe')
    ax.set_xlim(left=0.)
    ax.axhline(0., color='#A1ACB5', lw=.8, ls=':')
    ax.margins(x=.2, y=.2)
    _axis(ax)


def _fig_eth(accounts, periods):
    fig, axes = plt.subplots(3, 2, figsize=(17., 14.))
    for column, window in enumerate(WINDOWS):
        equity, exposure = axes[0, column], axes[1, column]
        for version in PAIRED:
            chart = accounts['ETH', window, version]['chart_daily']
            kwargs = {'color': COLORS[version], 'lw': 1.45, 'label': LABELS[version],
                      'ls': '--' if version.startswith('P_') else '-'}
            equity.plot(chart.index, chart['equity'].to_numpy() / 1000., **kwargs)
            exposure.plot(chart.index, chart['actual_leverage_mean'].to_numpy(), **kwargs)
        equity.axhline(100., color='#A1ACB5', lw=.8, ls=':')
        equity.set_title(f'ETH {window} | independent 100,000 USD start', loc='left')
        equity.set_ylabel('Cash-comparable equity (USD thousands)')
        equity.set_ylim(bottom=0.)
        equity.legend(frameon=False, ncol=3, fontsize=8, loc='best')
        exposure.set_title('Same target exposure | actual daily mean may drift', loc='left')
        exposure.set_ylabel('Actual mean price exposure (x equity)')
        for ax in (equity, exposure):
            if window == 'Post-IS':
                _post_boundaries(ax)
            _axis(ax, dates=True)
        _risk_overview(axes[2, column], periods, window)
    return _finish(fig, 'ETH | Same-target-exposure carriers and passive controls',
                   'Solid = trend (T); dashed = corresponding continuous-budget passive (P). V4 is a separate exposure group in Figure 3.\n'
                   'IS and Post are independently initialized. A compound reference multiplies their wealth ratios; it is not a continuous 69-month account.')


def build_figures(accounts, schedules, stats, comparison, diagnostics):
    """Build four figures from supplied tables, leaving their lifetime to caller.

``schedules`` and ``stats`` preserve the runtime call interface. The plots do
not recompute features, risk statistics or inference from either argument.
Validation completes before any figure is created.
    """
    selected, periods = _prepare(accounts, comparison, diagnostics)
    figures = {}
    with plt.rc_context(STYLE):
        try:
            figures[FIGURES[0]] = _fig_equity(selected)
            figures[FIGURES[1]] = _fig_funding(selected, periods)
            figures[FIGURES[2]] = _fig_coin(selected)
            figures[FIGURES[3]] = _fig_eth(selected, periods)
        except BaseException:
            for figure in figures.values():
                plt.close(figure)
            raise
    return figures


def render_outputs(accounts, schedules, stats, comparison, diagnostics, out):
    """Write the four fixed PNGs and captions without reading or replacing data."""
    out = Path(out)
    if not out.is_dir():
        raise ValueError('existing result directory required for CP3 charts')
    for name in (*FIGURES, 'figure_captions.md'):
        if (out / name).exists():
            raise FileExistsError(out / name)
    figures = build_figures(accounts, schedules, stats, comparison, diagnostics)
    try:
        for name, figure in figures.items():
            # Exclusive creation also protects against a concurrent writer.
            with (out / name).open('xb') as handle:
                figure.savefig(handle, format='png', dpi=160,
                               metadata={'Software': 'ppvp_quant CP3 descriptive evaluation'})
        with (out / 'figure_captions.md').open('x', encoding='utf-8') as handle:
            handle.write(CAPTIONS)
    finally:
        for figure in figures.values():
            plt.close(figure)
    return list(FIGURES)
