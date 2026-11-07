"""Four CP2 charts from caller-owned result tables; no market loading or inference.

The plotted primary return is cash-comparable equity with the real terminal
conversion. Extra collateral exposure is shown in its own panels, never ranked
as an equal-risk substitute for the three delta-matched venues.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

FIGURES = (
    'figure_1_equity_and_coin.png',
    'figure_2_cost_steps.png',
    'figure_3_funding_friction_and_gates.png',
    'figure_4_signal_and_rounding_diagnostics.png',
)
COSTS = ('gross_0', 'main_17', 'harsh_34')
COST_NAMES = ('Gross: funding off', 'Main', 'Harsh')
COST_COLORS = ('#8B98A7', '#287EA5', '#CF7835')
EQUAL_VERSIONS = ('V1_USDM', 'P_USDM', 'V2_SPOT', 'P_SPOT',
                  'V3_CM_DeltaMatched', 'P_CM_DeltaMatched')
LABELS = {
    'V1_USDM': 'T UM', 'P_USDM': 'P UM',
    'V2_SPOT': 'T Spot', 'P_SPOT': 'P Spot',
    'V3_CM_DeltaMatched': 'T CM hedge', 'P_CM_DeltaMatched': 'P CM hedge',
    'V4_CM_CoinAccum': 'V4 coin accumulation', 'HODL_1x_Coin': 'HODL collateral',
}
VERSION_COLORS = {
    'V1_USDM': '#246BA0', 'P_USDM': '#246BA0',
    'V2_SPOT': '#287E62', 'P_SPOT': '#287E62',
    'V3_CM_DeltaMatched': '#A66A22', 'P_CM_DeltaMatched': '#A66A22',
    'V4_CM_CoinAccum': '#9B486A', 'HODL_1x_Coin': '#6A7480',
}
FAMILIES = ('IS-Full__UM', 'IS-Full__SPOT', 'IS-Common__UM',
            'IS-Common__SPOT', 'IS-Common__CM_DeltaMatched')
GATES = ('V1', 'V2', 'V3', 'V4', 'V5', 'V6')
STYLE = {
    'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12,
    'axes.labelsize': 10, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#AEB8C1', 'axes.labelcolor': '#233544',
    'text.color': '#233544', 'xtick.color': '#425665', 'ytick.color': '#425665',
    'grid.color': '#DCE2E7', 'grid.linewidth': .6, 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'savefig.facecolor': 'white',
}
CAPTIONS = '''# CP2 BTC IS figure guide

1. **Equity and coin balances.** Main-cost cash-comparable daily equity for each
   strategy and its within-venue passive control. The Full and Common windows
   use separate initial accounts. V4 carries additional collateral exposure
   (`1 + gL`) and is compared separately with HODL; its coin index is not a USD
   risk-adjusted ranking. Terminal cash-equity points include the actual closing
   trade and Spot conversion, while the coin index retains final coin balances.
2. **Cost steps.** Returns use each account's actual endpoint. In gross_0,
   funding is disabled; gross-to-main differences therefore combine funding and
   transaction friction, not friction alone. V4/HODL coin growth is kept in a
   separate panel from equal-target-exposure cash returns.
3. **Funding, friction and gates.** Funding is signed and translated at each
   event's Spot reference for CM. Friction is positive fees minus signed
   slippage and conversion PnL. MaxDD is the positive loss fraction from valid
   minute equity. The V1–V6 matrix is an IS diagnostic; Holm controls only two
   hypotheses within one venue/window, not a global cross-venue family.
4. **Signal and rounding diagnostics.** The 38-month panel compares canonical
   UM and native CM decision schedules, not independently initialized account
   PnL. Whole-contract differences are whole minus continuous face, from the
   same Common-window start and main cost. These are diagnostics, not parameter
   selection or additional promotion gates.
'''


def _required(frame, fields, label):
    if not isinstance(frame, pd.DataFrame) or not set(fields).issubset(frame.columns):
        raise ValueError(f'{label}: missing required columns')
    if frame.columns.has_duplicates:
        raise ValueError(f'{label}: duplicate columns')


def _values(frame, column, label):
    values = frame[column].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f'{label}: nonfinite {column}')
    return values


def _prepare(accounts, stats, comparison, diagnostics):
    fields = ('job_id', 'window', 'version', 'cost', 'signal_source', 'whole_contracts',
              'total_return', 'coin_growth', 'funding_usd', 'fee_usd', 'slippage_usd',
              'conversion_usd', 'max_drawdown', 'cash_clip_count')
    _required(comparison, fields, 'comparison')
    if comparison['job_id'].duplicated().any():
        raise ValueError('comparison: duplicate job_id')
    if not pd.api.types.is_bool_dtype(comparison['whole_contracts']):
        raise ValueError('comparison: whole_contracts requires booleans')
    primary = comparison.loc[(comparison['signal_source'] == 'UM') & ~comparison['whole_contracts']].copy()
    selected = {}
    for window, versions in [('IS-Full', EQUAL_VERSIONS[:4]),
                             ('IS-Common', EQUAL_VERSIONS + ('V4_CM_CoinAccum', 'HODL_1x_Coin'))]:
        for version in versions:
            for cost in COSTS:
                rows = primary.loc[(primary.window == window) & (primary.version == version) & (primary.cost == cost)]
                if len(rows) != 1:
                    raise ValueError(f'primary account comparison is not unique: {window}/{version}/{cost}')
                selected[(window, version, cost)] = rows.iloc[0]
                if cost != 'main_17':
                    continue
                job_id = rows.iloc[0]['job_id']
                if job_id not in accounts:
                    raise ValueError(f'missing plotted account: {job_id}')
                account = accounts[job_id]
                daily = account['daily']
                _required(daily, ('equity',), 'account daily')
                index = daily.index
                if not isinstance(index, pd.DatetimeIndex) or index.tz is None or not len(index) or index.has_duplicates or not index.is_monotonic_increasing:
                    raise ValueError('account daily: ordered aware calendar required')
                if np.any(_values(daily, 'equity', 'account daily') < 0.):
                    raise ValueError('account daily: negative equity')
                if version in ('V4_CM_CoinAccum', 'HODL_1x_Coin'):
                    _required(daily, ('coin_equity',), 'coin account')
                    _values(daily, 'coin_equity', 'coin account')
                    if not np.isfinite(account['metadata']['initial_coin']) or account['metadata']['initial_coin'] <= 0.:
                        raise ValueError('coin account: positive initial coin required')
    for column in ('total_return', 'funding_usd', 'fee_usd', 'slippage_usd', 'conversion_usd', 'max_drawdown', 'cash_clip_count'):
        _values(primary, column, 'primary comparison')
    for family in FAMILIES:
        if family not in stats or 'gates' not in stats[family]:
            raise ValueError('missing gate family: '+family)
        frame = stats[family]['gates']
        _required(frame, ('passed',), 'gates')
        if not frame.index.equals(pd.Index(GATES)):
            raise ValueError('gates: ordered V1-V6 required')
    signal = diagnostics['self_signal']
    _required(signal, ('g_UM', 'g_CM', 'budget_UM', 'budget_CM'), 'self signal')
    if not isinstance(signal.index, pd.DatetimeIndex) or signal.index.tz is None or not len(signal):
        raise ValueError('self signal: nonempty aware calendar required')
    for column in ('g_UM', 'g_CM'):
        if not pd.api.types.is_bool_dtype(signal[column]) or signal[column].isna().any():
            raise ValueError('self signal: boolean switches required')
    _values(signal, 'budget_UM', 'self signal')
    _values(signal, 'budget_CM', 'self signal')
    rounding = diagnostics['rounding']
    _required(rounding, ('version', 'continuous_final_equity', 'whole_final_equity',
                        'continuous_coin_growth', 'whole_coin_growth'), 'rounding')
    expected = ('V3_CM_DeltaMatched', 'P_CM_DeltaMatched', 'V4_CM_CoinAccum')
    if len(rounding) != 3 or set(rounding['version']) != set(expected):
        raise ValueError('rounding: exactly three registered diagnostic versions required')
    rounding = rounding.set_index('version').loc[list(expected)].copy()
    for column in ('continuous_final_equity', 'whole_final_equity', 'continuous_coin_growth', 'whole_coin_growth'):
        _values(rounding, column, 'rounding')
    return selected, signal, rounding


def _style_axis(ax, *, dates=False):
    ax.grid(axis='y', alpha=.8)
    ax.set_axisbelow(True)
    if dates:
        locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _save(fig, path, title, footer):
    fig.suptitle(title, fontsize=17, fontweight='bold', x=.055, ha='left')
    fig.text(.055, .016, footer, fontsize=9, color='#526473')
    fig.tight_layout(rect=(.02, .06, .99, .94), h_pad=3., w_pad=2.5)
    try:
        fig.savefig(path, dpi=160, metadata={'Software': 'ppvp_quant CP2 BTC IS'})
    finally:
        plt.close(fig)


def _account(accounts, selected, window, version):
    row = selected[(window, version, 'main_17')]
    return accounts[row['job_id']]


def _equity_panel(ax, accounts, selected, window, versions, title):
    for version in versions:
        daily = _account(accounts, selected, window, version)['daily']
        ax.plot(daily.index, daily['equity'].to_numpy()/1000.,
                color=VERSION_COLORS[version], lw=1.45,
                ls='--' if version.startswith('P_') else '-', label=LABELS[version])
    ax.axhline(100., color='#A7B2BC', lw=.8, ls=':')
    ax.set_title(title, loc='left')
    ax.set_ylabel('Cash-comparable equity (USD thousands)')
    ax.set_ylim(bottom=0.)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc='upper left')
    _style_axis(ax, dates=True)


def _cost_panel(ax, selected, window, versions, metric, title):
    x = np.arange(len(versions), dtype=float)
    width = .24
    for c, cost in enumerate(COSTS):
        values = np.array([float(selected[(window, v, cost)][metric]) for v in versions])
        if not np.isfinite(values).all():
            raise ValueError('nonfinite selected cost-panel metric')
        ax.bar(x+(c-1)*width, values*100., width, label=COST_NAMES[c], color=COST_COLORS[c])
    ax.axhline(0., color='#8897A4', lw=.8)
    ax.set_xticks(x, [LABELS[v] for v in versions], rotation=25, ha='right')
    ax.set_title(title, loc='left')
    ax.set_ylabel('Coin growth (%)' if metric == 'coin_growth' else 'Cumulative cash return (%)')
    ax.legend(frameon=False, fontsize=8, loc='best')
    _style_axis(ax)


def render_outputs(accounts, features, stats, comparison, diagnostics, out):
    """Create the fixed four charts from completed results, without mutating inputs."""
    out = Path(out)
    if not out.is_dir():
        raise ValueError('existing result directory required for charts')
    for name in (*FIGURES, 'figure_captions.md'):
        if (out/name).exists():
            raise FileExistsError(out/name)
    selected, signal, rounding = _prepare(accounts, stats, comparison, diagnostics)
    # Features are accepted for a stable runtime interface; diagnostics already
    # contain the precise aligned signal comparison, so no feature recomputation.
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(16., 10.))
        _equity_panel(axes[0,0], accounts, selected, 'IS-Full', EQUAL_VERSIONS[:4],
                      'Full window | UM and Spot, T versus P')
        _equity_panel(axes[0,1], accounts, selected, 'IS-Common', EQUAL_VERSIONS,
                      'Common window | same target-exposure group')
        extra = ('V4_CM_CoinAccum', 'HODL_1x_Coin')
        _equity_panel(axes[1,0], accounts, selected, 'IS-Common', extra,
                      'Separate exposure group | V4 versus HODL')
        ax = axes[1,1]
        for version in extra:
            account = _account(accounts, selected, 'IS-Common', version)
            daily = account['daily']
            index = daily['coin_equity'].to_numpy()/float(account['metadata']['initial_coin'])*100.
            ax.plot(daily.index, index, lw=1.5, color=VERSION_COLORS[version], label=LABELS[version])
        ax.axhline(100., color='#A7B2BC', lw=.8, ls=':')
        ax.set_title('Coin balance | initial actual conversion = 100', loc='left')
        ax.set_ylabel('Coin balance index (pre-FX endpoint)')
        ax.set_ylim(bottom=0.)
        ax.legend(frameon=False, fontsize=8)
        _style_axis(ax, dates=True)
        _save(fig, out/FIGURES[0], 'BTC IS | Cross-venue equity and coin balances',
              'Main cost. T = trend switch; P = continuous budget. USD = USDT assumed. V4 includes an additional collateral coin exposure.')

        fig, axes = plt.subplots(1, 3, figsize=(18., 7.5))
        _cost_panel(axes[0], selected, 'IS-Full', EQUAL_VERSIONS[:4], 'total_return', 'Full | cash returns')
        _cost_panel(axes[1], selected, 'IS-Common', EQUAL_VERSIONS, 'total_return', 'Common | same target exposure')
        _cost_panel(axes[2], selected, 'IS-Common', extra, 'coin_growth', 'Common | extra-exposure coin growth')
        _save(fig, out/FIGURES[1], 'BTC IS | Registered cost steps',
              'Gross also disables funding. Every account includes its actual terminal liquidation/conversion; coin growth is measured before final FX.')

        fig, axes = plt.subplots(2, 2, figsize=(16., 10.))
        rows = [selected[('IS-Common', v, 'main_17')] for v in EQUAL_VERSIONS]
        x = np.arange(len(rows))
        labels = [LABELS[v] for v in EQUAL_VERSIONS]
        funds = np.array([float(r['funding_usd']) for r in rows])
        friction = np.array([float(r['fee_usd'])-float(r['slippage_usd'])-float(r['conversion_usd']) for r in rows])
        for ax, values, title, ylabel in (
                (axes[0,0], funds, 'Common main | net funding (signed)', 'Funding received / paid (USD)'),
                (axes[0,1], friction, 'Common main | trading and conversion friction', 'Total friction expense (USD)'),
                (axes[1,0], np.array([float(r['max_drawdown']) for r in rows])*100.,
                 'Common main | valid-minute maximum drawdown', 'Maximum drawdown (%)')):
            ax.bar(x, values, color=[VERSION_COLORS[v] for v in EQUAL_VERSIONS], alpha=.85)
            ax.set_xticks(x, labels, rotation=20, ha='right')
            ax.axhline(0., color='#8897A4', lw=.8)
            ax.set_title(title, loc='left')
            ax.set_ylabel(ylabel)
            _style_axis(ax)
        clips = ', '.join(f"{LABELS[v]}: {int(selected[('IS-Common',v,'main_17')]['cash_clip_count'])}"
                          for v in ('V2_SPOT', 'P_SPOT'))
        axes[0,1].text(.01, .98, 'Cash clips | '+clips, transform=axes[0,1].transAxes,
                       va='top', fontsize=8, bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
        values = np.empty((len(FAMILIES), len(GATES)), float)
        for i, family in enumerate(FAMILIES):
            for j, gate in enumerate(GATES):
                value = stats[family]['gates'].loc[gate, 'passed']
                values[i,j] = .5 if pd.isna(value) else float(bool(value))
        ax = axes[1,1]
        ax.imshow(values, vmin=0., vmax=1., cmap=ListedColormap(['#F2D8CC','#E6EBEE','#35846D']), aspect='auto')
        ax.set_xticks(np.arange(len(GATES)), GATES)
        ax.set_yticks(np.arange(len(FAMILIES)), [f.replace('IS-', '').replace('__', ' / ').replace('CM_DeltaMatched','CM hedge') for f in FAMILIES])
        ax.set_title('IS V1-V6 diagnostics | each venue/window separately', loc='left')
        for i in range(len(FAMILIES)):
            for j in range(len(GATES)):
                v = values[i,j]
                ax.text(j, i, 'NA' if v == .5 else 'Pass' if v == 1. else 'Fail', ha='center', va='center',
                        color='white' if v == 1. else '#674638', fontsize=9)
        _save(fig, out/FIGURES[2], 'BTC IS | Funding, execution costs and diagnostics',
              'CM funding uses the event Spot valuation. Gate families are descriptive within venue/window; there is no global cross-venue Holm claim.')

        fig, axes = plt.subplots(2, 2, figsize=(16., 10.))
        ax = axes[0,0]
        ax.plot(signal.index, signal['budget_UM'], color='#246BA0', lw=1.3, label='Canonical UM budget')
        ax.plot(signal.index, signal['budget_CM'], color='#A66A22', lw=1.3, alpha=.85, label='Native CM budget')
        ax.set_title('Self-signal window | held weekly budgets', loc='left')
        ax.set_ylabel('Ex ante target leverage budget')
        ax.set_ylim(bottom=0., top=1.05)
        ax.legend(frameon=False, fontsize=8)
        _style_axis(ax, dates=True)
        agreement = (signal['g_UM'] == signal['g_CM']).astype(float)
        monthly = agreement.resample('MS').mean()*100.
        ax = axes[0,1]
        ax.plot(monthly.index, monthly.to_numpy(), color='#287E62', marker='o', markersize=3., lw=1.2)
        concordance = float(agreement.mean()*100.)
        ax.set_title(f'Trend-switch agreement | all days: {concordance:.2f}%', loc='left')
        ax.set_ylabel('Monthly agreement (%)')
        ax.set_ylim(-2., 102.)
        _style_axis(ax, dates=True)
        labels = [LABELS[v] for v in rounding.index]
        eqdiff = rounding['whole_final_equity']-rounding['continuous_final_equity']
        coindiff = (rounding['whole_coin_growth']-rounding['continuous_coin_growth'])*100.
        for ax, values, title, ylabel in (
                (axes[1,0], eqdiff, 'Common main | whole-contract cash difference', 'Whole minus continuous final equity (USD)'),
                (axes[1,1], coindiff, 'Common main | whole-contract coin-growth difference', 'Whole minus continuous coin growth (percentage points)')):
            ax.bar(np.arange(len(labels)), values.to_numpy(), color=['#A66A22','#A66A22','#9B486A'])
            ax.set_xticks(np.arange(len(labels)), labels, rotation=15, ha='right')
            ax.axhline(0., color='#8897A4', lw=.8)
            ax.set_title(title, loc='left')
            ax.set_ylabel(ylabel)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.3g}'))
            _style_axis(ax)
        _save(fig, out/FIGURES[3], 'BTC IS | Native CM signal and contract-rounding diagnostics',
              'Signal concordance compares aligned decision dates. Rounding is toward zero at USD 100/contract; these diagnostics add no selection gate.')
    (out/'figure_captions.md').write_text(CAPTIONS)
    return list(FIGURES)
