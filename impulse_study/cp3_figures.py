"""Frozen §10 figures; descriptive intervals use 2,000 shared month draws.

Pre-unlock choices: for each figure/version, one ascending union of entry months
and one draw matrix serves every horizon/category/metric. Empty replicates are
discarded without replacement. Fewer than 12 active months gives an undefined
interval (§8.7). Onsets are transitions in each direction's eligible entry
signal (VersionInputs.entry_dir); they are simulated independently, without
portfolio position/cooldown. Histogram bins are common within each panel,
cover the entire observed range, and use numpy's automatic bin selection.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import engine as E, features as F, versions as VS
from .cp3_statistics import assert_statistics_scope

COLORS = {'gross': '#2471A3', 'net': '#B9770E', 'positive': '#2471A3', 'nonpositive': '#B9770E'}
NAMES = list(VS.VERSIONS)


def shared_month_intervals(times, matrix, seed):
    """Columns share exactly one frozen 2000 x G integer draw matrix."""
    values = np.asarray(matrix, float)
    if values.ndim == 1:
        values = values[:, None]
    idx = pd.DatetimeIndex(times)
    keys = idx.strftime('%Y-%m').to_numpy()
    months, inverse = np.unique(keys, return_inverse=True)
    g, m = len(months), values.shape[1]
    sums, counts = np.zeros((g, m)), np.zeros((g, m), dtype=np.int64)
    for j in range(m):
        good = np.isfinite(values[:, j])
        np.add.at(sums[:, j], inverse[good], values[good, j])
        np.add.at(counts[:, j], inverse[good], 1)
    means = np.divide(sums.sum(axis=0), counts.sum(axis=0),
                      out=np.full(m, np.nan), where=counts.sum(axis=0) > 0)
    lo, hi = np.full(m, np.nan), np.full(m, np.nan)
    discarded = np.full(m, 2000, dtype=int)
    active = (counts > 0).sum(axis=0)
    if g:
        draws = np.random.default_rng(seed).integers(0, g, size=(2000, g))
        for j in range(m):
            den = counts[:, j][draws].sum(axis=1)
            ok = den > 0
            discarded[j] = int((~ok).sum())
            if active[j] >= 12 and ok.any():
                boot = sums[:, j][draws].sum(axis=1)[ok] / den[ok]
                lo[j], hi[j] = np.percentile(boot, [2.5, 97.5])
    return pd.DataFrame({'mean': means, 'lo': lo, 'hi': hi, 'n': counts.sum(axis=0),
                         'active_months': active, 'discarded': discarded})


def lifecycle_rows(ctx, trades, pulse_direction):
    assert_statistics_scope(trades=trades)
    i = trades['trigger_idx'].to_numpy(dtype=np.int64)
    e = i + 1
    paths = np.full((len(i), 45), np.nan)
    for h in range(1, 46):
        target = i + h
        good = target < len(ctx.C)
        paths[good, h - 1] = pulse_direction * (ctx.C[target[good]] / ctx.O[e[good]] - 1) * 1e4
    return paths


def onset_events(ctx, vin):
    selected = np.zeros(len(ctx.C), dtype=bool)
    for side in (1, -1):
        signal = vin.entry_dir == side
        selected |= signal & ~np.r_[False, signal[:-1]]
    i = np.flatnonzero(selected)
    events = E.simulate_events(ctx, i, vin.entry_dir[i], vin.sl[i], vin.eref[i],
                               vin.spec.hold == 'fixed1', VS.COSTS[VS.MAIN_COST])
    events['trigger_idx'] = i
    events['trigger_time'] = ctx.index[i]
    events['entry_time'] = ctx.index[i + 1]
    # Independent simulator has no exit_time column; infer only for scope check.
    x = events['exit_bar'].to_numpy(dtype=int)
    opened = events['reason'].isin([E.GAP_ENTRY, E.GAP_STOP, E.HALT]).to_numpy()
    events['exit_time'] = ctx.index[0] + pd.to_timedelta(x + (~opened).astype(int), unit='min')
    events['category'] = vin.cat[i]
    assert_statistics_scope(trades=events)
    return events


def category_rows(trades, seed):
    assert_statistics_scope(trades=trades)
    columns, metadata = [], []
    for cat in (1, 2, 3):
        for metric in ('gross_bps', 'net_bps'):
            columns.append(np.where(trades['category'].to_numpy() == cat, trades[metric], np.nan))
            metadata.append((cat, metric))
    intervals = shared_month_intervals(trades['entry_time'], np.column_stack(columns), seed)
    intervals['category'] = [x[0] for x in metadata]
    intervals['metric'] = [x[1] for x in metadata]
    return intervals


def _finish(fig, path):
    for ax in fig.axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#D5D8DC', linewidth=0.6, alpha=0.65)
        ax.set_axisbelow(True)
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def make_figures(ctx, main_results, out):
    out = Path(out)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 12, 'figure.facecolor': 'white'})
    tables = {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained', sharex=True)
    samples = [('A1_box_opt', 1, 'Up impulse / box'), ('B1_pure_long', 1, 'Up impulse / pure'),
               ('A4_box_flush', -1, 'Down impulse / box (pulse direction)'),
               ('B3_pure_bi_mom', -1, 'Down impulse / pure shorts')]
    rows = []
    for ax, (name, d, title) in zip(axes.flat, samples):
        tr = main_results[name].trades
        if name == 'B3_pure_bi_mom':
            tr = tr[tr.side == -1]
        r = shared_month_intervals(tr.entry_time, lifecycle_rows(ctx, tr, d),
                                    VS.SEED_BASE + 1100 + NAMES.index(name))
        r['minute'] = np.arange(1, 46)
        r['version'] = name
        rows.append(r)
        ax.plot(r.minute, r['mean'], color=COLORS['gross'], label='Gross pulse-direction return')
        ax.fill_between(r.minute, r.lo, r.hi, color=COLORS['gross'], alpha=.2, label='95% month-cluster CI')
        for cost in (-17, 17):
            ax.axhline(cost, color='#566573', ls='--', lw=1)
        ax.axhline(0, color='#777777', lw=.6)
        ax.set(title=f'{title} | n={len(tr):,}', xlabel='Minutes after entry', ylabel='Gross return (bp)', xticks=[1, 15, 30, 45])
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Figure 1 | Impulse lifecycle; dashed lines = ±17 bp\nBTCUSDT IS, main-window actual fills; returns continue after strategy exit')
    _finish(fig, out / 'figure1_lifecycle.png')
    tables['figure1'] = pd.concat(rows, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    rows = []
    lo5, _ = F.origin_extremes(ctx.C, ctx.L, ctx.H, 5)
    lo15, _ = F.origin_extremes(ctx.C, ctx.L, ctx.H, 15)
    for ax, name in zip(axes, ['A1_box_opt', 'B1_pure_long']):
        tr = main_results[name].trades
        assert_statistics_scope(trades=tr)
        i = tr.trigger_idx.to_numpy(dtype=int)
        atr, ref = tr.atr15.to_numpy(), tr.e_ref.to_numpy()
        mae = np.maximum(0, tr.side.to_numpy() * (ref - tr.entry_open.to_numpy()) - tr.mae_px.to_numpy()) / atr
        origin = np.where(ctx.cond['pct']['up2'][i], lo15[i], lo5[i])
        distances = {'SL_A': np.full(len(tr), 2.5), 'SL_B': (ref - origin + .1 * atr) / atr,
                     'SL_C': .5 * (ref - origin) / atr}
        wins = tr.net_pnl.to_numpy() > 0
        bins = np.histogram_bin_edges(mae, bins='auto') if len(mae) else np.array([0, 1])
        for yes, label in [(True, 'Net P&L > 0'), (False, 'Net P&L ≤ 0')]:
            subset = wins == yes
            if subset.any():
                ax.hist(mae[subset], bins=bins, histtype='step', linewidth=1.8,
                        color=COLORS['positive' if yes else 'nonpositive'], label=f'{label} (n={subset.sum():,})')
            for stop, dist in distances.items():
                rows.append({'version': name, 'profitable': yes, 'stop': stop, 'n': int(subset.sum()),
                             'share_mae_reaching_stop': float(np.mean(mae[subset] >= dist[subset])) if subset.any() else np.nan})
        for stop, ls in [('SL_A', '-'), ('SL_B', '--'), ('SL_C', ':')]:
            if len(tr):
                q1, med, q3 = np.quantile(distances[stop], [.25, .5, .75])
                ax.axvline(med, color='#566573', ls=ls, label=f'{stop} median {med:.2f}')
                if stop != 'SL_A':
                    ax.axvspan(q1, q3, color='#566573', alpha=.08)
        ax.set(title=name, xlabel='Actual-holding MAE / ATR15 (E_ref basis)', ylabel='Trades')
        ax.legend(fontsize=8)
    fig.suptitle('Figure 2 | Stop distances and adverse excursion\nSL_B / SL_C bands show per-trade hypothetical stop-distance IQR')
    _finish(fig, out / 'figure2_stops.png')
    tables['figure2'] = pd.DataFrame(rows)

    annual = []
    for name in NAMES:
        result = main_results[name]
        tr = result.trades
        assert_statistics_scope(trades=tr)
        daily = result.daily_equity
        prior = pd.concat([pd.Series([VS.INITIAL_CAPITAL], index=[daily.index[0] - pd.Timedelta(days=1)]), daily])
        ret = prior.pct_change().iloc[1:]
        for year in (2020, 2021, 2022, 2023):
            t = tr[pd.DatetimeIndex(tr.entry_time).year == year]
            r = ret[ret.index.year == year]
            annual.append({'version': name, 'year': year, 'n': len(t),
                           'gross_bps': t.gross_bps.mean(), 'net_bps': t.net_bps.mean(),
                           'gross_R': (t.gross_pnl / t.risk_amount).mean(), 'net_R': t.net_R.mean(),
                           'sharpe': float(r.mean() / r.std(ddof=1) * np.sqrt(365)) if len(r) > 1 and r.std(ddof=1) > 0 else np.nan})
    annual = pd.DataFrame(annual)
    order = ['A1_box_opt','B1_pure_long','A2_box_origin','B2_pure_long_sl','A3_box_half',
             'A4_box_flush','A5_box_origin_z','B3_pure_bi_mom','B4_pure_1bar','B5_pure_bi_mom_z']
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), layout='constrained')
    year_colors = ['#2471A3','#B9770E','#7D3C98','#566573']
    for ax, metric, title in zip(axes.flat, ['gross_bps','net_bps','gross_R','net_R','n','sharpe'],
                                ['Gross mean (bp)','Net mean (bp)','Gross mean (R)','Net mean (R)','Trades','Annualized daily Sharpe']):
        for j, year in enumerate((2020,2021,2022,2023)):
            vals = annual[annual.year == year].set_index('version').loc[order, metric]
            ax.plot(np.arange(10), vals, marker=['o','s','^','D'][j], color=year_colors[j], label=str(year), lw=1)
        ax.axhline(0, color='#777777', lw=.7)
        ax.set(title=title, xticks=np.arange(10), xticklabels=[n.split('_')[0] for n in order])
    axes[0,0].legend(ncols=4, fontsize=8)
    fig.suptitle('Figure 3 | Annual main-configuration comparison\n2020 = March–December; A5/B5 use full-IS volatility calibration')
    _finish(fig, out / 'figure3_annual.png')
    tables['figure3'] = annual

    fig, axes = plt.subplots(5, 2, figsize=(13, 18), layout='constrained')
    rows = []
    for v, (ax, name) in enumerate(zip(axes.flat, NAMES)):
        actual = main_results[name].trades
        vin = F.build_version_inputs(ctx, VS.VERSIONS[name])
        onset = onset_events(ctx, vin)
        onset.to_csv(out / f'onset_{name}.csv.gz', index=False, compression='gzip')
        for mode, frame, seed, shift, marker in [('actual', actual, VS.SEED_BASE + 1400 + v, -.16, 'o'),
                                                  ('onset', onset, VS.SEED_BASE + 1500 + v, .16, 's')]:
            r = category_rows(frame, seed)
            r['version'], r['sample'] = name, mode
            rows.append(r)
            for metric, color, offset in [('gross_bps', COLORS['gross'], -.05), ('net_bps', COLORS['net'], .05)]:
                sub = r[r.metric == metric]
                x = sub.category.to_numpy() + shift + offset
                y, lo, hi = sub['mean'].to_numpy(), sub.lo.to_numpy(), sub.hi.to_numpy()
                ax.scatter(x, y, color=color, marker=marker, label=f'{mode} {metric.split("_")[0]}', s=24)
                ok = np.isfinite(lo) & np.isfinite(hi)
                ax.vlines(x[ok], lo[ok], hi[ok], color=color, linewidth=1)
        ax.axhline(0, color='#777777', lw=.7)
        ax.set(title=f'{name} | fills={len(actual):,}, onsets={len(onset):,}',
               xticks=[1,2,3], xticklabels=['Only ①','Only ②','① + ②'], ylabel='Mean (bp)')
        if v == 0:
            ax.legend(fontsize=8, ncols=2)
    fig.suptitle('Figure 4 | Trigger-category decomposition and independent onset simulations\n95% month-cluster intervals; A5/B5 = full-IS volatility-calibrated robustness versions')
    _finish(fig, out / 'figure4_categories.png')
    tables['figure4'] = pd.concat(rows, ignore_index=True)
    for key, table in tables.items():
        table.to_csv(out / f'{key}_data.csv', index=False)
    return {key: len(value) for key, value in tables.items()}
