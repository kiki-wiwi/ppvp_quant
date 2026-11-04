#!/usr/bin/env python3
"""
tools/run_phase2.py — Phase 2：BTC 基线回测（IS / VAL）与 IS 段亏损归因。

用法（仓库根目录）：python tools/run_phase2.py
输出：reports/phase2_baseline.json、reports/phase2_is_trades.csv、reports/figures/phase2_*.png

隔离约定
* 只经 pipeline.run 读取研究数据（默认截断在 2025-07-01 之前），OOS K 线不进入引擎；
  ETH 是跨品种零改参验证集，本阶段不对它计算任何绩效（pipeline.run 对 HOLDOUT_SYMBOLS 的分段报告加锁）。
* VAL 只报告总体指标（同口径分段指标、夏普置信区间、买入持有基准、成本对照）；交易级归因、分桶与假设生成只用 IS，
  使 VAL 在 Phase 3 的熔断判定中保持“未参与假设生成”的身份。
* 分桶边界是事先选定的整数 / 常用阈值（BUCKETS / REGIME_BUCKETS），未按结果调整。n≈100 的分桶比较是探索性的，
  置信区间很宽，只作为 Phase 3 的待检验假设，不作为采纳依据。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import attribution as A  # noqa: E402
import pipeline  # noqa: E402
from config import (BASELINE_BACKTEST, BASELINE_STRATEGY, IS_END, IS_START, SEGMENTS, VAL_END,  # noqa: E402
                    VAL_START)
from metrics import daily_equity, drawdown_series, monthly_returns  # noqa: E402

REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
SYMBOL = "BTCUSDT"
N_NULL = 5000
N_BOOT = 5000
SEED = 20261104

HEADLINE_KEYS = ("start", "end", "days", "sharpe", "sortino", "calmar", "cagr", "total_return", "max_dd", "max_dd_hourly",
                 "exposure", "ann_vol", "trades", "win_rate", "profit_factor", "avg_r", "avg_hold_bars", "avg_win_pct",
                 "avg_loss_pct", "fees", "funding", "exit_reasons", "trades_open_at_end", "capped_entries")
VARIANTS = {
    "baseline": {},
    "no_funding": {"include_funding": False},
    "no_slippage": {"slippage": 0.0},
    "no_fees": {"perp_taker_fee": 0.0, "perp_maker_fee": 0.0},
    "frictionless": {"include_funding": False, "slippage": 0.0, "perp_taker_fee": 0.0, "perp_maker_fee": 0.0},
    "spot_10bp": {"venue": "spot"},
}
BUCKETS = {                                   # 左闭右开，事先选定
    "rr_fill": [0.0, 0.5, 1.0, 2.0, np.inf],
    "width_atr": [0.0, 2.5, 4.0, np.inf],
    "sl_atr": [0.0, 1.5, 2.5, np.inf],
    "d_sl": [0.0, 0.01, 0.02, np.inf],
    "entry_vs_val_atr": [-np.inf, 0.0, 0.5, np.inf],
    "touch_gap_bars": [0, 12, 48, np.inf],
    "pattern_lag_bars": [0, 1, 3, np.inf],
    "box_age_h": [0.0, 12.0, 48.0, np.inf],
}
CATEGORICAL = ("pattern_name", "sfp", "pivot_pair", "reentry_after_loss_same_zone")
REGIME_BUCKETS = {
    "above_sma200": [-0.5, 0.5, 1.5],
    "sma200_up": [-0.5, 0.5, 1.5],
    "mom20d": [-np.inf, 0.0, np.inf],
    "vol30d_pct": [0.0, 1 / 3, 2 / 3, 1.0001],
    "dd365": [-1.0, -0.4, -0.2, 1e-12],
    "hours_since_crash": [0.0, 48.0, 168.0, np.inf],
    "ret24h": [-np.inf, -0.02, 0.0, np.inf],
}

# 图表令牌（dataviz 参考调色板，浅色）
SURF, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, pd.DataFrame):
        return [_jsonable(r) for r in x.to_dict(orient="records")]
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    if isinstance(x, pd.Interval):
        return str(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if not np.isfinite(v) else round(v, 6)
    if x is None or isinstance(x, (str, int)):
        return x
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    return str(x)


def _git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # pragma: no cover
        return "unknown"


def headline(report_seg: dict) -> dict:
    return {k: report_seg[k] for k in HEADLINE_KEYS if k in report_seg}


def signal_budget(sig: pd.DataFrame, trades: pd.DataFrame) -> dict:
    close_t = sig.index + pd.Timedelta(hours=1)
    out = {}
    for name in ("IS", "VAL"):
        s, e = SEGMENTS[name]
        m = (close_t >= s) & (close_t < e)
        t = trades[(trades["entry_time"] >= s) & (trades["entry_time"] < e)]
        out[name] = {"raw_signals": int(sig["sig_raw"][m].sum()), "tradable_signals": int(sig["signal"][m].sum()),
                     "blocked_rr": int(sig["block_rr"][m].sum()), "entries": int(len(t)),
                     "skipped_while_in_position": int(sig["signal"][m].sum() - len(t))}
    return out


def expectancy(tf: pd.DataFrame) -> dict:
    w, lo = tf[tf["win"]], tf[~tf["win"]]
    aw, al = float(w["r_multiple"].mean()), float(lo["r_multiple"].mean())
    exp_w = float(tf["p_rw"].sum())
    var_w = float((tf["p_rw"] * (1 - tf["p_rw"])).sum())
    return {
        "n": len(tf), "wins": int(len(w)), "win_rate": len(w) / len(tf), "avg_win_r": aw, "avg_loss_r": al,
        "mean_r": float(tf["r_multiple"].mean()), "sum_r": float(tf["r_multiple"].sum()),
        "breakeven_win_rate": -al / (aw - al),
        "median_rr_fill_all": float(tf["rr_fill"].median()), "median_rr_fill_winners": float(w["rr_fill"].median()),
        "median_rr_fill_losers": float(lo["rr_fill"].median()),
        "mean_cost_r": float(tf["cost_r"].mean()), "mean_pre_cost_r": float((tf["pre_cost_pnl"] / tf["risk_usd"]).mean()),
        "rw_expected_wins_driftless": exp_w, "rw_wins_sd": float(np.sqrt(var_w)),
        "rw_z": float((len(w) - exp_w) / np.sqrt(var_w)),
        "median_d_sl": float(tf["d_sl"].median()), "median_d_tp": float(tf["d_tp"].median()),
    }


def exit_table(tf: pd.DataFrame) -> pd.DataFrame:
    g = tf.groupby("exit_reason")
    return pd.DataFrame({
        "n": g.size(), "mean_r": g["r_multiple"].mean(), "sum_pnl": g["pnl"].sum(),
        "median_hold_bars": g["hold_bars"].median(), "mean_rr_fill": g["rr_fill"].mean(), "mean_p_rw": g["p_rw"].mean(),
    }).reset_index()


def cost_decomposition(tf: pd.DataFrame) -> dict:
    return {"pre_cost_pnl": float(tf["pre_cost_pnl"].sum()), "slippage": float(tf["slip_usd"].sum()),
            "fees": float(tf["fees"].sum()), "funding": float(tf["funding"].sum()), "net_pnl": float(tf["pnl"].sum()),
            "mean_cost_r": float(tf["cost_r"].mean()),
            "mean_slip_r": float((tf["slip_usd"] / tf["risk_usd"]).mean()),
            "mean_fee_r": float((tf["fees"] / tf["risk_usd"]).mean()),
            "mean_funding_r": float((tf["funding"] / tf["risk_usd"]).mean())}


def forward_drift(df1h: pd.DataFrame, entry_bars: np.ndarray, end_bar: int, first_bar: int) -> list[dict]:
    lc = np.log(df1h["close"].to_numpy(dtype=np.float64))
    rows = []
    for h in (1, 4, 12, 24, 48):
        allb = np.arange(first_bar, end_bar - h)
        fr_all = lc[allb + h] - lc[allb]
        eb = entry_bars[entry_bars + h < end_bar]
        fr_e = lc[eb + h] - lc[eb]
        se = fr_e.std(ddof=1) / np.sqrt(len(fr_e))
        rows.append({"h": h, "n": int(len(fr_e)), "mean_entries": float(fr_e.mean()), "mean_all_bars": float(fr_all.mean()),
                     "diff": float(fr_e.mean() - fr_all.mean()), "t": float((fr_e.mean() - fr_all.mean()) / se),
                     "frac_negative_entries": float((fr_e < 0).mean()), "frac_negative_all": float((fr_all < 0).mean())})
    return rows


def quantiles(s: pd.Series) -> dict:
    s = s.dropna()
    return {f"q{int(q * 100)}": float(s.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)} | {"n": int(len(s))}


def exit_diagnostics(tf: pd.DataFrame) -> dict:
    sl, tp = tf[tf["exit_reason"] == "SL"], tf[tf["exit_reason"] == "TP"]
    sl_poc = sl[sl["poc_above_entry"]]
    return {
        "sl_n": int(len(sl)), "sl_with_poc_above_entry": int(len(sl_poc)),
        "sl_reached_poc_strict": int(sl_poc["reached_poc_strict"].sum()),
        "sl_reached_half_r_strict": int(sl["reached_half_r_strict"].sum()),
        "sl_reached_1r_strict": int(sl["reached_1r_strict"].sum()),
        "sl_no_full_bar_before_exit": int(sl["mfe_r_strict"].isna().sum()),
        "sl_ambiguous_exit_bar": int(sl["ambiguous_exit_bar"].sum()),
        "sl_r_below_minus_1_2": int((sl["r_multiple"] < -1.2).sum()),
        "sl_mfe_r_strict": quantiles(sl["mfe_r_strict"]), "sl_hold_bars": quantiles(sl["hold_bars"]),
        "sl_r_multiple": quantiles(sl["r_multiple"]),
        "tp_n": int(len(tp)), "tp_mae_r_strict": quantiles(tp["mae_r_strict"]),
        "tp_mae_gt_half_r": int((tp["mae_r_strict"] > 0.5).sum()), "tp_mae_gt_0_8r": int((tp["mae_r_strict"] > 0.8).sum()),
        "tp_hold_bars": quantiles(tp["hold_bars"]), "tp_r_multiple": quantiles(tp["r_multiple"]),
        "tp_with_poc_above_entry": int(tp["poc_above_entry"].sum()),
        "all_hold_le_4": int((tf["hold_bars"] <= 4).sum()), "all_hold_gt_24": int((tf["hold_bars"] > 24).sum()),
        "all_hold_gt_48": int((tf["hold_bars"] > 48).sum()),
        "pnl_hold_gt_24": float(tf.loc[tf["hold_bars"] > 24, "pnl"].sum()),
        "mean_r_hold_gt_24": float(tf.loc[tf["hold_bars"] > 24, "r_multiple"].mean()),
    }


def bucketize(tf: pd.DataFrame, col: str, edges) -> pd.DataFrame:
    cut = pd.cut(tf[col], bins=edges, right=False)
    t = A.bucket_stats(tf, cut, seed=SEED)
    t["share_of_total_pnl"] = t["sum_pnl"] / tf["pnl"].sum()
    t.insert(0, "feature", col)
    return t


def categorical_table(tf: pd.DataFrame, col: str) -> pd.DataFrame:
    t = A.bucket_stats(tf, col, seed=SEED)
    t["share_of_total_pnl"] = t["sum_pnl"] / tf["pnl"].sum()
    t.insert(0, "feature", col)
    return t


def streaks(win: pd.Series) -> dict:
    runs, cur = [], 0
    for w in win.to_numpy():
        if not w:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    runs = np.array(runs) if runs else np.zeros(0, int)
    return {"max_consecutive_losses": int(runs.max()) if len(runs) else 0, "loss_runs_ge_3": int((runs >= 3).sum()),
            "loss_runs_ge_5": int((runs >= 5).sum())}


# ---------------------------------------------------------------------------
# 图表
# ---------------------------------------------------------------------------
def _style():
    import logging
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # WenQuanYi 只有一种字重，回退提示无害
    plt.rcParams.update({
        "font.family": ["Inter", "WenQuanYi Zen Hei", "DejaVu Sans"], "font.size": 9.5,
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": BASE, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 10.5, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.fontsize": 9, "savefig.dpi": 170, "axes.unicode_minus": False,
    })


def fig_equity(d_base: pd.Series, d_free: pd.Series, worst_months: list, share: float, n_tr: int, path: Path) -> None:
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.2, 5.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    for a in (ax, ax2):
        for y, m in worst_months:                    # IS 最差 5 个月（浅色底纹）
            m0 = pd.Timestamp(year=int(y), month=int(m), day=1, tz="UTC")
            a.axvspan(m0, m0 + pd.offsets.MonthBegin(1), color=GRID, alpha=0.9, linewidth=0, zorder=0)
        a.axvline(VAL_START, color=MUTED, linewidth=1.0)
    ax.plot(d_base.index, d_base.to_numpy() / 1e3, color=C1, label="基线（含手续费、滑点、资金费）")
    ax.plot(d_free.index, d_free.to_numpy() / 1e3, color=C2, label="同一组交易、零成本对照")
    ax.set_ylabel("日收盘净值（千 USDT）")
    ax.set_title("BTC 基线净值与水下回撤（研究区间：IS 2020–2023 | VAL 2024-01 至 2025-06）")
    ax.legend(loc="upper right")
    ax.text(0.01, 0.03, "IS", transform=ax.transAxes, color=INK2, fontsize=9)
    ax.text(VAL_START + pd.Timedelta(days=15), ax.get_ylim()[0] + 0.03 * np.diff(ax.get_ylim())[0], "VAL",
            color=INK2, fontsize=9)
    ax.text(0.30, 0.78, f"灰色底纹：IS 最差的 5 个月（{n_tr} 笔交易，占 IS 对数亏损的 {share:.0%}）",
            transform=ax.transAxes, color=INK2, fontsize=8.5)
    dd = drawdown_series(d_base) * 100
    ax2.fill_between(dd.index, dd.to_numpy(), 0, color=C1, alpha=0.10, linewidth=0)
    ax2.plot(dd.index, dd.to_numpy(), color=C1, linewidth=1.6)
    ax2.set_ylabel("基线回撤（%）")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_null(nulls: dict, actual: float, path: Path) -> None:
    keys = [("global", "全局随机入场（IS 任意时刻）"), ("local_15d", "局部随机入场（真实入场 ±15 天）")]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    for ax, (k, title) in zip(axes, keys):
        dist = nulls[k]["null_r_dist"]
        ax.hist(dist, bins=50, color=C1, alpha=0.85, rwidth=0.88)
        ax.axvline(actual, color=C2, linewidth=2.0)
        p = nulls[k]["p_mean_r_le"]
        ax.text(0.98, 0.95, f"橙线：真实入场 {actual:+.2f}R\n零分布均值 {np.mean(dist):+.2f}R\n单侧 p = {p:.3f}",
                transform=ax.transAxes, ha="right", va="top", color=INK, fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("109 笔交易的平均 R（不含资金费）")
    axes[0].set_ylabel("零分布频数（5,000 次重抽样）")
    fig.suptitle("同一止损 / 止盈几何下，真实入场时刻 vs 随机入场时刻（IS）", x=0.01, ha="left", fontsize=10.5,
                 fontweight="semibold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_rr(tab: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    x = np.arange(len(tab))
    labels = [str(b).replace("inf", "∞") for b in tab["bucket"]]
    h1 = ax.errorbar(x - 0.08, tab["win_rate"] * 100, yerr=[(tab["win_rate"] - tab["win_lo"]) * 100,
                                                          (tab["win_hi"] - tab["win_rate"]) * 100],
                     fmt="o", color=C1, markersize=7, elinewidth=1.6, capsize=0, label="实际胜率（Wilson 95% 区间）")
    h2, = ax.plot(x + 0.08, tab["rw_win_rate"] * 100, "o", markersize=7, markerfacecolor=SURF, markeredgecolor=C2,
                  markeredgewidth=2, label="同一止损 / 止盈距离下，无漂移随机游走的理论胜率")
    for xi, n in zip(x, tab["n"]):
        ax.text(xi, -6, f"n={n}", ha="center", va="top", color=MUTED, fontsize=8.5)
    ax.set_xticks(x, labels)
    ax.set_ylim(-14, 105)
    ax.set_xlabel("入场时的初始盈亏比 (VAH − 入场价) / (入场价 − SL)")
    ax.set_ylabel("胜率（%）")
    ax.set_title("IS：按初始盈亏比分组的胜率")
    ax.legend(handles=[h1, h2], loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_exits(tf: pd.DataFrame, path: Path) -> None:
    sl = tf[(tf["exit_reason"] == "SL") & tf["poc_above_entry"] & tf["mfe_r_strict"].notna()]
    tp = tf[tf["exit_reason"] == "TP"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.7))
    hit = sl["reached_poc_strict"]
    ax.scatter(sl.loc[~hit, "poc_r"], sl.loc[~hit, "mfe_r_strict"], s=36, color=C1, edgecolor=SURF, linewidth=1.5,
               label=f"未先到达 POC（{int((~hit).sum())}）")
    ax.scatter(sl.loc[hit, "poc_r"], sl.loc[hit, "mfe_r_strict"], s=36, color=C2, edgecolor=SURF, linewidth=1.5,
               label=f"止损前已到达 POC（{int(hit.sum())}）")
    lim = max(sl["poc_r"].max(), sl["mfe_r_strict"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color=MUTED, linewidth=1.0)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("POC 距入场价（R）；对角线以上 = 止损前已到达 POC")
    ax.set_ylabel("止损前的最大有利波动（R，严格口径）")
    ax.set_title("止损交易：离场前是否先到过 POC")
    ax.legend(loc="upper left")
    mae = tp["mae_r_strict"].dropna()
    ax2.hist(mae, bins=np.arange(0, 1.05, 0.05), color=C1, rwidth=0.88)
    ax2.set_xlabel("止盈前的最大不利波动（R，严格口径；1R = 触及止损）")
    ax2.set_ylabel("交易数")
    ax2.set_title(f"止盈交易：到达 VAH 前离止损多近（{len(mae)}/{len(tp)} 笔有完整 K 线）")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


REGIME_LABELS = {
    "above_sma200": ("日线收盘 vs SMA200", {-0.5: "下方", 0.5: "上方"}),
    "sma200_up": ("SMA200 的 20 日斜率", {-0.5: "向下", 0.5: "向上"}),
    "mom20d": ("20 日动量", {-np.inf: "< 0", 0.0: "≥ 0"}),
    "vol30d_pct": ("30 日波动率（过去一年分位）", {0.0: "低三分位", 1 / 3: "中三分位", 2 / 3: "高三分位"}),
    "dd365": ("距 365 日高点", {-1.0: "< −40%", -0.4: "−40% 至 −20%", -0.2: "≥ −20%"}),
    "hours_since_crash": ("距上次 4H 放量大阴线", {0.0: "< 48 小时", 48.0: "48 小时至 7 天", 168.0: "≥ 7 天"}),
    "ret24h": ("入场前 24 小时涨跌", {-np.inf: "< −2%", -0.02: "−2% 至 0", 0.0: "≥ 0"}),
}


def _regime_label(feature: str, interval) -> str:
    name, m = REGIME_LABELS[feature]
    for k, v in m.items():
        if np.isclose(interval.left, k, atol=1e-3) or (np.isinf(k) and np.isinf(interval.left)):   # pd.cut 的区间端点按 3 位小数取整
            return f"{name} · {v}"
    return f"{name} · {interval}"


def fig_regime(tab: pd.DataFrame, overall: float, path: Path) -> None:
    rows = [(_regime_label(r["feature"], r["bucket"]), r) for _, r in tab.iterrows()]
    fig, ax = plt.subplots(figsize=(8.8, 0.34 * len(rows) + 1.5))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, r) in zip(y, rows):
        ax.plot([r["mean_r_lo"], r["mean_r_hi"]], [yi, yi], color=C1, linewidth=1.6, zorder=2)
        ax.plot(r["mean_r"], yi, "o", color=C1, markersize=6.5, markeredgecolor=SURF, markeredgewidth=1.2, zorder=3)
        ax.plot(r["null_mean_r"], yi, "o", markersize=6.5, markerfacecolor=SURF, markeredgecolor=C2,
                markeredgewidth=1.8, zorder=4)
        ax.text(1.01, yi, f"n={int(r['n'])}", transform=ax.get_yaxis_transform(), va="center", color=MUTED, fontsize=8.5)
    ax.set_yticks(y, [lab for lab, _ in rows])
    ax.axvline(0, color=BASE, linewidth=1.0, zorder=1)
    ax.axvline(overall, color=MUTED, linewidth=1.2, zorder=1)
    ax.text(overall, y.max() + 0.75, f"IS 全体 {overall:+.2f}R", color=INK2, fontsize=8.5, ha="center")
    h1 = plt.Line2D([], [], color=C1, marker="o", linewidth=1.6, label="实际平均 R（自助法 95% 区间）")
    h2 = plt.Line2D([], [], color=C2, marker="o", markerfacecolor=SURF, linewidth=0,
                    label="同一批交易的局部随机入场基准（±15 天）")
    ax.legend(handles=[h1, h2], loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=2)
    ax.set_xlabel("平均 R")
    ax.set_title("IS：按入场时刻市场状态分组的平均 R", pad=34)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    FIG.mkdir(parents=True, exist_ok=True)
    scfg, bcfg = BASELINE_STRATEGY, BASELINE_BACKTEST
    out = pipeline.run(SYMBOL, scfg, bcfg)
    res, df1h, sig = out["result"], out["df1h"], out["signals"]
    R: dict = {"meta": {"symbol": SYMBOL, "git_head": _git_head(), "strategy_key": scfg.key(), "backtest_key": bcfg.key(),
                        "data_first_bar": df1h.index[0], "data_last_bar": df1h.index[-1], "n_null": N_NULL,
                        "n_boot": N_BOOT, "seed": SEED,
                        "policy": "IS-only trade-level attribution; VAL headline only; no OOS; no ETH performance"}}

    # ---- 1. 总体指标（IS / VAL）、夏普置信区间、买入持有基准 ----
    R["headline"] = {}
    for name in ("IS", "VAL"):
        s, e = SEGMENTS[name]
        r = A.segment_daily_returns(res, s, e)
        R["headline"][name] = {"strategy": headline(out["report"][name]),
                               "sharpe_ci": A.stationary_bootstrap_sharpe(r, n_boot=N_BOOT, seed=SEED),
                               "buy_and_hold": A.buy_and_hold(df1h, s, e)}
    R["signal_budget"] = signal_budget(sig, res.trades)

    # ---- 2. 成本对照（同一组交易）----
    R["cost_variants"] = {}
    ref_bars = res.trades[["entry_bar", "exit_bar"]].to_numpy()
    d_free = None
    for vname, kw in VARIANTS.items():
        o = out if vname == "baseline" else pipeline.run(SYMBOL, scfg, bcfg.with_(**kw))
        assert np.array_equal(o["result"].trades[["entry_bar", "exit_bar"]].to_numpy(), ref_bars), vname
        R["cost_variants"][vname] = {seg: {k: o["report"][seg][k] for k in ("sharpe", "cagr", "total_return", "max_dd",
                                                                             "profit_factor", "win_rate", "avg_r", "trades")}
                                     for seg in ("IS", "VAL")}
        if vname == "frictionless":
            d_free = daily_equity(o["result"])

    # ---- 3. IS 交易级归因 ----
    tf = A.trade_frame(res, df1h, out["boxes"], scfg)
    ist = A.segment_trades(tf, IS_START, IS_END).reset_index(drop=True)
    ist = pd.concat([ist, A.regime_features(df1h, out["df4h"], pipeline.spot_daily_warmup(SYMBOL),
                                            ist["entry_bar"].to_numpy())], axis=1)
    R["is_expectancy"] = expectancy(ist)
    rt_cost = 2 * bcfg.slippage + bcfg.taker_fee + bcfg.maker_fee             # 止盈离场的往返成本（≈ 11 bp）
    tp_tr = ist[ist["exit_reason"] == "TP"]
    R["is_expectancy"].update({"tp_exits_net_loss": int((tp_tr["pnl"] <= 0).sum()),
                               "trades_d_tp_lt_3x_rt_cost": int((ist["d_tp"] < 3 * rt_cost).sum()),
                               "round_trip_cost_tp": rt_cost})
    R["is_exit_reasons"] = exit_table(ist)
    R["is_costs"] = cost_decomposition(ist)

    nulls = {"global": A.random_entry_null(df1h, ist, bcfg, IS_START, IS_END, n_rep=N_NULL, seed=SEED)}
    for wd in (3, 15, 60):
        nulls[f"local_{wd}d"] = A.random_entry_null(df1h, ist, bcfg, IS_START, IS_END, n_rep=N_NULL, seed=SEED + wd,
                                                     window_days=wd)
    ist["null_r"] = nulls["local_15d"]["per_trade_null_mean_r"]       # 分桶的超额 R 以 ±15 天局部零假设为基准
    ist["null_r_global"] = nulls["global"]["per_trade_null_mean_r"]
    act = float(ist["r_ex_funding"].mean())
    R["is_null"] = {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, np.ndarray)} for k, v in nulls.items()}
    R["is_null_decomposition"] = {
        "actual_mean_r_ex_funding": act,
        "geometry_costs_fill_r": nulls["global"]["null_mean_r_mean"],
        "macro_timing_r": nulls["local_15d"]["null_mean_r_mean"] - nulls["global"]["null_mean_r_mean"],
        "micro_timing_r": act - nulls["local_15d"]["null_mean_r_mean"],
    }
    first_is = int(np.searchsorted(df1h.index, IS_START))
    end_is = int(np.searchsorted(df1h.index, IS_END))
    R["is_forward_drift"] = forward_drift(df1h, ist["entry_bar"].to_numpy(), end_is, first_is)
    R["is_exits"] = exit_diagnostics(ist)
    o_arr = df1h["open"].to_numpy(dtype=np.float64)
    sl_tr = ist[ist["exit_reason"] == "SL"]
    R["is_exits"]["sl_gap_fills"] = int((o_arr[sl_tr["exit_bar"].to_numpy()] < sl_tr["sl"].to_numpy()).sum())
    big_loss = sl_tr["r_multiple"] < -1.2
    R["is_exits"]["sl_r_below_minus_1_2_d_sl_range"] = [float(sl_tr.loc[big_loss, "d_sl"].min()),
                                                          float(sl_tr.loc[big_loss, "d_sl"].max())]
    R["is_exits"]["sl_r_below_minus_1_2_cost_r_range"] = [float(sl_tr.loc[big_loss, "cost_r"].min()),
                                                           float(sl_tr.loc[big_loss, "cost_r"].max())]

    ent = [bucketize(ist, c, e) for c, e in BUCKETS.items()] + [categorical_table(ist, c) for c in CATEGORICAL]
    R["is_entry_buckets"] = pd.concat(ent, ignore_index=True)
    reg = pd.concat([bucketize(ist, c, e) for c, e in REGIME_BUCKETS.items()], ignore_index=True)
    R["is_regime_buckets"] = reg
    allb = pd.concat([R["is_entry_buckets"], reg], ignore_index=True)
    big = allb[allb["n"] >= 5]
    R["is_bucket_summary"] = {
        "n_features": int(allb["feature"].nunique()), "n_groups": int(len(allb)), "n_groups_n_ge_5": int(len(big)),
        "positive_mean_r_groups_n_ge_5": big.loc[big["mean_r"] > 0, ["feature", "bucket", "n", "mean_r"]],
        "overall_excess_r": float((ist["r_ex_funding"] - ist["null_r"]).mean()),
        "spearman_width_atr_vs_rr_fill": float(ist["width_atr"].corr(ist["rr_fill"], method="spearman")),
    }
    R["is_regime_coverage"] = {c: int(ist[c].notna().sum()) for c in REGIME_BUCKETS}
    # 现有 regime_filter 定义在 IS 交易上的覆盖面预览（只看被标记交易的结果，不是回测实验：
    # 过滤器释放的仓位可能让此前因持仓而跳过的信号成交，真实效果以 Phase 3 实验为准）
    bear = (ist["above_sma200"] == 0) & (ist["sma200_up"] == 0)
    crash48 = ist["hours_since_crash"] < 48
    R["is_filter_preview"] = {
        name: {"flagged": int(m.sum()), "flagged_mean_r": float(ist.loc[m, "r_multiple"].mean()),
               "flagged_pnl": float(ist.loc[m, "pnl"].sum()), "kept_mean_r": float(ist.loc[~m, "r_multiple"].mean()),
               "flagged_excess_r": float((ist.loc[m, "r_ex_funding"] - ist.loc[m, "null_r"]).mean()),
               "kept_excess_r": float((ist.loc[~m, "r_ex_funding"] - ist.loc[~m, "null_r"]).mean())}
        for name, m in (("daily_sma_bear", bear), ("crash_cooldown_48h", crash48), ("mom20d_negative", ist["mom20d"] < 0))}

    # ---- 4. 回撤来源与时间分布（IS）----
    d_eq = daily_equity(res)
    d_is = d_eq[(d_eq.index >= IS_START - pd.Timedelta(days=1)) & (d_eq.index <= IS_END - pd.Timedelta(days=1))]
    eps = A.drawdown_episodes(d_is, top=5)
    for ep in eps:
        end_t = ep["trough"] + pd.Timedelta(days=1)
        m = (ist["exit_time"] > ep["peak"] + pd.Timedelta(days=1)) & (ist["exit_time"] <= end_t)
        ep.update({"trades_closed": int(m.sum()), "wins": int(ist.loc[m, "win"].sum()),
                   "pnl": float(ist.loc[m, "pnl"].sum()), "sum_r": float(ist.loc[m, "r_multiple"].sum())})
    R["is_drawdown_episodes"] = eps
    yr = ist.groupby(ist["entry_time"].dt.year)
    R["is_by_year"] = pd.DataFrame({"n": yr.size(), "win_rate": yr["win"].mean(), "mean_r": yr["r_multiple"].mean(),
                                    "sum_pnl": yr["pnl"].sum()}).reset_index()
    mr = monthly_returns(d_is).stack().rename("ret").reset_index()
    R["is_worst_months"] = mr.nsmallest(5, "ret")
    R["is_months_with_trades"] = int(ist["entry_time"].dt.strftime("%Y-%m").nunique())
    w5 = R["is_worst_months"]["ret"].to_numpy()
    total_is = d_is.iloc[-1] / d_is.iloc[0]
    R["is_worst5_months_share_of_log_loss"] = float(np.log1p(w5).sum() / np.log(total_is))
    in_w5 = ist["entry_time"].dt.strftime("%Y-%m").isin(
        [f"{int(y)}-{int(m):02d}" for y, m in R["is_worst_months"][["year", "month"]].to_numpy()])
    R["is_trades_in_worst5_months"] = {"n": int(in_w5.sum()), "mean_r": float(ist.loc[in_w5, "r_multiple"].mean()),
                                       "sum_pnl": float(ist.loc[in_w5, "pnl"].sum()),
                                       "sum_r": float(ist.loc[in_w5, "r_multiple"].sum()),
                                       "exit_reasons": ist.loc[in_w5, "exit_reason"].value_counts().to_dict(),
                                       "median_d_sl": float(ist.loc[in_w5, "d_sl"].median()),
                                       "median_d_sl_other": float(ist.loc[~in_w5, "d_sl"].median()),
                                       "median_atr_pct": float((ist.loc[in_w5, "atr"] / ist.loc[in_w5, "entry_px"]).median()),
                                       "median_atr_pct_other": float((ist.loc[~in_w5, "atr"] / ist.loc[~in_w5, "entry_px"]).median()),
                                       "mean_r_other": float(ist.loc[~in_w5, "r_multiple"].mean())}
    R["is_streaks"] = streaks(ist["win"])

    # ---- 5. 仓位 / 止损距离 ----
    lo = ist[~ist["win"]]
    big = ist["d_sl"] >= 0.02
    R["is_sizing"] = {
        "spearman_dsl_vs_loss_pct": float(lo["d_sl"].corr(-lo["ret_on_base"], method="spearman")),
        "d_sl": quantiles(ist["d_sl"]),
        "trades_d_sl_ge_2pct": int(big.sum()), "loss_share_d_sl_ge_2pct": float(
            ist.loc[big & ~ist["win"], "pnl"].sum() / lo["pnl"].sum()),
        "pnl_d_sl_ge_2pct": float(ist.loc[big, "pnl"].sum()), "mean_r_d_sl_ge_2pct": float(ist.loc[big, "r_multiple"].mean()),
        "mean_r_d_sl_lt_2pct": float(ist.loc[~big, "r_multiple"].mean()),
        "worst10": ist.nsmallest(10, "pnl")[["entry_time", "exit_reason", "d_sl", "ret_on_base", "r_multiple",
                                              "above_sma200", "ret24h", "hours_since_crash"]],
    }

    # ---- 6. 输出 ----
    keep = ["entry_time", "exit_time", "exit_reason", "pattern_name", "entry_px", "exit_px", "sl", "tp", "poc", "VAL", "VAH",
            "atr", "d_sl", "d_tp", "rr_fill", "p_rw", "pnl", "ret_on_base", "r_multiple", "r_ex_funding", "cost_r",
            "hold_bars", "mfe_r_strict", "mae_r_strict", "reached_poc_strict", "ambiguous_exit_bar", "width_atr", "sl_atr",
            "entry_vs_val_atr", "touch_gap_bars", "pattern_lag_bars", "box_age_h", "sfp", "pivot_pair",
            "reentry_after_loss_same_zone", "above_sma200", "sma200_up", "mom20d", "vol30d_pct", "dd365",
            "hours_since_crash", "ret24h", "ret168h"]
    ist[keep].to_csv(REPORTS / "phase2_is_trades.csv", index=False, float_format="%.10g")
    R["meta"]["runtime_s"] = time.time() - t0
    (REPORTS / "phase2_baseline.json").write_text(json.dumps(_jsonable(R), ensure_ascii=False, indent=1))

    _style()
    d_res = d_eq[d_eq.index <= VAL_END - pd.Timedelta(days=1)]
    fig_equity(d_res, d_free[d_free.index <= VAL_END - pd.Timedelta(days=1)],
               R["is_worst_months"][["year", "month"]].to_numpy().tolist(), R["is_worst5_months_share_of_log_loss"],
               R["is_trades_in_worst5_months"]["n"], FIG / "phase2_equity_underwater.png")
    fig_null(nulls, act, FIG / "phase2_is_null_test.png")
    fig_rr(R["is_entry_buckets"][R["is_entry_buckets"]["feature"] == "rr_fill"].reset_index(drop=True),
           FIG / "phase2_is_winrate_by_rr.png")
    fig_exits(ist, FIG / "phase2_is_exit_paths.png")
    fig_regime(reg, float(ist["r_multiple"].mean()), FIG / "phase2_is_regime.png")
    print(f"phase 2 done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
