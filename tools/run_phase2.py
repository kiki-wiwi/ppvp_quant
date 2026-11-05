#!/usr/bin/env python3
"""
tools/run_phase2.py — Phase 2：BTC 基线回测（IS / VAL）与 IS 段亏损归因。

用法（仓库根目录）：python tools/run_phase2.py
输出：reports/phase2_baseline.json、reports/phase2_tables.md（阶段记录中的表格由它逐字生成）、
reports/phase2_is_trades.csv、reports/figures/phase2_*.png

隔离约定
* 只经 pipeline.run 读取研究数据（默认截断在 2025-07-01 之前），OOS K 线不进入任何计算；
  ETH 是跨品种零改参验证集，本脚本不对它做任何运行（pipeline.run 对 HOLDOUT_SYMBOLS 不运行回测）。
* VAL 只报告总体指标：同口径分段指标、夏普置信区间、买入持有基准、成本对照，以及 Phase 1 记录事先承诺的
  carry / roll 四种组合。逐笔归因、分桶、随机入场检验、配对夏普差与全部假设只来自 IS；图表只画 IS。
* 分桶边界是事先选定的整数 / 常用阈值（BUCKETS / REGIME_BUCKETS），未按结果调整。n≈100 的分桶比较是探索性的，
  只作为 Phase 3 的待检验假设，不作为采纳依据。
"""
from __future__ import annotations

import hashlib
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
N_THIN = 300
SEED = 20261104
DAY = pd.Timedelta(days=1)

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
# Phase 1 记录承诺：carry 与 roll 是规格之外的状态机扩展，Phase 2 同时报告二者开、关时的基线（只报告总体指标）
STRUCT_VARIANTS = {
    "carry_on_roll_on (baseline)": {},
    "carry_off": {"carry_touches_on_box_update": False},
    "roll_off": {"roll_failed_second_touch": False},
    "carry_off_roll_off": {"carry_touches_on_box_update": False, "roll_failed_second_touch": False},
}
# 随机入场零假设：(键, 时间窗天数, 取样侧, 剔除信号 K 线前后多少根)；分桶的超额 R 以 local_15d_x24 为基准
# (键, 时间窗天数, 取样侧, 剔除信号 K 线前后多少根, 候选池, 几何)：
# 候选池 'enterable' = 若此根出现看涨形态即会在收盘发出信号的 K 线：处理后状态为 TOUCH2 且触及支撑带、收盘 ≥ 带下沿，
# 或本身就是原始信号 K 线（信号消费序列后状态回到 ARMED）。test_attribution 用“在该根强加 Pinbar”的反事实逐根核对。
NULL_SPECS = (
    ("global", None, "both", None, None, "pct"),
    ("local_60d", 60, "both", None, None, "pct"),
    ("local_15d", 15, "both", None, None, "pct"),
    ("local_15d_x24", 15, "both", 24, None, "pct"),
    ("local_3d", 3, "both", None, None, "pct"),
    ("before_3d", 3, "before", None, None, "pct"),
    ("after_3d", 3, "after", None, None, "pct"),
    ("after_3d_x24", 3, "after", 24, None, "pct"),
    ("enterable_global", None, "both", None, "enterable", "pct"),
    ("enterable_15d", 15, "both", None, "enterable", "pct"),
    ("atr_global", None, "both", None, None, "atr"),
    ("atr_15d_x24", 15, "both", 24, None, "atr"),
)
EXCESS_BASE = "local_15d_x24"
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
PROVENANCE_FILES = ("attribution.py", "tools/run_phase2.py", "config.py", "pipeline.py", "backtester.py",
                    "strategy_state_machine.py", "ppvp_engine.py", "metrics.py", "data_loader.py")

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


def _provenance() -> dict:
    """生成本报告的代码版本：git describe（含 -dirty 标记）与关键源文件的 SHA-256。"""
    try:
        desc = subprocess.run(["git", "-C", str(ROOT), "describe", "--always", "--dirty"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # pragma: no cover
        desc = "unknown"
    hashes = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()[:16] for f in PROVENANCE_FILES}
    return {"git_describe": desc, "sha256_16": hashes}


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


def expectancy(tf: pd.DataFrame, bcfg) -> dict:
    """期望值结构与随机游走对照。p_rw = d_sl/(d_sl+d_tp) 是无漂移连续随机游走“先触及止盈”的概率，不含成本，
    因此与之同口径的是止盈离场笔数；含成本的口径把“止盈也净亏”的交易的 p_rw 置 0，再与净盈利笔数比较。"""
    w, lo = tf[tf["win"]], tf[~tf["win"]]
    aw, al = float(w["r_multiple"].mean()), float(lo["r_multiple"].mean())
    p = tf["p_rw"].to_numpy(dtype=np.float64)
    n_tp = int((tf["exit_reason"] == "TP").sum())
    tp_net_ok = (tf["tp"] * (1 - bcfg.slippage) * (1 - bcfg.maker_fee) > tf["entry_px"] * (1 + bcfg.taker_fee)).to_numpy()
    p_net = np.where(tp_net_ok, p, 0.0)
    n_net = int((tf["r_ex_funding"] > 0).sum())

    def z(obs, q):
        return float((obs - q.sum()) / np.sqrt((q * (1 - q)).sum()))

    return {
        "n": len(tf), "wins": int(len(w)), "win_rate": len(w) / len(tf), "avg_win_r": aw, "avg_loss_r": al,
        "mean_r": float(tf["r_multiple"].mean()), "sum_r": float(tf["r_multiple"].sum()),
        "breakeven_win_rate": -al / (aw - al),
        "median_rr_fill_all": float(tf["rr_fill"].median()), "median_rr_fill_winners": float(w["rr_fill"].median()),
        "median_rr_fill_losers": float(lo["rr_fill"].median()),
        "mean_cost_r": float(tf["cost_r"].mean()), "mean_pre_cost_r": float((tf["pre_cost_pnl"] / tf["risk_usd"]).mean()),
        "median_d_sl": float(tf["d_sl"].median()), "median_d_tp": float(tf["d_tp"].median()),
        # 同口径 1：止盈离场笔数 vs Σp_rw（先触及止盈的期望笔数）
        "tp_exits": n_tp, "rw_expected_tp_first": float(p.sum()), "rw_tp_sd": float(np.sqrt((p * (1 - p)).sum())),
        "rw_z_tp": z(n_tp, p), "rw_p_tp_le_exact": A.poisson_binomial_cdf(p, n_tp),
        # 同口径 2：不含资金费的净盈利笔数 vs 含成本的随机游走期望（止盈也净亏的交易 p 置 0）
        "net_wins_ex_funding": n_net, "rw_expected_net_wins": float(p_net.sum()), "rw_z_net": z(n_net, p_net),
        "rw_p_net_le_exact": A.poisson_binomial_cdf(p_net, n_net),
        "trades_tp_cannot_net_win": int((~tp_net_ok).sum()),
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


def exit_diagnostics(tf: pd.DataFrame, o_arr: np.ndarray) -> dict:
    sl, tp = tf[tf["exit_reason"] == "SL"], tf[tf["exit_reason"] == "TP"]
    sl_poc = sl[sl["poc_above_entry"]]
    big = sl["r_multiple"] < -1.2
    return {
        "sl_n": int(len(sl)), "sl_with_poc_above_entry": int(len(sl_poc)),
        "sl_with_poc_above_entry_and_full_bar": int(sl_poc["mfe_r_strict"].notna().sum()),
        "sl_reached_poc_strict": int(sl_poc["reached_poc_strict"].sum()),
        "sl_reached_half_r_strict": int(sl["reached_half_r_strict"].sum()),
        "sl_reached_1r_strict": int(sl["reached_1r_strict"].sum()),
        "sl_no_full_bar_before_exit": int(sl["mfe_r_strict"].isna().sum()),
        "sl_negative_mfe_strict": int((sl["mfe_r_strict"] < 0).sum()),
        "sl_ambiguous_exit_bar": int(sl["ambiguous_exit_bar"].sum()),
        "sl_gap_fills": int((o_arr[sl["exit_bar"].to_numpy()] < sl["sl"].to_numpy()).sum()),
        "sl_r_below_minus_1_2": int(big.sum()),
        "sl_r_below_minus_1_2_d_sl_range": [float(sl.loc[big, "d_sl"].min()), float(sl.loc[big, "d_sl"].max())],
        "sl_r_below_minus_1_2_cost_r_range": [float(sl.loc[big, "cost_r"].min()), float(sl.loc[big, "cost_r"].max())],
        "sl_mfe_r_strict": quantiles(sl["mfe_r_strict"]), "sl_hold_bars": quantiles(sl["hold_bars"]),
        "sl_r_multiple": quantiles(sl["r_multiple"]),
        "tp_n": int(len(tp)), "tp_first_bar": int((tp["hold_bars"] == 1).sum()),
        "tp_mae_r_strict": quantiles(tp["mae_r_strict"]),
        "tp_mae_gt_half_r": int((tp["mae_r_strict"] > 0.5).sum()), "tp_mae_gt_0_8r": int((tp["mae_r_strict"] > 0.8).sum()),
        "tp_hold_bars": quantiles(tp["hold_bars"]), "tp_r_multiple": quantiles(tp["r_multiple"]),
        "tp_with_poc_above_entry": int(tp["poc_above_entry"].sum()),
        "all_hold_le_4": int((tf["hold_bars"] <= 4).sum()),
    }


def hold_profile(tf: pd.DataFrame, total_pnl: float, total_log: float) -> list[dict]:
    """持仓时长 > h 根的交易（时间止损的作用对象）：笔数、平均 R、平均净值收益、占 IS 净亏损与对数亏损的比例。"""
    rows = []
    for h in (12, 24, 48):
        m = tf["hold_bars"] > h
        rows.append({"hold_gt": h, "n": int(m.sum()), "mean_r": float(tf.loc[m, "r_multiple"].mean()),
                     "kept_mean_r": float(tf.loc[~m, "r_multiple"].mean()), "kept_mean_ret": float(tf.loc[~m, "ret_on_base"].mean()),
                     "median_d_sl": float(tf.loc[m, "d_sl"].median()), "median_d_sl_rest": float(tf.loc[~m, "d_sl"].median()),
                     "mean_ret": float(tf.loc[m, "ret_on_base"].mean()), "pnl": float(tf.loc[m, "pnl"].sum()),
                     "share_of_net_loss": float(tf.loc[m, "pnl"].sum() / total_pnl),
                     "share_of_log_loss": float(np.log1p(tf.loc[m, "ret_on_base"]).sum() / total_log)})
    return rows


def _with_shares(t: pd.DataFrame, tf: pd.DataFrame, feature: str, total_log: float) -> pd.DataFrame:
    t["share_of_total_pnl"] = t["sum_pnl"] / tf["pnl"].sum()
    t["share_of_log_loss"] = t["sum_log_ret"] / total_log
    t.insert(0, "feature", feature)
    return t


def bucketize(tf: pd.DataFrame, col: str, edges, total_log: float) -> pd.DataFrame:
    cut = pd.cut(tf[col], bins=edges, right=False)
    return _with_shares(A.bucket_stats(tf, cut, n_boot=N_BOOT, seed=SEED), tf, col, total_log)


def categorical_table(tf: pd.DataFrame, col: str, total_log: float) -> pd.DataFrame:
    return _with_shares(A.bucket_stats(tf, col, n_boot=N_BOOT, seed=SEED), tf, col, total_log)


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


def _row(tab: pd.DataFrame, feature: str, left: float) -> pd.Series:
    sub = tab[tab["feature"] == feature]
    for _, r in sub.iterrows():
        b = r["bucket"]
        if isinstance(b, pd.Interval) and (np.isclose(b.left, left, atol=1e-3) or (np.isinf(left) and np.isinf(b.left))):
            return r
        if not isinstance(b, pd.Interval) and b == left:
            return r
    raise KeyError((feature, left))


def gap(tab: pd.DataFrame, a: tuple, b: tuple) -> dict:
    """两组的超额 R 与不含资金费平均 R 之差（a - b）及其标准误倍数（两组自助法标准误按独立合成）。"""
    ra, rb = _row(tab, *a), _row(tab, *b)
    d_ex = ra["excess_r"] - rb["excess_r"]
    d_xf = ra["mean_r_xf"] - rb["mean_r_xf"]
    se_ex = float(np.hypot(ra["excess_r_se"], rb["excess_r_se"]))
    se_xf = float(np.hypot(ra["mean_r_xf_se"], rb["mean_r_xf_se"]))
    null_share = (ra["null_mean_r"] - rb["null_mean_r"]) / d_xf if d_xf != 0 else float("nan")
    return {"a": f"{a[0]}={a[1]}", "b": f"{b[0]}={b[1]}", "n_a": int(ra["n"]), "n_b": int(rb["n"]),
            "excess_diff": float(d_ex), "excess_diff_se_ratio": float(d_ex / se_ex),
            "mean_r_xf_diff": float(d_xf), "mean_r_xf_diff_se_ratio": float(d_xf / se_xf),
            "null_share_of_mean_gap": float(null_share)}


# ---------------------------------------------------------------------------
# 表格（阶段记录逐字使用）
# ---------------------------------------------------------------------------
def _f(x, nd=3, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if round(x, nd) == 0:                       # 取整后为零时不带符号（避免“−0%”）
        x = 0.0
        return f"{0:.{nd}f}"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "−")


def _pct(x, nd=2, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return _f(100 * x, nd, sign) + "%"


def _ci(lo, hi, nd=2):
    return f"[{_f(lo, nd, True)}, {_f(hi, nd, True)}]"


def tables_md(R: dict, ent: pd.DataFrame, reg: pd.DataFrame) -> str:
    h = R["headline"]
    L = ["<!-- 由 tools/run_phase2.py 生成，请勿手改 -->", "", "### T1 基线绩效", "",
         "| 指标 | IS 策略 | VAL 策略 | IS 买入持有 | VAL 买入持有 |", "|---|---|---|---|---|"]
    si, sv = h["IS"]["strategy"], h["VAL"]["strategy"]
    bi, bv = h["IS"]["buy_and_hold"], h["VAL"]["buy_and_hold"]
    ci, cv = h["IS"]["sharpe_ci"], h["VAL"]["sharpe_ci"]
    rows = [
        ("日度夏普", _f(si["sharpe"]), _f(sv["sharpe"]), _f(bi["sharpe"]), _f(bv["sharpe"])),
        ("夏普 95% 区间", _ci(ci["lo"], ci["hi"], 3), _ci(cv["lo"], cv["hi"], 3), "", ""),
        ("自助样本中夏普 > 0 的比例", _pct(ci["p_gt0"], 1), _pct(cv["p_gt0"], 1), "", ""),
        ("Sortino", _f(si["sortino"]), _f(sv["sortino"]), _f(bi["sortino"]), _f(bv["sortino"])),
        ("Calmar", _f(si["calmar"]), _f(sv["calmar"]), _f(bi["calmar"]), _f(bv["calmar"])),
        ("CAGR", _pct(si["cagr"]), _pct(sv["cagr"]), _pct(bi["cagr"]), _pct(bv["cagr"])),
        ("累计收益", _pct(si["total_return"], 2, True), _pct(sv["total_return"], 2, True),
         _pct(bi["total_return"], 2, True), _pct(bv["total_return"], 2, True)),
        ("最大回撤（日收盘，分段内）", _pct(si["max_dd"]), _pct(sv["max_dd"]), _pct(bi["max_dd"]), _pct(bv["max_dd"])),
        ("最大回撤（逐小时盯市，分段内）", _pct(si["max_dd_hourly"]), _pct(sv["max_dd_hourly"]), "", ""),
        ("年化波动", _pct(si["ann_vol"]), _pct(sv["ann_vol"]), _pct(bi["ann_vol"]), _pct(bv["ann_vol"])),
        ("持仓时间占比", _pct(si["exposure"]), _pct(sv["exposure"]), "100%", "100%"),
        ("交易笔数（止损 / 止盈）", f"{si['trades']}（{si['exit_reasons'].get('SL', 0)} / {si['exit_reasons'].get('TP', 0)}）",
         f"{sv['trades']}（{sv['exit_reasons'].get('SL', 0)} / {sv['exit_reasons'].get('TP', 0)}）", "", ""),
        ("胜率（净盈亏 > 0）", _pct(si["win_rate"]), _pct(sv["win_rate"]), "", ""),
        ("盈亏比（Profit Factor）", _f(si["profit_factor"]), _f(sv["profit_factor"]), "", ""),
        ("平均 R", _f(si["avg_r"]), _f(sv["avg_r"]), "", ""),
        ("平均持仓（1H 根数）", _f(si["avg_hold_bars"], 1), _f(sv["avg_hold_bars"], 1), "", ""),
        ("手续费 / 资金费（USDT）", f"{si['fees']:,.0f} / {si['funding']:,.0f}", f"{sv['fees']:,.0f} / {sv['funding']:,.0f}",
         "", ""),
    ]
    L += [f"| {' | '.join(r)} |" for r in rows]

    L += ["", "### T2 成本对照（同一组交易）", "", "| 对照组 | IS 夏普 | IS 累计 | VAL 夏普 | VAL 累计 |", "|---|---|---|---|---|"]
    names = {"baseline": "基线（taker 5bp、maker 2bp、滑点 2bp、资金费）", "no_funding": "不计资金费",
             "no_slippage": "不计滑点", "no_fees": "不计手续费", "frictionless": "零成本",
             "spot_10bp": "现货等效（taker / maker 各 10bp、滑点 2bp、无资金费、名义 ≤ 净值 / (1 + taker)）"}
    for k, v in R["cost_variants"].items():
        L.append(f"| {names[k]} | {_f(v['IS']['sharpe'])} | {_pct(v['IS']['total_return'], 2, True)} | "
                 f"{_f(v['VAL']['sharpe'])} | {_pct(v['VAL']['total_return'], 2, True)} |")

    L += ["", "### T3 carry / roll 四种组合（总体指标；IS 配对夏普差与零技能稀释基准）", "",
          "| 状态机扩展 | IS 原始信号 / 入场 | IS 夏普 | IS 配对夏普差 [95%] | IS 稀释基准（经验 / 公式；稀释后平均笔数） | IS 扣除稀释后的增量 | 稀释下不差于该变体的概率 | IS 平均 R | VAL 原始信号 / 入场 | VAL 夏普 | VAL 稀释公式基准 | VAL 平均 R |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    snames = {"carry_on_roll_on (baseline)": "carry 开、roll 开（基线）", "carry_off": "carry 关", "roll_off": "roll 关",
              "carry_off_roll_off": "二者都关"}
    for k, v in R["struct_variants"].items():
        pdiff = R["struct_paired_is"].get(k)
        th = R["struct_thinning"].get(k)
        pd_s = "—" if pdiff is None else f"{_f(pdiff['diff'], 3, True)} {_ci(pdiff['lo'], pdiff['hi'])}"
        th_s = "—" if th is None else f"{_f(th['thinned_mean_sharpe'])} / {_f(th['formula_is'])}；{_f(th['thinned_mean_trades'], 1)}"
        mg_s = "—" if th is None else _f(th["margin_is_vs_empirical"], 3, True)
        pp_s = "—" if th is None else _f(th["p_thinned_ge_variant"], 3)
        vf_s = "—" if th is None else _f(th["formula_val"])
        L.append(f"| {snames[k]} | {v['IS']['raw_signals']} / {v['IS']['entries']} | {_f(v['IS']['sharpe'])} | {pd_s} | "
                 f"{th_s} | {mg_s} | {pp_s} | {_f(v['IS']['avg_r'])} | {v['VAL']['raw_signals']} / {v['VAL']['entries']} | "
                 f"{_f(v['VAL']['sharpe'])} | {vf_s} | {_f(v['VAL']['avg_r'])} |")

    L += ["", "### T4 随机入场零假设（IS，109 笔，各 5,000 次）", "",
          "| 零假设的取样范围 | 零假设平均 R | 零假设 90% 区间 | 单侧 p（平均 R） | 零假设胜笔数（真实 40） | 单侧 p（胜笔数） |",
          "|---|---|---|---|---|---|"]
    nn = {"global": "IS 任意时刻", "local_60d": "信号前后 60 天", "local_15d": "信号前后 15 天",
          "local_15d_x24": "信号前后 15 天，剔除前后 24 根", "local_3d": "信号前后 3 天", "before_3d": "信号前 3 天",
          "after_3d": "信号后 3 天", "after_3d_x24": "信号后第 25 根至 3 天",
          "enterable_global": "可执行 K 线池，IS 任意时刻", "enterable_15d": "可执行 K 线池，信号前后 15 天",
          "atr_global": "ATR 几何，IS 任意时刻", "atr_15d_x24": "ATR 几何，信号前后 15 天，剔除前后 24 根"}
    for k, v in R["is_null"].items():
        L.append(f"| {nn[k]} | {_f(v['null_mean_r_mean'], 3, True)} | {_ci(v['null_mean_r_p05'], v['null_mean_r_p95'], 3)} | "
                 f"{_f(v['p_mean_r_le'], 3)} | {_f(v['null_wins_mean'], 1)} | {_f(v['p_wins_le'], 3)} |")
    cs = R["is_null_common_shift"]
    L.append(f"| 全部入场整体平移 ±{cs['min_shift_bars']}–{cs['max_shift_bars']} 根（保留聚集结构，{cs['n_shifts']} 次） | "
             f"{_f(cs['null_mean_r_mean'], 3, True)} | "
             f"{_ci(cs['null_mean_r_p05'], cs['null_mean_r_p95'], 3)} | {_f(cs['p_mean_r_le'], 3)} | — | — |")

    def bucket_lines(tab, spec):
        out = []
        for feat, label, items in spec:
            first = True
            for left, name in items:
                r = _row(tab, feat, left)
                out.append(f"| {label if first else ''} | {name} | {int(r['n'])} | {_pct(r['tp_rate'], 1)} | "
                           f"{_pct(r['rw_win_rate'], 1)} | {_f(r['mean_r_xf'], 2, True)} {_ci(r['mean_r_xf_lo'], r['mean_r_xf_hi'])} | "
                           f"{_f(r['null_mean_r'], 2, True)} | {_f(r['excess_r'], 2, True)} {_ci(r['excess_r_lo'], r['excess_r_hi'])} | "
                           f"{_pct(r['mean_ret'], 2, True)} | {_pct(r['share_of_log_loss'], 0)} |")
                first = False
        return out

    head = ["| 特征 | 分组 | 笔数 | 止盈命中率 | 随机游走理论 | 平均 R（不含资金费）[95%] | 零假设平均 R | 超额 R [95%] | 每笔净值盈亏 | 占 IS 对数亏损 |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    ent_spec = [
        ("rr_fill", "初始盈亏比", [(0.0, "< 0.5"), (0.5, "0.5–1"), (1.0, "1–2"), (2.0, "≥ 2")]),
        ("width_atr", "箱体宽度 (VAH−VAL)/ATR", [(0.0, "< 2.5"), (2.5, "2.5–4"), (4.0, "≥ 4")]),
        ("d_sl", "止损距离 d_sl", [(0.0, "< 1%"), (0.01, "1%–2%"), (0.02, "≥ 2%")]),
        ("pattern_name", "形态", [("pinbar", "Pinbar"), ("engulfing", "吞没")]),
        ("pivot_pair", "箱体枢轴方向", [("HL", "高→低（下跌段）"), ("LH", "低→高（上涨段）")]),
        ("sfp", "第 2 次测试扫破第 1 次低点", [(False, "否"), (True, "是")]),
        ("box_age_h", "箱体可用后多久入场", [(0.0, "< 12 小时"), (12.0, "12–48 小时")]),
    ]
    L += ["", "### T5 结构与形态分组（IS 摘录）", ""] + head + bucket_lines(ent, ent_spec)
    reg_spec = [
        ("above_sma200", "日线收盘 vs SMA200", [(-0.5, "下方"), (0.5, "上方")]),
        ("mom20d", "20 日动量", [(-np.inf, "< 0"), (0.0, "≥ 0")]),
        ("dd365", "距 365 日高点", [(-1.0, "< −40%"), (-0.4, "−40% 至 −20%"), (-0.2, "≥ −20%")]),
        ("hours_since_crash", "距上次 4H 放量大阴线", [(0.0, "< 48 小时"), (48.0, "48 小时至 7 天"), (168.0, "≥ 7 天")]),
        ("ret24h", "入场前 24 小时涨跌", [(-np.inf, "< −2%"), (-0.02, "−2% 至 0"), (0.0, "≥ 0")]),
        ("vol30d_pct", "30 日波动率分位", [(0.0, "低三分位"), (1 / 3, "中三分位"), (2 / 3, "高三分位")]),
        ("sma200_up", "SMA200 的 20 日斜率", [(-0.5, "向下"), (0.5, "向上")]),
    ]
    L += ["", "### T6 市场状态分组（IS）", ""] + head + bucket_lines(reg, reg_spec)

    L += ["", "### T7 过滤器与时间止损的覆盖面预览（IS，只看被标记交易本身，不是回测实验）", "",
          "| 预览 | 标记笔数 | 标记组平均 R | 保留组平均 R | 标记组每笔净值盈亏 | 保留组每笔净值盈亏 | 标记组占 IS 净亏损 | 标记组占 IS 对数亏损 |",
          "|---|---|---|---|---|---|---|---|"]
    fn = {"daily_sma_bear": "daily_sma（收盘低于 SMA200 且 SMA200 的 20 日斜率为负）",
          "crash_cooldown_48h": "crash_cooldown（4H 放量大阴线后 48 小时）", "mom20d_negative": "20 日动量为负（未实现）"}
    for k, v in R["is_filter_preview"].items():
        L.append(f"| {fn[k]} | {v['flagged']} | {_f(v['flagged_mean_r'], 2, True)} | {_f(v['kept_mean_r'], 2, True)} | "
                 f"{_pct(v['flagged_mean_ret'], 2, True)} | {_pct(v['kept_mean_ret'], 2, True)} | "
                 f"{_pct(v['flagged_share_of_net_loss'], 0)} | {_pct(v['flagged_share_of_log_loss'], 0)} |")
    for r in R["is_hold_profile"]:
        L.append(f"| 持仓超过 {r['hold_gt']} 小时（时间止损的作用对象） | {r['n']} | {_f(r['mean_r'], 2, True)} | "
                 f"{_f(r['kept_mean_r'], 2, True)} | "
                 f"{_pct(r['mean_ret'], 2, True)} | {_pct(r['kept_mean_ret'], 2, True)} | {_pct(r['share_of_net_loss'], 0)} | "
                 f"{_pct(r['share_of_log_loss'], 0)} |")

    L += ["", "### T8 年度分布（IS）", "", "| 年份 | 笔数 | 胜率 | 平均 R | 盈亏（USDT） |", "|---|---|---|---|---|"]
    for r in R["is_by_year"]:
        L.append(f"| {r['entry_time']} | {r['n']} | {_pct(r['win_rate'], 1)} | {_f(r['mean_r'], 2, True)} | "
                 f"{r['sum_pnl']:,.0f} |".replace("-", "−"))

    L += ["", "### T9 按亏损金额排序的最差 10 笔（IS，全部止损离场）", "",
          "| 入场时刻（UTC） | 止损距离 | 占净值亏损 | R | 日线 vs SMA200 | 入场前 24h |", "|---|---|---|---|---|---|"]
    for r in R["is_sizing"]["worst10"]:
        t = pd.Timestamp(r["entry_time"]).strftime("%Y-%m-%d %H:%M")
        side = {1.0: "上方", 0.0: "下方"}.get(r["above_sma200"], "—")
        L.append(f"| {t} | {_pct(r['d_sl'])} | {_pct(r['ret_on_base'], 2)} | {_f(r['r_multiple'], 2, True)} | {side} | "
                 f"{_pct(r['ret24h'], 1, True)} |")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# 图表（全部只画 IS）
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
    ax.plot(d_base.index, d_base.to_numpy() / 1e3, color=C1, label="基线（含手续费、滑点、资金费）")
    ax.plot(d_free.index, d_free.to_numpy() / 1e3, color=C2, label="同一组交易、零成本对照")
    ax.set_ylabel("日收盘净值（千 USDT）")
    ax.set_title("BTC 基线净值与水下回撤（IS：2020-01 至 2023-12；VAL 只报告总体指标，不画路径）")
    ax.legend(loc="upper right")
    ax.text(0.30, 0.78, f"灰色底纹：IS 最差的 5 个月（{n_tr} 笔交易，占 IS 对数亏损的 {share:.0%}）",
            transform=ax.transAxes, color=INK2, fontsize=8.5)
    dd = drawdown_series(d_base) * 100
    ax2.fill_between(dd.index, dd.to_numpy(), 0, color=C1, alpha=0.10, linewidth=0)
    ax2.plot(dd.index, dd.to_numpy(), color=C1, linewidth=1.6)
    ax2.set_ylabel("IS 内回撤（%）")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_null(nulls: dict, actual: float, path: Path) -> None:
    keys = [("global", "全局随机入场（IS 任意时刻）"), (EXCESS_BASE, "局部随机入场（±15 天，剔除前后 24 根）")]
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
    h1 = ax.errorbar(x - 0.08, tab["tp_rate"] * 100, yerr=[(tab["tp_rate"] - tab["tp_lo"]) * 100,
                                                         (tab["tp_hi"] - tab["tp_rate"]) * 100],
                     fmt="o", color=C1, markersize=7, elinewidth=1.6, capsize=0, label="实际止盈命中率（Wilson 95% 区间）")
    h2, = ax.plot(x + 0.08, tab["rw_win_rate"] * 100, "o", markersize=7, markerfacecolor=SURF, markeredgecolor=C2,
                  markeredgewidth=2, label="同一止损 / 止盈距离下，无漂移随机游走先触及止盈的概率")
    for xi, n in zip(x, tab["n"]):
        ax.text(xi, -6, f"n={n}", ha="center", va="top", color=MUTED, fontsize=8.5)
    ax.set_xticks(x, labels)
    ax.set_ylim(-14, 105)
    ax.set_xlabel("入场时的初始盈亏比 (VAH − 入场价) / (入场价 − SL)")
    ax.set_ylabel("先触及止盈的比例（%）")
    ax.set_title("IS：按初始盈亏比分组的止盈命中率（与随机游走同口径，不含成本）")
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
    low = min(0.0, float(sl["mfe_r_strict"].min())) - 0.08
    ax.plot([0, lim], [0, lim], color=MUTED, linewidth=1.0)
    ax.set_xlim(0, lim)
    ax.set_ylim(low, lim)
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
        if np.isclose(interval.left, k, atol=1e-3) or (np.isinf(k) and np.isinf(interval.left)):   # pd.cut 端点按 3 位小数取整
            return f"{name} · {v}"
    return f"{name} · {interval}"


def fig_regime(tab: pd.DataFrame, overall: float, path: Path) -> None:
    rows = [(_regime_label(r["feature"], r["bucket"]), r) for _, r in tab.iterrows()]
    fig, ax = plt.subplots(figsize=(8.8, 0.34 * len(rows) + 1.5))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, r) in zip(y, rows):
        ax.plot([r["mean_r_xf_lo"], r["mean_r_xf_hi"]], [yi, yi], color=C1, linewidth=1.6, zorder=2)
        ax.plot(r["mean_r_xf"], yi, "o", color=C1, markersize=6.5, markeredgecolor=SURF, markeredgewidth=1.2, zorder=3)
        ax.plot(r["null_mean_r"], yi, "o", markersize=6.5, markerfacecolor=SURF, markeredgecolor=C2,
                markeredgewidth=1.8, zorder=4)
        ax.text(1.01, yi, f"n={int(r['n'])}", transform=ax.get_yaxis_transform(), va="center", color=MUTED, fontsize=8.5)
    ax.set_yticks(y, [lab for lab, _ in rows])
    ax.axvline(0, color=BASE, linewidth=1.0, zorder=1)
    ax.axvline(overall, color=MUTED, linewidth=1.2, zorder=1)
    ax.text(overall, y.max() + 0.75, f"IS 全体 {overall:+.2f}R", color=INK2, fontsize=8.5, ha="center")
    h1 = plt.Line2D([], [], color=C1, marker="o", linewidth=1.6, label="实际平均 R（不含资金费；自助法 95% 区间）")
    h2 = plt.Line2D([], [], color=C2, marker="o", markerfacecolor=SURF, linewidth=0,
                    label="同一批交易的局部随机入场基准（±15 天，剔除前后 24 根）")
    ax.legend(handles=[h1, h2], loc="upper center", bbox_to_anchor=(0.45, -0.07), ncol=1)
    ax.set_xlabel("平均 R（不含资金费）")
    ax.set_title("IS：按入场时刻市场状态分组的平均 R", pad=16)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    FIG.mkdir(parents=True, exist_ok=True)
    scfg, bcfg = BASELINE_STRATEGY, BASELINE_BACKTEST
    out = pipeline.run(SYMBOL, scfg, bcfg)
    res, df1h, sig = out["result"], out["df1h"], out["signals"]
    R: dict = {"meta": {"symbol": SYMBOL, **_provenance(), "strategy_key": scfg.key(), "backtest_key": bcfg.key(),
                        "data_first_bar": df1h.index[0], "data_last_bar": df1h.index[-1], "n_null": N_NULL,
                        "n_boot": N_BOOT, "seed": SEED,
                        "policy": "IS-only trade-level attribution; VAL headline only (plus the carry/roll variants "
                                  "pre-committed in Phase 1); no OOS; no ETH runs"}}

    # ---- 1. 总体指标（IS / VAL）、夏普置信区间、买入持有基准 ----
    R["headline"] = {}
    rets = {}
    for name in ("IS", "VAL"):
        s, e = SEGMENTS[name]
        rets[name] = A.segment_daily_returns(res, s, e)
        R["headline"][name] = {"strategy": headline(out["report"][name]),
                               "sharpe_ci": A.stationary_bootstrap_sharpe(rets[name], n_boot=N_BOOT, seed=SEED),
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

    # ---- 2b. 基线定义的敏感性：carry / roll 开关（总体指标；配对夏普差只算 IS）----
    R["struct_variants"], R["struct_paired_is"] = {}, {}
    for vname, kw in STRUCT_VARIANTS.items():
        o = out if not kw else pipeline.run(SYMBOL, scfg.with_(**kw), bcfg)
        bud = signal_budget(o["signals"], o["result"].trades)
        R["struct_variants"][vname] = {seg: {"raw_signals": bud[seg]["raw_signals"], "entries": bud[seg]["entries"],
                                             **{k: o["report"][seg][k] for k in ("sharpe", "cagr", "total_return", "max_dd",
                                                                                  "profit_factor", "win_rate", "avg_r")}}
                                       for seg in ("IS", "VAL")}
        if kw:
            r_is = A.segment_daily_returns(o["result"], IS_START, IS_END)
            R["struct_paired_is"][vname] = A.paired_sharpe_diff(r_is, rets["IS"], n_boot=N_BOOT, seed=SEED)

    # 零技能稀释基准：随机丢弃信号到与变体相同的入场笔数（IS 经验分布）；VAL 只用公式 S_ref * sqrt(n / n_ref)
    n_ref_is = R["struct_variants"]["carry_on_roll_on (baseline)"]["IS"]["entries"]
    n_ref_val = R["struct_variants"]["carry_on_roll_on (baseline)"]["VAL"]["entries"]
    s_ref_is, s_ref_val = out["report"]["IS"]["sharpe"], out["report"]["VAL"]["sharpe"]
    _, fund_research, _ = pipeline.load_market(SYMBOL)
    R["struct_thinning"] = {}
    for j, vname in enumerate(k for k in STRUCT_VARIANTS if STRUCT_VARIANTS[k]):
        v = R["struct_variants"][vname]
        th = A.random_thinning(df1h, sig, bcfg, fund_research, IS_START, IS_END, target_trades=v["IS"]["entries"],
                               n_rep=N_THIN, seed=SEED + 100 + j)
        R["struct_thinning"][vname] = {
            "keep": th["keep"], "thinned_mean_trades": th["mean_trades"], "thinned_mean_sharpe": th["mean_sharpe"],
            "thinned_sd_sharpe": th["sd_sharpe"],
            "p_thinned_ge_variant": float((1 + (th["sharpe"] >= v["IS"]["sharpe"]).sum()) / (N_THIN + 1)),
            "formula_is": s_ref_is * np.sqrt(v["IS"]["entries"] / n_ref_is),
            "margin_is_vs_formula": v["IS"]["sharpe"] - s_ref_is * np.sqrt(v["IS"]["entries"] / n_ref_is),
            "margin_is_vs_empirical": v["IS"]["sharpe"] - th["mean_sharpe"],
            "formula_val": s_ref_val * np.sqrt(v["VAL"]["entries"] / n_ref_val),
            "margin_val_vs_formula": v["VAL"]["sharpe"] - s_ref_val * np.sqrt(v["VAL"]["entries"] / n_ref_val)}
    th1 = A.random_thinning(df1h, sig, bcfg, fund_research, IS_START, IS_END, keep=1.0, n_rep=1, seed=SEED)
    assert abs(th1["sharpe"][0] - s_ref_is) < 1e-12 and th1["trades"][0] == n_ref_is   # keep = 1 复现基线

    # ---- 3. IS 交易级归因 ----
    tf = A.trade_frame(res, df1h, out["boxes"], scfg)
    ist = A.segment_trades(tf, IS_START, IS_END).reset_index(drop=True)
    ist = pd.concat([ist, A.regime_features(df1h, out["df4h"], pipeline.spot_daily_warmup(SYMBOL),
                                            ist["entry_bar"].to_numpy())], axis=1)
    total_pnl = float(ist["pnl"].sum())
    total_log = float(np.log1p(ist["ret_on_base"]).sum())
    R["is_log_identity"] = {"sum_log1p_ret_on_base": total_log,
                            "log_is_equity_ratio": float(np.log(1 + R["headline"]["IS"]["strategy"]["total_return"]))}
    R["is_expectancy"] = expectancy(ist, bcfg)
    rt_cost = 2 * bcfg.slippage + bcfg.taker_fee + bcfg.maker_fee             # 止盈离场的往返成本（≈ 11 bp）
    near = ist[ist["d_tp"] < 3 * rt_cost]
    tp_tr = ist[ist["exit_reason"] == "TP"]
    pos = (ist["entry_px"] / (1 + bcfg.slippage) - ist["VAL"]) / (ist["VAH"] - ist["VAL"])   # 信号收盘在箱体中的高度
    R["is_near_cost"] = {
        "round_trip_cost_tp": rt_cost, "n_d_tp_lt_3x_cost": int(len(near)),
        "tp_exits_net_loss": int((tp_tr["pnl"] <= 0).sum()),
        "tp_net_loss_d_tp_range": [float(tp_tr.loc[tp_tr["pnl"] <= 0, "d_tp"].min()),
                                   float(tp_tr.loc[tp_tr["pnl"] <= 0, "d_tp"].max())],
        "width_atr_range": [float(near["width_atr"].min()), float(near["width_atr"].max())],
        "width_atr_median_all": float(ist["width_atr"].median()),
        "box_position_min_median_max": [float(pos[near.index].min()), float(pos[near.index].median()),
                                        float(pos[near.index].max())],
        "vah_minus_close_atr_max": float(((ist["VAH"] - ist["entry_px"] / (1 + bcfg.slippage)) / ist["atr"])[near.index].max()),
        "vah_minus_close_atr_median_all": float(((ist["VAH"] - ist["entry_px"] / (1 + bcfg.slippage)) / ist["atr"]).median()),
        "rr_lt_05_median_d_tp": float(ist.loc[ist["rr_fill"] < 0.5, "d_tp"].median()),
        "rr_lt_05_median_hold": float(ist.loc[ist["rr_fill"] < 0.5, "hold_bars"].median()),
        "rr_lt_05_sum_pnl": float(ist.loc[ist["rr_fill"] < 0.5, "pnl"].sum()),
    }
    R["is_exit_reasons"] = exit_table(ist)
    R["is_costs"] = cost_decomposition(ist)

    sgl = out["signals"]
    lows = df1h["low"].to_numpy(dtype=np.float64)
    closes = df1h["close"].to_numpy(dtype=np.float64)
    st = sgl["state"].to_numpy()
    enterable = A.executable_bars(df1h, sgl)
    assert np.isin(ist["entry_bar"].to_numpy(), enterable).all()           # 全部真实入场都在候选池内
    atr_bar = sgl["atr"].to_numpy(dtype=np.float64)                         # 第 i 根开盘时已知的 ATR
    nulls = {}
    for j, (key, wd, side, excl, pool, geom) in enumerate(NULL_SPECS):
        nulls[key] = A.random_entry_null(df1h, ist, bcfg, IS_START, IS_END, n_rep=N_NULL, seed=SEED + j,
                                         window_days=wd, side=side, exclude_bars=excl,
                                         pool=enterable if pool == "enterable" else None, geometry=geom,
                                         atr=atr_bar if geom == "atr" else None)
    R["is_enterable_pool"] = {"n_bars_is": int(((enterable >= int(np.searchsorted(df1h.index, IS_START)))
                                                 & (enterable < int(np.searchsorted(df1h.index, IS_END)))).sum()),
                              "contains_all_entries": True}
    shift = A.common_shift_null(df1h, ist, bcfg, IS_START, IS_END, max_shift_bars=360, min_shift_bars=25)
    ist["null_r"] = nulls[EXCESS_BASE]["per_trade_null_mean_r"]       # 分桶的超额 R 基准：±15 天、剔除前后 24 根
    act = float(ist["r_ex_funding"].mean())
    R["is_null"] = {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, np.ndarray)} for k, v in nulls.items()}
    R["is_null_common_shift"] = {k: v for k, v in shift.items() if not isinstance(v, np.ndarray)}
    R["is_null_actual_mean_r_ex_funding"] = act
    first_is = int(np.searchsorted(df1h.index, IS_START))
    end_is = int(np.searchsorted(df1h.index, IS_END))
    R["is_forward_drift"] = forward_drift(df1h, ist["entry_bar"].to_numpy(), end_is, first_is)
    R["is_exits"] = exit_diagnostics(ist, df1h["open"].to_numpy(dtype=np.float64))
    R["is_hold_profile"] = hold_profile(ist, total_pnl, total_log)

    ent = pd.concat([bucketize(ist, c, e, total_log) for c, e in BUCKETS.items()]
                    + [categorical_table(ist, c, total_log) for c in CATEGORICAL], ignore_index=True)
    reg = pd.concat([bucketize(ist, c, e, total_log) for c, e in REGIME_BUCKETS.items()], ignore_index=True)
    R["is_entry_buckets"], R["is_regime_buckets"] = ent, reg
    R["is_regime_coverage"] = {c: int(ist[c].notna().sum()) for c in REGIME_BUCKETS}
    allb = pd.concat([ent, reg], ignore_index=True)
    big = allb[allb["n"] >= 5]
    R["is_bucket_summary"] = {
        "n_features": int(allb["feature"].nunique()), "n_groups": int(len(allb)), "n_groups_n_ge_5": int(len(big)),
        "positive_mean_r_xf_groups_n_ge_5": big.loc[big["mean_r_xf"] > 0, ["feature", "bucket", "n", "mean_r_xf"]],
        "positive_mean_ret_groups_n_ge_5": big.loc[big["mean_ret"] > 0, ["feature", "bucket", "n", "mean_ret"]],
        "overall_excess_r": float((ist["r_ex_funding"] - ist["null_r"]).mean()),
        "spearman_width_atr_vs_rr_fill": float(ist["width_atr"].corr(ist["rr_fill"], method="spearman")),
    }
    ist["excess_r"] = ist["r_ex_funding"] - ist["null_r"]
    groupings = {c: pd.cut(ist[c], bins=e, right=False) for c, e in BUCKETS.items()}
    groupings.update({c: ist[c] for c in CATEGORICAL})
    groupings.update({c: pd.cut(ist[c], bins=e, right=False) for c, e in REGIME_BUCKETS.items()})
    R["is_bucket_permutation"] = A.max_gap_permutation(ist, groupings, "excess_r", n_perm=2000, min_n=15, seed=SEED)
    R["is_bucket_gaps"] = [
        gap(ent, ("width_atr", 0.0), ("width_atr", 4.0)), gap(ent, ("rr_fill", 0.0), ("rr_fill", 2.0)),
        gap(ent, ("pivot_pair", "LH"), ("pivot_pair", "HL")), gap(ent, ("pattern_name", "engulfing"), ("pattern_name", "pinbar")),
        gap(reg, ("above_sma200", 0.5), ("above_sma200", -0.5)), gap(reg, ("mom20d", 0.0), ("mom20d", -np.inf)),
    ]
    # 距 365 日高点：≥ −20% 一组 vs 其余两组合并（零假设解释的均值差份额）
    lowdd = ist["dd365"] < -0.2
    R["is_dd365_null_share"] = float((ist.loc[~lowdd, "null_r"].mean() - ist.loc[lowdd, "null_r"].mean())
                                     / (ist.loc[~lowdd, "r_ex_funding"].mean() - ist.loc[lowdd, "r_ex_funding"].mean()))
    # 现有 regime_filter 定义在 IS 交易上的覆盖面预览（只看被标记交易本身，不是回测实验）。
    # 固定 1 倍名义仓位下夏普取决于净值口径的盈亏，因此同时给出 R 与净值两种口径。
    bear = (ist["above_sma200"] == 0) & (ist["sma200_up"] == 0)
    crash48 = ist["hours_since_crash"] < 48
    R["is_filter_preview"] = {}
    for name, m in (("daily_sma_bear", bear), ("crash_cooldown_48h", crash48), ("mom20d_negative", ist["mom20d"] < 0)):
        R["is_filter_preview"][name] = {
            "flagged": int(m.sum()), "flagged_mean_r": float(ist.loc[m, "r_multiple"].mean()),
            "kept_mean_r": float(ist.loc[~m, "r_multiple"].mean()),
            "flagged_mean_ret": float(ist.loc[m, "ret_on_base"].mean()), "kept_mean_ret": float(ist.loc[~m, "ret_on_base"].mean()),
            "flagged_pnl": float(ist.loc[m, "pnl"].sum()), "flagged_share_of_net_loss": float(ist.loc[m, "pnl"].sum() / total_pnl),
            "flagged_share_of_log_loss": float(np.log1p(ist.loc[m, "ret_on_base"]).sum() / total_log),
            "flagged_excess_r": float((ist.loc[m, "r_ex_funding"] - ist.loc[m, "null_r"]).mean()),
            "kept_excess_r": float((ist.loc[~m, "r_ex_funding"] - ist.loc[~m, "null_r"]).mean())}

    # ---- 4. 回撤来源与时间分布（IS）----
    d_eq = daily_equity(res)
    d_is = d_eq[(d_eq.index >= IS_START - DAY) & (d_eq.index <= IS_END - DAY)]
    eps = A.drawdown_episodes(d_is, top=5)
    for ep in eps:
        end_t = ep["trough"] + DAY
        m = (ist["exit_time"] > ep["peak"] + DAY) & (ist["exit_time"] <= end_t)
        ep.update({"trades_closed": int(m.sum()), "wins": int(ist.loc[m, "win"].sum()),
                   "pnl": float(ist.loc[m, "pnl"].sum()), "sum_r": float(ist.loc[m, "r_multiple"].sum())})
    R["is_drawdown_episodes"] = eps
    dd_is = d_is / d_is.cummax() - 1.0
    R["is_dd_path"] = {"min_2020": float(dd_is[dd_is.index < "2021-01-01"].min()),
                       "min_2020_date": dd_is[dd_is.index < "2021-01-01"].idxmin(),
                       "max_2021q1": float(dd_is[(dd_is.index >= "2021-01-01") & (dd_is.index < "2021-04-01")].max()),
                       "max_2021q1_date": dd_is[(dd_is.index >= "2021-01-01") & (dd_is.index < "2021-04-01")].idxmax()}
    yr = ist.groupby(ist["entry_time"].dt.year)
    R["is_by_year"] = pd.DataFrame({"n": yr.size(), "win_rate": yr["win"].mean(), "mean_r": yr["r_multiple"].mean(),
                                    "sum_pnl": yr["pnl"].sum()}).reset_index()
    mr = monthly_returns(d_is).stack().rename("ret").reset_index()
    R["is_worst_months"] = mr.nsmallest(5, "ret")
    R["is_months_with_trades"] = int(ist["entry_time"].dt.strftime("%Y-%m").nunique())
    w5 = R["is_worst_months"]["ret"].to_numpy()
    R["is_worst5_months_share_of_log_loss"] = float(np.log1p(w5).sum() / np.log(d_is.iloc[-1] / d_is.iloc[0]))
    in_w5 = ist["entry_time"].dt.strftime("%Y-%m").isin(
        [f"{int(y)}-{int(m):02d}" for y, m in R["is_worst_months"][["year", "month"]].to_numpy()])
    R["is_trades_in_worst5_months"] = {
        "n": int(in_w5.sum()), "mean_r": float(ist.loc[in_w5, "r_multiple"].mean()),
        "sum_pnl": float(ist.loc[in_w5, "pnl"].sum()), "sum_r": float(ist.loc[in_w5, "r_multiple"].sum()),
        "exit_reasons": ist.loc[in_w5, "exit_reason"].value_counts().to_dict(),
        "median_d_sl": float(ist.loc[in_w5, "d_sl"].median()), "median_d_sl_other": float(ist.loc[~in_w5, "d_sl"].median()),
        "median_atr_pct": float((ist.loc[in_w5, "atr"] / ist.loc[in_w5, "entry_px"]).median()),
        "median_atr_pct_other": float((ist.loc[~in_w5, "atr"] / ist.loc[~in_w5, "entry_px"]).median()),
        "mean_r_other": float(ist.loc[~in_w5, "r_multiple"].mean()), "n_other": int((~in_w5).sum())}
    R["is_streaks"] = streaks(ist["win"])

    # ---- 5. 仓位 / 止损距离 ----
    lo = ist[~ist["win"]]
    sl_only = ist[ist["exit_reason"] == "SL"]
    wide = ist["d_sl"] >= 0.02
    # 以损定仓的机制（逐笔、忽略复利与杠杆上限）：固定名义仓位下每笔净值收益 ≈ R × d_sl，等风险下 ≈ R × 常数。
    # 期望为负时，权重从宽止损组移向窄止损组；逐笔“均值 / 标准差”给出夏普方向的近似判断。
    ret, rmul, dsl = ist["ret_on_base"], ist["r_multiple"], ist["d_sl"]
    w_fixed = {name: float(dsl[m].mean() / dsl.mean()) for name, m in
               (("d_sl<1%", dsl < 0.01), ("1%<=d_sl<2%", (dsl >= 0.01) & (dsl < 0.02)), ("d_sl>=2%", dsl >= 0.02))}
    x_fixed = rmul * dsl / dsl.mean()                  # 固定名义仓位、按相同平均风险归一后的逐笔收益（以 R 计）
    groups = (("d_sl<1%", dsl < 0.01), ("1%<=d_sl<2%", (dsl >= 0.01) & (dsl < 0.02)), ("d_sl>=2%", dsl >= 0.02))
    R["is_sizing_mechanism"] = {
        "per_trade_ir_fixed_notional": float(ret.mean() / ret.std(ddof=1)),
        "per_trade_ir_equal_risk": float(rmul.mean() / rmul.std(ddof=1)),
        "equal_avg_risk_mean_fixed": float(x_fixed.mean()), "equal_avg_risk_sd_fixed": float(x_fixed.std(ddof=1)),
        "equal_avg_risk_mean_equal": float(rmul.mean()), "equal_avg_risk_sd_equal": float(rmul.std(ddof=1)),
        "dsl_weighted_mean_r_by_group": {name: float((rmul[m] * dsl[m]).sum() / dsl[m].sum()) for name, m in groups},
        "relative_risk_weight_fixed_notional": w_fixed,
        "mean_r_by_group": {name: float(rmul[m].mean()) for name, m in
                            (("d_sl<1%", dsl < 0.01), ("1%<=d_sl<2%", (dsl >= 0.01) & (dsl < 0.02)), ("d_sl>=2%", dsl >= 0.02))},
    }
    neg24 = ist["ret24h"] < -0.02
    worst_bar = ist.loc[ist["pnl"].idxmin(), "entry_bar"]
    R["is_ret24h_drop_bucket"] = {"n": int(neg24.sum()), "mean_ret": float(ist.loc[neg24, "ret_on_base"].mean()),
                                  "mean_ret_without_worst_trade": float(ist.loc[neg24 & (ist["entry_bar"] != worst_bar),
                                                                                "ret_on_base"].mean()),
                                  "worst_trade_in_bucket": bool(neg24[ist["entry_bar"] == worst_bar].any())}
    R["is_sizing"] = {
        "spearman_dsl_vs_loss_pct_losers": float(lo["d_sl"].corr(-lo["ret_on_base"], method="spearman")),
        "spearman_dsl_vs_loss_pct_sl_only": float(sl_only["d_sl"].corr(-sl_only["ret_on_base"], method="spearman")),
        "d_sl": quantiles(ist["d_sl"]),
        "trades_d_sl_ge_2pct": int(wide.sum()),
        "loss_share_d_sl_ge_2pct": float(ist.loc[wide & ~ist["win"], "pnl"].sum() / lo["pnl"].sum()),
        "pnl_d_sl_ge_2pct": float(ist.loc[wide, "pnl"].sum()), "net_loss_share_d_sl_ge_2pct": float(ist.loc[wide, "pnl"].sum() / total_pnl),
        "mean_r_d_sl_ge_2pct": float(ist.loc[wide, "r_multiple"].mean()),
        "mean_r_d_sl_lt_2pct": float(ist.loc[~wide, "r_multiple"].mean()),
        "worst10": ist.nsmallest(10, "pnl")[["entry_time", "exit_reason", "d_sl", "ret_on_base", "r_multiple",
                                              "above_sma200", "ret24h", "hours_since_crash"]],
    }

    # ---- 6. 输出 ----
    keep = ["entry_time", "exit_time", "exit_reason", "pattern_name", "entry_px", "exit_px", "sl", "tp", "poc", "VAL", "VAH",
            "atr", "d_sl", "d_tp", "rr_fill", "p_rw", "pnl", "ret_on_base", "r_multiple", "r_ex_funding", "cost_r",
            "hold_bars", "mfe_r_strict", "mae_r_strict", "reached_poc_strict", "ambiguous_exit_bar", "width_atr", "sl_atr",
            "entry_vs_val_atr", "touch_gap_bars", "pattern_lag_bars", "box_age_h", "sfp", "pivot_pair",
            "reentry_after_loss_same_zone", "above_sma200", "sma200_up", "mom20d", "vol30d_pct", "dd365",
            "hours_since_crash", "ret24h", "ret168h", "null_r"]
    ist[keep].to_csv(REPORTS / "phase2_is_trades.csv", index=False, float_format="%.10g")
    R["meta"]["runtime_s"] = time.time() - t0
    (REPORTS / "phase2_baseline.json").write_text(json.dumps(_jsonable(R), ensure_ascii=False, indent=1))
    (REPORTS / "phase2_tables.md").write_text(tables_md(json.loads(json.dumps(_jsonable(R))), ent, reg))

    _style()
    fig_equity(d_is, d_free[(d_free.index >= IS_START - DAY) & (d_free.index <= IS_END - DAY)],
               R["is_worst_months"][["year", "month"]].to_numpy().tolist(), R["is_worst5_months_share_of_log_loss"],
               R["is_trades_in_worst5_months"]["n"], FIG / "phase2_equity_underwater.png")
    fig_null(nulls, act, FIG / "phase2_is_null_test.png")
    fig_rr(ent[ent["feature"] == "rr_fill"].reset_index(drop=True), FIG / "phase2_is_winrate_by_rr.png")
    fig_exits(ist, FIG / "phase2_is_exit_paths.png")
    fig_regime(reg, act, FIG / "phase2_is_regime.png")
    print(f"phase 2 done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
