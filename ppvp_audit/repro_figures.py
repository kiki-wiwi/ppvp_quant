"""
ppvp_audit/repro_figures.py — 在 IS 截断数据上从原始输出复现 Phase 2 的五张诊断图，并补算截图未给出的精确数值。

只读 IS（ppvp_audit.common.is_run）。随机入场零假设沿用 Phase 2 的种子（20261104 + 规格序号），以便逐位复现原图。
输出：reports/audit_round/figure_checks.json 与 reports/audit_round/figures/fig{1..5}_*.png（核验版）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import attribution as A
from config import BASELINE_BACKTEST, BASELINE_STRATEGY, IS_END, IS_START
from metrics import daily_equity, daily_returns, drawdown_series, monthly_returns, sharpe

from .common import FIG, OUT, is_run, load_is_market, write_json

P2_SEED = 20261104            # Phase 2 run_phase2.py 的 SEED；global 为第 0 个规格、local_15d_x24 为第 3 个
DAY = pd.Timedelta(days=1)


def baseline_frames():
    out = is_run(BASELINE_STRATEGY, BASELINE_BACKTEST)
    df1h, _, spot = load_is_market()
    tf = A.trade_frame(out["result"], out["df1h"], out["boxes"], BASELINE_STRATEGY)
    tf = pd.concat([tf, A.regime_features(out["df1h"], out["df4h"], spot, tf["entry_bar"].to_numpy())], axis=1)
    return out, tf


def resolve_exit_bar_poc_with_1m(tf: pd.DataFrame, df1m: pd.DataFrame) -> pd.DataFrame:
    """止损交易离场 K 线内是否先到 POC、再到 SL：用 1m 路径判定（同一分钟内同时触及记为“不可判定”）。"""
    rows = []
    for _, r in tf[tf["exit_reason"] == "SL"].iterrows():
        t0 = r["exit_bar_open"]
        m = df1m[(df1m.index >= t0) & (df1m.index < t0 + pd.Timedelta(hours=1))]
        hit_sl = np.flatnonzero(m["low"].to_numpy() <= r["sl"])
        hit_poc = np.flatnonzero(m["high"].to_numpy() >= r["poc"]) if r["poc"] > r["entry_px"] else np.array([], int)
        hit_tp = np.flatnonzero(m["high"].to_numpy() >= r["tp"])
        first_sl = int(hit_sl[0]) if len(hit_sl) else -1
        first_poc = int(hit_poc[0]) if len(hit_poc) else -1
        first_tp = int(hit_tp[0]) if len(hit_tp) else -1
        def order(a, b):
            if a < 0:
                return "never"
            if b < 0 or a < b:
                return "before_sl"
            return "same_minute" if a == b else "after_sl"
        rows.append({"entry_time": r["entry_time"], "poc_in_exit_bar_vs_sl": order(first_poc, first_sl),
                     "tp_in_exit_bar_vs_sl": order(first_tp, first_sl), "sl_minute": first_sl})
    return pd.DataFrame(rows)


def fig1_numbers(tf: pd.DataFrame, df1m: pd.DataFrame) -> dict:
    sl = tf[tf["exit_reason"] == "SL"]
    tp = tf[tf["exit_reason"] == "TP"]
    sl_poc = sl[sl["poc_above_entry"]]
    sl_plot = sl_poc[sl_poc["mfe_r_strict"].notna()]
    exb = resolve_exit_bar_poc_with_1m(tf, df1m)
    m = sl.merge(exb, on="entry_time", how="left")
    in_plot = m["poc_above_entry"] & m["mfe_r_strict"].notna()
    not_strict = in_plot & ~m["reached_poc_strict"]
    return {
        "trades": int(len(tf)), "sl_trades": int(len(sl)), "tp_trades": int(len(tp)),
        "sl_poc_above_entry": int(len(sl_poc)), "sl_poc_not_above_entry": int((~sl["poc_above_entry"]).sum()),
        "sl_poc_above_without_full_bar": int(sl_poc["mfe_r_strict"].isna().sum()),
        "sl_in_left_panel": int(len(sl_plot)), "sl_reached_poc_strict": int(sl_plot["reached_poc_strict"].sum()),
        "sl_not_reached_poc_strict": int((~sl_plot["reached_poc_strict"]).sum()),
        "tp_with_full_bar_for_strict_mae": int(tp["mae_r_strict"].notna().sum()),
        "tp_exit_on_first_bar_after_entry": int((tp["exit_bar"] - tp["entry_bar"] == 1).sum()),
        "entry_and_exit_same_bar": int((tf["exit_bar"] == tf["entry_bar"]).sum()),
        "sl_exit_bar_also_touches_tp": int(tf["ambiguous_exit_bar"].sum()),
        "sl_exit_bar_tp_order_1m": m.loc[m["tp_in_exit_bar_vs_sl"] != "never", "tp_in_exit_bar_vs_sl"].value_counts().to_dict(),
        # 严格口径未计入、但离场 K 线内到过 POC 的止损交易：用 1m 判定先后
        "left_panel_not_strict_but_poc_in_exit_bar_1m": m.loc[not_strict, "poc_in_exit_bar_vs_sl"].value_counts().to_dict(),
        "share_reached_poc_strict": float(sl_plot["reached_poc_strict"].mean()),
    }


def fig4_numbers(tf: pd.DataFrame) -> list[dict]:
    edges = [0.0, 0.5, 1.0, 2.0, np.inf]
    cut = pd.cut(tf["rr_fill"], bins=edges, right=False)
    rows = []
    for b, sub in tf.groupby(cut, observed=True):
        n, k = len(sub), int((sub["exit_reason"] == "TP").sum())
        lo, hi = A.wilson(k, n)
        p = sub["p_rw"].to_numpy()
        exp = float(p.sum())
        cdf_le = A.poisson_binomial_cdf(p, k)
        cdf_lt = A.poisson_binomial_cdf(p, k - 1) if k > 0 else 0.0
        rows.append({"bucket": str(b), "n": n, "tp_hits": k, "tp_rate": k / n, "wilson_lo": lo, "wilson_hi": hi,
                     "rw_mean_p": float(p.mean()), "rw_expected_hits": exp,
                     "rw_inside_wilson": bool(lo <= p.mean() <= hi),
                     "p_lower_exact": float(cdf_le), "p_upper_exact": float(1.0 - cdf_lt),
                     "rw_from_mean_rr": float(1.0 / (1.0 + sub["rr_fill"].mean()))})
    return rows


def frictionless_fixed_quantity(out: dict) -> dict:
    """固定原交易的入场 / 离场 K 线、数量与离场事件，去掉全部成本（入场按信号收盘、离场按参考价，不收手续费与资金费）；
    逐小时按同一数量盯市，得到日度净值。与“取消成本后按净值重新计算复利仓位”的对照不同，两者分开报告。"""
    res = out["result"]
    df1h = out["df1h"]
    c = df1h["close"].to_numpy(dtype=np.float64)
    tr = res.trades
    slip = res.config.slippage
    n = len(df1h)
    pnl_closed = np.zeros(n)
    mtm = np.zeros(n)
    for _, t in tr.iterrows():
        eb, xb, q = int(t["entry_bar"]), int(t["exit_bar"]), float(t["qty"])
        e_ref = t["entry_px"] / (1 + slip)
        x_ref = t["exit_px"] / (1 - slip)
        mtm[eb:xb] += q * (c[eb:xb] - e_ref)
        pnl_closed[xb:] += q * (x_ref - e_ref)
    eq = res.initial_capital + pnl_closed + mtm
    s = pd.Series(eq, index=res.equity.index)
    from backtester import BacktestResult
    fake = BacktestResult(equity=s, position=res.position, trades=tr.iloc[:0], config=res.config,
                          initial_capital=res.initial_capital, start_time=res.start_time)
    d = daily_equity(fake)
    d_is = d[(d.index >= IS_START - DAY) & (d.index <= IS_END - DAY)]
    r = daily_returns(d_is)
    return {"final_equity": float(d_is.iloc[-1]), "total_return": float(d_is.iloc[-1] / d_is.iloc[0] - 1),
            "sharpe": sharpe(r), "daily": d_is}


def fig5_numbers(out: dict, tf: pd.DataFrame) -> tuple[dict, pd.Series, pd.Series, pd.Series]:
    res = out["result"]
    d = daily_equity(res)
    d_is = d[(d.index >= IS_START - DAY) & (d.index <= IS_END - DAY)]
    free = is_run(BASELINE_STRATEGY, BASELINE_BACKTEST.with_(perp_taker_fee=0.0, perp_maker_fee=0.0, slippage=0.0,
                                                            include_funding=False))
    same = np.array_equal(free["result"].trades[["entry_bar", "exit_bar"]].to_numpy(),
                          res.trades[["entry_bar", "exit_bar"]].to_numpy())
    d_free = daily_equity(free["result"])
    d_free = d_free[(d_free.index >= IS_START - DAY) & (d_free.index <= IS_END - DAY)]
    fq = frictionless_fixed_quantity(out)
    mr = monthly_returns(d_is).stack().rename("ret").reset_index()
    w5 = mr.nsmallest(5, "ret")
    share = float(np.log1p(w5["ret"].to_numpy()).sum() / np.log(d_is.iloc[-1] / d_is.iloc[0]))
    months = [f"{int(y)}-{int(m):02d}" for y, m in w5[["year", "month"]].to_numpy()]
    in_w5 = tf["entry_time"].dt.strftime("%Y-%m").isin(months)
    # 同一 5 个月：按“这 13 笔交易本身”的对数收益占比（与按月净值的口径不同，二者都报告）
    trade_share = float(np.log1p(tf.loc[in_w5, "ret_on_base"]).sum() / np.log1p(tf["ret_on_base"]).sum())
    costs = {"gross_pnl_at_fill": float(res.trades["gross_pnl"].sum()), "fees": float(res.trades["fees"].sum()),
             "funding": float(res.trades["funding"].sum()), "slippage": float(tf["slip_usd"].sum()),
             "net_pnl": float(res.trades["pnl"].sum())}
    num = {
        "start_equity": float(d_is.iloc[0]), "final_equity": float(d_is.iloc[-1]),
        "total_return": float(d_is.iloc[-1] / d_is.iloc[0] - 1), "sharpe": sharpe(daily_returns(d_is)),
        "frictionless_recompounded": {"same_entry_exit_bars": bool(same), "final_equity": float(d_free.iloc[-1]),
                                      "total_return": float(d_free.iloc[-1] / d_free.iloc[0] - 1),
                                      "sharpe": sharpe(daily_returns(d_free))},
        "frictionless_fixed_quantity": {k: v for k, v in fq.items() if k != "daily"},
        "worst5_months": months, "worst5_month_returns": w5["ret"].tolist(),
        "worst5_share_of_is_log_loss_by_month": share,
        "trades_entered_in_worst5_months": int(in_w5.sum()),
        "worst5_trades_share_of_trade_log_loss": trade_share,
        "costs_usd": costs,
    }
    return num, d_is, d_free, fq["daily"]


def run(df1m: pd.DataFrame | None = None) -> dict:
    from .data_1m import load_1m
    df1m = load_1m() if df1m is None else df1m
    out, tf = baseline_frames()
    rep = out["report"]["IS"]
    res = {"is_headline": {k: rep[k] for k in ("sharpe", "trades", "total_return", "max_dd", "win_rate", "avg_r",
                                               "profit_factor", "exit_reasons", "trades_open_at_end")}}
    res["fig1"] = fig1_numbers(tf, df1m)
    # ---- 图 2：随机入场零假设（Phase 2 的方法、种子与 5,000 次重抽样）----
    nulls = {}
    for key, j, wd, excl in (("global", 0, None, None), ("local_15d_x24", 3, 15, 24)):
        nul = A.random_entry_null(out["df1h"], tf, BASELINE_BACKTEST, IS_START, IS_END, n_rep=5000, seed=P2_SEED + j,
                                  window_days=wd, exclude_bars=excl)
        nulls[key] = nul
    actual = float(tf["r_ex_funding"].mean())
    res["fig2"] = {
        "actual_mean_r_ex_funding": actual, "actual_mean_net_r_incl_funding": float(tf["r_multiple"].mean()),
        "actual_mean_gross_r_before_costs": float((tf["pre_cost_pnl"] / tf["risk_usd"]).mean()),
        "r_denominator": "qty * (entry_fill - SL_initial), entry_fill = signal close * (1 + slippage)",
        "costs_in_null_and_actual": "taker fee on entry and SL, maker fee on TP, 2bp slippage on every fill; funding excluded",
        **{k: {kk: v[kk] for kk in ("null_mean_r_mean", "null_mean_r_p05", "null_mean_r_p95", "p_mean_r_le",
                                    "unresolved_frac", "n_rep", "seed", "window_days", "exclude_bars",
                                    "same_bar_sl_tp_frac", "draw_share_own_bar", "draw_share_within_24")}
           for k, v in nulls.items()},
    }
    # ---- 图 3：市场状态分组 ----
    tf["null_r"] = nulls["local_15d_x24"]["per_trade_null_mean_r"]
    rows = []
    edges = {"above_sma200": [-0.5, 0.5, 1.5], "sma200_up": [-0.5, 0.5, 1.5], "mom20d": [-np.inf, 0.0, np.inf],
             "vol30d_pct": [0.0, 1 / 3, 2 / 3, 1.0001], "dd365": [-1.0, -0.4, -0.2, 1e-12],
             "hours_since_crash": [0.0, 48.0, 168.0, np.inf], "ret24h": [-np.inf, -0.02, 0.0, np.inf]}
    for col, e in edges.items():
        t = A.bucket_stats(tf, pd.cut(tf[col], bins=e, right=False), n_boot=2000, seed=0)
        for _, r in t.iterrows():
            rows.append({"feature": col, "bucket": str(r["bucket"]), "n": int(r["n"]), "mean_r_xf": r["mean_r_xf"],
                         "lo": r["mean_r_xf_lo"], "hi": r["mean_r_xf_hi"], "ci_excludes_zero": bool(r["mean_r_xf_hi"] < 0),
                         "null_mean_r": r["null_mean_r"]})
    res["fig3"] = {"groups": rows, "n_groups": len(rows), "groups_with_ci_below_zero": int(sum(r["ci_excludes_zero"] for r in rows)),
                   "groups_n_lt_15": [f"{r['feature']} {r['bucket']} (n={r['n']})" for r in rows if r["n"] < 15]}
    res["fig4"] = fig4_numbers(tf)
    num5, d_is, d_free, d_fq = fig5_numbers(out, tf)
    res["fig5"] = num5
    write_json(OUT / "figure_checks.json", res)
    _plots(tf, nulls, actual, res, d_is, d_free, d_fq)
    return res


def _plots(tf, nulls, actual, res, d_is, d_free, d_fq) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .plotstyle import C1, C2, C3, GRID, INK2, MUTED, SURF, style
    style()
    FIG.mkdir(parents=True, exist_ok=True)
    # 图 1 核验版
    sl = tf[(tf["exit_reason"] == "SL") & tf["poc_above_entry"] & tf["mfe_r_strict"].notna()]
    tp = tf[tf["exit_reason"] == "TP"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    hit = sl["reached_poc_strict"].astype(bool)
    ax.scatter(sl.loc[~hit, "poc_r"], sl.loc[~hit, "mfe_r_strict"], s=34, color=C1, edgecolor=SURF, linewidth=1.2,
               label=f"止损前未到 POC（{int((~hit).sum())}）")
    ax.scatter(sl.loc[hit, "poc_r"], sl.loc[hit, "mfe_r_strict"], s=34, color=C2, edgecolor=SURF, linewidth=1.2,
               label=f"止损前已到 POC（{int(hit.sum())}）")
    lim = max(sl["poc_r"].max(), sl["mfe_r_strict"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color=MUTED, linewidth=1.0)
    ax.set_xlim(0, lim)
    ax.set_ylim(min(0.0, float(sl["mfe_r_strict"].min())) - 0.08, lim)
    f1 = res["fig1"]
    ax.set_title(f"止损 {f1['sl_trades']} 笔：POC 高于入场 {f1['sl_poc_above_entry']} 笔，有完整 K 线 {f1['sl_in_left_panel']} 笔")
    ax.set_xlabel("POC 距入场价（R）")
    ax.set_ylabel("止损前最大有利波动（R，严格口径）")
    ax.legend(loc="upper left")
    mae = tp["mae_r_strict"].dropna()
    ax2.hist(mae, bins=np.arange(0, 1.05, 0.05), color=C1, rwidth=0.88)
    ax2.set_title(f"止盈 {len(tp)} 笔：{len(mae)} 笔有完整 K 线，{f1['tp_exit_on_first_bar_after_entry']} 笔入场后第一根即止盈")
    ax2.set_xlabel("止盈前最大不利波动（R，严格口径）")
    ax2.set_ylabel("交易数")
    fig.suptitle("图 1 核验版（IS）：严格口径只用入场后、离场前的完整 1H K 线", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_poc_and_stop_paths_verified.png")
    plt.close(fig)
    # 图 2 核验版
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, (k, title) in zip(axes, (("global", "全局：IS 任意 1H 收盘"), ("local_15d_x24", "局部：信号 ±15 天，剔除前后 24 根"))):
        dist = nulls[k]["null_r_dist"]
        ax.hist(dist, bins=50, color=C1, alpha=0.85, rwidth=0.88)
        ax.axvline(actual, color=C2, linewidth=2.0)
        ax.text(0.98, 0.95, f"真实 {actual:+.3f}R\n零分布均值 {np.mean(dist):+.3f}R\n下尾 p = {nulls[k]['p_mean_r_le']:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("109 笔的平均 R（含手续费与滑点，不含资金费）")
    axes[0].set_ylabel("频数（5,000 次）")
    fig.suptitle("图 2 核验版（IS）：同一止损 / 止盈几何下的事件级随机入场（交易可重叠，不是组合净值）", x=0.01, ha="left",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_random_entry_verified.png")
    plt.close(fig)
    # 图 4 核验版
    t4 = pd.DataFrame(res["fig4"])
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    x = np.arange(len(t4))
    ax.errorbar(x - 0.08, t4["tp_rate"] * 100, yerr=[(t4["tp_rate"] - t4["wilson_lo"]) * 100, (t4["wilson_hi"] - t4["tp_rate"]) * 100],
                fmt="o", color=C1, markersize=7, elinewidth=1.6, capsize=0, label="止盈命中率（Wilson 95%）")
    ax.plot(x + 0.08, t4["rw_mean_p"] * 100, "o", markersize=7, markerfacecolor=SURF, markeredgecolor=C2, markeredgewidth=2,
            label="逐笔随机游走参考值的组均值")
    for xi, r in zip(x, t4.itertuples()):
        ax.text(xi, -6, f"n={r.n}\n下尾 p={r.p_lower_exact:.2f}", ha="center", va="top", color=MUTED, fontsize=8)
    ax.set_xticks(x, [b.replace("inf", "∞") for b in t4["bucket"]])
    ax.set_ylim(-20, 105)
    ax.set_ylabel("先触及止盈的比例（%）")
    ax.set_title("图 4 核验版（IS）：参考值为连续、无漂移、无成本、无期限假设下的边界触碰概率")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_initial_rr_verified.png")
    plt.close(fig)
    # 图 5 核验版
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    ax.plot(d_is.index, d_is / 1e3, color=C1, label="基线（含手续费、滑点、资金费）")
    ax.plot(d_fq.index, d_fq / 1e3, color=C3, label="零成本：固定原交易的时间与数量")
    ax.plot(d_free.index, d_free / 1e3, color=C2, label="零成本：按净值重新计算复利仓位")
    ax.set_ylabel("日收盘净值（千 USDT）")
    ax.set_title("图 5 核验版（IS）：两种零成本对照口径")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_is_equity_verified.png")
    plt.close(fig)


if __name__ == "__main__":
    import json
    r = run()
    print(json.dumps({k: v for k, v in r.items() if k != "fig3"}, ensure_ascii=False, indent=1, default=str)[:6000])
