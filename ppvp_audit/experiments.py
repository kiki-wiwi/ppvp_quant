"""
ppvp_audit/experiments.py — 按 reports/audit_round/experiment_plan.md 运行 9 个预注册版本并生成指标、配对比较、
匹配随机基准、成本压力、内部稳定性、实验台账与图表。BTC IS only，1H 数据，最坏路径。

用法：python -m ppvp_audit.experiments
输出：reports/audit_round/experiments/（逐版本交易表、results.json）、reports/audit_round/experiment_ledger.csv、
reports/audit_round/figures/exp_*.png
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import attribution as A
import pipeline
from backtester import run_backtest
from config import IS_END, IS_START
from metrics import daily_equity, daily_returns, segment_report, sharpe

from . import nulls as N
from . import variants as V
from .common import FIG, OUT, assert_is_only, is_run, load_is_market, sha256_file, write_json
from .engine_path import as_backtest_result, run_path, tick_size
from .event_sim import simulate_events

EXP = OUT / "experiments"
SEED_PAIRED, SEED_CLUSTER, SEED_THIN, SEED_MATCH = 20261125, 20261126, 20261130, 20261140
DAY = pd.Timedelta(days=1)
LOSS_MONTHS = ("2020-02", "2020-03", "2021-04", "2021-05", "2021-06")
HALVES = (("2020-2021", "2020-01-01", "2022-01-01"), ("2022-2023", "2022-01-01", "2024-01-01"))


def base_configs():
    """B_audited 的配置。审计确认的修复在生产代码中实施；若无修复，B_audited 与 B_original 逐位相同。"""
    from config import BASELINE_BACKTEST, BASELINE_STRATEGY
    return BASELINE_STRATEGY, BASELINE_BACKTEST


def stressed(bcfg):
    return bcfg.with_(perp_taker_fee=2 * bcfg.perp_taker_fee, perp_maker_fee=2 * bcfg.perp_maker_fee,
                      slippage=2 * bcfg.slippage)


# ---------------------------------------------------------------------------
# 运行单个版本（组合层面，各自的持仓状态机）
# ---------------------------------------------------------------------------
def run_version(v: str, base: dict, scfg, bcfg, fill_rule: str = "through") -> dict:
    df1h = base["df1h"]
    _, fund, _ = load_is_market()
    if v == "base":
        return {"out": base, "engine": "backtester"}
    if v in ("A1", "A2", "A3", "E1"):
        sig = V.apply_filter(base, bcfg, v)
        res = run_backtest(df1h, sig, bcfg, fund)
        out = dict(base, signals=sig, result=res, report=segment_report(res, ("IS",)))
        assert_is_only(out)
        return {"out": out, "engine": "backtester"}
    if v == "B1":
        with V.b1_patched(df1h, on=True):
            out = pipeline.run("BTCUSDT", scfg, bcfg, end=IS_END, segments=("IS",))
        assert_is_only(out)
        return {"out": out, "engine": "backtester (B1 state-machine copy)"}
    if v == "B2":
        return {"out": is_run(V.b2_strategy(scfg), bcfg), "engine": "backtester"}
    if v in ("D1", "D2"):
        return {"out": is_run(scfg, V.d_backtest(bcfg, v == "D2")), "engine": "backtester"}
    if v == "C1":
        lim = V.c1_limit_prices(df1h, base["signals"])
        r = run_path(df1h, base["signals"], bcfg, fund, limit_price=lim, valid_bars=V.C1_VALID_BARS, fill_rule=fill_rule)
        res = as_backtest_result(r, bcfg, df1h)
        out = dict(base, result=res, report=segment_report(res, ("IS",)), orders=r["orders"], path_trades=r["trades"])
        assert_is_only(out)
        return {"out": out, "engine": "engine_path (1H sub-bars, worst case)"}
    raise ValueError(v)


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def _daily(res):
    d = daily_equity(res)
    return d[(d.index >= IS_START - DAY) & (d.index <= IS_END - DAY)]


def ambiguity(out: dict, v: str) -> dict:
    """K 线内路径歧义（最坏路径已处理）：离场那根同时触及止损与止盈 / 未成交的 POC；同根分批后保本离场；限价成交那根即止损或触及止盈。"""
    tr = out["result"].trades
    tr = tr[tr["exit_reason"] != "OPEN"]
    h = out["df1h"]["high"].to_numpy(dtype=np.float64)
    if len(tr) == 0:
        return {"n": 0, "ambiguous": 0, "share": float("nan")}
    xb = tr["exit_bar"].to_numpy(dtype=np.int64)
    stop_exit = tr["exit_reason"].isin(["SL", "BE"]).to_numpy()
    amb = stop_exit & (h[xb] >= tr["tp"].to_numpy())
    pb = tr["partial_bar"].to_numpy() if "partial_bar" in tr else np.full(len(tr), -1)
    poc_ok = (tr["poc"] > tr["entry_px"]) & (tr["poc"] < tr["tp"])
    amb |= stop_exit & poc_ok.to_numpy() & (pb < 0) & (h[xb] >= tr["poc"].to_numpy())
    amb |= (pb >= 0) & (pb == xb) & (tr["exit_reason"] == "BE").to_numpy()
    if "fill_and_exit_same_bar" in tr:
        fb = tr["fill_bar"].to_numpy(dtype=np.int64)
        amb |= tr["fill_and_exit_same_bar"].to_numpy(dtype=bool)
        amb |= (~tr["entry_is_market"].to_numpy(dtype=bool)) & (h[fb] >= tr["tp"].to_numpy())
    return {"n": int(len(tr)), "ambiguous": int(amb.sum()), "share": float(amb.mean())}


def version_metrics(out: dict, v: str, parents: int, rejections: dict) -> dict:
    res = out["result"]
    rep = out["report"]["IS"]
    tr = res.trades
    cl = tr[tr["exit_reason"] != "OPEN"].copy()
    slip = res.config.slippage
    cl["entry_ref"] = cl["entry_px"] / (1 + slip)
    cl["exit_ref"] = cl["exit_px"] / (1 - slip)
    risk = cl["qty"] * (cl["entry_px"] - cl["sl"])
    gross_ref = cl["qty"] * (cl["exit_ref"] - cl["entry_ref"])
    cl["gross_R"] = gross_ref / risk
    cl["net_R"] = cl["pnl"] / risk
    month = pd.DatetimeIndex(cl["entry_time"]).strftime("%Y-%m").to_numpy()
    ci = N.cluster_bootstrap_mean(cl["net_R"].to_numpy(), month, seed=SEED_CLUSTER) if len(cl) else {}
    win = cl["pnl"] > 0
    d = _daily(res)
    pos = res.position[(res.position.index > IS_START) & (res.position.index <= IS_END)]
    return {
        "version": v, "parent_signals": parents, "fills": int(len(tr)), "closed_trades": int(len(cl)),
        "fill_rate": float(len(tr) / parents) if parents else float("nan"), "rejections": rejections,
        "total_net_return": rep["total_return"], "total_net_pnl": float(cl["pnl"].sum()),
        "per_opportunity_net_pnl": float(cl["pnl"].sum() / parents) if parents else float("nan"),
        "mean_gross_R": float(cl["gross_R"].mean()), "median_gross_R": float(cl["gross_R"].median()),
        "mean_net_R": float(cl["net_R"].mean()), "median_net_R": float(cl["net_R"].median()),
        "mean_net_R_ci_month_cluster": [ci.get("lo"), ci.get("hi")],
        "win_rate": rep["win_rate"], "avg_win_usd": float(cl.loc[win, "pnl"].mean()) if win.any() else float("nan"),
        "avg_loss_usd": float(cl.loc[~win, "pnl"].mean()) if (~win).any() else float("nan"),
        "avg_win_pct_equity": rep["avg_win_pct"], "avg_loss_pct_equity": rep["avg_loss_pct"],
        "profit_factor": rep["profit_factor"], "avg_hours_to_exit_bar_close": rep["avg_hold_bars"], "exposure": rep["exposure"],
        "mean_nominal_leverage": float((cl["notional"] / cl["equity_at_entry"]).mean()),
        "mean_risk_pct_equity": float((risk / cl["equity_at_entry"]).mean()),
        "commission_usd": float(cl["fees"].sum()), "slippage_impact_usd": float((gross_ref - cl["gross_pnl"]).sum()),
        "funding_usd": float(cl["funding"].sum()),
        "max_dd": rep["max_dd"], "sharpe": rep["sharpe"], "sortino": rep["sortino"], "cagr": rep["cagr"],
        "calmar": rep["calmar"], "exit_reasons": rep["exit_reasons"], "final_equity": float(d.iloc[-1]),
        "intrabar_ambiguity": ambiguity(out, v),
    }


# ---------------------------------------------------------------------------
# 比较
# ---------------------------------------------------------------------------
def paired(res_v, res_b, seed=SEED_PAIRED) -> dict:
    rv, rb = daily_returns(_daily(res_v)), daily_returns(_daily(res_b))
    pdx = A.paired_sharpe_diff(rv, rb, n_boot=5000, seed=seed)
    return {"sharpe_diff": pdx["diff"], "lo": pdx["lo"], "hi": pdx["hi"], "p_gt0": pdx["p_gt0"], "p_one_sided": 1.0 - pdx["p_gt0"]}


def period_table(res_v, res_b) -> list[dict]:
    rows = []
    dv, db = _daily(res_v), _daily(res_b)
    periods = [(str(y), f"{y}-01-01", f"{y + 1}-01-01") for y in range(2020, 2024)] + list(HALVES)
    for name, s, e in periods:
        s, e = pd.Timestamp(s, tz="UTC"), pd.Timestamp(e, tz="UTC")
        sv = dv[(dv.index >= s - DAY) & (dv.index <= e - DAY)]
        sb = db[(db.index >= s - DAY) & (db.index <= e - DAY)]
        rows.append({"period": name, "sharpe_version": sharpe(daily_returns(sv)), "sharpe_base": sharpe(daily_returns(sb)),
                     "return_version": float(sv.iloc[-1] / sv.iloc[0] - 1), "return_base": float(sb.iloc[-1] / sb.iloc[0] - 1)})
    for r in rows:
        r["sharpe_diff"] = r["sharpe_version"] - r["sharpe_base"]
        r["return_diff"] = r["return_version"] - r["return_base"]
    return rows


def loss_month_split(res_v, res_b) -> dict:
    out = {}
    for name, res in (("version", res_v), ("base", res_b)):
        tr = res.trades[res.trades["exit_reason"] != "OPEN"]
        m = pd.DatetimeIndex(tr["entry_time"]).strftime("%Y-%m").isin(LOSS_MONTHS)
        out[name] = {"pnl_loss_months": float(tr.loc[m, "pnl"].sum()), "trades_loss_months": int(m.sum()),
                     "pnl_other": float(tr.loc[~m, "pnl"].sum()), "trades_other": int((~m).sum())}
    return out


def box_concentration(res_v, res_b, sig_v, sig_b) -> dict:
    """按箱体（box_id）的净盈亏差；去掉贡献最大的 3 个箱体后的剩余差。"""
    def by_box(res, sig):
        tr = res.trades[res.trades["exit_reason"] != "OPEN"]
        bid = sig["box_id"].to_numpy()[tr["entry_bar"].to_numpy(dtype=np.int64)]
        return pd.Series(tr["pnl"].to_numpy(), index=bid).groupby(level=0).sum()
    a, b = by_box(res_v, sig_v), by_box(res_b, sig_b)
    d = a.reindex(a.index.union(b.index), fill_value=0.0) - b.reindex(a.index.union(b.index), fill_value=0.0)
    top = d.sort_values(ascending=False).head(3)
    return {"total_pnl_diff": float(d.sum()), "top3_boxes": {int(k): float(v) for k, v in top.items()},
            "pnl_diff_without_top3": float(d.sum() - top[top > 0].sum()), "n_boxes_changed": int((d != 0).sum())}


def opportunity_pairs(base: dict, bcfg, v: str, fill_rule: str = "through") -> dict | None:
    """同一批父信号（B_audited 的全部可交易信号）逐一独立撮合：基线执行 vs 版本执行。只对过滤类、C1、D1、D2 有定义。"""
    if v not in ("A1", "A2", "A3", "E1", "C1", "D1", "D2"):
        return None
    df1h, sig = base["df1h"], base["signals"]
    bars = np.flatnonzero(sig["signal"].to_numpy())
    sl, tp, poc = (sig[k].to_numpy()[bars] for k in ("sl", "tp", "poc_tp"))
    b = simulate_events(df1h, bars, sl, tp, poc, bcfg=bcfg)
    if v in ("A1", "A2", "A3", "E1"):
        m = V.filter_masks(base, bcfg)[f"pass_{v}"].to_numpy()[bars]
        ret_v = np.where(m, b["ret"], 0.0)
        extra = {"accepted": int(m.sum())}
    elif v == "C1":
        lim = V.c1_limit_prices(df1h, sig)[bars]
        e = simulate_events(df1h, bars, sl, tp, poc, limit=lim, tick=tick_size(df1h.index)[bars], valid=V.C1_VALID_BARS,
                            fill_rule=fill_rule, bcfg=bcfg)
        ret_v = np.where(e["filled"], e["ret"], 0.0)
        extra = {"filled": int(e["filled"].sum()), "marketable_at_placement": int((lim >= df1h["close"].to_numpy()[bars]).sum()),
                 "wait_bars": (e["fill_bar"][e["filled"]] - bars[e["filled"]]).tolist()}
    else:
        e = simulate_events(df1h, bars, sl, tp, poc, partial=True, frac=V.D_PARTIAL_FRAC, move_be=(v == "D2"), bcfg=bcfg)
        ret_v = e["ret"]
        extra = {"partial_applicable": int(((poc > df1h["close"].to_numpy()[bars] * (1 + bcfg.slippage)) & (poc < tp)).sum())}
    month = df1h.index[bars].strftime("%Y-%m").to_numpy()
    dlt = ret_v - b["ret"]
    ci = N.cluster_bootstrap_mean(dlt, month, seed=SEED_CLUSTER)
    return {"parents": int(len(bars)), "mean_ret_base": float(b["ret"].mean()), "mean_ret_version": float(ret_v.mean()),
            "mean_diff": ci["mean"], "diff_ci_month_cluster": [ci["lo"], ci["hi"]], **extra}


def matched_random_improvement(base: dict, bcfg, v: str, fill_rule: str = "through") -> dict | None:
    """C1 / D1 / D2：同一执行 / 出场规则用在匹配的随机时刻上；比较真实父信号上的改善与随机时刻上的改善。"""
    if v not in ("C1", "D1", "D2"):
        return None
    df1h, sig = base["df1h"], base["signals"]
    c = df1h["close"].to_numpy(dtype=np.float64)
    h, l = df1h["high"].to_numpy(dtype=np.float64), df1h["low"].to_numpy(dtype=np.float64)
    bars = np.flatnonzero(sig["signal"].to_numpy())
    ep = c[bars] * (1 + bcfg.slippage)
    d_sl = 1 - sig["sl"].to_numpy()[bars] / ep
    d_tp = sig["tp"].to_numpy()[bars] / ep - 1
    d_poc = sig["poc_tp"].to_numpy()[bars] / ep - 1
    feats = N.bar_features(base)
    pools, _ = N.candidate_sets(bars, feats, df1h)
    ticks = tick_size(df1h.index)

    def improvement(dec, dsl, dtp, dpoc):
        e = c[dec] * (1 + bcfg.slippage)
        sl_, tp_, poc_ = e * (1 - dsl), e * (1 + dtp), e * (1 + dpoc)
        b = simulate_events(df1h, dec, sl_, tp_, poc_, bcfg=bcfg)
        if v == "C1":
            x = simulate_events(df1h, dec, sl_, tp_, poc_, limit=0.5 * (h[dec] + l[dec]), tick=ticks[dec],
                                valid=V.C1_VALID_BARS, fill_rule=fill_rule, bcfg=bcfg)
            rv = np.where(x["filled"], x["ret"], 0.0)
        else:
            x = simulate_events(df1h, dec, sl_, tp_, poc_, partial=True, frac=V.D_PARTIAL_FRAC, move_be=(v == "D2"), bcfg=bcfg)
            rv = x["ret"]
        return rv - b["ret"]
    real = improvement(bars, d_sl, d_tp, d_poc)
    exp_rand = np.empty(len(bars))
    for j, pool in enumerate(pools):
        k = len(pool)
        exp_rand[j] = improvement(pool, np.full(k, d_sl[j]), np.full(k, d_tp[j]), np.full(k, d_poc[j])).mean()
    month = df1h.index[bars].strftime("%Y-%m").to_numpy()
    ci = N.cluster_bootstrap_mean(real - exp_rand, month, seed=SEED_MATCH)
    return {"mean_improvement_real": float(real.mean()), "mean_improvement_random": float(exp_rand.mean()),
            "real_minus_random": ci["mean"], "ci_month_cluster": [ci["lo"], ci["hi"]],
            "pool_size_median": float(np.median([len(p) for p in pools]))}


def poc_tie_trades(base: dict, bcfg) -> dict:
    """计划第 7 节第 3 条：列出所用箱体的 POC 存在并列（ppvp_engine.TIE_RTOL 容差）的成交，以及取价格最高的并列桶时
    “Entry < POC < VAH”是否翻转；对翻转的成交，用事件撮合给出 D1 / D2 在两种并列规则下的单笔结果（只是标注，
    不改变 D1 / D2 的定义：定义仍取箱体表中的 POC，即价格最低的并列桶）。"""
    from ppvp_engine import TIE_RTOL, volume_profile
    df1h, boxes, sig = base["df1h"], base["boxes"], base["signals"]
    lo1, hi1, v1 = (df1h[k].to_numpy(dtype=np.float64) for k in ("low", "high", "volume"))
    scfg, _ = base_configs()
    tr = base["result"].trades
    rows = []
    for _, t in tr.iterrows():
        k = int(sig["box_id"].to_numpy()[int(t["entry_bar"])])
        bx = boxes.loc[k]
        s0 = int(df1h.index.searchsorted(pd.Timestamp(bx["window_start"]), side="left"))
        e0 = int(df1h.index.searchsorted(pd.Timestamp(bx["window_end"]), side="left"))
        prof = volume_profile(lo1[s0:e0], hi1[s0:e0], v1[s0:e0], scfg.n_bins, scfg.va_pct)
        hist, edges = prof["hist"], prof["edges"]
        tied = np.flatnonzero(hist >= hist.max() - TIE_RTOL * hist.max())
        if len(tied) < 2:
            continue
        p_lo, p_hi = 0.5 * (edges[tied[0]] + edges[tied[0] + 1]), 0.5 * (edges[tied[-1]] + edges[tied[-1] + 1])
        if not np.isclose(p_lo, t["poc"], rtol=0, atol=1e-9 * abs(p_lo)):
            raise RuntimeError("recomputed lowest tied POC differs from the box table")
        ok_lo = bool(t["entry_px"] < p_lo < t["tp"])
        ok_hi = bool(t["entry_px"] < p_hi < t["tp"])
        row = {"entry_time": str(t["entry_time"]), "box_id": k, "n_tied_bins": int(len(tied)), "poc_lowest_tied": float(p_lo),
               "poc_highest_tied": float(p_hi), "entry_px": float(t["entry_px"]), "tp": float(t["tp"]),
               "eligible_lowest_rule": ok_lo, "eligible_highest_rule": ok_hi, "flips": ok_lo != ok_hi}
        if ok_lo != ok_hi:
            bar = np.array([int(t["entry_bar"])])
            sl_, tp_ = np.array([t["sl"]]), np.array([t["tp"]])
            for name, be in (("D1", False), ("D2", True)):
                r_lo = simulate_events(df1h, bar, sl_, tp_, np.array([p_lo]), partial=True, frac=V.D_PARTIAL_FRAC,
                                       move_be=be, bcfg=bcfg)["ret"][0]
                r_hi = simulate_events(df1h, bar, sl_, tp_, np.array([p_hi]), partial=True, frac=V.D_PARTIAL_FRAC,
                                       move_be=be, bcfg=bcfg)["ret"][0]
                row[f"{name}_event_ret_lowest_rule"] = float(r_lo)
                row[f"{name}_event_ret_highest_rule"] = float(r_hi)
        rows.append(row)
    return {"n_trades_with_tied_poc": len(rows), "n_flipping": int(sum(r["flips"] for r in rows)), "trades": rows}


def thinning_benchmark(base: dict, bcfg, entries: int, k: int) -> dict:
    _, fund, _ = load_is_market()
    th = A.random_thinning(base["df1h"], base["signals"], bcfg, fund, IS_START, IS_END, target_trades=int(entries),
                           n_rep=300, seed=SEED_THIN + k)
    return {"keep": th["keep"], "thinned_mean_trades": th["mean_trades"], "thinned_mean_sharpe": th["mean_sharpe"],
            "thinned_sd_sharpe": th["sd_sharpe"], "sharpe_dist": th["sharpe"]}


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, run = {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        adj[k] = run
    return adj


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _rejections(v: str, base: dict, out_v: dict) -> tuple[int, dict]:
    sig_b = base["signals"]
    raw_b = int(sig_b["sig_raw"].sum())
    tradable_b = int(sig_b["signal"].sum())
    if v in ("A1", "A2", "A3", "E1"):
        sv = out_v["signals"]
        return tradable_b, {"blocked_vah_not_above_close": int(sig_b["block_rr"].sum()),
                            f"blocked_{v}": int(sv[f"block_{v}"].sum()),
                            "skipped_position_open": int(sv["signal"].sum() - len(out_v["result"].trades))}
    if v == "C1":
        st = out_v["orders"]["status"].value_counts().to_dict()
        return tradable_b, {"order_status": st}
    sv = out_v["signals"]
    tr = out_v["result"].trades
    return int(sv["signal"].sum()), {"raw_signals": int(sv["sig_raw"].sum()),
                                     "blocked_vah_not_above_close": int(sv["block_rr"].sum()),
                                     "skipped_position_open": int(sv["signal"].sum() - len(tr)),
                                     "baseline_raw_signals": raw_b}


def run() -> dict:
    t_all = time.time()
    EXP.mkdir(parents=True, exist_ok=True)
    scfg, bcfg = base_configs()
    base = is_run(scfg, bcfg)
    base_s = is_run(scfg, stressed(bcfg))
    parents_b = int(base["signals"]["signal"].sum())
    res = {"meta": {"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "base_strategy_key": scfg.key(), "base_backtest_key": bcfg.key(),
                    "plan": "reports/audit_round/experiment_plan.md", "plan_sha256": sha256_file(OUT / "experiment_plan.md")},
           "base": version_metrics(base, "B_audited", parents_b, {"blocked_vah_not_above_close": int(base["signals"]["block_rr"].sum())})}
    ledger = []
    pvals = {}
    for k, v in enumerate(V.VERSIONS):
        t0 = time.time()
        spec = V.spec_for(v, scfg, bcfg)
        row = {"run_id": f"R{k + 1:02d}", "version": v, "spec_key": spec.key(), "params": spec.params,
               "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        try:
            rv = run_version(v, base, scfg, bcfg)
            out_v = rv["out"]
            parents, rej = _rejections(v, base, out_v)
            m = version_metrics(out_v, v, parents, rej)
            p = paired(out_v["result"], base["result"])
            pvals[v] = p["p_one_sided"]
            # 成本压力（taker 10bp、maker 4bp、滑点 4bp；资金费不变）
            rs = run_version(v, base_s, scfg, stressed(bcfg))
            ps = paired(rs["out"]["result"], base_s["result"])
            entry = {"spec": spec.params, "spec_key": spec.key(), "engine": rv["engine"], "metrics": m, "paired_vs_base": p,
                     "cost_stress": {"sharpe_version": rs["out"]["report"]["IS"]["sharpe"],
                                     "sharpe_base": base_s["report"]["IS"]["sharpe"], "paired": ps},
                     "periods": period_table(out_v["result"], base["result"]),
                     "loss_months": loss_month_split(out_v["result"], base["result"]),
                     "box_concentration": box_concentration(out_v["result"], base["result"],
                                                            out_v.get("signals", base["signals"]), base["signals"]),
                     "opportunity_pairs": opportunity_pairs(base, bcfg, v)}
            if v == "C1":
                rt = run_version("C1", base, scfg, bcfg, fill_rule="touch")
                entry["touch_sensitivity"] = {"metrics": version_metrics(rt["out"], "C1_touch", parents_b,
                                                                         {"order_status": rt["out"]["orders"]["status"].value_counts().to_dict()}),
                                              "paired_vs_base": paired(rt["out"]["result"], base["result"]),
                                              "opportunity_pairs": opportunity_pairs(base, bcfg, "C1", fill_rule="touch")}
                entry["c1_orders"] = c1_order_stats(base, out_v)
            if v in ("A1", "A2", "A3", "B1", "E1"):
                n_v, n_b = len(out_v["result"].trades), len(base["result"].trades)
                if n_v >= n_b:
                    entry["matched_random"] = {"kind": "calibrated random thinning", "not_applicable":
                                               f"version has {n_v} fills, not fewer than B_audited's {n_b}"}
                else:
                    th = thinning_benchmark(base, bcfg, n_v, k)
                    sv = out_v["report"]["IS"]["sharpe"]
                    entry["matched_random"] = {"kind": "calibrated random thinning of B_audited's parent signals (300 reps)",
                                               **{kk: vv for kk, vv in th.items() if kk != "sharpe_dist"},
                                               "p": float((1 + (th["sharpe_dist"] >= sv).sum()) / 301),
                                               "margin_vs_thinned_mean": float(sv - th["thinned_mean_sharpe"])}
            elif v == "B2":
                tfv = A.trade_frame(out_v["result"], out_v["df1h"], out_v["boxes"], V.b2_strategy(scfg))
                ev = N.matched_event_null(out_v, tfv, seed=SEED_MATCH + k)
                dd = tfv["r_ex_funding"].to_numpy() - ev["per_trade_expected_r"]
                ci = N.cluster_bootstrap_mean(dd, tfv["entry_time"].dt.strftime("%Y-%m").to_numpy(), seed=SEED_CLUSTER)
                tfb = A.trade_frame(base["result"], base["df1h"], base["boxes"], scfg)
                eb = N.matched_event_null(base, tfb, seed=SEED_MATCH + k)
                db_ = tfb["r_ex_funding"].to_numpy() - eb["per_trade_expected_r"]
                cib = N.cluster_bootstrap_mean(db_, tfb["entry_time"].dt.strftime("%Y-%m").to_numpy(), seed=SEED_CLUSTER)
                entry["matched_random"] = {"kind": "matched event-level random timing of B2's trades",
                                           "actual_mean_r_ex_funding": ev["actual_mean_r"],
                                           "null_mean_r": ev["null_mean_r_mean"], "p_lower": ev["p_lower"],
                                           "mean_diff": ci["mean"], "diff_ci_month_cluster": [ci["lo"], ci["hi"]],
                                           "same_statistic_for_B_audited": {
                                               "actual_mean_r_ex_funding": eb["actual_mean_r"], "null_mean_r": eb["null_mean_r_mean"],
                                               "p_lower": eb["p_lower"], "mean_diff": cib["mean"],
                                               "diff_ci_month_cluster": [cib["lo"], cib["hi"]]}}
            else:
                entry["matched_random"] = {"kind": "same rule at matched random times (improvement real minus random)",
                                           **matched_random_improvement(base, bcfg, v)}
            if v in ("D1", "D2"):
                entry["poc_tie_annotation"] = poc_tie_trades(base, bcfg)
            tr = out_v["result"].trades
            tr.to_csv(EXP / f"{v}_trades.csv", index=False, float_format="%.10g")
            res[v] = entry
            row.update({"status": "completed", "engine": rv["engine"], "parents": parents, "fills": int(len(tr)),
                        "is_sharpe": out_v["report"]["IS"]["sharpe"], "trades_file": f"reports/audit_round/experiments/{v}_trades.csv",
                        "trades_sha256": sha256_file(EXP / f"{v}_trades.csv")})
        except Exception as exc:  # 失败照实记录，不替换版本
            row.update({"status": f"failed: {type(exc).__name__}: {exc}"})
            res[v] = {"error": row["status"]}
        row["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        row["seconds"] = round(time.time() - t0, 1)
        ledger.append(row)
    adj = holm(pvals)
    for v in pvals:
        res[v]["paired_vs_base"]["p_holm"] = adj[v]
        res[v]["total_net_return_diff_vs_base"] = res[v]["metrics"]["total_net_return"] - res["base"]["total_net_return"]
        res[v]["per_opportunity_net_pnl_diff_vs_base"] = (res[v]["metrics"]["per_opportunity_net_pnl"]
                                                          - res["base"]["per_opportunity_net_pnl"])
        res[v]["decision"] = decide(res[v])
    res["holm_adjusted_p"] = adj
    res["meta"]["runtime_s"] = time.time() - t_all
    write_json(EXP / "results.json", res)
    write_ledger(ledger, res)
    plots(res)
    return res


def c1_order_stats(base: dict, out_v: dict) -> dict:
    """C1：全部父信号的挂单结果、等待根数、未成交后的走势与成交后的最大不利波动。"""
    df1h = base["df1h"]
    o = out_v["orders"]
    c, h, l = (df1h[k].to_numpy(dtype=np.float64) for k in ("close", "high", "low"))
    bars = np.flatnonzero(base["signals"]["signal"].to_numpy())
    st = o.reindex(df1h.index[bars])["status"].fillna("none").to_numpy()
    lim = V.c1_limit_prices(df1h, base["signals"])[bars]
    unf = bars[st == "expired_unfilled"]
    after = []
    for b in unf:
        a, e = b + 1 + V.C1_VALID_BARS, min(b + 1 + V.C1_VALID_BARS + 24, len(df1h))
        if e > a:
            after.append({"max_up_vs_signal_close": float(h[a:e].max() / c[b] - 1),
                          "min_down_vs_limit": float(l[a:e].min() / lim[np.flatnonzero(bars == b)[0]] - 1)})
    pt = out_v["path_trades"]
    lim_fill = pt[~pt["entry_is_market"]]
    wait = (lim_fill["fill_bar"] - lim_fill["parent_bar"]).to_numpy()
    mae = []
    for _, t in pt.iterrows():
        a, e = int(t["fill_bar"]) + 1, int(t["exit_bar"]) if t["exit_bar"] >= 0 else len(df1h)
        if e > a:
            mae.append(float((t["entry_px"] - l[a:e].min()) / (t["entry_px"] - t["sl"])))
    after_df = pd.DataFrame(after)
    return {"parents": int(len(bars)), "status_counts": pd.Series(st).value_counts().to_dict(),
            "limit_fills": int(len(lim_fill)), "marketable_fills": int(pt["entry_is_market"].sum()),
            "wait_bars_counts": pd.Series(wait).value_counts().sort_index().to_dict(),
            "unfilled_next24h_median_max_up_vs_signal_close": float(after_df["max_up_vs_signal_close"].median()) if len(after_df) else None,
            "unfilled_next24h_median_min_down_vs_limit": float(after_df["min_down_vs_limit"].median()) if len(after_df) else None,
            "post_fill_strict_mae_R_median": float(np.median(mae)) if mae else None,
            "fill_and_stop_same_bar": int(pt["fill_and_exit_same_bar"].sum())}


def decide(e: dict) -> dict:
    """计划第 5 节的六条判定。"""
    p = e["paired_vs_base"]
    m = e["metrics"]
    per = {r["period"]: r for r in e["periods"]}
    c1 = p.get("p_holm", 1.0) <= 0.05
    c2 = e["total_net_return_diff_vs_base"] > 0           # 总净收益（按 IS 期末盯市净值）高于 B_audited
    mr = e.get("matched_random", {})
    if "p" in mr:
        c3 = mr["p"] <= 0.10
    elif "ci_month_cluster" in mr:
        c3 = mr["ci_month_cluster"][0] > 0
    elif "diff_ci_month_cluster" in mr:
        c3 = mr["diff_ci_month_cluster"][0] > 0
    else:
        c3 = False
    c4 = per["2020-2021"]["sharpe_diff"] > 0 and per["2022-2023"]["sharpe_diff"] > 0
    c5 = e["cost_stress"]["paired"]["sharpe_diff"] > 0
    c6 = m["closed_trades"] >= 30 and e["box_concentration"]["pnl_diff_without_top3"] > 0
    crit = {"1_holm_p_le_0.05": c1, "2_total_net_pnl_higher": c2, "3_beats_matched_random": bool(c3),
            "4_both_halves_positive": bool(c4), "5_positive_under_2x_costs": bool(c5), "6_n_ge_30_and_not_top3_boxes": bool(c6)}
    if all(crit.values()):
        label = "mechanism candidate worth follow-up validation"
    elif all(v for k, v in crit.items() if not k.startswith("1_")) and p["p_one_sided"] <= 0.05:
        label = "insufficient evidence (nominally significant, fails multiple-comparison correction)"
    else:
        label = "no credible improvement"
    return {"criteria": crit, "label": label}


def write_ledger(ledger: list[dict], res: dict) -> None:
    rows = [{"run_id": "R00", "version": "pre-plan exposure", "status": "disclosed",
             "notes": "Engine parity check run before the plan was committed printed the IS Sharpe of the D1 and D2 "
                      "exit settings (about -1.166 each; 109 and 113 trades). Parameters are user-fixed; nothing changed."}]
    for r in ledger:
        rr = dict(r)
        rr["params"] = str(rr.get("params"))
        v = r["version"]
        if v in res and "decision" in res[v]:
            rr["sharpe_diff_vs_base"] = res[v]["paired_vs_base"]["sharpe_diff"]
            rr["p_one_sided"] = res[v]["paired_vs_base"]["p_one_sided"]
            rr["p_holm"] = res[v]["paired_vs_base"]["p_holm"]
            rr["decision"] = res[v]["decision"]["label"]
        rows.append(rr)
    pd.DataFrame(rows).to_csv(OUT / "experiment_ledger.csv", index=False)


def plots(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .plotstyle import C1, C2, GRID, INK2, MUTED, SURF, style
    style()
    vs = [v for v in V.VERSIONS if v in res and "paired_vs_base" in res[v]]
    base_sh = res["base"]["sharpe"]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    y = np.arange(len(vs))[::-1]
    zero_lab = True
    for yi, v in zip(y, vs):
        p = res[v]["paired_vs_base"]
        ax.plot([p["lo"], p["hi"]], [yi, yi], color=C1, linewidth=1.8)
        ax.plot(p["sharpe_diff"], yi, "o", color=C1, markersize=7, markeredgecolor=SURF,
                label="版本 − B_audited（点估计与 95% 区间）" if yi == y[0] else None)
        mr = res[v].get("matched_random", {})
        if "thinned_mean_sharpe" in mr:            # 零技能对照：随机保留到同样笔数时夏普的平均变化
            ax.plot(mr["thinned_mean_sharpe"] - base_sh, yi, "D", color=C2, markersize=6, markeredgecolor=SURF,
                    label="同笔数随机保留的平均夏普差（零技能对照）" if zero_lab else None)
            zero_lab = False
        ax.text(1.01, yi, f"Holm p = {p['p_holm']:.2f}", transform=ax.get_yaxis_transform(), va="center", fontsize=8.5, color=MUTED)
    ax.axvline(0, color=INK2, linewidth=1.0)
    ax.set_yticks(y, vs)
    ax.set_xlabel("相对 B_audited 的日度夏普差（配对平稳自助法 95% 区间）")
    ax.set_title("IS：9 个预注册版本相对 B_audited 的夏普差")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "exp_sharpe_diff.png")
    plt.close(fig)
    years = ["2020", "2021", "2022", "2023"]
    mat = np.array([[next(r["sharpe_diff"] for r in res[v]["periods"] if r["period"] == yv) for yv in years] for v in vs])
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    lim = np.nanmax(np.abs(mat))
    im = ax.imshow(mat, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(vs)):
        for j in range(len(years)):
            dark = abs(mat[i, j]) > 0.6 * lim
            ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=8.5, color="#ffffff" if dark else "#111111")
    ax.set_xticks(range(len(years)), years)
    ax.set_yticks(range(len(vs)), vs)
    ax.grid(False)
    ax.set_title("按自然年份的夏普差（版本 − B_audited；内部稳定性检查）")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIG / "exp_sharpe_diff_by_year.png")
    plt.close(fig)


SEEDS = {"paired_sharpe_stationary_bootstrap": SEED_PAIRED, "month_cluster_bootstrap": SEED_CLUSTER,
         "random_thinning": "20261130 + version index (A1=0 ... E1=8)", "matched_random_C1_D1_D2": SEED_MATCH,
         "matched_event_null_B2_and_B_audited": "20261140 + 4 (B2's index)"}


def annotate_ledger() -> None:
    """实验台账补充种子与 results.json 的哈希（只读运行结果，不重跑任何版本）。"""
    led = pd.read_csv(OUT / "experiment_ledger.csv")
    k = {v: i for i, v in enumerate(V.VERSIONS)}
    seeds = []
    for v in led["version"]:
        if v not in k:
            seeds.append("")
            continue
        s = [f"paired={SEED_PAIRED}", f"cluster={SEED_CLUSTER}"]
        if v in ("A1", "A2", "A3", "B1", "E1"):
            s.append(f"thinning={SEED_THIN + k[v]}")
        elif v == "B2":
            s.append(f"matched_event={SEED_MATCH + k[v]}")
        else:
            s.append(f"matched_random={SEED_MATCH}")
        seeds.append(";".join(s))
    led["seeds"] = seeds
    led["results_file"] = "reports/audit_round/experiments/results.json"
    led["results_sha256"] = sha256_file(EXP / "results.json")
    led.to_csv(OUT / "experiment_ledger.csv", index=False)


if __name__ == "__main__":
    import json
    import sys
    if "--plots-only" in sys.argv:                 # 只从 results.json 重画图与补充台账，不重跑任何版本
        plots(json.loads((EXP / "results.json").read_text()))
        annotate_ledger()
    else:
        r = run()
        annotate_ledger()
        print(json.dumps({v: r[v].get("decision") for v in V.VERSIONS}, ensure_ascii=False, indent=1, default=str))
