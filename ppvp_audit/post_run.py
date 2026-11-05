"""
ppvp_audit/post_run.py — 实验运行之后的补充核算（审计复核的修正与补记）。不改变任何版本的定义、交易或判定，
也不改写 experiments/results.json（它是 2026-11-05 那次运行的原始记录，哈希登记在实验台账中）。

1. 路径歧义（复核 EC-1、EC-2）：results.json 的 intrabar_ambiguity 对每个版本都计入“离场 K 线同时触及止损与 POC”，
   但只有 D1 / D2 有 POC 分批，其余版本这一情形不影响盈亏；C1 的“成交即同根止损”在成交那根没有触及止盈时先后是
   确定的。这里给出只计影响该版本盈亏的歧义：离场 K 线同时触及止损（含保本）与止盈；D1 / D2 的离场 K 线在分批之前
   同时触及止损与 POC，以及分批与保本发生在同一根；C1 限价成交那一根触及止盈。原口径一并列出。
2. C1 逐单描述统计（复核 DF-1 至 DF-4）：限价距离、入场改善、RR、未成交 / 成交父信号在基线执行下的结果。
3. C1 匹配随机对照的同类比较（复核 EC-3）：随机时刻的挂单价 (high + low) / 2 有近一半不低于收盘、按市价处理，
   改善恒为 0；这里另给出只用挂单价低于收盘的随机时刻的结果。
4. D1 / D2 的适用性计数（复核 PC-4）：Entry < POC < VAH 的适用与不适用笔数、分批与保本笔数。
5. 补记运行（复核 PC-7）：2 倍成本下的 B_audited 与 9 个版本、C1 触价敏感性、B_audited 本身，在原运行中已经计算
   但没有逐笔落盘；这里用同一代码与配置重算并落盘，核对夏普与 results.json 逐位一致，向实验台账追加记录。
6. 按入场月份聚类（复核 EC-4）：计划写“按入场月份聚类”，运行时用的是信号 K 线开盘所在月份；这里按入场时刻
   （信号 K 线收盘）重算机会配对与匹配随机的区间，并核对原口径能逐位复现 results.json。
8. 描述性核对（复核 LC-2、LC-3、LC-6）：B_audited 按入场高度分组的平均净 R；亏损月份的价格走势与 2022 年的对照；
   过滤类版本按运行时的保留概率与种子重建 300 次随机保留，统计零技能保留通过判定 (2)、(4)、(5)、(6) 的比例。
7. 判定第 (3) 条对 K 线内先后的敏感性（复核 LC-1）：第 (3) 条比较“真实信号上的改善”与“同一规则在随机时刻上的改善”，
   两边都按最坏路径撮合；随机时刻更常遇到 D2 的“同根分批后回落到保本”，最坏路径对这一差值并不保守。这里用一个
   可切换先后的单事件撮合副本（最坏路径时与 event_sim 逐位一致）重算 C1 / D1 / D2 的匹配随机对照：只把同根分批与
   保本按有利顺序处理（先低后高），以及把全部歧义 K 线都按有利顺序处理（止损与止盈 / POC 同根时先到高点、限价成交
   那一根先成交后止盈）。

用法：python -m ppvp_audit.post_run
输出：reports/audit_round/experiments/post_run_checks.json、experiments/rematerialised/*.csv，实验台账追加 R10 起的行。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from numba_compat import njit

from metrics import segment_report

from . import experiments as X
from . import nulls as N
from . import variants as V
from .common import OUT, is_run, sha256_file, write_json
from .engine_path import tick_size
from .event_sim import simulate_events

EXP = OUT / "experiments"
REM = EXP / "rematerialised"


# ---------------------------------------------------------------------------
# 1. 路径歧义
# ---------------------------------------------------------------------------
def ambiguity_split(tr: pd.DataFrame, h: np.ndarray, poc_mode: bool) -> dict:
    tr = tr[tr["exit_reason"] != "OPEN"]
    if len(tr) == 0:
        return {"n": 0}
    xb = tr["exit_bar"].to_numpy(dtype=np.int64)
    stop_exit = tr["exit_reason"].isin(["SL", "BE"]).to_numpy()
    tp = tr["tp"].to_numpy()
    poc = tr["poc"].to_numpy()
    pb = tr["partial_bar"].to_numpy() if "partial_bar" in tr else np.full(len(tr), -1)
    poc_ok = ((tr["poc"] > tr["entry_px"]) & (tr["poc"] < tr["tp"])).to_numpy()
    exit_sl_tp = stop_exit & (h[xb] >= tp)
    exit_sl_poc = stop_exit & poc_ok & (pb < 0) & (h[xb] >= poc)
    be_same_bar = (pb >= 0) & (pb == xb) & (tr["exit_reason"] == "BE").to_numpy()
    if "fill_bar" in tr:
        fb = tr["fill_bar"].to_numpy(dtype=np.int64)
        limit_fill = ~tr["entry_is_market"].to_numpy(dtype=bool)
        fill_tp = limit_fill & (h[fb] >= tp)
        fill_stop_same = tr["fill_and_exit_same_bar"].to_numpy(dtype=bool)
        # 成交那根触及止盈、按最坏路径不计，但下一根就按同一止盈价离场且没有资金费：单笔盈亏与先后无关，只差一根的离场时刻
        same_pnl = fill_tp & (tr["exit_reason"] == "TP").to_numpy() & (xb == fb + 1) & (tr["funding"].to_numpy() == 0.0)
    else:
        fill_tp = fill_stop_same = same_pnl = np.zeros(len(tr), bool)
    pnl = exit_sl_tp | fill_tp | ((exit_sl_poc | be_same_bar) if poc_mode else False)
    strict = pnl & ~(same_pnl & ~exit_sl_tp)
    orig = exit_sl_tp | exit_sl_poc | be_same_bar | fill_tp | fill_stop_same
    return {"n": int(len(tr)), "pnl_relevant": int(pnl.sum()), "pnl_relevant_share": float(pnl.mean()),
            "pnl_relevant_excluding_same_pnl_fill_bar_tp": int(strict.sum()),
            "as_in_results_json": int(orig.sum()),
            "components": {"exit_bar_stop_and_tp": int(exit_sl_tp.sum()), "exit_bar_stop_and_poc": int(exit_sl_poc.sum()),
                           "partial_and_breakeven_same_bar": int(be_same_bar.sum()), "limit_fill_bar_touches_tp": int(fill_tp.sum()),
                           "limit_fill_bar_touches_tp_but_same_trade_pnl": int(same_pnl.sum()),
                           "limit_fill_then_stop_same_bar": int(fill_stop_same.sum())},
            "stop_and_poc_counts_for_pnl": bool(poc_mode)}


def _logit_lr_test(y: np.ndarray, x_red: np.ndarray, x_extra: np.ndarray) -> dict:
    """逻辑回归的似然比检验：y ~ x_red + x_extra 对 y ~ x_red（x_red 含截距列）。不依赖 statsmodels。"""
    from scipy.optimize import minimize
    from scipy.stats import chi2

    def fit(X):
        def nll(beta):
            z = X @ beta
            return float(np.sum(np.logaddexp(0.0, z) - y * z))
        r = minimize(nll, np.zeros(X.shape[1]), method="BFGS")
        return r.x, -r.fun
    b_red, ll_red = fit(x_red)
    X = np.column_stack([x_red, x_extra])
    b_full, ll_full = fit(X)
    lr = 2.0 * (ll_full - ll_red)
    return {"coef_extra": float(b_full[-1]), "lr_stat": float(lr), "p_value": float(chi2.sf(lr, 1))}


# ---------------------------------------------------------------------------
# 2. C1 逐单描述统计
# ---------------------------------------------------------------------------
def c1_descriptives(base: dict, bcfg) -> dict:
    df1h, sig = base["df1h"], base["signals"]
    bars = np.flatnonzero(sig["signal"].to_numpy())
    sl, tp, poc = (sig[k].to_numpy()[bars] for k in ("sl", "tp", "poc_tp"))
    c = df1h["close"].to_numpy()[bars]
    slip = bcfg.slippage
    b = simulate_events(df1h, bars, sl, tp, poc, bcfg=bcfg)
    lim = V.c1_limit_prices(df1h, sig)[bars]
    e = simulate_events(df1h, bars, sl, tp, poc, limit=lim, tick=tick_size(df1h.index)[bars], valid=V.C1_VALID_BARS,
                        fill_rule="through", bcfg=bcfg)
    f = e["filled"].astype(bool)
    passive = lim < c
    ep = c * (1 + slip)
    fill_px = np.where(passive, lim, c) * (1 + slip)
    rr_base = (tp - ep) / (ep - sl)
    rr_c1 = (tp - fill_px) / (fill_px - sl)
    tp_b = b["reason"] == 2
    pos_b = b["ret"] > 0
    month = (df1h.index[bars] + pd.Timedelta(hours=1)).strftime("%Y-%m").to_numpy()
    # 未成交组与成交组在基线执行下平均事件收益之差的按月聚类区间（两组各自按月重抽样的簇差，固定种子）
    rng = np.random.default_rng(X.SEED_CLUSTER)
    months = np.unique(month)
    diffs = []
    for _ in range(10000):
        pick = rng.choice(months, size=len(months), replace=True)
        idx = np.concatenate([np.flatnonzero(month == m) for m in pick])
        u, fl = idx[~f[idx]], idx[f[idx]]
        if len(u) and len(fl):
            diffs.append(b["ret"][u].mean() - b["ret"][fl].mean())
    diffs = np.array(diffs)
    imp = np.where(f, e["ret"], 0.0) - b["ret"]
    # 止盈比例之差的按月聚类区间，以及控制 RR 之后“未成交”是否还解释止盈（逻辑回归似然比检验）
    rng2 = np.random.default_rng(X.SEED_CLUSTER + 1)
    tp_diffs = []
    for _ in range(10000):
        pick = rng2.choice(months, size=len(months), replace=True)
        idx = np.concatenate([np.flatnonzero(month == m) for m in pick])
        u, fl = idx[~f[idx]], idx[f[idx]]
        if len(u) and len(fl):
            tp_diffs.append(tp_b[u].mean() - tp_b[fl].mean())
    ones = np.ones(len(bars))
    lr_test = _logit_lr_test(tp_b.astype(float), np.column_stack([ones, np.log(rr_base)]), (~f).astype(float))
    return {
        "tp_share_gap_unfilled_minus_filled": float(tp_b[~f].mean() - tp_b[f].mean()),
        "tp_share_gap_ci_month_cluster": [float(np.percentile(tp_diffs, 2.5)), float(np.percentile(tp_diffs, 97.5))],
        "tp_on_unfilled_controlling_log_rr_logit_lr": lr_test,
        "parents": int(len(bars)), "event_level_filled": int(f.sum()), "event_level_unfilled": int((~f).sum()),
        "passive_limits": int(passive.sum()), "marketable_at_placement": int((~passive).sum()),
        "median_limit_below_close_bp_passive": float(np.median((1 - lim[passive] / c[passive]) * 1e4)),
        "median_entry_improvement_R_filled_passive": float(np.median(((ep - fill_px) / (ep - sl))[f & passive])),
        "median_rr_base_all": float(np.median(rr_base)), "median_rr_base_filled": float(np.median(rr_base[f])),
        "median_rr_base_unfilled": float(np.median(rr_base[~f])), "median_rr_c1_filled": float(np.median(rr_c1[f])),
        "base_execution_unfilled": {"n": int((~f).sum()), "tp_exits": int(tp_b[~f].sum()), "tp_share": float(tp_b[~f].mean()),
                                    "net_positive": int(pos_b[~f].sum()), "net_positive_share": float(pos_b[~f].mean()),
                                    "mean_event_return": float(b["ret"][~f].mean())},
        "base_execution_filled": {"n": int(f.sum()), "tp_exits": int(tp_b[f].sum()), "tp_share": float(tp_b[f].mean()),
                                  "net_positive": int(pos_b[f].sum()), "net_positive_share": float(pos_b[f].mean()),
                                  "mean_event_return": float(b["ret"][f].mean())},
        "random_walk_tp_reference_mean": {"unfilled": float((1 / (1 + rr_base[~f])).mean()),
                                          "filled": float((1 / (1 + rr_base[f])).mean())},
        "fisher_exact_p_tp_share": float(fisher_exact([[tp_b[~f].sum(), (~tp_b[~f]).sum()], [tp_b[f].sum(), (~tp_b[f]).sum()]])[1]),
        "unfilled_minus_filled_mean_return": float(b["ret"][~f].mean() - b["ret"][f].mean()),
        "unfilled_minus_filled_ci_month_cluster": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "per_opportunity_improvement": float(imp.mean()),
        "per_opportunity_improvement_from_filled_entries": float(imp[f].sum() / len(bars)),
        "per_opportunity_improvement_from_not_trading_unfilled": float(imp[~f].sum() / len(bars)),
        "note": "event level (each parent independent, no funding); 'filled' = filled under the C1 'through' rule",
    }


# ---------------------------------------------------------------------------
# 3 / 6. 机会配对与匹配随机：复现 results.json，再按入场月份与同类比较重算
# ---------------------------------------------------------------------------
def _opp_pairs(base: dict, bcfg, v: str, month: np.ndarray) -> dict:
    df1h, sig = base["df1h"], base["signals"]
    bars = np.flatnonzero(sig["signal"].to_numpy())
    sl, tp, poc = (sig[k].to_numpy()[bars] for k in ("sl", "tp", "poc_tp"))
    b = simulate_events(df1h, bars, sl, tp, poc, bcfg=bcfg)
    if v in ("A1", "A2", "A3", "E1"):
        m = V.filter_masks(base, bcfg)[f"pass_{v}"].to_numpy()[bars]
        ret_v = np.where(m, b["ret"], 0.0)
    elif v == "C1":
        lim = V.c1_limit_prices(df1h, sig)[bars]
        e = simulate_events(df1h, bars, sl, tp, poc, limit=lim, tick=tick_size(df1h.index)[bars], valid=V.C1_VALID_BARS,
                            fill_rule="through", bcfg=bcfg)
        ret_v = np.where(e["filled"], e["ret"], 0.0)
    else:
        e = simulate_events(df1h, bars, sl, tp, poc, partial=True, frac=V.D_PARTIAL_FRAC, move_be=(v == "D2"), bcfg=bcfg)
        ret_v = e["ret"]
    ci = N.cluster_bootstrap_mean(ret_v - b["ret"], month, seed=X.SEED_CLUSTER)
    return {"mean_diff": ci["mean"], "ci": [ci["lo"], ci["hi"]]}


def _matched(base: dict, bcfg, v: str, month: np.ndarray, passive_only: bool = False) -> dict:
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
                                valid=V.C1_VALID_BARS, fill_rule="through", bcfg=bcfg)
            rv = np.where(x["filled"], x["ret"], 0.0)
        else:
            x = simulate_events(df1h, dec, sl_, tp_, poc_, partial=True, frac=V.D_PARTIAL_FRAC, move_be=(v == "D2"), bcfg=bcfg)
            rv = x["ret"]
        return rv - b["ret"]
    real = improvement(bars, d_sl, d_tp, d_poc)
    exp_rand = np.empty(len(bars))
    kept = []
    for j, pool in enumerate(pools):
        if passive_only:
            pool = pool[0.5 * (h[pool] + l[pool]) < c[pool]]
        kept.append(len(pool))
        k = len(pool)
        exp_rand[j] = improvement(pool, np.full(k, d_sl[j]), np.full(k, d_tp[j]), np.full(k, d_poc[j])).mean()
    ci = N.cluster_bootstrap_mean(real - exp_rand, month, seed=X.SEED_MATCH)
    return {"mean_improvement_real": float(real.mean()), "mean_improvement_random": float(exp_rand.mean()),
            "real_minus_random": ci["mean"], "ci": [ci["lo"], ci["hi"]], "pool_size_median": float(np.median(kept)),
            "pool_size_min": int(np.min(kept))}


def clustering_and_like_for_like(base: dict, bcfg, res: dict) -> dict:
    df1h, sig = base["df1h"], base["signals"]
    bars = np.flatnonzero(sig["signal"].to_numpy())
    m_run = df1h.index[bars].strftime("%Y-%m").to_numpy()                                 # 运行时的口径
    m_entry = (df1h.index[bars] + pd.Timedelta(hours=1)).strftime("%Y-%m").to_numpy()     # 计划的口径（入场 = 信号收盘）
    out = {"parents_whose_signal_month_differs_from_entry_month": int((m_run != m_entry).sum())}
    for v in ("A1", "A2", "A3", "E1", "C1", "D1", "D2"):
        a = _opp_pairs(base, bcfg, v, m_run)
        ref = res[v]["opportunity_pairs"]
        if not (a["mean_diff"] == ref["mean_diff"] and a["ci"] == ref["diff_ci_month_cluster"]):
            raise RuntimeError(f"opportunity pairs for {v} do not reproduce results.json")
        out[f"opportunity_pairs_{v}"] = {"run_signal_month": a, "entry_month": _opp_pairs(base, bcfg, v, m_entry)}
    for v in ("C1", "D1", "D2"):
        a = _matched(base, bcfg, v, m_run)
        ref = res[v]["matched_random"]
        if not (a["real_minus_random"] == ref["real_minus_random"] and a["ci"] == ref["ci_month_cluster"]):
            raise RuntimeError(f"matched random for {v} does not reproduce results.json")
        out[f"matched_random_{v}"] = {"run_signal_month": a, "entry_month": _matched(base, bcfg, v, m_entry)}
    out["matched_random_C1_passive_random_limits_only"] = _matched(base, bcfg, "C1", m_entry, passive_only=True)
    return out


# ---------------------------------------------------------------------------
# 7. K 线内先后敏感性：可切换先后的单事件撮合副本
# ---------------------------------------------------------------------------
FAV_BE_SAME_BAR, FAV_HIGH_FIRST, FAV_FILL_THEN_TP = 1, 2, 8


@njit(cache=False)
def _events_order(o, h, l, c, dec_bar, sl_a, tp_a, poc_a, lim_a, tick_a, valid, through, partial, frac, move_be,
                  taker, maker, slip, end_bar, fav):
    """event_sim._events 的副本，fav = 0 时逐位相同。fav 位：1 = 同根分批后回落到保本价按“先低后高”处理（不离场）；
    2 = 同一根同时触及止损与止盈 / POC 时，若开盘在止损之上，按“先到高点”处理（先分批 / 止盈，余仓再按止损价离场）；
    8 = 限价成交那一根也触及止盈时按“先成交、后止盈”处理。"""
    m = dec_bar.shape[0]
    ret = np.zeros(m)
    filled = np.zeros(m, np.bool_)
    for j in range(m):
        b = dec_bar[j]
        sl = sl_a[j]
        tp = tp_a[j]
        poc = poc_a[j]
        lim = lim_a[j]
        if lim != lim or lim >= c[b]:
            e_ref = c[b]
            efill = c[b] * (1.0 + slip)
            fee_in = taker
            fb = b
            start = b + 1
        else:
            fb = -1
            last = min(b + valid, end_bar - 1)
            for i in range(b + 1, last + 1):
                hit = (l[i] <= lim - tick_a[j]) if through else (l[i] <= lim)
                if hit:
                    fb = i
                    break
            if fb < 0:
                continue
            e_ref = lim
            efill = lim * (1.0 + slip)
            fee_in = maker
            start = fb
        if not efill > sl:
            continue
        filled[j] = True
        q = 1.0
        stop = sl
        part = False
        pnl = -efill * fee_in
        done = False
        for i in range(start, end_bar):
            first = (i == fb) and (lim == lim and lim < c[b])
            elig = partial and (not part) and poc > efill and poc < tp and h[i] >= poc
            if first and (fav & 8) != 0 and h[i] >= tp:
                px = tp * (1.0 - slip)
                pnl += q * (px - efill) - q * px * maker
                done = True
                break
            if (fav & 2) != 0 and (not first) and l[i] <= stop and o[i] > stop and (h[i] >= tp or elig):
                if elig:
                    q1 = q * frac
                    px = poc * (1.0 - slip)
                    pnl += q1 * (px - efill) - q1 * px * maker
                    q -= q1
                    part = True
                    if move_be:
                        stop = efill
                if h[i] >= tp:
                    px = tp * (1.0 - slip)
                    pnl += q * (px - efill) - q * px * maker
                    done = True
                    break
                px = stop * (1.0 - slip)
                pnl += q * (px - efill) - q * px * taker
                done = True
                break
            if l[i] <= stop:
                px_ref = stop if first else min(o[i], stop)
                px = px_ref * (1.0 - slip)
                pnl += q * (px - efill) - q * px * taker
                done = True
                break
            if first:
                continue
            if elig:
                q1 = q * frac
                px = poc * (1.0 - slip)
                pnl += q1 * (px - efill) - q1 * px * maker
                q -= q1
                part = True
                if move_be:
                    stop = efill
                    if l[i] <= stop and (fav & 1) == 0:
                        px = stop * (1.0 - slip)
                        pnl += q * (px - efill) - q * px * taker
                        done = True
                        break
            if h[i] >= tp:
                px = tp * (1.0 - slip)
                pnl += q * (px - efill) - q * px * maker
                done = True
                break
        if not done:
            i = end_bar - 1
            px = c[i] * (1.0 - slip)
            pnl += q * (px - efill) - q * px * taker
        ret[j] = pnl / efill
    return filled, ret


def _sim_order(df1h, dec, sl, tp, poc=None, limit=None, tick=None, valid=3, partial=False, frac=0.5, move_be=False,
               bcfg=None, fav=0):
    m = len(dec)
    f = lambda x, fill: np.full(m, fill) if x is None else np.asarray(x, dtype=np.float64)  # noqa: E731
    return _events_order(df1h["open"].to_numpy(np.float64), df1h["high"].to_numpy(np.float64),
                         df1h["low"].to_numpy(np.float64), df1h["close"].to_numpy(np.float64), np.asarray(dec, dtype=np.int64),
                         f(sl, np.nan), f(tp, np.nan), f(poc, np.nan), f(limit, np.nan), f(tick, 0.0), int(valid), True,
                         bool(partial), float(frac), bool(move_be), float(bcfg.taker_fee), float(bcfg.maker_fee),
                         float(bcfg.slippage), int(len(df1h)), int(fav))


def order_sensitivity(base: dict, bcfg, res: dict) -> dict:
    df1h, sig = base["df1h"], base["signals"]
    c = df1h["close"].to_numpy(dtype=np.float64)
    h, l = df1h["high"].to_numpy(dtype=np.float64), df1h["low"].to_numpy(dtype=np.float64)
    bars = np.flatnonzero(sig["signal"].to_numpy())
    ep = c[bars] * (1 + bcfg.slippage)
    d_sl = 1 - sig["sl"].to_numpy()[bars] / ep
    d_tp = sig["tp"].to_numpy()[bars] / ep - 1
    d_poc = sig["poc_tp"].to_numpy()[bars] / ep - 1
    pools, _ = N.candidate_sets(bars, N.bar_features(base), df1h)
    ticks = tick_size(df1h.index)
    month = df1h.index[bars].strftime("%Y-%m").to_numpy()      # 与运行时相同的口径，便于和 results.json 对照

    def improvement(v, dec, dsl, dtp, dpoc, fav_v, fav_b):
        e = c[dec] * (1 + bcfg.slippage)
        sl_, tp_, poc_ = e * (1 - dsl), e * (1 + dtp), e * (1 + dpoc)
        _, rb = _sim_order(df1h, dec, sl_, tp_, poc_, bcfg=bcfg, fav=fav_b)
        if v == "C1":
            fv, rv = _sim_order(df1h, dec, sl_, tp_, poc_, limit=0.5 * (h[dec] + l[dec]), tick=ticks[dec],
                                valid=V.C1_VALID_BARS, bcfg=bcfg, fav=fav_v)
            rv = np.where(fv, rv, 0.0)
        else:
            _, rv = _sim_order(df1h, dec, sl_, tp_, poc_, partial=True, frac=V.D_PARTIAL_FRAC, move_be=(v == "D2"),
                               bcfg=bcfg, fav=fav_v)
        return rv - rb

    def matched(v, fav_v, fav_b):
        real = improvement(v, bars, d_sl, d_tp, d_poc, fav_v, fav_b)
        rnd = np.array([improvement(v, pool, np.full(len(pool), d_sl[j]), np.full(len(pool), d_tp[j]),
                                    np.full(len(pool), d_poc[j]), fav_v, fav_b).mean() for j, pool in enumerate(pools)])
        ci = N.cluster_bootstrap_mean(real - rnd, month, seed=X.SEED_MATCH)
        return {"mean_improvement_real": float(real.mean()), "mean_improvement_random": float(rnd.mean()),
                "real_minus_random": ci["mean"], "ci": [ci["lo"], ci["hi"]], "passes_criterion_3": bool(ci["lo"] > 0)}
    out = {}
    all_fav = FAV_BE_SAME_BAR | FAV_HIGH_FIRST | FAV_FILL_THEN_TP
    for v in ("C1", "D1", "D2"):
        w = matched(v, 0, 0)
        ref = res[v]["matched_random"]
        if not (w["real_minus_random"] == ref["real_minus_random"] and w["ci"] == ref["ci_month_cluster"]):
            raise RuntimeError(f"order-switchable copy does not reproduce results.json for {v}")
        out[v] = {"worst_case_as_run": w, "all_ambiguous_bars_favourable": matched(v, all_fav, FAV_HIGH_FIRST)}
        if v == "D2":
            out[v]["only_same_bar_partial_then_breakeven_favourable"] = matched(v, FAV_BE_SAME_BAR, 0)
    return out


# ---------------------------------------------------------------------------
# 8. 机制假设的描述性核对（复核 LC-2、LC-3）与过滤类版本的零技能判定通过率（复核 LC-6）
# ---------------------------------------------------------------------------
def entry_height_outcomes() -> dict:
    """B_audited 109 笔按入场高度分组的平均净 R（含资金费）：检验“入场偏高导致亏损”的说法是否与账本一致。"""
    t = pd.read_csv(OUT / "trades.csv")
    out = {}

    def grp(name, mask):
        out[name] = {"n": int(mask.sum()), "mean_net_R": float(t.loc[mask, "net_R"].mean()),
                     "n_other": int((~mask).sum()), "mean_net_R_other": float(t.loc[~mask, "net_R"].mean())}
    grp("rr_fill_lt_1", t["rr_fill"] < 1)
    grp("normalized_entry_position_gt_0.5", t["normalized_entry_position"] > 0.5)
    grp("entry_above_poc", t["entry_px"] > t["POC"])
    grp("entry_above_band_top", t["entry_above_band"].astype(bool))
    q = t["confirmation_range_atr"].quantile([1 / 3, 2 / 3]).to_numpy()
    terc = np.digitize(t["confirmation_range_atr"], q)
    out["confirmation_range_atr_terciles"] = [
        {"median_atr": float(t.loc[terc == k, "confirmation_range_atr"].median()), "n": int((terc == k).sum()),
         "mean_net_R": float(t.loc[terc == k, "net_R"].mean())} for k in range(3)]
    return out


def loss_period_composition(df1h: pd.DataFrame) -> dict:
    """亏损月份的价格走势与 2022 年的对照（描述 H2 时只用实测量）。月收益 = 月末收盘 / 上月末收盘 − 1（2020-01 用首根开盘）。"""
    close_m = df1h["close"].resample("MS").last()
    prev = close_m.shift(1)
    prev.iloc[0] = float(df1h["open"].iloc[0])
    mret = close_m / prev - 1.0
    mret.index = mret.index.strftime("%Y-%m")
    t = pd.read_csv(OUT / "trades.csv", parse_dates=["entry_time"])
    t["m"] = t["entry_time"].dt.strftime("%Y-%m")
    total = float(t["net_pnl"].sum())
    pnl_m = t.groupby("m")["net_pnl"].sum()
    loss5 = list(X.LOSS_MONTHS)
    w = t[t["m"].isin(loss5)]
    worst10 = mret.nsmallest(10)
    neg = mret[mret < 0].index
    y22 = t[t["entry_time"].dt.year == 2022]
    return {
        "btc_month_return_loss_months": {m: float(mret[m]) for m in loss5},
        "loss_month_trades": int(len(w)), "loss_month_pnl": float(w["net_pnl"].sum()),
        "loss_month_trades_above_sma200": int(w["regime_above_sma200"].fillna(0).astype(bool).sum()),
        "trades_2022": int(len(y22)), "pnl_2022": float(y22["net_pnl"].sum()), "share_of_total_loss_2022": float(y22["net_pnl"].sum() / total),
        "mean_net_R_2022": float(y22["net_R"].mean()), "btc_return_2022": float(close_m["2022-12"].iloc[0] / close_m["2021-12"].iloc[0] - 1.0),
        "worst10_btc_months": {m: float(r) for m, r in worst10.items()},
        "pnl_in_each_worst10_month": {m: float(pnl_m.get(m, 0.0)) for m in worst10.index},
        "trades_in_each_worst10_month": {m: int((t["m"] == m).sum()) for m in worst10.index},
        "worst10_months_also_loss_months": [m for m in worst10.index if m in loss5],
        "pnl_in_worst10_months_outside_loss_months": float(pnl_m.reindex([m for m in worst10.index if m not in loss5]).fillna(0).sum()),
        "share_of_total_loss_in_negative_return_months": float(pnl_m.reindex(neg).fillna(0).sum() / total),
        "n_negative_return_months": int(len(neg)), "total_net_pnl": total,
    }


def thinning_criteria_pass_rates(base: dict, base_s: dict, bcfg, res: dict) -> dict:
    """过滤类版本：用运行时同样的保留概率与种子重建 300 次随机保留，统计零技能保留能否通过判定 (2)、(4)、(5)、(6)。"""
    import attribution as A
    from backtester import run_backtest
    from metrics import daily_returns
    from .common import load_is_market
    _, fund, _ = load_is_market()
    sig = base["signals"]
    sig_bars = np.flatnonzero(sig["signal"].to_numpy(dtype=bool))
    rb_ret = res["base"]["total_net_return"]
    sb = X.stressed(bcfg)

    def sh(r):
        a = daily_returns(X._daily(r)).to_numpy(dtype=np.float64)
        return float(a.mean() / a.std(ddof=1) * np.sqrt(A.ANN))
    sh_base_s = sh(base_s["result"])
    out = {}
    for v in ("A1", "A2", "A3", "E1"):
        k = V.VERSIONS.index(v)
        keep = res[v]["matched_random"]["keep"]
        rng = np.random.default_rng(X.SEED_THIN + k)
        u = rng.random((300, len(sig_bars)))
        c2 = c4 = c5 = c6 = allc = 0
        n_tr = []
        for i in range(300):
            col = sig["signal"].to_numpy(dtype=bool).copy()
            col[sig_bars[u[i] >= keep]] = False
            s_i = sig.assign(signal=col)
            r = run_backtest(base["df1h"], s_i, bcfg, fund)
            rs = run_backtest(base["df1h"], s_i, sb, fund)
            rep = segment_report(r, ("IS",))["IS"]
            per = {x["period"]: x for x in X.period_table(r, base["result"])}
            bc = X.box_concentration(r, base["result"], s_i, sig)
            ok2 = rep["total_return"] - rb_ret > 0
            ok4 = per["2020-2021"]["sharpe_diff"] > 0 and per["2022-2023"]["sharpe_diff"] > 0
            ok5 = sh(rs) - sh_base_s > 0
            ncl = int((r.trades["exit_reason"] != "OPEN").sum())
            ok6 = ncl >= 30 and bc["pnl_diff_without_top3"] > 0
            c2 += ok2; c4 += ok4; c5 += ok5; c6 += ok6; allc += ok2 and ok4 and ok5 and ok6
            n_tr.append(ncl)
        out[v] = {"keep": keep, "seed": X.SEED_THIN + k, "reps": 300, "mean_trades": float(np.mean(n_tr)),
                  "share_pass_2": c2 / 300, "share_pass_4": c4 / 300, "share_pass_5": c5 / 300, "share_pass_6": c6 / 300,
                  "share_pass_2_4_5_6": allc / 300}
    return out


# ---------------------------------------------------------------------------
# 4. D1 / D2 适用性
# ---------------------------------------------------------------------------
def d_applicability(v: str) -> dict:
    t = pd.read_csv(EXP / f"{v}_trades.csv")
    app = (t["poc"] > t["entry_px"]) & (t["poc"] < t["tp"])
    part = t["partial_bar"] >= 0
    be = t["exit_reason"] == "BE"
    return {"trades": int(len(t)), "applicable": int(app.sum()), "not_applicable": int((~app).sum()),
            "partial_filled": int(part.sum()), "breakeven_exits": int(be.sum()),
            "partial_and_breakeven_same_bar": int((part & be & (t["partial_bar"] == t["exit_bar"])).sum()),
            "exit_reasons": t["exit_reason"].value_counts().to_dict()}


# ---------------------------------------------------------------------------
# 5. 补记运行
# ---------------------------------------------------------------------------
def rematerialise(base: dict, base_s: dict, scfg, bcfg, res: dict) -> list[dict]:
    REM.mkdir(parents=True, exist_ok=True)
    rows = []

    def save(name: str, tr: pd.DataFrame) -> tuple[str, str]:
        p = REM / f"{name}.csv"
        tr.to_csv(p, index=False, float_format="%.10g")
        return f"reports/audit_round/experiments/rematerialised/{name}.csv", sha256_file(p)

    def row(rid, version, spec_key, bkey, sharpe_now, sharpe_ref, path, sha, note):
        if sharpe_now != sharpe_ref:
            raise RuntimeError(f"{version}: Sharpe {sharpe_now} differs from results.json {sharpe_ref}")
        rows.append({"run_id": rid, "version": version, "status": "re-materialised post-run (identical Sharpe)",
                     "notes": note, "spec_key": spec_key, "backtest_key": bkey, "is_sharpe": sharpe_now,
                     "trades_file": path, "trades_sha256": sha})
    sb = X.stressed(bcfg)
    k = 10
    path, sha = save("B_audited_trades", base["result"].trades)
    row(f"R{k:02d}", "B_audited", scfg.key(), bcfg.key(), base["report"]["IS"]["sharpe"], res["base"]["sharpe"], path, sha,
        "comparison baseline of the run (bit-identical to B_original)")
    k += 1
    path, sha = save("B_audited_trades_2x_costs", base_s["result"].trades)
    row(f"R{k:02d}", "B_audited @2x costs", scfg.key(), sb.key(), base_s["report"]["IS"]["sharpe"],
        res["A1"]["cost_stress"]["sharpe_base"], path, sha, "taker 10bp, maker 4bp, slippage 4bp; real funding")
    k += 1
    for v in V.VERSIONS:
        rs = X.run_version(v, base_s, scfg, sb)
        path, sha = save(f"{v}_trades_2x_costs", rs["out"]["result"].trades)
        row(f"R{k:02d}", f"{v} @2x costs", V.spec_for(v, scfg, bcfg).key(), sb.key(), rs["out"]["report"]["IS"]["sharpe"],
            res[v]["cost_stress"]["sharpe_version"], path, sha, "criterion (5) input")
        k += 1
    rt = X.run_version("C1", base, scfg, bcfg, fill_rule="touch")
    path, sha = save("C1_touch_trades", rt["out"]["result"].trades)
    row(f"R{k:02d}", "C1 touch sensitivity", V.spec_for("C1", scfg, bcfg).key(), bcfg.key(), rt["out"]["report"]["IS"]["sharpe"],
        res["C1"]["touch_sensitivity"]["metrics"]["sharpe"], path, sha, "fill rule low <= L instead of low <= L - tick")
    return rows


def run() -> dict:
    res = json.loads((EXP / "results.json").read_text())
    scfg, bcfg = X.base_configs()
    base = is_run(scfg, bcfg)
    base_s = is_run(scfg, X.stressed(bcfg))
    h = base["df1h"]["high"].to_numpy(dtype=np.float64)
    amb = {"B_audited": ambiguity_split(base["result"].trades, h, poc_mode=False)}
    for v in V.VERSIONS:
        amb[v] = ambiguity_split(pd.read_csv(EXP / f"{v}_trades.csv"), h, poc_mode=v in ("D1", "D2"))
        if amb[v]["as_in_results_json"] != res[v]["metrics"]["intrabar_ambiguity"]["ambiguous"]:
            raise RuntimeError(f"{v}: original ambiguity count not reproduced")
    rows = rematerialise(base, base_s, scfg, bcfg, res)
    out = {
        "note": "post-run checks; results.json is left as written by the run",
        "results_json_sha256": sha256_file(EXP / "results.json"),
        "intrabar_ambiguity": amb,
        "c1_descriptives": c1_descriptives(base, bcfg),
        "d_applicability": {v: d_applicability(v) for v in ("D1", "D2")},
        "clustering_and_like_for_like": clustering_and_like_for_like(base, bcfg, res),
        "criterion3_intrabar_order_sensitivity": order_sensitivity(base, bcfg, res),
        "entry_height_outcomes_B_audited": entry_height_outcomes(),
        "loss_period_composition": loss_period_composition(base["df1h"]),
        "filter_versions_zero_skill_pass_rates": thinning_criteria_pass_rates(base, base_s, bcfg, res),
        "rematerialised_runs": rows,
    }
    write_json(EXP / "post_run_checks.json", out)
    led = pd.read_csv(OUT / "experiment_ledger.csv")
    led = led[~led["run_id"].isin([r["run_id"] for r in rows])]
    led = pd.concat([led, pd.DataFrame(rows)], ignore_index=True)
    led.to_csv(OUT / "experiment_ledger.csv", index=False)
    return out


if __name__ == "__main__":
    o = run()
    print(json.dumps({k: o[k] for k in ("intrabar_ambiguity", "c1_descriptives", "d_applicability")}, indent=1, default=str)[:6000])
