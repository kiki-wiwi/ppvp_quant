"""
ppvp_audit/nulls.py — 随机基准复核（用户第 5 节）：原方法复现之外的匹配对照、组合级对照与聚类区间。IS only。

事先写定的规则（种子 = SEED + 10 起）：
* 事件级匹配对照（matched_event_null）：对每笔真实交易（信号 K 线 b），候选 K 线 c 须同时满足
  (1) c 在 IS 内且之后至少还有一根 K 线可撮合；(2) |c − b| > 24 根且 c 距 b 不超过 30 天（候选不足 20 根时放宽到 60 天，
  仍不足则 90 天，并记录放宽次数）；(3) 决策时已知的市场状态与 b 相同：日线收盘相对 SMA200 的位置（above_sma200）与
  30 日波动率在过去一年中的三分位（vol30d_pct，均为 c 之前已收盘日线的值）；(4) c 收盘时基线没有持仓（可交易时刻）。
  不使用候选之后的任何信息（收益、持有时间）筛选候选。几何沿用真实交易相对成交价的止损 / 止盈百分比距离，
  撮合、手续费、滑点与原方法相同（不含资金费），持有到止损 / 止盈或 IS 末（按最后收盘离场）。
  每笔交易的零假设期望 = 其全部候选的平均 R；抽样分布按“每笔交易各抽一个候选”重抽样 5,000 次。
  交易可重叠，这是事件级比较，不是组合净值。
* 组合级匹配对照（matched_portfolio_null）：把 IS 的每个原始信号（120 个，含被跳过与被拦下的）换成一个按上述规则匹配的
  随机 K 线（几何沿用该信号的百分比距离；被拦下的信号按其收盘口径的距离），用真实回测器运行（持仓互斥、
  1 倍名义、复利、全部成本含资金费），重复 1,000 次，比较日度夏普、总收益与平均 R。
* 区间：平均差值与平均净 R 用按入场月份聚类的自助法（月份有放回重抽样，10,000 次，百分位区间）；同时报告按箱体聚类。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import attribution as A
from backtester import run_backtest
from config import BASELINE_BACKTEST, BASELINE_STRATEGY, IS_END, IS_START
from metrics import daily_equity, daily_returns, sharpe

from .common import SEED, is_run, load_is_market

DAY_NS = np.int64(86400 * 10**9)


def bar_features(out: dict) -> pd.DataFrame:
    """每根 IS 1H K 线收盘决策时已知的状态特征（日线来源，因果）与基线是否持仓。"""
    df1h = out["df1h"]
    _, _, spot = load_is_market()
    bars = np.arange(len(df1h))
    reg = A.regime_features(df1h, out["df4h"], spot, bars)
    tr = out["result"].trades
    inpos = np.zeros(len(df1h), bool)
    for e, x in zip(tr["entry_bar"], tr["exit_bar"]):
        inpos[int(e):int(x)] = True                       # [entry_bar, exit_bar)：收盘时仍持仓
    f = pd.DataFrame({"above_sma200": reg["above_sma200"].to_numpy(),
                      "vol_tercile": np.floor(np.clip(reg["vol30d_pct"].to_numpy(), 0, 0.999999) * 3),
                      "flat_at_close": ~inpos}, index=df1h.index)
    return f


def candidate_sets(sig_bars: np.ndarray, feats: pd.DataFrame, df1h: pd.DataFrame, min_pool: int = 20,
                   windows=(30, 60, 90), excl: int = 24) -> tuple[list[np.ndarray], list[int]]:
    t = df1h.index.as_unit("ns").asi8
    n = len(df1h)
    ok_feat = feats["above_sma200"].notna().to_numpy() & np.isfinite(feats["vol_tercile"].to_numpy())
    base_ok = ok_feat & feats["flat_at_close"].to_numpy()
    base_ok[n - 1] = False                                  # 至少还有一根 K 线可撮合
    sets, used_w = [], []
    a_s = feats["above_sma200"].to_numpy()
    v_t = feats["vol_tercile"].to_numpy()
    for b in sig_bars:
        same = base_ok & (a_s == a_s[b]) & (v_t == v_t[b])
        idx = np.arange(n)
        far = np.abs(idx - b) > excl
        pool = np.array([], dtype=np.int64)
        w_used = -1
        for w in windows:
            near = np.abs(t - t[b]) <= w * DAY_NS
            pool = np.flatnonzero(same & far & near)
            w_used = w
            if len(pool) >= min_pool:
                break
        sets.append(pool)
        used_w.append(w_used)
    return sets, used_w


def _r_for_bars(df1h, bars, d_sl, d_tp, bcfg, end_bar):
    c = df1h["close"].to_numpy(dtype=np.float64)
    ef = c[bars] * (1.0 + bcfg.slippage)
    _, _, r = A.bracket_outcomes(df1h, bars, ef * (1 - d_sl), ef * (1 + d_tp), end_bar, bcfg)
    return r


def matched_event_null(out: dict, tf: pd.DataFrame, n_rep: int = 5000, seed: int = SEED + 10) -> dict:
    df1h = out["df1h"]
    bcfg = out["result"].config
    feats = bar_features(out)
    end_bar = len(df1h)
    sets, used_w = candidate_sets(tf["entry_bar"].to_numpy(), feats, df1h)
    rng = np.random.default_rng(seed)
    exp_r = np.empty(len(tf))
    draws = np.empty((n_rep, len(tf)))
    for j, (pool, (_, row)) in enumerate(zip(sets, tf.iterrows())):
        r = _r_for_bars(df1h, pool, float(row["d_sl"]), float(row["d_tp"]), bcfg, end_bar)
        exp_r[j] = r.mean()
        draws[:, j] = r[rng.integers(0, len(r), n_rep)]
    actual = tf["r_ex_funding"].to_numpy()
    null_means = draws.mean(axis=1)
    p = float((1 + (null_means <= actual.mean() + 1e-12).sum()) / (n_rep + 1))
    return {"pool_sizes": [len(s) for s in sets], "window_days_used": used_w, "per_trade_expected_r": exp_r,
            "null_mean_r_mean": float(null_means.mean()), "null_mean_r_p05": float(np.percentile(null_means, 5)),
            "null_mean_r_p95": float(np.percentile(null_means, 95)), "actual_mean_r": float(actual.mean()),
            "p_lower": p, "n_rep": n_rep, "seed": seed, "null_dist": null_means}


def cluster_bootstrap_mean(x: np.ndarray, clusters: np.ndarray, n_boot: int = 10000, seed: int = SEED + 11) -> dict:
    """按簇有放回重抽样的平均值区间（簇内全部交易一起进出），百分位 95%。"""
    rng = np.random.default_rng(seed)
    cl = pd.Series(x).groupby(pd.Series(clusters)).agg(["sum", "count"])
    s, k = cl["sum"].to_numpy(), cl["count"].to_numpy()
    m = len(cl)
    idx = rng.integers(0, m, size=(n_boot, m))
    means = s[idx].sum(axis=1) / k[idx].sum(axis=1)
    return {"mean": float(np.mean(x)), "lo": float(np.percentile(means, 2.5)), "hi": float(np.percentile(means, 97.5)),
            "p_le_0": float((means <= 0).mean()), "n_clusters": int(m), "n": int(len(x))}


def matched_portfolio_null(out: dict, n_rep: int = 1000, seed: int = SEED + 12) -> dict:
    df1h = out["df1h"]
    sig = out["signals"]
    bcfg = out["result"].config
    _, fund, _ = load_is_market()
    feats = bar_features(out)
    raw = np.flatnonzero(sig["sig_raw"].to_numpy())
    c = df1h["close"].to_numpy(dtype=np.float64)
    ef = c[raw] * (1 + bcfg.slippage)
    d_sl = (ef - sig["sl"].to_numpy()[raw]) / ef
    d_tp = (sig["tp"].to_numpy()[raw]) / ef - 1.0
    sets, used_w = candidate_sets(raw, feats, df1h)
    rng = np.random.default_rng(seed)
    n = len(df1h)
    sharpe_d, tot, avg_r, ntr = [], [], [], []
    for _ in range(n_rep):
        pick = np.array([s[rng.integers(0, len(s))] for s in sets])
        order = np.argsort(pick, kind="stable")
        s_sig = np.zeros(n, bool)
        sl = np.full(n, np.nan)
        tp = np.full(n, np.nan)
        for j in order:
            b = pick[j]
            if s_sig[b]:
                continue                                  # 两个原始信号匹配到同一根 K 线：只保留一个
            e = c[b] * (1 + bcfg.slippage)
            s_sig[b] = d_tp[j] > 0                        # 被拦下的信号（VAH 不高于收盘）在随机时刻同样不交易
            sl[b] = e * (1 - d_sl[j])
            tp[b] = e * (1 + d_tp[j])
        frame = pd.DataFrame({"signal": s_sig, "sl": sl, "tp": tp, "poc_tp": np.nan}, index=df1h.index)
        res = run_backtest(df1h, frame, bcfg, fund)
        d = daily_equity(res)
        r = daily_returns(d[(d.index >= IS_START - pd.Timedelta(days=1)) & (d.index <= IS_END - pd.Timedelta(days=1))])
        sharpe_d.append(sharpe(r))
        tot.append(float(d.iloc[-1] / d.iloc[0] - 1))
        cl = res.trades[res.trades["exit_reason"] != "OPEN"]
        avg_r.append(float(cl["r_multiple"].mean()) if len(cl) else np.nan)
        ntr.append(int(len(cl)))
    real = out["report"]["IS"]
    sharpe_d, tot, avg_r = np.array(sharpe_d), np.array(tot), np.array(avg_r)
    return {"n_rep": n_rep, "seed": seed, "raw_signals": int(len(raw)),
            "window_days_used_counts": pd.Series(used_w).value_counts().to_dict(),
            "real": {"sharpe": real["sharpe"], "total_return": real["total_return"], "avg_r": real["avg_r"],
                     "trades": real["trades"]},
            "null_sharpe_mean": float(np.nanmean(sharpe_d)), "null_sharpe_p05": float(np.nanpercentile(sharpe_d, 5)),
            "null_sharpe_p95": float(np.nanpercentile(sharpe_d, 95)),
            "p_sharpe_lower": float((1 + (sharpe_d <= real["sharpe"]).sum()) / (n_rep + 1)),
            "null_total_return_mean": float(np.mean(tot)), "null_avg_r_mean": float(np.nanmean(avg_r)),
            "p_avg_r_lower": float((1 + (avg_r <= real["avg_r"]).sum()) / (n_rep + 1)),
            "null_trades_mean": float(np.mean(ntr)), "sharpe_dist": sharpe_d}


def run() -> dict:
    from .common import OUT, write_json
    from .repro_figures import baseline_frames
    out, tf = baseline_frames()
    ev = matched_event_null(out, tf)
    month = tf["entry_time"].dt.strftime("%Y-%m").to_numpy()
    box = tf["box_id"].to_numpy()
    d = tf["r_ex_funding"].to_numpy() - ev["per_trade_expected_r"]
    res = {
        "matched_event": {k: v for k, v in ev.items() if k not in ("per_trade_expected_r", "null_dist", "pool_sizes", "window_days_used")},
        "pool_size_quantiles": pd.Series(ev["pool_sizes"]).quantile([0, 0.1, 0.5, 0.9, 1]).to_dict(),
        "window_days_used_counts": pd.Series(ev["window_days_used"]).value_counts().to_dict(),
        "paired_difference_vs_matched_by_month": cluster_bootstrap_mean(d, month),
        "paired_difference_vs_matched_by_box": cluster_bootstrap_mean(d, box),
        "mean_net_r_incl_funding_vs_zero_by_month": cluster_bootstrap_mean(tf["r_multiple"].to_numpy(), month),
        "mean_r_ex_funding_vs_zero_by_month": cluster_bootstrap_mean(tf["r_ex_funding"].to_numpy(), month),
    }
    pf = matched_portfolio_null(out)
    res["matched_portfolio"] = {k: v for k, v in pf.items() if k != "sharpe_dist"}
    write_json(OUT / "random_baseline_review.json", res)
    np.save(OUT / "ledgers" / "matched_event_null_dist.npy", ev["null_dist"])
    np.save(OUT / "ledgers" / "matched_portfolio_sharpe_dist.npy", pf["sharpe_dist"])
    return res


if __name__ == "__main__":
    import json
    r = run()
    print(json.dumps(r, default=str, ensure_ascii=False, indent=1))
