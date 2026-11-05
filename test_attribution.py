"""
test_attribution.py — Phase 2 研究诊断（attribution.py、tools/run_phase2.py）的正确性测试。

* 括号单撮合在真实入场 K 线上逐笔复现回测器（离场 K 线、原因、不含资金费的 R），并在合成数据上按手算校验
  止损优先、跳空成交、止盈 maker、未了结的处理。
* 局部随机入场零假设在极小时间窗下只能抽到真实信号 K 线本身，零分布必须退化为真实结果（校验窗口中心与撮合口径）。
* 市场状态特征的因果性：在截断到决策 K 线（含）的数据上重算，与全量研究数据上的结果逐位一致。
* 交易表的会计恒等式、严格 MFE 不含离场 K 线、回撤区间、分段日度收益与 metrics 一致、买入持有手算一致。
* 滑点与手续费拆分由原始 K 线独立重算核对（不依赖 trade_frame 自身的定义式）。
* 随机入场的抽样范围（时间窗、信号前 / 后、剔除共享路径的 K 线）与泊松二项分布的精确尾概率。
* OOS 锁（每个接收交易表或 K 线的入口）与 ETH 留出锁（结构性运行不运行回测）；Phase 2 基线数值钉在测试内的常量上
  （不读 reports/phase2_baseline.json：该文件由 tools/run_phase2.py 重写，钉在它上面会自我循环）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import attribution as A
import pipeline
from backtester import BacktestResult
from config import (BASELINE_BACKTEST, BASELINE_STRATEGY, IS_END, IS_START, OOS_START, SEGMENTS, VAL_END, VAL_START,
                    BacktestConfig)
from metrics import OOSLockedError, sharpe
from ppvp_engine import build_4h

@pytest.fixture(scope="module")
def base():
    out = pipeline.run("BTCUSDT", BASELINE_STRATEGY, BASELINE_BACKTEST)
    tf = A.trade_frame(out["result"], out["df1h"], out["boxes"], BASELINE_STRATEGY)
    return out, tf


# ---------------------------------------------------------------------------
# 撮合
# ---------------------------------------------------------------------------
def test_bracket_reproduces_backtester_on_research_trades(base):
    out, tf = base
    df1h = out["df1h"]
    xb, reason, r = A.bracket_outcomes(df1h, tf["entry_bar"], tf["sl"], tf["tp"], len(df1h), BASELINE_BACKTEST)
    assert np.array_equal(xb, tf["exit_bar"].to_numpy())
    assert np.array_equal(reason, tf["exit_reason"].map({"SL": 1, "TP": 2}).to_numpy())
    assert np.allclose(r, tf["r_ex_funding"].to_numpy(), rtol=0, atol=1e-12)
    o, h, l, c = (df1h[k].to_numpy(dtype=np.float64) for k in ("open", "high", "low", "close"))
    py = A._bracket.py_func(o, h, l, c, tf["entry_bar"].to_numpy(dtype=np.int64), tf["sl"].to_numpy(dtype=np.float64),
                            tf["tp"].to_numpy(dtype=np.float64), len(df1h), 0.0005, 0.0002, 0.0002)
    assert np.array_equal(py[0], xb) and np.array_equal(py[1], reason) and np.array_equal(py[2], r)


def _df(rows):
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="h", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def test_bracket_rules_hand_computed():
    cfg = BacktestConfig()
    slip, tk, mk = cfg.slippage, cfg.taker_fee, cfg.maker_fee
    e = 100.0 * (1 + slip)
    df = _df([(100, 100, 100, 100),
              (100.2, 101.5, 98.9, 100.0),            # 同根触及 SL 与 TP：悲观按止损，成交 min(open, SL)
              (100, 100, 100, 100)])
    xb, rs, r = A.bracket_outcomes(df, [0], [99.0], [101.0], len(df), cfg)
    px = 99.0 * (1 - slip)
    assert xb[0] == 1 and rs[0] == 1
    assert np.isclose(r[0], (px - e - tk * e - tk * px) / (e - 99.0), rtol=0, atol=1e-12)
    df = _df([(100, 100, 100, 100), (98.0, 98.5, 97.5, 98.2), (100, 100, 100, 100)])      # 跳空低开：按开盘价成交
    xb, rs, r = A.bracket_outcomes(df, [0], [99.0], [101.0], len(df), cfg)
    px = 98.0 * (1 - slip)
    assert rs[0] == 1 and np.isclose(r[0], (px - e - tk * e - tk * px) / (e - 99.0), rtol=0, atol=1e-12)
    df = _df([(100, 100, 100, 100), (100, 100.5, 99.5, 100), (100.6, 102, 100.4, 101.5)])  # 止盈：maker
    xb, rs, r = A.bracket_outcomes(df, [0], [99.0], [101.0], len(df), cfg)
    px = 101.0 * (1 - slip)
    assert xb[0] == 2 and rs[0] == 2 and np.isclose(r[0], (px - e - tk * e - mk * px) / (e - 99.0), rtol=0, atol=1e-12)
    df = _df([(100, 100, 100, 100), (100, 100.5, 99.5, 100.2), (100, 100.5, 99.5, 100.3)])  # 未了结：末根收盘 taker
    xb, rs, r = A.bracket_outcomes(df, [0], [99.0], [101.0], len(df), cfg)
    px = 100.3 * (1 - slip)
    assert xb[0] == 2 and rs[0] == 0 and np.isclose(r[0], (px - e - tk * e - tk * px) / (e - 99.0), rtol=0, atol=1e-12)
    with pytest.raises(ValueError):
        A.bracket_outcomes(df, [2], [99.0], [101.0], len(df), cfg)


# ---------------------------------------------------------------------------
# 随机入场零假设
# ---------------------------------------------------------------------------
def test_local_null_with_tiny_window_degenerates_to_actual(base):
    out, tf = base
    ist = A.segment_trades(tf, IS_START, IS_END)
    nul = A.random_entry_null(out["df1h"], ist, BASELINE_BACKTEST, IS_START, IS_END, n_rep=20, seed=1,
                              window_days=1e-6)
    assert nul["null_wins_mean"] == nul["actual_wins"]
    assert np.allclose(nul["null_r_dist"], nul["actual_mean_r_ex_funding"], rtol=0, atol=1e-12)
    assert np.allclose(nul["per_trade_null_mean_r"], ist["r_ex_funding"].to_numpy(), rtol=0, atol=1e-12)
    assert nul["p_mean_r_le"] == 1.0


def test_null_is_seeded_and_bounded(base):
    out, tf = base
    ist = A.segment_trades(tf, IS_START, IS_END)
    a = A.random_entry_null(out["df1h"], ist, BASELINE_BACKTEST, IS_START, IS_END, n_rep=50, seed=7)
    b = A.random_entry_null(out["df1h"], ist, BASELINE_BACKTEST, IS_START, IS_END, n_rep=50, seed=7)
    assert np.array_equal(a["null_r_dist"], b["null_r_dist"])
    assert 0.0 < a["p_mean_r_le"] <= 1.0 and a["n_rep"] == 50


# ---------------------------------------------------------------------------
# 市场状态特征的因果性
# ---------------------------------------------------------------------------
def test_regime_features_are_causal(base):
    out, tf = base
    df1h = out["df1h"]
    spot = pipeline.spot_daily_warmup("BTCUSDT")
    rng = np.random.default_rng(3)
    idx = df1h.index
    day_edges = np.flatnonzero((idx.hour == 23) | (idx.hour == 0))
    bars = np.unique(np.concatenate([tf["entry_bar"].to_numpy()[::12], rng.choice(day_edges, 6, replace=False),
                                     rng.integers(200, len(df1h) - 1, 6)]))
    full = A.regime_features(df1h, out["df4h"], spot, bars)
    for j, b in enumerate(bars):
        cut = df1h.iloc[: b + 1]
        part = A.regime_features(cut, build_4h(cut), spot, np.array([b]))
        for col in full.columns:
            x, y = full[col].iloc[j], part[col].iloc[0]
            same = (x == y) or (pd.isna(x) and pd.isna(y))
            assert same, (col, str(idx[b]))


# ---------------------------------------------------------------------------
# 交易表与回撤 / 分段 / 基准
# ---------------------------------------------------------------------------
def test_trade_frame_identities(base):
    out, tf = base
    h = out["df1h"]["high"].to_numpy()
    assert np.allclose(tf["pre_cost_pnl"] - tf["slip_usd"] - tf["fees"] - tf["funding"], tf["pnl"], rtol=0, atol=1e-6)
    assert np.allclose(tf["r_ex_funding"] * tf["risk_usd"], tf["gross_pnl"] - tf["fees"], rtol=0, atol=1e-6)
    ok = tf["mfe_r_strict"].notna()
    assert (tf.loc[ok, "mfe_r_strict"] <= tf.loc[ok, "mfe_r"] + 1e-12).all()
    for _, t in tf.iloc[::7].iterrows():                         # 严格 MFE = entry_bar+1 .. exit_bar-1 的最高价
        a, b = int(t["entry_bar"]) + 1, int(t["exit_bar"])
        if b > a:
            assert np.isclose(t["mfe_r_strict"], (h[a:b].max() - t["entry_px"]) / (t["entry_px"] - t["sl"]))
        else:
            assert np.isnan(t["mfe_r_strict"])
    amb = tf["ambiguous_exit_bar"]
    assert (tf.loc[amb, "exit_reason"] == "SL").all()
    assert (h[tf.loc[amb, "exit_bar"].to_numpy()] >= tf.loc[amb, "tp"].to_numpy()).all()
    # 滑点与手续费拆分：由原始 K 线的参考价独立重算（入场参考 = 信号收盘；止损参考 = min(开盘, SL)；止盈参考 = TP）
    o = out["df1h"]["open"].to_numpy()
    c = out["df1h"]["close"].to_numpy()
    cfg = BASELINE_BACKTEST
    is_sl = (tf["exit_reason"] == "SL").to_numpy()
    xb = tf["exit_bar"].to_numpy()
    ref_in = c[tf["entry_bar"].to_numpy()]
    ref_out = np.where(is_sl, np.minimum(o[xb], tf["sl"].to_numpy()), tf["tp"].to_numpy())
    q = tf["qty"].to_numpy()
    assert np.allclose(tf["entry_px"], ref_in * (1 + cfg.slippage), rtol=1e-12, atol=0)
    assert np.allclose(tf["exit_px"], ref_out * (1 - cfg.slippage), rtol=1e-12, atol=0)
    slip = q * (tf["entry_px"].to_numpy() - ref_in) + q * (ref_out - tf["exit_px"].to_numpy())
    assert np.allclose(tf["slip_usd"], slip, rtol=1e-9, atol=1e-9)
    fees = q * tf["entry_px"].to_numpy() * cfg.taker_fee + q * tf["exit_px"].to_numpy() * np.where(is_sl, cfg.taker_fee,
                                                                                                    cfg.maker_fee)
    assert np.allclose(tf["fees"], fees, rtol=1e-9, atol=1e-9)


def test_drawdown_episodes_synthetic():
    idx = pd.date_range("2024-01-01", periods=7, freq="D", tz="UTC")
    eq = pd.Series([100.0, 110.0, 99.0, 105.0, 111.0, 90.0, 95.0], index=idx)
    eps = A.drawdown_episodes(eq, top=5)
    assert len(eps) == 2
    assert np.isclose(eps[0]["depth"], 90 / 111 - 1) and eps[0]["peak"] == idx[4] and eps[0]["recovery"] is None
    assert np.isclose(eps[1]["depth"], 99 / 110 - 1) and eps[1]["trough"] == idx[2] and eps[1]["recovery"] == idx[4]


def test_segment_returns_and_buy_and_hold(base):
    out, _ = base
    res, df1h = out["result"], out["df1h"]
    for name in ("IS", "VAL"):
        r = A.segment_daily_returns(res, *SEGMENTS[name])
        assert sharpe(r) == out["report"][name]["sharpe"]
        ci = A.stationary_bootstrap_sharpe(r, n_boot=200, seed=5)
        assert ci == A.stationary_bootstrap_sharpe(r, n_boot=200, seed=5) and ci["lo"] < ci["hi"]
    c = df1h["close"]
    bh_is = A.buy_and_hold(df1h, IS_START, IS_END)
    assert np.isclose(bh_is["total_return"], c.loc["2023-12-31 23:00"] / df1h["open"].iloc[0] - 1, rtol=1e-12)
    bh_val = A.buy_and_hold(df1h, VAL_START, VAL_END)
    assert np.isclose(bh_val["total_return"], c.loc["2025-06-30 23:00"] / c.loc["2023-12-31 23:00"] - 1, rtol=1e-12)


def test_sample_entry_bars_respects_window_side_and_exclusion():
    idx_ns = np.arange(400, dtype=np.int64) * np.int64(3600 * 10**9)
    e = np.array([100, 250], dtype=np.int64)
    rng = np.random.default_rng(0)
    both = A._sample_entry_bars(idx_ns, e, 10, 390, 4000, rng, 3, "both", 24)
    for j in range(2):
        col = both[:, j]
        assert col.min() >= e[j] - 72 and col.max() <= e[j] + 72
        assert (np.abs(col - e[j]) > 24).all()
        assert set(np.unique(col)) == set(range(e[j] - 72, e[j] - 24)) | set(range(e[j] + 25, e[j] + 73))
    after = A._sample_entry_bars(idx_ns, e, 10, 390, 2000, rng, 3, "after", None)
    assert (after > e[None, :]).all() and (after <= e[None, :] + 72).all()
    before = A._sample_entry_bars(idx_ns, e, 10, 390, 2000, rng, 3, "before", None)
    assert (before < e[None, :]).all() and (before >= e[None, :] - 72).all()
    glob = A._sample_entry_bars(idx_ns, e, 10, 390, 2000, rng, None, "both", None)
    assert glob.min() >= 10 and glob.max() <= 388                 # [first, end_bar - 1)：至少留一根后续 K 线
    with pytest.raises(ValueError):
        A._sample_entry_bars(idx_ns, np.array([385]), 10, 390, 10, rng, 3, "after", 24)


def test_poisson_binomial_cdf_matches_enumeration():
    p = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
    outs = np.array(np.meshgrid(*[[0, 1]] * 5, indexing="ij")).reshape(5, -1).T
    prob = np.prod(np.where(outs == 1, p, 1 - p), axis=1)
    for k in range(6):
        assert np.isclose(A.poisson_binomial_cdf(p, k), prob[outs.sum(axis=1) <= k].sum(), rtol=0, atol=1e-12)


def test_bucket_stats_null_identity_and_tp_rate(base):
    out, tf = base
    ist = A.segment_trades(tf, IS_START, IS_END).copy()
    ist["null_r"] = 0.1 * np.sin(np.arange(len(ist)))
    t = A.bucket_stats(ist, "pattern_name", n_boot=200)
    assert np.allclose(t["mean_r_xf"] - t["null_mean_r"], t["excess_r"], rtol=0, atol=1e-12)
    for _, r in t.iterrows():
        sub = ist[ist["pattern_name"] == r["bucket"]]
        assert r["tp_hits"] == int((sub["exit_reason"] == "TP").sum())
        assert np.isclose(r["sum_log_ret"], np.log1p(sub["ret_on_base"]).sum())


def test_paired_sharpe_diff_properties(base):
    out, _ = base
    r = A.segment_daily_returns(out["result"], VAL_START, VAL_END)
    zero = A.paired_sharpe_diff(r, r, n_boot=300, seed=2)
    assert zero["diff"] == 0.0 and zero["lo"] == 0.0 and zero["hi"] == 0.0
    r2 = r * 0.5 + 0.0002                                       # 同日强相关的另一条序列
    ab = A.paired_sharpe_diff(r2, r, n_boot=300, seed=2)
    ba = A.paired_sharpe_diff(r, r2, n_boot=300, seed=2)
    assert np.isclose(ab["diff"], sharpe(r2) - sharpe(r)) and np.isclose(ab["diff"], -ba["diff"])
    assert np.isclose(ab["lo"], -ba["hi"]) and np.isclose(ab["hi"], -ba["lo"])
    with pytest.raises(ValueError):
        A.paired_sharpe_diff(r.iloc[1:], r)


def test_bucket_stats_wilson():
    t0 = pd.Timestamp("2021-01-01", tz="UTC")
    tf = pd.DataFrame({"g": ["a"] * 10, "win": [True] * 8 + [False] * 2, "r_multiple": [1.0] * 8 + [-1.0] * 2,
                       "pnl": [1.0] * 8 + [-1.0] * 2, "p_rw": [0.5] * 10, "entry_time": [t0] * 10,
                       "exit_time": [t0 + pd.Timedelta(hours=5)] * 10})
    t = A.bucket_stats(tf, "g", n_boot=100)
    assert t.loc[0, "n"] == 10 and t.loc[0, "wins"] == 8
    assert np.isclose(t.loc[0, "win_lo"], 0.4902, atol=1e-4) and np.isclose(t.loc[0, "win_hi"], 0.9433, atol=1e-4)


# ---------------------------------------------------------------------------
# 隔离
# ---------------------------------------------------------------------------
def test_attribution_respects_oos_lock():
    idx = pd.date_range(OOS_START - pd.Timedelta(hours=3), periods=6, freq="h", tz="UTC")
    tr = pd.DataFrame({"entry_time": [idx[-2]], "exit_time": [idx[-1]], "exit_reason": ["SL"]})
    res = BacktestResult(equity=pd.Series(1.0, index=idx), position=pd.Series(0.0, index=idx), trades=tr,
                         config=BacktestConfig(), initial_capital=1.0, start_time=idx[0])
    with pytest.raises(OOSLockedError):
        A.trade_frame(res, pd.DataFrame(), pd.DataFrame(), BASELINE_STRATEGY)
    with pytest.raises(OOSLockedError):
        A.segment_trades(tr, IS_START, IS_END)
    with pytest.raises(OOSLockedError):
        A.stationary_bootstrap_sharpe(pd.Series(0.01, index=idx), n_boot=10)
    with pytest.raises(OOSLockedError):
        A.drawdown_episodes(pd.Series(1.0, index=idx))
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}, index=idx)
    with pytest.raises(OOSLockedError):
        A.buy_and_hold(df, idx[0].floor("D"), OOS_START + pd.Timedelta(days=1))
    tfo = pd.DataFrame({"entry_time": [idx[-2]], "exit_time": [idx[-1]], "exit_reason": ["SL"], "win": [False],
                        "r_multiple": [-1.0], "pnl": [-1.0], "p_rw": [0.5], "entry_bar": [4], "d_sl": [0.01], "d_tp": [0.01],
                        "r_ex_funding": [-1.0]})
    with pytest.raises(OOSLockedError):
        A.bucket_stats(tfo, "exit_reason", n_boot=10)
    with pytest.raises(OOSLockedError):
        A.random_entry_null(df, tfo, BacktestConfig(), idx[0], idx[-1], n_rep=5)
    tf_ok = tfo.assign(entry_time=[idx[1]], exit_time=[idx[2]], entry_bar=[1])
    with pytest.raises(OOSLockedError):                         # 撮合截止时刻越过 OOS_START
        A.random_entry_null(df, tf_ok, BacktestConfig(), idx[0], OOS_START + pd.Timedelta(hours=2), n_rep=5)
    with pytest.raises(OOSLockedError):                         # 撮合用到的 K 线越过 OOS_START
        A.bracket_outcomes(df, [0], [0.5], [2.0], len(df), BacktestConfig())
    with pytest.raises(OOSLockedError):
        A.regime_features(df, pd.DataFrame(), None, np.array([0]))
    xb, rs, r = A.bracket_outcomes(df, [0], [0.5], [2.0], 3, BacktestConfig())   # 只用 OOS 之前的 K 线：允许
    assert rs[0] == 0


def test_eth_holdout_guard(monkeypatch):
    with pytest.raises(OOSLockedError):
        pipeline.run("ETHUSDT")

    def forbidden(*a, **k):
        raise AssertionError("run_backtest must not be called for a holdout symbol")

    monkeypatch.setattr(pipeline, "run_backtest", forbidden)
    out = pipeline.run("ETHUSDT", segments=())                  # 结构性运行：只有箱体与信号，不运行回测
    assert out["result"] is None and out["report"] == {}
    assert len(out["signals"]) == len(out["df1h"]) and len(out["boxes"]) > 0
    pipeline.clear_cache()


# Phase 2 基线（StrategyConfig() + BacktestConfig()，研究数据）的钉住值：引擎或数据的任何无意改动都会使其失败
PINNED = {
    "IS": {"sharpe": -1.174067431234681, "total_return": -0.4563376830466489, "max_dd": -0.4674881770067414,
           "win_rate": 0.3669724770642202, "profit_factor": 0.4501157788201149, "fees": 6389.8965746348385,
           "funding": 1438.8106007914257, "trades": 109, "exit_reasons": {"SL": 65, "TP": 44}},
    "VAL": {"sharpe": -0.5254644149426314, "total_return": -0.06922176727513596, "max_dd": -0.13585709147858727,
            "win_rate": 0.4, "profit_factor": 0.6982496620375542, "fees": 1414.882575250229,
            "funding": 375.3191845306527, "trades": 30, "exit_reasons": {"SL": 17, "TP": 13}},
}


def test_phase2_baseline_pinned(base):
    out, _ = base
    for name, want in PINNED.items():
        got = out["report"][name]
        assert got["trades"] == want["trades"] and got["exit_reasons"] == want["exit_reasons"], name
        for k in ("sharpe", "total_return", "max_dd", "win_rate", "profit_factor", "fees", "funding"):
            assert abs(got[k] - want[k]) <= 1e-9 * max(1.0, abs(want[k])), (name, k)
