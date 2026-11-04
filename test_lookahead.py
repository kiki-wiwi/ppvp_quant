"""
test_lookahead.py — 防未来函数与撮合真实性单元测试（pytest）。

核心测试 ``test_no_lookahead``：随机选取 5 个历史截断时点 T_cut，分别在“仅含 open_time ≤ T_cut 的截断数据”
与“2020-2026 全量数据”上运行 4H PPVP 箱体、1H 状态机信号与回测，断言 t ≤ T_cut 的全部
VAL/POC/VAH/支撑带/状态/信号/止损止盈价位、净值曲线与已平仓交易逐位（bitwise）一致。
覆盖基线配置与两套开启全部可选模块的配置。

运行：``python -m pytest test_lookahead.py -v -s``
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtester import run_backtest
from config import BacktestConfig, StrategyConfig
from data_loader import load_funding, load_klines, resample_ohlcv
from metrics import daily_equity, daily_returns, sharpe
from numba_compat import HAVE_NUMBA
from pipeline import spot_daily_warmup
from ppvp_engine import build_4h, build_boxes, detect_pivots, map_boxes_to_1h, to_ns
from strategy_state_machine import _run_state_machine, generate_signals, state_machine_args

ROOT = Path(__file__).resolve().parent
SEED = 20261104
SYMBOL = "BTCUSDT"

CONFIGS = {
    "baseline": (StrategyConfig(), BacktestConfig()),
    "all_features_A": (
        StrategyConfig(pivot_left=8, pivot_right=3, profile_pivots=3, regime_filter="daily_sma", min_rr=1.0,
                       max_box_width_atr=8.0, max_box_width_pct=0.05, reject_broken_at_activation=True),
        BacktestConfig(sizing="risk", tp_mode="poc_partial", time_stop_bars=48, exit_on_breakdown=True)),
    "all_features_B": (
        StrategyConfig(regime_filter="crash_cooldown", roll_failed_second_touch=False,
                       carry_touches_on_box_update=False, va_pct=0.65, n_bins=40, engulf_ratio=0.8),
        BacktestConfig(sizing="inv_vol", venue="spot", compounding=False)),
}


# ----------------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def market():
    df = load_klines(SYMBOL, "1h", "perp")
    df = df[df.index < pd.Timestamp("2026-11-01", tz="UTC")]
    return df, load_funding(SYMBOL), spot_daily_warmup(SYMBOL)


def _pipeline(df1h, funding, spot_d, scfg, bcfg):
    boxes, df4h = build_boxes(df1h, scfg)
    sig = generate_signals(df1h, scfg, boxes, df4h, spot_d)
    fund = funding[funding.index <= df1h.index[-1] + pd.Timedelta(hours=1)]
    res = run_backtest(df1h, sig, bcfg, fund)
    return boxes, sig, res


def _bitwise_equal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """逐元素“逐位”相等：浮点按 int64 位模式比较；时间按纳秒整数比较（NaT 与 NaT 视为相等）。"""
    if x.shape != y.shape:
        return np.zeros(max(len(x), len(y)), dtype=bool)
    if x.dtype.kind == "f":
        return x.view(np.int64) == y.view(np.int64)
    if x.dtype.kind == "M":
        return x.view(np.int64) == y.view(np.int64)
    if x.dtype == object:
        return np.array([(a == b) or (pd.isna(a) and pd.isna(b)) for a, b in zip(x, y)], dtype=bool)
    return x == y


def _col_array(s: pd.Series) -> np.ndarray:
    if isinstance(s.dtype, pd.DatetimeTZDtype) or s.dtype.kind == "M":
        return pd.DatetimeIndex(s).as_unit("ns").tz_localize(None).to_numpy() if getattr(s.dt, "tz", None) else \
            pd.DatetimeIndex(s).as_unit("ns").to_numpy()
    return s.to_numpy()


def assert_frames_bitwise(a: pd.DataFrame, b: pd.DataFrame, ctx: str) -> None:
    assert list(a.columns) == list(b.columns), f"{ctx}: columns differ"
    assert len(a) == len(b), f"{ctx}: length {len(a)} vs {len(b)}"
    assert a.index.equals(b.index), f"{ctx}: index differs"
    for col in a.columns:
        eq = _bitwise_equal(_col_array(a[col]), _col_array(b[col]))
        if not eq.all():
            j = int(np.flatnonzero(~eq)[0])
            raise AssertionError(f"{ctx}: column '{col}' differs at row {j} ({a.index[j]}): {a[col].iloc[j]!r} vs {b[col].iloc[j]!r}")


def compare_truncated_vs_full(full_out, trunc_out, df_full, t_cut, ctx) -> dict:
    boxes_f, sig_f, res_f = full_out
    boxes_t, sig_t, res_t = trunc_out
    last_close = t_cut + pd.Timedelta(hours=1)
    # 1) 截断时点前可见的全部箱体逐位一致
    vis_f = boxes_f[boxes_f["available_time"] <= last_close].reset_index(drop=True)
    assert_frames_bitwise(vis_f, boxes_t.reset_index(drop=True), f"{ctx} boxes")
    # 2) t ≤ T_cut 的逐根信号与诊断（VAL/POC/VAH/支撑带/状态/信号/SL/TP ...）逐位一致
    assert_frames_bitwise(sig_f.loc[sig_f.index <= t_cut], sig_t, f"{ctx} signals")
    # 3) 净值与持仓逐位一致
    n = len(sig_t)
    assert np.array_equal(res_f.equity.to_numpy()[:n].view(np.int64), res_t.equity.to_numpy().view(np.int64)), f"{ctx} equity"
    assert np.array_equal(res_f.position.to_numpy()[:n].view(np.int64), res_t.position.to_numpy().view(np.int64)), f"{ctx} position"
    # 4) 截断前已平仓交易全部字段一致；截断时仍持仓的交易入场字段一致
    tf, tt = res_f.trades, res_t.trades
    closed_t = tt[(tt["exit_bar"] >= 0)].reset_index(drop=True)
    closed_f = tf[(tf["exit_bar"] >= 0) & (tf["exit_bar"] < n)].reset_index(drop=True)
    assert_frames_bitwise(closed_f, closed_t, f"{ctx} closed trades")
    entry_cols = ["entry_bar", "entry_time", "entry_px", "qty", "notional", "leverage", "equity_at_entry", "sl", "tp", "poc"]
    ent_f = tf[tf["entry_bar"] < n][entry_cols].reset_index(drop=True)
    assert_frames_bitwise(ent_f, tt[entry_cols].reset_index(drop=True), f"{ctx} trade entries")
    return {"t_cut": str(t_cut), "boxes_compared": int(len(vis_f)), "bars_compared": int(n),
            "signals_compared": int(sig_t["sig_raw"].sum()), "closed_trades_compared": int(len(closed_t)),
            "entries_compared": int(len(ent_f))}


# ----------------------------------------------------------------------------
# 1. 核心：随机截断防未来函数测试
# ----------------------------------------------------------------------------
def test_no_lookahead(market):
    df, fund, spot_d = market
    rng = np.random.default_rng(SEED)
    pool = df.index[(df.index >= pd.Timestamp("2020-03-01", tz="UTC")) & (df.index < pd.Timestamp("2026-10-25", tz="UTC"))]
    picked = np.sort(rng.choice(to_ns(pool), size=5, replace=False))
    cuts = [pd.Timestamp(int(x), unit="ns", tz="UTC") for x in picked]
    log = {"seed": SEED, "cuts": [str(c) for c in cuts], "results": []}
    for name, (scfg, bcfg) in CONFIGS.items():
        full_out = _pipeline(df, fund, spot_d, scfg, bcfg)
        for t_cut in cuts:
            trunc = df[df.index <= t_cut]
            trunc_out = _pipeline(trunc, fund, spot_d, scfg, bcfg)
            rec = compare_truncated_vs_full(full_out, trunc_out, df, t_cut, f"[{name} @ {t_cut}]")
            rec["config"] = name
            log["results"].append(rec)
            print(f"PASS {name:15s} T_cut={t_cut}  boxes={rec['boxes_compared']:5d} bars={rec['bars_compared']:6d} "
                  f"signals={rec['signals_compared']:4d} closed_trades={rec['closed_trades_compared']:4d}")
    out = ROOT / "reports" / "lookahead_test_log.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(log, indent=2, ensure_ascii=False))


def test_no_lookahead_adversarial_cuts(market):
    """在最容易泄露未来信息的时点截断：箱体确认前 1 小时 / 确认当时、信号 K 线及其前一根、持仓中途、4H 未走完时。"""
    df, fund, spot_d = market
    scfg, bcfg = CONFIGS["all_features_A"]
    full_out = _pipeline(df, fund, spot_d, scfg, bcfg)
    boxes_f, sig_f, res_f = full_out
    cuts = []
    for k in (50, 400, 1200):
        t = boxes_f["available_time"].iloc[k]
        cuts += [t - pd.Timedelta(hours=1), t]
    sig_times = sig_f.index[sig_f["sig_raw"]]
    for k in (10, 120):
        cuts += [sig_times[k] - pd.Timedelta(hours=1), sig_times[k]]
    tr = res_f.trades
    for k in (5, 60):
        mid = int((tr["entry_bar"].iloc[k] + max(tr["exit_bar"].iloc[k], tr["entry_bar"].iloc[k] + 1)) // 2)
        cuts.append(df.index[mid])
    cuts.append(pd.Timestamp("2023-06-14 13:00", tz="UTC"))       # 4H 只走完 2 根 1H
    for t_cut in sorted(set(cuts)):
        trunc_out = _pipeline(df[df.index <= t_cut], fund, spot_d, scfg, bcfg)
        compare_truncated_vs_full(full_out, trunc_out, df, t_cut, f"[adversarial @ {t_cut}]")
        print(f"PASS adversarial T_cut={t_cut}")


# ----------------------------------------------------------------------------
# 2. 时间对齐：4H 绝对 UTC 边界、维护缺口、可见时间
# ----------------------------------------------------------------------------
def test_4h_utc_boundaries_and_gap_bar(market):
    df, _, _ = market
    df4 = build_4h(df)
    ns = to_ns(df4.index)
    assert np.all(ns % (4 * 3600 * 10**9) == 0), "4H open times must sit on UTC 00/04/08/12/16/20"
    assert (df4["available_time"] == df4.index + pd.Timedelta(hours=4)).all(), "available_time must equal open + 4h"
    gap = pd.Timestamp("2024-10-28 20:00", tz="UTC")
    assert gap not in df.index, "maintenance placeholder 1H bar must be absent"
    bar = df4.loc[gap]
    parts = df.loc[(df.index >= gap) & (df.index < gap + pd.Timedelta(hours=4))]
    assert len(parts) == 3 and int(bar["n_bars"]) == 3
    assert bar["open"] == parts["open"].iloc[0] and bar["close"] == parts["close"].iloc[-1]
    assert bar["high"] == parts["high"].max() and bar["low"] == parts["low"].min()
    assert np.isclose(bar["volume"], parts["volume"].sum(), rtol=0, atol=1e-9)
    nxt = df4.loc[gap + pd.Timedelta(hours=4)]                      # 缺口后的 4H 仍锚定 00:00，不发生行号错位
    blk = df.loc[(df.index >= gap + pd.Timedelta(hours=4)) & (df.index < gap + pd.Timedelta(hours=8))]
    assert int(nxt["n_bars"]) == 4 and nxt["open"] == blk["open"].iloc[0] and nxt["close"] == blk["close"].iloc[-1]
    # 截断在 4H 中途：未走完的 4H 必须被剔除
    cut = df[df.index <= pd.Timestamp("2023-06-14 13:00", tz="UTC")]
    df4c = build_4h(cut)
    assert df4c["available_time"].iloc[-1] <= cut.index[-1] + pd.Timedelta(hours=1)
    assert df4c.index[-1] == pd.Timestamp("2023-06-14 08:00", tz="UTC")
    # 与 data_loader 的通用聚合一致
    ref = resample_ohlcv(df[["open", "high", "low", "close", "volume", "quote_volume"]], "4h")
    assert (ref.index == df4.index).all()


def test_pivot_right_bar_confirmation_lag(market):
    df, _, _ = market
    scfg = StrategyConfig()
    L, R = scfg.pivot_left, scfg.pivot_right
    df4 = build_4h(df[df.index < pd.Timestamp("2021-01-01", tz="UTC")])
    h, l = df4["high"].to_numpy(), df4["low"].to_numpy()
    ph, pl = detect_pivots(h, l, L, R)
    idx = np.flatnonzero(ph | pl)[:200]
    for i in idx:
        p0, q0 = detect_pivots(h[: i + R], l[: i + R], L, R)        # 第 i+R 根尚未收盘
        assert not (p0[i] if i < len(p0) else False) and not (q0[i] if i < len(q0) else False), f"pivot {i} visible before i+R"
        p1, q1 = detect_pivots(h[: i + R + 1], l[: i + R + 1], L, R)
        assert (p1[i] or q1[i]), f"pivot {i} not confirmed at i+R"
    boxes, _ = build_boxes(df[df.index < pd.Timestamp("2021-01-01", tz="UTC")], scfg)
    exp = df4["available_time"].to_numpy()[boxes["pivot_b_idx"].to_numpy() + R]
    assert np.array_equal(boxes["available_time"].to_numpy(), exp)
    assert (boxes["available_time"] - boxes["window_end"] == pd.Timedelta(hours=4 * R)).all()


def test_1h_reads_only_closed_4h(market):
    df, _, _ = market
    boxes, _ = build_boxes(df, StrategyConfig())
    bi = map_boxes_to_1h(df, boxes)
    ok = bi >= 0
    avail = to_ns(boxes["available_time"])[bi[ok]]
    assert np.all(avail <= to_ns(df.index)[ok]), "a 1H bar used a box confirmed after its open"
    nxt = np.minimum(bi[ok] + 1, len(boxes) - 1)
    later = (bi[ok] + 1 < len(boxes))
    assert np.all(to_ns(boxes["available_time"])[nxt][later] > to_ns(df.index)[ok][later]), "a newer visible box was skipped"


# ----------------------------------------------------------------------------
# 3. 撮合与成本（合成数据，结果可手算）
# ----------------------------------------------------------------------------
def _mk_df(ohlc, start="2024-01-01 00:00"):
    idx = pd.date_range(start, periods=len(ohlc), freq="1h", tz="UTC")
    df = pd.DataFrame(ohlc, columns=["open", "high", "low", "close"], index=idx, dtype=float)
    df["volume"] = 100.0
    df["quote_volume"] = 1e6
    return df


def _mk_sig(df, entries: dict):
    s = pd.DataFrame({"signal": False, "sl": np.nan, "tp": np.nan, "poc_tp": np.nan, "breakdown": False}, index=df.index)
    for bar, (sl, tp, poc) in entries.items():
        s.iloc[bar, s.columns.get_loc("signal")] = True
        s.iloc[bar, s.columns.get_loc("sl")] = sl
        s.iloc[bar, s.columns.get_loc("tp")] = tp
        s.iloc[bar, s.columns.get_loc("poc_tp")] = poc
    return s


def test_intrabar_pessimistic_stop_first():
    df = _mk_df([[100, 101, 99, 100], [100, 106, 94, 100], [100, 101, 99, 100]])
    res = run_backtest(df, _mk_sig(df, {0: (95.0, 105.0, 102.0)}), BacktestConfig(include_funding=False))
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "SL" and t["exit_bar"] == 1
    assert t["exit_px"] == pytest.approx(95.0 * (1 - 0.0002), rel=1e-12)


def test_no_exit_evaluated_on_entry_bar():
    df = _mk_df([[100, 120, 80, 100], [100, 101, 99, 100], [100, 101, 99, 100]])
    res = run_backtest(df, _mk_sig(df, {0: (95.0, 105.0, 102.0)}), BacktestConfig(include_funding=False))
    assert res.trades.iloc[0]["exit_reason"] == "OPEN", "entry bar's own high/low must not trigger exits"


def test_gap_open_below_stop_fills_at_open():
    df = _mk_df([[100, 101, 99, 100], [93, 94, 92, 93]])
    res = run_backtest(df, _mk_sig(df, {0: (95.0, 105.0, 102.0)}), BacktestConfig(include_funding=False))
    assert res.trades.iloc[0]["exit_px"] == pytest.approx(93.0 * (1 - 0.0002), rel=1e-12)


def test_poc_partial_same_bar_breakeven_is_pessimistic():
    df = _mk_df([[100, 101, 99, 100], [100, 106, 99.5, 100], [100, 101, 99, 100]])
    res = run_backtest(df, _mk_sig(df, {0: (95.0, 105.0, 102.0)}),
                       BacktestConfig(include_funding=False, tp_mode="poc_partial", partial_frac=0.5))
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "BE" and t["exit_bar"] == 1 and t["partial_bar"] == 1


def _funding_case(entry_bar, exit_bar, rates, venue="perp", include=True):
    df = _mk_df([[100, 100.5, 99.5, 100]] * 24)
    fund = pd.DataFrame({"funding_rate": [rates[0], rates[1], rates[2]], "funding_interval_hours": 8},
                        index=pd.DatetimeIndex(["2024-01-01 08:00", "2024-01-01 16:00", "2024-01-02 00:00"], tz="UTC"))
    cfg = BacktestConfig(venue=venue, include_funding=include, time_stop_bars=exit_bar - entry_bar)
    res = run_backtest(df, _mk_sig(df, {entry_bar: (50.0, 200.0, 150.0)}), cfg, fund)
    return res.trades.iloc[0], cfg


def test_funding_settles_only_when_crossing_settlement_time():
    r = (0.0001, -0.0002, 0.0003)
    # 05:00 K 线收盘(06:00)入场，15:00 K 线收盘(16:00)时间止损离场：跨越 08:00 与 16:00 两个结算点
    t, _ = _funding_case(5, 15, r)
    assert t["funding"] == pytest.approx(t["qty"] * 100.0 * r[0] + t["qty"] * 100.0 * r[1], rel=1e-12)
    # 07:00 K 线收盘(08:00)入场：成交在 08:00 结算快照之后，不承担 08:00 资金费；仅 16:00
    t, _ = _funding_case(7, 15, r)
    assert t["funding"] == pytest.approx(t["qty"] * 100.0 * r[1], rel=1e-12)
    # 08:00~14:00 之间开平：不跨越任何结算点 → 0
    t, _ = _funding_case(8, 13, r)
    assert t["funding"] == 0.0
    # 关闭资金费开关 → 0
    t, _ = _funding_case(5, 15, r, include=False)
    assert t["funding"] == 0.0


def test_spot_mode_no_funding_spot_fees_and_1x_cap():
    r = (0.0001, -0.0002, 0.0003)
    df = _mk_df([[100, 100.5, 99.5, 100]] * 24)
    fund = pd.DataFrame({"funding_rate": list(r), "funding_interval_hours": 8},
                        index=pd.DatetimeIndex(["2024-01-01 08:00", "2024-01-01 16:00", "2024-01-02 00:00"], tz="UTC"))
    cfg = BacktestConfig(venue="spot", fixed_leverage=1.5, time_stop_bars=10)
    res = run_backtest(df, _mk_sig(df, {5: (50.0, 200.0, 150.0)}), cfg, fund)
    t = res.trades.iloc[0]
    assert t["funding"] == 0.0
    assert t["leverage"] == pytest.approx(1.0, rel=1e-12)
    exp_fees = t["notional"] * 0.001 + t["qty"] * t["exit_px"] * 0.001
    assert t["fees"] == pytest.approx(exp_fees, rel=1e-12)


def test_risk_sizing_leverage_cap():
    df = _mk_df([[100, 100.5, 99.5, 100]] * 5)
    res = run_backtest(df, _mk_sig(df, {0: (99.9, 110.0, 105.0)}),
                       BacktestConfig(sizing="risk", risk_per_trade=0.01, max_leverage=1.5, include_funding=False))
    assert res.trades.iloc[0]["leverage"] == pytest.approx(1.5, rel=1e-12)


# ----------------------------------------------------------------------------
# 4. JIT 与纯 Python 路径逐位一致；日度夏普口径
# ----------------------------------------------------------------------------
@pytest.mark.skipif(not HAVE_NUMBA, reason="numba not installed")
def test_numba_and_python_paths_identical(market):
    df, fund, spot_d = market
    sub = df[df.index < pd.Timestamp("2021-07-01", tz="UTC")]
    for name, (scfg, bcfg) in CONFIGS.items():
        boxes, df4 = build_boxes(sub, scfg)
        args = state_machine_args(sub, scfg, boxes)
        a = _run_state_machine(*args)
        b = _run_state_machine.py_func(*args)
        for x, y in zip(a, b):
            assert np.array_equal(np.asarray(x).view(np.uint8), np.asarray(y).view(np.uint8)), f"{name}: state machine differs"
        sig = generate_signals(sub, scfg, boxes, df4, spot_d)
        r1 = run_backtest(sub, sig, bcfg, fund, use_jit=True)
        r2 = run_backtest(sub, sig, bcfg, fund, use_jit=False)
        assert np.array_equal(r1.equity.to_numpy().view(np.int64), r2.equity.to_numpy().view(np.int64)), name
        assert_frames_bitwise(r1.trades, r2.trades, f"{name} trades jit vs python")


def test_daily_sharpe_counts_flat_days():
    from backtester import BacktestResult
    idx = pd.date_range("2024-01-01 01:00", periods=24 * 10, freq="1h", tz="UTC")
    eq = np.full(len(idx), 100_000.0)
    for d in range(10):                         # 偶数日 +1%（在当日最后一根收盘），奇数日空仓 0 收益
        if d % 2 == 0:
            eq[d * 24 + 23:] *= 1.01
    res = BacktestResult(equity=pd.Series(eq, index=idx), position=pd.Series(0.0, index=idx), trades=pd.DataFrame(),
                         config=BacktestConfig(), initial_capital=100_000.0, start_time=pd.Timestamp("2024-01-01", tz="UTC"))
    r = daily_returns(daily_equity(res))
    assert len(r) == 10 and (r == 0).sum() == 5
    exp = np.array([0.01, 0.0] * 5)
    assert sharpe(r) == pytest.approx(exp.mean() / exp.std(ddof=1) * np.sqrt(365), rel=1e-9)


# ----------------------------------------------------------------------------
# 5. 性能基准：完整 2020-01 ~ 2026-10 单次回测
# ----------------------------------------------------------------------------
def test_full_period_runtime_benchmark(market):
    df, fund, spot_d = market
    scfg, bcfg = StrategyConfig(), BacktestConfig()
    _pipeline(df, fund, spot_d, scfg, bcfg)                       # JIT 预热
    timings = {}
    for rep in range(3):
        t0 = time.perf_counter()
        boxes, df4 = build_boxes(df, scfg)
        t1 = time.perf_counter()
        sig = generate_signals(df, scfg, boxes, df4, spot_d)
        t2 = time.perf_counter()
        run_backtest(df, sig, bcfg, fund)
        t3 = time.perf_counter()
        timings[f"run{rep}"] = {"ppvp_boxes_s": t1 - t0, "state_machine_s": t2 - t1, "backtest_s": t3 - t2, "total_s": t3 - t0}
    best = min(v["total_s"] for v in timings.values())
    info = {"bars_1h": len(df), "bars_4h": int(len(df4)), "boxes": int(len(boxes)), "numba": HAVE_NUMBA,
            "timings": timings, "best_total_s": best}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "phase1_benchmark.json").write_text(json.dumps(info, indent=2))
    print(f"full-period 1H bars={len(df)} boxes={len(boxes)} best total={best:.3f}s numba={HAVE_NUMBA}")
    assert best < 5.0, f"single 7-year backtest took {best:.2f}s"
