"""
test_audit_round.py — 本轮（BTC IS only，1H 数据）的小型可手算测试与 IS 内前缀截断一致性测试。

不读取任何 VAL / OOS / ETH 数据：真实数据只经 ppvp_audit.common（截断在 IS_END 之前），其余为合成数据。
覆盖用户第 3.4 节列出的风险：枢轴可用时间、箱体版本、跨周期收盘、下一根的即时止损、入场 / 离场同根、跳空、
资金费边界、分批平仓、未实现盈亏、不完整 K 线，以及本轮新增的路径撮合器、单事件模拟器、B1 状态机副本与过滤器。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester import run_backtest
from config import BASELINE_BACKTEST, BASELINE_STRATEGY, IS_END, BacktestConfig, StrategyConfig
from ppvp_audit import variants as V
from ppvp_audit.common import ISBoundaryError, assert_is_only, is_run, load_is_market
from ppvp_audit.engine_path import run_path
from ppvp_audit.event_sim import simulate_events
from ppvp_engine import build_4h, build_boxes, map_boxes_to_1h
from strategy_state_machine import bullish_engulfing, bullish_pinbar, generate_signals, state_machine_params

H = pd.Timedelta(hours=1)
T0 = pd.Timestamp("2021-03-01 00:00", tz="UTC")


@pytest.fixture(scope="module")
def base():
    return is_run(BASELINE_STRATEGY, BASELINE_BACKTEST)


def bars(rows, start=T0) -> pd.DataFrame:
    a = np.asarray([tuple(r) + ((100.0,) if len(r) == 4 else ()) for r in rows], dtype=np.float64)
    idx = pd.date_range(start, periods=len(a), freq="h", tz="UTC")
    return pd.DataFrame({"open": a[:, 0], "high": a[:, 1], "low": a[:, 2], "close": a[:, 3], "volume": a[:, 4],
                         "quote_volume": a[:, 4] * a[:, 3]}, index=idx)


def one_signal(df, at, sl, tp, poc=np.nan) -> pd.DataFrame:
    n = len(df)
    s = pd.DataFrame({"signal": np.zeros(n, bool), "sl": np.nan, "tp": np.nan, "poc_tp": np.nan}, index=df.index)
    s.iloc[at, 0] = True
    s.iloc[at, 1], s.iloc[at, 2], s.iloc[at, 3] = sl, tp, poc
    return s


NOFUND = BASELINE_BACKTEST.with_(include_funding=False)


# ---------------------------------------------------------------------------
# 跨周期与箱体可用时间
# ---------------------------------------------------------------------------
def test_4h_uses_utc_boundaries_and_counts_incomplete_bars():
    df = bars([(100, 101, 99, 100)] * 11, start=pd.Timestamp("2021-03-01 01:00", tz="UTC"))   # 01:00 .. 11:00
    d4 = build_4h(df)
    assert list(d4.index) == [pd.Timestamp("2021-03-01 00:00", tz="UTC"), pd.Timestamp("2021-03-01 04:00", tz="UTC"),
                              pd.Timestamp("2021-03-01 08:00", tz="UTC")]
    assert list(d4["n_bars"]) == [3, 4, 4]                      # 缺 00:00 的 4H 只有 3 根：n_bars 显式记录
    assert (d4["available_time"] == d4.index + 4 * H).all()
    d4b = build_4h(df.iloc[:-1])                                # 最后一个 4H 只走了 3 根：尚未收盘，被剔除
    assert d4b.index[-1] == pd.Timestamp("2021-03-01 04:00", tz="UTC")


def test_box_availability_and_versions_on_is(base):
    boxes, df4h, df1h = base["boxes"], base["df4h"], base["df1h"]
    R = BASELINE_STRATEGY.pivot_right
    av = pd.DatetimeIndex(boxes["available_time"]).as_unit("ns")
    exp = pd.DatetimeIndex(df4h["available_time"].to_numpy()[boxes["pivot_b_idx"].to_numpy() + R]).as_unit("ns")
    assert np.array_equal(av.asi8, exp.asi8)                    # 箱体在枢轴 b 之后第 R 根 4H 收盘时才可用
    assert (np.diff(av.asi8) > 0).all() and (np.diff(pd.DatetimeIndex(boxes["window_start"]).asi8) >= 0).all()
    assert (pd.DatetimeIndex(boxes["window_end"]) <= av).all()   # 分布窗口不含确认之前的未来数据
    m = map_boxes_to_1h(df1h, boxes)
    for k in np.random.default_rng(0).choice(len(boxes) - 1, 25, replace=False):
        first = int(np.flatnonzero(m == k)[0])                  # 第一根使用箱体 k 的 1H：开盘 ≥ 可用时刻，前一根仍用 k−1
        assert df1h.index[first] >= av[k] and df1h.index[first] - H < av[k] and m[first - 1] == k - 1


# ---------------------------------------------------------------------------
# 成交、路径与记账（合成数据，可手算）
# ---------------------------------------------------------------------------
def test_entry_at_close_and_immediate_stop_on_next_bar():
    df = bars([(100, 101, 99, 100), (100, 100.5, 95.0, 99.5), (99.5, 100.0, 97.0, 98.0), (98, 99, 97, 98.5)])
    s = one_signal(df, 1, sl=97.5, tp=110.0)                    # 信号 K 线自身的低点 95 在入场之前，不得触发止损
    tr = run_backtest(df, s, NOFUND).trades
    slip, tk = NOFUND.slippage, NOFUND.taker_fee
    assert len(tr) == 1 and tr.at[0, "entry_bar"] == 1 and tr.at[0, "exit_bar"] == 2 and tr.at[0, "exit_reason"] == "SL"
    ep = 99.5 * (1 + slip)
    xp = min(99.5, 97.5) * (1 - slip)
    assert tr.at[0, "entry_px"] == pytest.approx(ep) and tr.at[0, "exit_px"] == pytest.approx(xp)
    q = tr.at[0, "qty"]
    assert tr.at[0, "pnl"] == pytest.approx(q * (xp - ep) - q * ep * tk - q * xp * tk)
    assert tr.at[0, "r_multiple"] == pytest.approx(tr.at[0, "pnl"] / (q * (ep - 97.5)))


def test_gap_below_stop_fills_at_open_and_same_bar_sl_tp_is_stop():
    df = bars([(100, 101, 99, 100), (100, 101, 99, 100), (96.0, 112.0, 95.5, 111.0), (111, 112, 110, 111)])
    tr = run_backtest(df, one_signal(df, 1, sl=97.0, tp=110.0), NOFUND).trades
    assert tr.at[0, "exit_reason"] == "SL" and tr.at[0, "exit_px"] == pytest.approx(96.0 * (1 - NOFUND.slippage))


def test_mark_to_market_equity_includes_unrealised_pnl():
    df = bars([(100, 101, 99, 100), (100, 101, 99.5, 100), (100, 103, 99.8, 102), (102, 103, 101, 101.5)])
    res = run_backtest(df, one_signal(df, 1, sl=98.0, tp=120.0), NOFUND)
    t = res.trades.iloc[0]
    wallet = NOFUND.initial_capital - t["qty"] * t["entry_px"] * NOFUND.taker_fee
    for i in (2, 3):
        assert res.equity.iloc[i] == pytest.approx(wallet + t["qty"] * (df["close"].iloc[i] - t["entry_px"]))


def test_funding_boundaries():
    # 2021-03-01 07:00 开盘的信号 K 线在 08:00 收盘入场：08:00 的结算不收；16:00 的结算按 16:00 那根开盘价收
    rows = [(100, 101, 99, 100)] * 7 + [(100, 101, 99.5, 100)] + [(100, 101, 99.5, 100.5)] * 9 + [(100.5, 101, 90, 95)]
    df = bars(rows)
    fund = pd.DataFrame({"funding_rate": [0.001, 0.002, 0.003], "funding_interval_hours": [8, 8, 8]},
                        index=pd.DatetimeIndex([T0, T0 + 8 * H, T0 + 16 * H], name="funding_time"))
    fund = pd.concat([fund, pd.DataFrame({"funding_rate": [0.0], "funding_interval_hours": [8]},
                                         index=pd.DatetimeIndex([T0 + 24 * H], name="funding_time"))])
    cfg = BASELINE_BACKTEST
    res = run_backtest(df, one_signal(df, 7, sl=95.0, tp=120.0), cfg, fund)
    t = res.trades.iloc[0]
    assert t["entry_time"] == T0 + 8 * H and t["exit_bar"] == 17          # 17 = 16:00 开盘、止损所在 K 线
    assert t["n_funding"] == 1 and t["funding"] == pytest.approx(t["qty"] * df["open"].iloc[16] * 0.003)


def test_poc_partial_then_breakeven_same_bar_is_worst_case():
    df = bars([(100, 101, 99, 100), (100, 101, 99.5, 100), (100, 104.0, 99.0, 100.5), (100.5, 101, 100, 100.5)])
    cfg = NOFUND.with_(tp_mode="poc_partial", partial_frac=0.5, move_to_breakeven=True)
    tr = run_backtest(df, one_signal(df, 1, sl=97.0, tp=110.0, poc=103.0), cfg).trades
    assert tr.at[0, "partial_bar"] == 2 and tr.at[0, "exit_bar"] == 2 and tr.at[0, "exit_reason"] == "BE"
    cfg1 = cfg.with_(move_to_breakeven=False)                          # D1：不上移止损，余仓继续持有
    tr1 = run_backtest(df, one_signal(df, 1, sl=97.0, tp=110.0, poc=103.0), cfg1).trades
    assert tr1.at[0, "partial_bar"] == 2 and tr1.at[0, "exit_reason"] == "OPEN"


# ---------------------------------------------------------------------------
# 路径撮合器（C1）与单事件模拟器
# ---------------------------------------------------------------------------
def test_engine_path_reproduces_backtester_on_is(base):
    _, fund, _ = load_is_market()
    for cfg in (BASELINE_BACKTEST, V.d_backtest(BASELINE_BACKTEST, False), V.d_backtest(BASELINE_BACKTEST, True)):
        sig = base["signals"]
        a = run_backtest(base["df1h"], sig, cfg, fund)
        b = run_path(base["df1h"], sig, cfg, fund)
        assert np.array_equal(a.equity.to_numpy(), b["equity"].to_numpy())
        for col in ("entry_bar", "exit_bar", "entry_px", "exit_px", "qty", "fees", "funding", "pnl"):
            assert np.array_equal(a.trades[col].to_numpy(), b["trades"][col].to_numpy()), col


def test_limit_order_rules():
    tick = 0.01
    rows = [(100, 101, 99, 100), (100, 101, 99.0, 100.6),          # 信号 K 线：(high + low) / 2 = 100.0 < 收盘 100.6
            (100.6, 101, 100.05, 100.5),                             # 未穿价（low 100.05 > 100 − tick）
            (100.5, 100.8, 99.99, 100.4),                            # 穿价成交（low 99.99 ≤ 99.99）
            (100.4, 120, 100.2, 119)]                                # 下一根触及止盈
    df = bars(rows)
    fund = pd.DataFrame({"funding_rate": [0.0], "funding_interval_hours": [8]},
                        index=pd.DatetimeIndex([T0 + 8 * H], name="funding_time"))
    cfg = NOFUND
    s = one_signal(df, 1, sl=98.0, tp=110.0)
    lim = np.full(len(df), np.nan)
    lim[1] = 100.0
    r = run_path(df, s, cfg, fund.iloc[:0], limit_price=lim, valid_bars=3, fill_rule="through")
    t = r["trades"].iloc[0]
    assert t["fill_bar"] == 3 and t["entry_px"] == pytest.approx(100.0 * (1 + cfg.slippage))
    assert t["fees"] == pytest.approx(t["qty"] * t["entry_px"] * cfg.maker_fee + t["qty"] * t["exit_px"] * cfg.maker_fee)
    assert t["exit_bar"] == 4 and t["exit_reason"] == "TP"
    rt = run_path(df, s, cfg, fund.iloc[:0], limit_price=lim, valid_bars=3, fill_rule="touch")
    assert rt["trades"].iloc[0]["fill_bar"] == 3                    # 触价口径：100.05 > 100 仍不成交，99.99 成交
    short = run_path(df, s, cfg, fund.iloc[:0], limit_price=lim, valid_bars=1, fill_rule="through")
    assert len(short["trades"]) == 0 and short["orders"]["status"].iloc[0] == "expired_unfilled"
    # 成交那一根触及止损：成交后再止损（最坏路径）；该根的止盈不计
    df2 = bars([(100, 101, 99, 100), (100, 101, 99.0, 100.6), (100.6, 115, 97.0, 99.0), (99, 100, 98.5, 99)])
    r2 = run_path(df2, s, cfg, fund.iloc[:0], limit_price=lim, valid_bars=3, fill_rule="through")
    t2 = r2["trades"].iloc[0]
    assert t2["fill_bar"] == 2 and t2["exit_bar"] == 2 and t2["exit_reason"] == "SL" and bool(t2["fill_and_exit_same_bar"])
    # 限价 ≥ 信号收盘：挂出即可成交，按市价单处理
    lim3 = lim.copy()
    lim3[1] = 100.7
    r3 = run_path(df, s, cfg, fund.iloc[:0], limit_price=lim3)
    assert r3["orders"]["status"].iloc[0] == "filled_marketable_at_placement"
    assert r3["trades"].iloc[0]["entry_px"] == pytest.approx(100.6 * (1 + cfg.slippage))


def test_event_sim_reproduces_backtester_on_is(base):
    tr = base["result"].trades
    ev = simulate_events(base["df1h"], tr["entry_bar"].to_numpy(), tr["sl"].to_numpy(), tr["tp"].to_numpy(),
                         tr["poc"].to_numpy(), bcfg=BASELINE_BACKTEST)
    assert np.array_equal(ev["exit_bar"], tr["exit_bar"].to_numpy())
    r_xf = (tr["gross_pnl"] - tr["fees"]) / (tr["qty"] * (tr["entry_px"] - tr["sl"]))
    assert np.allclose(ev["r"], r_xf.to_numpy(), rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# B1 状态机副本与过滤器
# ---------------------------------------------------------------------------
def test_b1_copy_disabled_is_identical_on_is(base):
    import pipeline
    with V.b1_patched(base["df1h"], on=False):
        off = pipeline.run("BTCUSDT", BASELINE_STRATEGY, BASELINE_BACKTEST, end=IS_END, segments=())
    for col in base["signals"].columns:
        assert base["signals"][col].equals(off["signals"][col]), col
    assert base["result"].trades.equals(off["result"].trades)


ABOVE = (101.0, 101.8, 100.8, 101.5)
TOUCH = (101.0, 101.2, 100.2, 100.6)                 # 第 1 次测试，低点 100.2
PIN_NO_SFP = (100.95, 101.05, 100.3, 101.0)          # 看涨 Pinbar，低点高于第 1 次低点
PIN_SFP = (100.9, 101.0, 99.7, 100.95)               # 看涨 Pinbar，刺破第 1 次低点后收回 VAL 上方
PIN_DEEP = (100.9, 101.0, 99.0, 100.95)              # 下影线刺穿支撑带下沿（99.5）后收回 VAL 上方


def run_b1(rows, on, cfg=BASELINE_STRATEGY):
    a = np.asarray([tuple(r) + (100.0,) for r in rows], dtype=np.float64)
    o, h, l, c, v = (np.ascontiguousarray(a[:, k]) for k in range(5))
    n = len(a)
    open_ns = np.int64(T0.value) + np.arange(n, dtype=np.int64) * np.int64(3600 * 10**9)
    args = (o, h, l, c, v, np.ones(n), np.full(n, 100.0), open_ns, np.zeros(n, np.int64), np.array([100.0]),
            np.array([110.0]), np.array([np.int64(T0.value) - np.int64(100 * 3600 * 10**9)]),
            bullish_pinbar(o, h, l, c, cfg), bullish_engulfing(o, h, l, c, cfg)) + state_machine_params(cfg)
    return V.b1_state_machine()(*args, on, np.full(n, 0.01))


@pytest.mark.parametrize("last", [PIN_SFP, PIN_DEEP])
def test_b1_requires_sfp_and_tolerates_the_wick(last):
    rows = [ABOVE, TOUCH, ABOVE, PIN_NO_SFP, last]
    off = run_b1(rows, False)
    on = run_b1(rows, True)
    assert off[4][3] and not off[4][4]                  # 基线：第 3 根已发信号并消费序列
    assert not on[4][3] and on[4][4]                    # B1：第 3 根不满足 SFP，序列继续；第 4 根刺破后收回 → 发信号
    assert on[0][4] != 5 and (on[0][:5] != 5).all()     # 盘中下影线不触发失效（INVALID = 5）
    assert on[6][4] == pytest.approx(min(100.2, last[2]) - 0.2)   # 止损公式不变


def test_filter_definitions_hand_computed():
    closes = np.array([100.0, 102.0, 101.0, 103.0, 102.0])
    d4 = pd.DataFrame({"close": closes})
    bx = pd.DataFrame({"pivot_a_idx": [0, 1], "pivot_b_idx": [4, 2]})
    er = V.box_efficiency_ratio(bx, d4)
    assert er[0] == pytest.approx(2.0 / (2 + 1 + 2 + 1))      # |102 − 100| / (2 + 1 + 2 + 1)
    assert er[1] == pytest.approx(1.0)                          # 单步：|101 − 102| / |101 − 102|
    flat = V.box_efficiency_ratio(pd.DataFrame({"pivot_a_idx": [0], "pivot_b_idx": [2]}), pd.DataFrame({"close": [5.0] * 3}))
    assert flat[0] == 0.0
    assert V.E_PRED_SLIP == 0.0002 and V.RR_MIN_A1 == 1.5 and V.POS_MAX_A2 == 0.5 and V.ER_MAX_A3 == 0.35 and V.C1_VALID_BARS == 3


# ---------------------------------------------------------------------------
# IS 边界与 IS 内前缀截断一致性
# ---------------------------------------------------------------------------
def test_is_boundary_assertion():
    df = bars([(100, 101, 99, 100)] * 3, start=IS_END - 2 * H)
    fake = {"df1h": df, "df4h": df.iloc[:0].assign(available_time=pd.DatetimeIndex([], tz="UTC")),
            "boxes": pd.DataFrame({"available_time": pd.DatetimeIndex([], tz="UTC")}), "signals": df, "result": None}
    with pytest.raises(ISBoundaryError):
        assert_is_only(fake)


def test_prefix_truncation_consistency_within_is(base):
    """在 IS 内的关键事件处截断：截断数据上算出的箱体、逐根信号诊断、已了结交易与净值，必须与完整 IS 结果的同一前缀逐位一致。"""
    df1h, _, spot = load_is_market()
    _, fund, _ = load_is_market()
    tr = base["result"].trades
    boxes = base["boxes"]
    rng = np.random.default_rng(20261105)
    ev = np.concatenate([tr["entry_bar"].to_numpy() + 1, tr["exit_bar"].to_numpy() + 1,
                         np.searchsorted(df1h.index, pd.DatetimeIndex(boxes["available_time"]))])
    cuts = np.unique(rng.choice(ev[(ev > 400) & (ev < len(df1h) - 1)], 12, replace=False))
    for t in cuts:
        cut = df1h.iloc[:t]
        bx, d4 = build_boxes(cut, BASELINE_STRATEGY)
        sig = generate_signals(cut, BASELINE_STRATEGY, bx, d4, spot)
        res = run_backtest(cut, sig, BASELINE_BACKTEST, fund[fund.index <= cut.index[-1] + H])
        known = boxes[pd.DatetimeIndex(boxes["available_time"]) <= cut.index[-1] + H]
        pd.testing.assert_frame_equal(bx.reset_index(drop=True), known.reset_index(drop=True), check_dtype=False)
        full_sig = base["signals"].iloc[:t]
        for col in sig.columns:
            assert np.array_equal(sig[col].to_numpy(), full_sig[col].to_numpy(), equal_nan=sig[col].dtype.kind == "f"), (t, col)
        done = res.trades[res.trades["exit_reason"] != "OPEN"]
        ref = tr[tr["exit_bar"] < t].reset_index(drop=True)
        for col in ("entry_bar", "exit_bar", "entry_px", "exit_px", "qty", "fees", "funding", "pnl"):
            assert np.array_equal(done[col].to_numpy(), ref[col].to_numpy()), (t, col)
        assert np.array_equal(res.equity.to_numpy(), base["result"].equity.to_numpy()[:t])


# ---------------------------------------------------------------------------
# 数据层修复（审计 D-1、D-4、D-5）：合成数据 + 本地桩，不访问网络
# ---------------------------------------------------------------------------
def _full_grid_frame():
    import data_loader as dl
    idx = pd.date_range(dl.RESEARCH_START, dl.RESEARCH_END - H, freq="h", tz="UTC")
    n = len(idx)
    return pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": np.full(n, 50.0),
                         "quote_volume": 5000.0, "count": 10.0, "taker_buy_volume": 25.0}, index=idx)


def _okx_stub(symbol, start, end, bar="1H"):
    idx = pd.date_range(start, end - H, freq="h", tz="UTC")
    return pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": np.full(len(idx), 25.0),
                         "quote_volume": 2500.0, "count": np.nan, "taker_buy_volume": np.nan}, index=idx)


def test_gap_fill_keeps_placeholder_maintenance_missing(monkeypatch):
    import data_loader as dl
    monkeypatch.setattr(dl, "fetch_okx_klines", _okx_stub)
    df = _full_grid_frame()
    maint = pd.date_range("2021-06-01 00:00", periods=7, freq="h", tz="UTC")       # 7 小时维护占位（已剔除）
    archive = pd.date_range("2021-08-01 00:00", periods=6, freq="h", tz="UTC")     # 6 小时归档缺失
    df = df.drop(maint).drop(archive)
    out, rec = dl.fill_archive_gaps(df, "BTCUSDT", placeholders=[str(t) for t in maint])
    assert not out.index.isin(maint).any()                                         # 停机保持缺失，不伪造 K 线
    assert archive.isin(out.index).all() and (out.loc[archive, "fill_source"] == "okx_swap").all()
    assert len(rec) == 1 and rec[0]["missing_hours"] == 6
    assert out.loc[archive, "volume"].notna().all()


def test_gap_fill_refuses_without_pre_gap_overlap(monkeypatch):
    import data_loader as dl
    monkeypatch.setattr(dl, "fetch_okx_klines", _okx_stub)
    df = _full_grid_frame()
    head = pd.date_range(dl.RESEARCH_START, periods=6, freq="h", tz="UTC")          # 缺口位于数据开头：没有缺口前样本
    with pytest.raises(ValueError):
        dl.fill_archive_gaps(df.drop(head), "BTCUSDT")


def test_missing_cache_raises_instead_of_rebuilding(monkeypatch, tmp_path):
    import data_loader as dl
    monkeypatch.setattr(dl, "_proc_path", lambda name: tmp_path / f"{name}.parquet")
    monkeypatch.setattr(dl, "build_cache", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network rebuild")))
    with pytest.raises(FileNotFoundError):
        dl.load_klines("BTCUSDT", "1h", "perp")


# ---------------------------------------------------------------------------
# 路径撮合器修复（审计 E-1、E-2）
# ---------------------------------------------------------------------------
def _gap_frame(gap_hours):
    t = [T0 + k * H for k in range(4)] + [T0 + (4 + gap_hours + k) * H for k in range(3)]
    rows = [(100, 101, 99, 100), (100, 101, 99, 100), (100, 101, 99, 100), (100, 101, 99.5, 100),
            (100.5, 102, 99.8, 101), (101, 102, 100, 101.5), (101.5, 102, 101, 101.8)]
    df = bars(rows)
    df.index = pd.DatetimeIndex(t)
    return df


def test_engine_path_charges_funding_inside_a_gap_like_the_backtester():
    df = _gap_frame(6)                                         # 04:00–09:00 缺失；08:00 的结算落在缺口内
    fund = pd.DataFrame({"funding_rate": [0.001], "funding_interval_hours": [8]},
                        index=pd.DatetimeIndex([T0 + 8 * H], name="funding_time"))
    s = one_signal(df, 3, sl=95.0, tp=120.0)
    a = run_backtest(df, s, BASELINE_BACKTEST, fund).trades
    b = run_path(df, s, BASELINE_BACKTEST, fund)["trades"]
    assert a.at[0, "funding"] > 0 and b.at[0, "funding"] == pytest.approx(a.at[0, "funding"], rel=0, abs=1e-9)
    df16 = _gap_frame(14)                                      # 08:00 与 16:00 两次结算落在同一根 → 报错
    fund16 = pd.DataFrame({"funding_rate": [0.001, 0.002], "funding_interval_hours": [8, 8]},
                          index=pd.DatetimeIndex([T0 + 8 * H, T0 + 16 * H], name="funding_time"))
    with pytest.raises(ValueError):
        run_path(df16, one_signal(df16, 3, sl=95.0, tp=120.0), BASELINE_BACKTEST, fund16)


def test_engine_path_rejects_unsupported_configs_and_honours_funding_switch():
    df = bars([(100, 101, 99, 100)] * 4)
    s = one_signal(df, 1, sl=95.0, tp=120.0)
    fund = pd.DataFrame({"funding_rate": [0.0], "funding_interval_hours": [8]},
                        index=pd.DatetimeIndex([T0], name="funding_time"))
    for bad in (dict(venue="spot"), dict(sizing="risk"), dict(compounding=False), dict(time_stop_bars=3),
                dict(exit_on_breakdown=True)):
        with pytest.raises(ValueError):
            run_path(df, s, BASELINE_BACKTEST.with_(**bad), fund)
    r = run_path(df, s, NOFUND, fund)
    assert (r["trades"]["funding"] == 0).all()
