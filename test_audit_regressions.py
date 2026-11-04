"""
test_audit_regressions.py — Phase 1 对抗式审计确认问题的回归测试（合成数据，结果可手算）。

每个测试对应审计报告中的一条发现（编号见 docs/phase_logs/PHASE1_engine.md 第 6 节），
先在审计时的代码（commit ee26506）上复现失败，再在修复后的代码上通过。

合成状态机场景统一采用：VAL = 100，ATR = 1，k1 = k2 = 0.5 → 支撑带 [99.5, 100.5]；均量 100，放量阈值 150。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pipeline
from backtester import BacktestResult, run_backtest
from config import OOS_END, OOS_START, RESEARCH_DATA_END, BacktestConfig, StrategyConfig
from metrics import (OOSLockedError, daily_equity, daily_returns, monthly_returns, segment_metrics, trade_stats)
from ppvp_engine import _value_area
from strategy_state_machine import (ARMED, DEPARTED, INVALID, TOUCH1, TOUCH2, _run_state_machine, bullish_engulfing,
                                    bullish_pinbar, state_machine_params)

HOUR = np.int64(3600 * 10**9)
BASE = np.int64(1_704_067_200) * np.int64(10**9)          # 2024-01-01 00:00 UTC
SM_OUT = ("state", "zlo", "zhi", "brk", "sig", "pat", "sl", "tp", "t1_low", "t1_bar", "t2_low", "t2_bar", "t2_start",
          "box", "need")

ABOVE = (101.0, 101.8, 100.8, 101.5)        # 整根位于带上方（low 100.8 > 带上沿 100.5）
TOUCH = (101.0, 101.2, 100.2, 100.6)        # 触及支撑带，无形态
PIN = (100.9, 101.0, 99.7, 100.95)          # 触及支撑带的看涨 Pinbar（下影 1.2 / 振幅 1.3）
BRK = (100.3, 100.4, 98.8, 99.0, 200.0)     # 放量实体跌穿带下沿（实体 1.3 ATR，量 2 倍均量）
BELOW = (99.0, 99.3, 98.6, 99.1)            # 带下方
RECLAIM = (99.4, 101.0, 99.3, 100.9)


def sm(bars, boxes, cfg: StrategyConfig | None = None, atr=1.0, vsma=100.0) -> dict:
    """在合成 1H 序列上直接运行状态机核心。boxes: [{'at': 激活 bar, 'val':, 'vah':, 'ws': 窗口起点 bar}]。"""
    cfg = cfg or StrategyConfig()
    a = [tuple(b) + ((100.0,) if len(b) == 4 else ()) for b in bars]
    arr = np.asarray(a, dtype=np.float64)
    o, h, l, c, v = (np.ascontiguousarray(arr[:, k]) for k in range(5))
    n = len(arr)
    open_ns = BASE + np.arange(n, dtype=np.int64) * HOUR
    box_idx = np.full(n, -1, dtype=np.int64)
    for k, bx in enumerate(boxes):
        box_idx[bx["at"]:] = k
    bval = np.array([bx["val"] for bx in boxes], dtype=np.float64)
    bvah = np.array([bx.get("vah", bx["val"] + 10.0) for bx in boxes], dtype=np.float64)
    bws = np.array([BASE + np.int64(bx.get("ws", -100)) * HOUR for bx in boxes], dtype=np.int64)
    pin, eng = bullish_pinbar(o, h, l, c, cfg), bullish_engulfing(o, h, l, c, cfg)
    args = (o, h, l, c, v, np.full(n, atr), np.full(n, vsma), open_ns, box_idx, bval, bvah, bws, pin, eng) \
        + state_machine_params(cfg)
    out = dict(zip(SM_OUT, _run_state_machine(*args)))
    py = dict(zip(SM_OUT, _run_state_machine.py_func(*args)))
    for k in SM_OUT:                                      # JIT 与纯 Python 路径必须逐位一致
        assert np.array_equal(np.asarray(out[k]).view(np.uint8), np.asarray(py[k]).view(np.uint8)), k
    return out


def _breakdown_then_new_box(new_box: dict, reclaim_close_in_band=False):
    bars = [ABOVE, TOUCH, BRK, (99.0, 99.8, 98.9, 99.7) if reclaim_close_in_band else BELOW, BELOW, RECLAIM,
            ABOVE, TOUCH] + [ABOVE] * 6 + [PIN]
    boxes = [{"at": 0, "val": 100.0, "ws": -50}, dict({"at": 4}, **new_box)]
    return bars, boxes


# ----------------------------------------------------------------------------
# SPEC-1：放量跌穿后的失效不应被“重叠的新箱体”解除
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("reject_broken", [False, True])
@pytest.mark.parametrize("close_in_band", [False, True])
def test_spec1_invalid_persists_on_overlapping_box(reject_broken, close_in_band):
    bars, boxes = _breakdown_then_new_box({"val": 100.0, "ws": -10}, close_in_band)
    out = sm(bars, boxes, StrategyConfig(reject_broken_at_activation=reject_broken))
    assert out["state"][2] == INVALID
    assert (out["state"][2:] == INVALID).all(), "an overlapping box built on pre-breakdown data must not re-arm"
    assert not out["sig"].any()


def test_spec1_invalid_lifted_by_box_built_after_breakdown():
    bars, boxes = _breakdown_then_new_box({"val": 100.0, "ws": 3})      # 窗口起点 = 跌穿 K 线收盘
    out = sm(bars, boxes)
    assert out["state"][4] == ARMED and out["need"][4]                  # 激活时价格在带下方 → 须先到带上方
    assert out["state"][7] == TOUCH1 and out["state"][13] == DEPARTED
    assert out["sig"][14] and out["sig"].sum() == 1
    assert out["sl"][14] == pytest.approx(min(100.2, 99.7) - 0.2)


def test_spec1_invalid_lifted_by_non_overlapping_box():
    bars, boxes = _breakdown_then_new_box({"val": 103.0, "ws": -10})
    out = sm(bars, boxes)
    assert out["state"][3] == INVALID and out["state"][4] == ARMED


def test_spec1_chain_of_overlapping_boxes_compares_with_broken_level():
    bars = [ABOVE, TOUCH, BRK, BELOW, BELOW, BELOW, RECLAIM, ABOVE, TOUCH] + [ABOVE] * 6 + [PIN]
    boxes = [{"at": 0, "val": 100.0, "ws": -50}, {"at": 4, "val": 100.8, "ws": -10}, {"at": 5, "val": 101.6, "ws": -10}]
    out = sm(bars, boxes)
    # 100.8 与被跌穿的 100 重叠 → 仍失效；101.6 与 100 相距 1.6 ATR > 1 ATR → 不同支撑区，重新 ARMED
    assert out["state"][4] == INVALID and out["state"][5] == ARMED


# ----------------------------------------------------------------------------
# SPEC-2：第 1 次测试必须是“回落”触及（激活时位于带下方的 K 线不算）
# ----------------------------------------------------------------------------
def test_spec2_first_test_must_come_from_above():
    bars = [(95.0, 95.4, 94.8, 95.0), (95.0, 95.5, 94.5, 95.2), (95.2, 101.2, 95.1, 101.0)] + [ABOVE] * 6 \
        + [PIN] + [ABOVE] * 6 + [PIN]
    out = sm(bars, [{"at": 1, "val": 100.0}])
    assert out["state"][1] == ARMED and out["need"][1]                   # 激活 K 线在带下方：不是第 1 次测试
    assert out["state"][3] == ARMED and not out["need"][3]               # 到带上方后才开始等待回落
    assert out["state"][9] == TOUCH1 and not out["sig"][9]               # 第一次真正的回落 = 第 1 次测试，不出信号
    assert out["sig"][16] and out["sig"].sum() == 1
    assert out["sl"][16] == pytest.approx(99.7 - 0.2)                    # 止损锚定真实的两次测试，而非 94.5


# ----------------------------------------------------------------------------
# SPEC-3：信号消费后必须先脱离到带上方，才能开始新的第 1 次测试
# ----------------------------------------------------------------------------
def test_spec3_new_sequence_requires_departure_after_signal():
    bars = [ABOVE, TOUCH] + [ABOVE] * 6 + [PIN, TOUCH] + [ABOVE] * 6 + [PIN]
    out = sm(bars, [{"at": 0, "val": 100.0}])
    assert out["sig"][8]
    assert out["state"][9] == ARMED and out["need"][9]                   # 仍在带内：不是新的第 1 次测试
    assert out["state"][16] == TOUCH1 and not out["sig"][16]             # 只需一次新测试就出信号的旧行为已消除
    assert out["sig"].sum() == 1


# ----------------------------------------------------------------------------
# SPEC-4：t2_bar / t1_bar 指向低点所在 K 线；t2_start 为第 2 次测试首根
# ----------------------------------------------------------------------------
def test_spec4_t2_bar_points_at_low_bar():
    bars = [ABOVE, TOUCH] + [ABOVE] * 6 + [(100.6, 100.7, 100.0, 100.1), (100.05, 100.9, 99.9, 100.8)]
    out = sm(bars, [{"at": 0, "val": 100.0}])
    assert out["sig"][9] and out["pat"][9] == 2                           # 看涨吞没
    assert out["t2_low"][9] == 99.9 and out["t2_bar"][9] == 9 and out["t2_start"][9] == 8
    assert out["t1_low"][9] == 100.2 and out["t1_bar"][9] == 1


def test_spec4_roll_carries_bar_of_low():
    bars = [ABOVE, TOUCH] + [ABOVE] * 6 + [(100.6, 100.7, 100.0, 100.1), (100.2, 100.4, 99.8, 100.1)] \
        + [ABOVE] * 6 + [PIN]
    out = sm(bars, [{"at": 0, "val": 100.0}])
    # 第 2 次测试（bar 8-9，无形态）离开后滚动为新的第 1 次测试，其低点 99.8 在 bar 9
    assert out["sig"][16] and out["t1_low"][16] == 99.8 and out["t1_bar"][16] == 9


# ----------------------------------------------------------------------------
# SPEC-12：carry 的重叠判定以“第 1 次测试所在箱体的 VAL”为基准，链式漂移不能保留进度
# ----------------------------------------------------------------------------
def test_spec12_carry_compares_with_first_test_val():
    bars = [ABOVE, TOUCH, ABOVE, (102.0, 102.6, 101.6, 102.3), (102.4, 103.0, 102.2, 102.8)]
    boxes = [{"at": 0, "val": 100.0}, {"at": 3, "val": 100.8}, {"at": 4, "val": 101.6}]
    out = sm(bars, boxes)
    assert out["state"][1] == TOUCH1 and out["state"][3] == DEPARTED     # 0.8 ATR：同一支撑区，保留进度
    assert out["state"][4] == ARMED                                       # 距第 1 次测试 VAL 1.6 ATR：重置


# ----------------------------------------------------------------------------
# SPEC-5 / SPEC-6 / SPEC-9：显式参数化的模糊规格
# ----------------------------------------------------------------------------
def _lag_case():
    return [ABOVE, TOUCH] + [ABOVE] * 6 + [(100.6, 100.7, 100.0, 100.1), (100.05, 100.9, 99.9, 100.8)]


def test_spec5_max_pattern_lag_bars():
    assert sm(_lag_case(), [{"at": 0, "val": 100.0}])["sig"][9]
    assert sm(_lag_case(), [{"at": 0, "val": 100.0}], StrategyConfig(max_pattern_lag_bars=1))["sig"][9]
    assert not sm(_lag_case(), [{"at": 0, "val": 100.0}], StrategyConfig(max_pattern_lag_bars=0))["sig"].any()


def test_spec6_pin_require_bull_close():
    o, h, l, c = (np.array([x]) for x in (100.7, 100.8, 99.6, 100.5))   # 阴线实体、下影 75%
    assert bullish_pinbar(o, h, l, c, StrategyConfig())[0]
    assert not bullish_pinbar(o, h, l, c, StrategyConfig(pin_require_bull_close=True))[0]


def test_spec9_sl_anchor_test_vs_pattern():
    bars = [ABOVE, TOUCH] + [ABOVE] * 6 + [(100.4, 100.45, 98.0, 98.3), (99.9, 100.0, 99.6, 99.95)]
    base = sm(bars, [{"at": 0, "val": 100.0}])
    pat = sm(bars, [{"at": 0, "val": 100.0}], StrategyConfig(sl_anchor="pattern"))
    assert base["sig"][9] and pat["sig"][9]
    assert base["sl"][9] == pytest.approx(98.0 - 0.2)                      # 第 2 次测试全部 K 线的最低点
    assert pat["sl"][9] == pytest.approx(99.6 - 0.2)                       # 仅形态 K 线低点（规格字面）


# ----------------------------------------------------------------------------
# SPEC-11：POC / VA 平局规则
# ----------------------------------------------------------------------------
def test_spec11_poc_plateau_takes_lowest_bin_and_va_ties_go_up():
    hist = np.array([1.0, 5.0, 5.0 * (1 + 1e-13), 5.0, 1.0])
    assert _value_area(hist, 0.2)[0] == 1
    poc, lo, hi = _value_area(np.array([1.0, 3.0, 10.0, 3.0 * (1 - 1e-13), 1.0]), 0.6)
    assert (poc, lo, hi) == (2, 2, 3)


# ----------------------------------------------------------------------------
# EXECUTION：持仓期破位参照入场箱体 / POC 全平 / 配置校验 / 现货不融资
# ----------------------------------------------------------------------------
def _mk_df(rows):
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="1h", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)
    df["volume"] = 100.0
    df["quote_volume"] = 1e6
    return df


def test_execution1_breakdown_exit_uses_entry_box():
    rows = [[101, 101.5, 100.5, 101]] * 3 + [[104, 104.2, 102.8, 103.0], [101, 101.5, 100.5, 101],
                                              [100.5, 100.6, 98.8, 99.0], [99, 99.5, 98.5, 99]]
    df = _mk_df(rows)
    n = len(df)
    sig = pd.DataFrame({"signal": False, "sl": np.nan, "tp": np.nan, "poc_tp": np.nan,
                        "VAL": [100.0] * 2 + [104.0] * (n - 2),            # 持仓期间新箱体 VAL=104 生效
                        "breakdown": [False, False, False, True, False, False, False],   # 新箱体口径的破位（不应被使用）
                        "brk_bodyvol": [False, False, False, True, False, True, False],
                        "k1_atr": 0.5}, index=df.index)
    sig.loc[sig.index[0], ["signal", "sl", "tp", "poc_tp"]] = [True, 90.0, 120.0, 115.0]
    res = run_backtest(df, sig, BacktestConfig(include_funding=False, exit_on_breakdown=True))
    t = res.trades.iloc[0]
    # bar 3 收盘 103 > 入场箱体带下沿 99.5 → 不离场；bar 5 收盘 99 < 99.5 且放量实体 → 破位离场
    assert t["exit_reason"] == "BREAKDOWN" and t["exit_bar"] == 5
    with pytest.raises(ValueError):
        run_backtest(df, sig.drop(columns=["brk_bodyvol"]), BacktestConfig(include_funding=False, exit_on_breakdown=True))


@pytest.mark.parametrize("bar1_low", [100.5, 99.9])
def test_execution2_full_exit_at_poc_is_labelled_and_counted(bar1_low):
    df = _mk_df([[100, 100.5, 99.5, 100.0], [100.5, 103, bar1_low, 102.5], [102, 102.5, 101.5, 102]])
    sig = pd.DataFrame({"signal": [True, False, False], "sl": [95.0, np.nan, np.nan], "tp": [110.0, np.nan, np.nan],
                        "poc_tp": [102.0, np.nan, np.nan]}, index=df.index)
    res = run_backtest(df, sig, BacktestConfig(include_funding=False, tp_mode="poc_partial", partial_frac=1.0))
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "POC" and t["exit_bar"] == 1 and res.position.iloc[1] == 0.0
    assert t["exit_px"] == pytest.approx(102.0 * (1 - 0.0002), rel=1e-15)
    assert trade_stats(res.trades)["trades"] == 1


@pytest.mark.parametrize("kw", [{"venue": "PERP"}, {"venue": "futures"}, {"partial_frac": 1.5}, {"partial_frac": 0.0},
                                {"sizing": "Risk"}, {"tp_mode": "poc"}, {"time_stop_bars": 0}, {"slippage": -0.001},
                                {"max_leverage": 0.0}])
def test_execution4_backtest_config_validation(kw):
    with pytest.raises(ValueError):
        BacktestConfig(**kw)


@pytest.mark.parametrize("kw", [{"regime_filter": "sma"}, {"sl_anchor": "x"}, {"va_pct": 1.2}, {"n_bins": 1},
                                {"max_pattern_lag_bars": -1}])
def test_execution4_strategy_config_validation(kw):
    with pytest.raises(ValueError):
        StrategyConfig(**kw)


def test_execution7_spot_mode_never_borrows_cash():
    df = _mk_df([[100, 100.5, 99.5, 100.0]] * 4)
    sig = pd.DataFrame({"signal": [True, False, False, False], "sl": [90.0] + [np.nan] * 3,
                        "tp": [120.0] + [np.nan] * 3, "poc_tp": [110.0] + [np.nan] * 3}, index=df.index)
    for lev in (1.0, 1.5):
        res = run_backtest(df, sig, BacktestConfig(venue="spot", fixed_leverage=lev))
        t = res.trades.iloc[0]
        cash_after = t["equity_at_entry"] - t["notional"] - t["notional"] * 0.001
        assert cash_after >= -1e-9


# ----------------------------------------------------------------------------
# METRICS：分段覆盖、月度表、OOS 锁、固定基数、午间起点、缓存别名
# ----------------------------------------------------------------------------
def _mk_res(start, end, values=None, compounding=True, start_time=None):
    idx = pd.date_range(start + pd.Timedelta(hours=1), end, freq="1h", tz="UTC")
    eq = np.full(len(idx), 100_000.0) if values is None else values(idx)
    return BacktestResult(equity=pd.Series(eq, index=idx), position=pd.Series(0.0, index=idx), trades=pd.DataFrame(),
                          config=BacktestConfig(compounding=compounding), initial_capital=100_000.0,
                          start_time=start_time if start_time is not None else start)


def test_metrics1_segment_end_must_be_covered():
    res = _mk_res(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-10", tz="UTC"))
    with pytest.raises(ValueError):
        segment_metrics(res, pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-02-01", tz="UTC"))
    with pytest.raises(ValueError):
        segment_metrics(res, pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-05 12:00", tz="UTC"))
    ok = segment_metrics(res, pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-10", tz="UTC"))
    assert ok["days"] == 9 and ok["end"] == "2024-01-09"


def test_metrics1_is_val_day_counts():
    res = _mk_res(pd.Timestamp("2020-01-01", tz="UTC"), RESEARCH_DATA_END)
    from config import SEGMENTS
    assert segment_metrics(res, *SEGMENTS["IS"])["days"] == 1461
    assert segment_metrics(res, *SEGMENTS["VAL"])["days"] == 547


def test_metrics2_monthly_returns_first_row_is_first_real_month():
    res = _mk_res(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-04-01", tz="UTC"),
                  values=lambda idx: 100_000.0 * (1 + 0.001) ** np.arange(len(idx)))
    tbl = monthly_returns(daily_equity(res))
    assert list(tbl.index) == [2024] and list(tbl.columns) == [1, 2, 3]
    assert int(tbl.notna().sum().sum()) == 3


def test_metrics5_oos_windows_are_locked():
    res = _mk_res(pd.Timestamp("2025-06-01", tz="UTC"), pd.Timestamp("2025-09-01", tz="UTC"))
    with pytest.raises(OOSLockedError):
        segment_metrics(res, pd.Timestamp("2025-06-01", tz="UTC"), pd.Timestamp("2025-08-01", tz="UTC"))
    with pytest.raises(OOSLockedError):
        monthly_returns(daily_equity(res))
    segment_metrics(res, pd.Timestamp("2025-06-01", tz="UTC"), pd.Timestamp("2025-08-01", tz="UTC"), allow_oos=True)


def test_metrics8_non_compounding_segments_use_fixed_capital_base():
    s, m, e = (pd.Timestamp(x, tz="UTC") for x in ("2024-01-01", "2024-02-01", "2024-03-01"))

    def path(carry):
        def f(idx):
            pnl = np.where(idx > m, 2_000.0 * np.sin(np.arange(len(idx)) / 50.0), 0.0)
            return np.where(idx <= m, 100_000.0 + carry * np.arange(len(idx)) / max(1, (idx <= m).sum() - 1),
                            100_000.0 + carry + pnl)
        return f
    a = segment_metrics(_mk_res(s, e, path(0.0), compounding=False), m, e)
    b = segment_metrics(_mk_res(s, e, path(100_000.0), compounding=False), m, e)
    for k in ("total_return", "max_dd", "ann_vol", "sharpe"):
        assert a[k] == pytest.approx(b[k], rel=1e-12, abs=1e-15), k


def test_metrics9_midday_start_keeps_initial_capital_anchor():
    st = pd.Timestamp("2024-01-01 13:00", tz="UTC")
    res = _mk_res(st, pd.Timestamp("2024-01-03", tz="UTC"), values=lambda idx: np.full(len(idx), 105_000.0))
    d = daily_equity(res)
    assert d.index[0] == pd.Timestamp("2023-12-31", tz="UTC") and d.iloc[0] == 100_000.0
    assert daily_returns(d).iloc[0] == pytest.approx(0.05)


def test_metrics3_get_boxes_lock_checked_before_cache():
    key = ("BTCUSDT", OOS_END, StrategyConfig().structural_key())
    pipeline._BOXES[key] = ("sentinel", "sentinel")
    try:
        with pytest.raises(OOSLockedError):
            pipeline.get_boxes("BTCUSDT", StrategyConfig(), OOS_END)
        with pytest.raises(OOSLockedError):
            pipeline.load_market("BTCUSDT", OOS_END)
    finally:
        pipeline._BOXES.pop(key, None)


def test_metrics10_mutating_run_output_does_not_poison_cache():
    pipeline.clear_cache()
    out = pipeline.run("BTCUSDT", segments=())
    ref_close = out["df1h"]["close"].to_numpy().copy()
    ref_val = out["boxes"]["VAL"].to_numpy().copy()
    out["df1h"]["close"] *= 1.003
    out["df1h"]["high"] *= 1.003
    out["boxes"]["VAL"] *= 0.5
    again = pipeline.run("BTCUSDT", segments=())
    assert np.array_equal(again["df1h"]["close"].to_numpy(), ref_close)
    assert np.array_equal(again["boxes"]["VAL"].to_numpy(), ref_val)
    assert again["df1h"].index[-1] + pd.Timedelta(hours=1) == RESEARCH_DATA_END
    assert pipeline.load_market("BTCUSDT")[1].index[-1] == OOS_START      # 资金费含 T = end 的结算
    pipeline.clear_cache()
