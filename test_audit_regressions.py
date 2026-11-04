"""
test_audit_regressions.py — Phase 1 对抗式审计确认问题的回归测试（合成数据，结果可手算）。

每个测试对应一条审计或复核发现（编号见 docs/phase_logs/PHASE1_engine.md 第 6 至 9 节）。修复类测试先在修复前的代码上
复现失败、再在修复后的代码上通过；第三、四轮的“钉住”类测试约束此前没有测试覆盖、但实现正确的既有行为，在修复前的代码上
同样通过，作用是让对应的突变体（tools/mutation_test.py）失败。reference_state_machine 是按文档规则独立编写的参考实现，
与 Numba 内核在随机合成场景上逐位对拍。

合成状态机场景统一采用：VAL = 100，ATR = 1，k1 = k2 = 0.5 → 支撑带 [99.5, 100.5]；均量 100，放量阈值 150。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pipeline
from backtester import BacktestResult, run_backtest
from config import OOS_END, OOS_START, RESEARCH_DATA_END, BacktestConfig, StrategyConfig
from metrics import (OOSLockedError, daily_equity, daily_returns, drawdown_series, max_drawdown, monthly_returns,
                     segment_metrics, sharpe, sortino, trade_stats)
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


# ============================================================================
# 第二轮（修复复核）确认问题的回归测试：SM-R1 … SM-R7、EXEC-R2 … EXEC-R9
# ============================================================================
def _r1a_bars():
    return [ABOVE, TOUCH, BRK, BELOW, BELOW, RECLAIM, ABOVE, TOUCH] + [ABOVE] * 6 + [PIN]


@pytest.mark.parametrize("reject_broken", [False, True])
def test_smr1_detour_through_other_zone_cannot_rearm_broken_level(reject_broken):
    # 100 被放量跌穿 → VAL 103 的新箱体（不同支撑区）→ VAL 100.2 且窗口含跌穿前数据的箱体：必须判为失效
    boxes = [{"at": 0, "val": 100.0, "ws": -50}, {"at": 4, "val": 103.0, "ws": -10}, {"at": 6, "val": 100.2, "ws": 1}]
    out = sm(_r1a_bars(), boxes, StrategyConfig(reject_broken_at_activation=reject_broken))
    assert out["state"][6] == INVALID and (out["state"][6:] == INVALID).all()
    assert not out["sig"].any()


def test_smr1_detour_cannot_carry_onto_broken_level():
    bars = [ABOVE, TOUCH, BRK, BELOW, (99.2, 102.0, 99.1, 101.9), (102.0, 102.6, 101.8, 102.3),
            (102.2, 102.3, 101.4, 101.6), (101.9, 102.6, 101.8, 102.4)] + [(101.9, 102.6, 101.8, 102.4)] * 6 \
        + [(101.5, 101.6, 100.4, 101.55)]
    boxes = [{"at": 0, "val": 100.0, "ws": -50}, {"at": 4, "val": 101.2, "ws": -10}, {"at": 8, "val": 100.6, "ws": 1}]
    out = sm(bars, boxes)
    assert out["state"][6] == TOUCH1 and out["state"][7] == DEPARTED     # 在 101.2 支撑区上完成第 1 次测试
    assert out["state"][8] == INVALID and not out["sig"].any()          # 回到被跌穿的 100 支撑区：不得 carry


def test_smr1_memory_keeps_every_broken_zone():
    # 100 被放量跌穿 → 价格跌到 95 支撑区并把它也放量跌穿 → 反弹后出现与 100 重叠、仍含首次跌穿前数据的箱体：
    # 两条记忆互不覆盖，100 支撑区仍被判为失效
    bars = [ABOVE, TOUCH, BRK, (99.0, 99.2, 96.0, 96.2), (96.2, 96.5, 95.8, 96.0), (96.0, 96.1, 95.2, 95.4),
            (95.4, 95.5, 93.8, 94.0, 200.0), (94.0, 97.0, 93.9, 96.9), (96.9, 99.0, 96.8, 98.9),
            (99.5, 101.5, 99.4, 101.3), (101.3, 102.0, 101.0, 101.8), (101.4, 101.5, 100.6, 100.9)] \
        + [(101.2, 102.0, 101.0, 101.8)] * 6 + [(101.2, 101.3, 100.0, 101.25)]
    boxes = [{"at": 0, "val": 100.0, "ws": -50}, {"at": 4, "val": 95.0, "ws": -10}, {"at": 9, "val": 100.3, "ws": -5}]
    out = sm(bars, boxes)
    assert out["state"][2] == INVALID and out["state"][4] == ARMED and out["state"][5] == TOUCH1
    assert out["state"][6] == INVALID                                     # 95 支撑区也被放量跌穿
    assert (out["state"][9:] == INVALID).all() and not out["sig"].any()   # 回到 100 支撑区：仍失效


def _exhausted_bars():
    return [ABOVE, TOUCH] + [ABOVE] * 6 + [TOUCH, ABOVE, BRK, BELOW, BELOW, RECLAIM, ABOVE, TOUCH] + [ABOVE] * 6 + [PIN]


def test_smr2_breakdown_while_exhausted_invalidates_zone():
    cfg = StrategyConfig(roll_failed_second_touch=False)
    out = sm(_exhausted_bars(), [{"at": 0, "val": 100.0, "ws": -50}, {"at": 12, "val": 100.1, "ws": 5}], cfg)
    assert out["state"][9] == 6 and out["state"][10] == INVALID          # EXHAUSTED 期间放量跌穿 → INVALID
    assert (out["state"][10:] == INVALID).all() and not out["sig"].any()
    fresh = sm(_exhausted_bars(), [{"at": 0, "val": 100.0, "ws": -50}, {"at": 12, "val": 100.1, "ws": 11}], cfg)
    assert fresh["state"][12] == ARMED and fresh["sig"][22]              # 完全由跌穿后数据构成的箱体：可重新 ARMED


def test_smr3_same_zone_box_update_keeps_post_signal_departure_requirement():
    bars = [ABOVE, TOUCH] + [ABOVE] * 6 + [PIN, TOUCH, TOUCH] + [ABOVE] * 6 + [PIN]
    out = sm(bars, [{"at": 0, "val": 100.0}, {"at": 10, "val": 100.0}])
    assert out["sig"][8] and out["state"][10] == ARMED and out["need"][10]
    assert out["state"][17] == TOUCH1 and out["sig"].sum() == 1


def _c_bars():
    return [(95.0, 95.4, 94.8, 95.0), (95.0, 95.5, 94.5, 95.2), (95.2, 100.4, 95.1, 100.2),
            (100.2, 100.45, 99.9, 100.3)] + [ABOVE] * 6 + [PIN]


def test_smr3_below_band_arm_then_same_zone_update_still_requires_departure():
    out = sm(_c_bars(), [{"at": 1, "val": 100.0}, {"at": 3, "val": 100.0}])          # C1：已知价格自下方而来
    assert out["state"][3] == ARMED and out["need"][3]
    assert out["state"][10] == TOUCH1 and not out["sig"].any()


def test_smr7_in_band_activation_counts_first_touch():
    out = sm(_c_bars(), [{"at": 3, "val": 100.0}])                                    # C2：首次激活时价格在带内
    assert not out["need"][3] and out["state"][3] == TOUCH1 and out["sig"][10]


def test_smr7_activation_threshold_is_band_low():
    bars = [(100.0, 100.3, 99.7, 99.8), (99.8, 100.2, 99.6, 100.0)]                  # 上一根收盘 99.8 ∈ [99.5, 100.5]
    out = sm(bars, [{"at": 1, "val": 100.0}])
    assert not out["need"][1] and out["state"][1] == TOUCH1


def test_smr4_value_area_terminates_with_huge_volumes():
    import signal

    def _timeout(*_):
        raise TimeoutError("value-area loop did not terminate")
    old = signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(10)
    try:
        assert _value_area.py_func(np.array([2e9, 0.0, 3e9, 1e9]), 0.7) == (2, 0, 3)
        assert _value_area.py_func(np.array([2e9, 0.5, 3e9, 1e9]), 0.7) == (2, 0, 3)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    assert _value_area(np.array([2e9, 0.0, 3e9, 1e9]), 0.7) == (2, 0, 3)
    # va_pct = 1 且浮点累加略小于总量：两侧都已无桶时必须停止，不得越界回绕到另一端
    h = np.array([0.1, 0.2, 0.3])
    assert 0.3 + 0.2 + 0.1 < h.sum()
    assert _value_area.py_func(h, 1.0) == (2, 0, 2) and _value_area(h, 1.0) == (2, 0, 2)


def test_smr5_reject_is_box_level_and_writes_no_breakdown_memory():
    bars = [BELOW, (99.2, 100.0, 99.1, 99.9), (99.9, 101.2, 99.8, 101.0)] + [ABOVE] * 3 + [TOUCH] + [ABOVE] * 6 + [PIN]
    boxes = [{"at": 1, "val": 100.0, "ws": -20}, {"at": 5, "val": 100.2, "ws": -10}]
    out = sm(bars, boxes, StrategyConfig(reject_broken_at_activation=True))
    assert out["state"][1] == INVALID                                     # 激活时上一根收盘在带下方：否决该箱体
    assert out["state"][5] == ARMED                                       # 下一个箱体按自身条件判定（无放量跌穿记忆）
    assert out["sig"][13]


def _rebreak_bars(rebreak_close):
    return [ABOVE, TOUCH, BRK, BELOW, (100.8, 101.5, 100.7, 100.8), (100.8, 100.85, rebreak_close - 0.1, rebreak_close, 200.0),
            BELOW, BELOW]


def test_smr6_rebreakdown_measured_against_broken_level_and_fresh_boundary():
    boxes = lambda ws7: [{"at": 0, "val": 100.0, "ws": -50}, {"at": 4, "val": 100.9, "ws": -10},
                         {"at": 7, "val": 100.3, "ws": ws7}]
    # bar 5 收于 99.9：低于新箱体带下沿 100.4，但高于被跌穿的带下沿 99.5 → 不是对被跌穿位的再次跌穿
    out = sm(_rebreak_bars(99.9), boxes(4))
    assert out["brk"][5] and out["state"][4] == INVALID and out["state"][7] == ARMED
    # bar 5 收于 99.3 < 99.5：同一支撑区再次放量跌穿 → 记忆时刻后移，窗口起于 bar 4 的箱体不再是“全新结构”
    out = sm(_rebreak_bars(99.3), boxes(4))
    assert out["state"][7] == INVALID
    # 全新结构的边界：窗口起点 = 跌穿 K 线开盘（含跌穿 K 线）→ 仍失效；= 其收盘 → 重新 ARMED
    edge = lambda ws: [{"at": 0, "val": 100.0, "ws": -50}, {"at": 4, "val": 100.0, "ws": ws}]
    assert sm(_r1a_bars(), edge(2))["state"][4] == INVALID
    assert sm(_r1a_bars(), edge(3))["state"][4] == ARMED


def test_smr7_roll_moves_carry_reference_to_second_test_val():
    up = (102.0, 102.6, 101.6, 102.3)
    bars = [ABOVE, TOUCH, ABOVE, up, up, up, up, up, up, (101.8, 101.9, 101.0, 101.2), up,
            (102.8, 103.3, 102.5, 103.0)]
    boxes = [{"at": 0, "val": 100.0}, {"at": 3, "val": 100.8}, {"at": 11, "val": 101.7}]
    out = sm(bars, boxes)
    assert out["state"][9] == TOUCH2 and out["state"][10] == DEPARTED    # 第 2 次测试（在 100.8 箱体上）无形态 → 滚动
    assert out["state"][11] == DEPARTED                                    # 101.7 与滚动后基准 100.8 相距 0.9 ATR → carry


def test_execr3_cache_isolated_from_array_writes():
    pipeline.clear_cache()
    df, fund, _ = pipeline.load_market("BTCUSDT")
    ref = pipeline.load_market("BTCUSDT")[0]["low"].to_numpy().copy()
    df["low"].array[:] = df["low"].to_numpy() * 0.5
    fund["funding_rate"].array[:] = 0.0
    again, fund2, _ = pipeline.load_market("BTCUSDT")
    assert np.array_equal(again["low"].to_numpy(), ref) and fund2["funding_rate"].abs().sum() > 0
    boxes, _ = pipeline.get_boxes("BTCUSDT", StrategyConfig())
    v0 = boxes["VAL"].to_numpy().copy()
    boxes["VAL"].array[:] = 0.0
    assert np.array_equal(pipeline.get_boxes("BTCUSDT", StrategyConfig())[0]["VAL"].to_numpy(), v0)
    pipeline.clear_cache()


def test_execr4_capped_entries_are_flagged():
    df = _mk_df([[100, 100.5, 99.5, 100.0], [100, 100.2, 79.0, 80.0], [80, 80.5, 79.5, 80.0], [80, 80.5, 79.5, 80.0]])
    sig = pd.DataFrame({"signal": [True, False, True, False], "sl": [85.0, np.nan, 70.0, np.nan],
                        "tp": [120.0, np.nan, 100.0, np.nan], "poc_tp": [110.0, np.nan, 90.0, np.nan]}, index=df.index)
    res = run_backtest(df, sig, BacktestConfig(include_funding=False, compounding=False, fixed_leverage=1.5))
    assert list(res.trades["capped"]) == [False, True]                    # 亏损后固定名义 150k > 1.5 × 当期净值
    st = trade_stats(res.trades)
    assert st["capped_entries"] == 1


def test_execr5_funding_grid_coverage():
    df = _mk_df([[100.0 + k * 0.01, 100.5 + k * 0.01, 99.5 + k * 0.01, 100.0 + k * 0.01] for k in range(24)])
    sig = pd.DataFrame({"signal": False, "sl": np.nan, "tp": np.nan, "poc_tp": np.nan}, index=df.index)
    sig.loc[sig.index[15], ["signal", "sl", "tp", "poc_tp"]] = [True, 50.0, 500.0, 400.0]
    full = pd.DataFrame({"funding_rate": [1e-4] * 4, "funding_interval_hours": 8},
                        index=pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 08:00", "2024-01-01 16:00",
                                                "2024-01-02 00:00"], tz="UTC"))
    cfg = BacktestConfig(time_stop_bars=8)                                 # bar 23 收盘(=次日 00:00 结算)时间止损
    r_full = run_backtest(df, sig, cfg, full)
    t = r_full.trades.iloc[0]
    assert t["exit_bar"] == 23 and t["n_funding"] == 1 and t["funding_complete"] and not r_full.funding_end_missing
    r_miss = run_backtest(df, sig, cfg, full.iloc[:3])                     # 缺少恰在最后收盘时刻的结算
    assert r_miss.funding_end_missing and not r_miss.trades.iloc[0]["funding_complete"]
    with pytest.raises(ValueError, match="misses the settlement"):
        run_backtest(df, sig, cfg, full.drop(full.index[2]))               # 缺少区间内的 16:00 结算
    with pytest.raises(ValueError, match="duplicate"):
        run_backtest(df, sig, cfg, pd.concat([full, full.iloc[[1]]]).sort_index())
    tiny = df.iloc[1:3]                                                    # (01:00, 03:00] 内无结算网格点：不检查
    assert not run_backtest(tiny, sig.iloc[1:3], BacktestConfig(), full).funding_end_missing
    with pytest.raises(ValueError, match="no settlement within one day"):
        run_backtest(tiny, sig.iloc[1:3], BacktestConfig(), full.iloc[:0])   # 无法确定结算网格：报错


def test_execr6_empty_and_nonempty_tables_share_dtypes():
    df = _mk_df([[100, 100.5, 99.5, 100.0]] * 4)
    base = pd.DataFrame({"signal": False, "sl": np.nan, "tp": np.nan, "poc_tp": np.nan}, index=df.index)
    one = base.copy()
    one.loc[one.index[0], ["signal", "sl", "tp", "poc_tp"]] = [True, 90.0, 110.0, 105.0]
    e = run_backtest(df, base, BacktestConfig(include_funding=False)).trades
    f = run_backtest(df, one, BacktestConfig(include_funding=False)).trades
    assert len(e) == 0 and len(f) == 1
    assert list(e.dtypes.astype(str)) == list(f.dtypes.astype(str))


def test_execr7_nan_val_on_signal_bar_is_rejected_for_breakdown_exit():
    df = _mk_df([[100, 100.5, 99.5, 100.0]] * 4)
    sig = pd.DataFrame({"signal": [True, False, False, False], "sl": [90.0] + [np.nan] * 3, "tp": [110.0] + [np.nan] * 3,
                        "poc_tp": [105.0] + [np.nan] * 3, "VAL": np.nan, "brk_bodyvol": False, "k1_atr": 0.5},
                       index=df.index)
    with pytest.raises(ValueError, match="finite VAL"):
        run_backtest(df, sig, BacktestConfig(include_funding=False, exit_on_breakdown=True))


@pytest.mark.parametrize("kw", [{"slippage": float("nan")}, {"perp_maker_fee": float("nan")}, {"max_leverage": float("inf")},
                                {"target_vol": float("inf")}])
def test_execr8_backtest_config_rejects_non_finite(kw):
    with pytest.raises(ValueError):
        BacktestConfig(**kw)


@pytest.mark.parametrize("kw", [{"regime_sma_days": 0}, {"regime_slope_days": -3}, {"max_box_width_atr": -1.0},
                                {"max_box_width_pct": 0.0}, {"crash_cooldown_hours": -5}, {"crash_body_atr4": -1.0},
                                {"zone_k1": float("nan")}])
def test_execr8_strategy_config_rejects_invalid(kw):
    with pytest.raises(ValueError):
        StrategyConfig(**kw)


def test_execr9_metrics_locked_on_results_extending_into_oos():
    from config import VAL_END, VAL_START
    res = _mk_res(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-08-01", tz="UTC"))
    with pytest.raises(OOSLockedError):
        segment_metrics(res, VAL_START, VAL_END)                         # 分段本身在 OOS 之前，但结果含 OOS 期间的离场
    segment_metrics(res, VAL_START, VAL_END, allow_oos=True)


# ============================================================================
# 第三轮复核确认问题的回归测试：SM3-1 … SM3-6、DOC3-2、DOC3-10（状态机）
# ============================================================================
IN = (100.2, 100.45, 99.9, 100.2)           # 带内、无形态
IN_LOW = (100.3, 100.4, 99.6, 99.7)         # 带内，收盘 99.7 < VAL 100.3 箱体的带下沿 99.8
UP = (101.5, 102.2, 101.3, 101.9)           # 整根位于 100 与 100.3 两个箱体的带上方


def _post_signal_chop():
    """bar 8 出信号（need 锚点 100）；bar 9-16 在带内震荡、从未整根到带上方；bar 17 起到带上方；bar 23 为 Pinbar。"""
    return [ABOVE, TOUCH] + [ABOVE] * 6 + [PIN] + [IN, IN, IN_LOW, IN, IN, TOUCH, TOUCH, TOUCH] + [ABOVE] * 6 + [PIN]


@pytest.mark.parametrize("reject_broken", [False, True])
def test_sm3_1_departure_requirement_survives_rejected_box(reject_broken):
    # 信号后的带内震荡中，VAL 100.3 的箱体（上一根收盘 99.7 < 其带下沿 99.8）被否决，再回到 VAL 100 的箱体：
    # 价格从未整根到带上方，震荡不能成为新的第 1 次测试（审计时代码在 reject=True 时于 bar 23 多出一个信号）
    boxes = [{"at": 0, "val": 100.0}, {"at": 12, "val": 100.3}, {"at": 13, "val": 100.0}]
    out = sm(_post_signal_chop(), boxes, StrategyConfig(reject_broken_at_activation=reject_broken))
    assert out["state"][12] == (INVALID if reject_broken else ARMED)
    assert out["need"][13] and out["state"][16] == ARMED and not out["need"][17]
    assert list(np.flatnonzero(out["sig"])) == [8]


@pytest.mark.parametrize("mid_val", [99.5, 99.9])
def test_sm3_1_departure_requirement_survives_memory_blocked_box(mid_val):
    # 98.8 支撑区已被放量跌穿；信号（bar 11）后的震荡中出现与 98.8 同区的 VAL 99.5 箱体（被记忆判为失效），
    # 再回到 VAL 100：要求不得因中间的失效箱体而丢失（99.9 为对照：不被阻断）
    pre = [(99.6, 99.8, 99.2, 99.4), (99.4, 99.5, 97.9, 98.0, 200.0), (98.0, 101.2, 97.9, 101.0)]
    bars = pre + _post_signal_chop()
    boxes = [{"at": 0, "val": 98.8, "ws": -50}, {"at": 2, "val": 100.0, "ws": -40},
             {"at": 15, "val": mid_val, "ws": -30}, {"at": 16, "val": 100.0, "ws": -20}]
    out = sm(bars, boxes)
    assert out["state"][15] == (INVALID if mid_val == 99.5 else ARMED)
    assert list(np.flatnonzero(out["sig"])) == [11]


def test_sm3_1_departure_requirement_survives_detour_through_other_zone():
    # A(100) → B(101.2，不同支撑区) → A(100)：A 区的要求在 B 期间保留（B 的窗口含信号前数据，不是全新结构）
    boxes = [{"at": 0, "val": 100.0}, {"at": 12, "val": 101.2}, {"at": 13, "val": 100.0}]
    out = sm(_post_signal_chop(), boxes)
    assert out["need"][12] and out["need"][13] and out["state"][16] == ARMED
    assert list(np.flatnonzero(out["sig"])) == [8]


@pytest.mark.parametrize("ws_b, signals", [(8, [8]), (9, [8, 23])])
def test_sm3_1_out_of_zone_requirement_expires_on_fresh_structure(ws_b, signals):
    # 有界记忆：B 的窗口起点 ≥ 写入时刻（信号 K 线收盘 = bar 9 开盘）时，不在当前支撑区的 A 区要求失效；
    # 回到 A 时按激活规则判定（上一根收盘在带内 → 首次触及即第 1 次测试）
    boxes = [{"at": 0, "val": 100.0}, {"at": 12, "val": 101.2, "ws": ws_b}, {"at": 13, "val": 100.0, "ws": ws_b}]
    out = sm(_post_signal_chop(), boxes)
    assert list(np.flatnonzero(out["sig"])) == signals


def test_sm3_1_claimed_requirement_does_not_expire_on_fresh_structure():
    # 同区箱体即使完全由信号后的数据构成，也只认领、不使要求失效：仍须先整根到带上方
    boxes = [{"at": 0, "val": 100.0}, {"at": 12, "val": 100.0, "ws": 10}]
    out = sm(_post_signal_chop(), boxes)
    assert out["need"][12] and list(np.flatnonzero(out["sig"])) == [8]


def test_sm3_1_departure_while_invalid_satisfies_requirement():
    # 否决路径：信号后 VAL 100.3 的箱体被否决，INVALID 期间价格整根到带上方，回到 VAL 100 后的回落是真正的第 1 次测试
    bars = [ABOVE, TOUCH] + [ABOVE] * 6 + [PIN] + [IN_LOW, IN_LOW] + [UP] * 4 + [TOUCH] + [ABOVE] * 6 + [PIN]
    boxes = [{"at": 0, "val": 100.0}, {"at": 10, "val": 100.3}, {"at": 15, "val": 100.0}]
    out = sm(bars, boxes, StrategyConfig(reject_broken_at_activation=True))
    assert out["state"][10] == INVALID and not out["need"][11] and out["state"][15] == TOUCH1
    assert list(np.flatnonzero(out["sig"])) == [8, 22]
    # 记忆路径：被记忆阻断的 VAL 99.5 箱体期间价格整根到带上方
    pre = [(99.6, 99.8, 99.2, 99.4), (99.4, 99.5, 97.9, 98.0, 200.0), (98.0, 101.2, 97.9, 101.0)]
    bars = pre + [ABOVE, TOUCH] + [ABOVE] * 6 + [PIN] + [IN, IN, IN_LOW, IN] + [UP] * 3 + [TOUCH] + [ABOVE] * 6 + [PIN]
    boxes = [{"at": 0, "val": 98.8, "ws": -50}, {"at": 2, "val": 100.0, "ws": -40},
             {"at": 15, "val": 99.5, "ws": -30}, {"at": 19, "val": 100.0, "ws": -20}]
    out = sm(bars, boxes)
    assert out["state"][15] == INVALID and out["state"][19] == TOUCH1
    assert list(np.flatnonzero(out["sig"])) == [11, 26]


def test_sm3_1_fresh_box_after_breakdown_keeps_unsatisfied_same_zone_requirement():
    # 激活时价格在带下方（要求写入）→ 未到带上方就被放量跌穿 → 完全由跌穿后数据构成的同区箱体：
    # 记忆失效、重新 ARMED，但同区要求仍未满足（价格从未整根到带上方）→ 须先到带上方
    bars = [(98.0, 98.4, 97.8, 98.2), (98.2, 99.8, 98.1, 99.7), (99.9, 100.0, 98.6, 98.9, 200.0),
            (98.9, 100.3, 98.8, 100.2), IN, IN, TOUCH, TOUCH] + [ABOVE] * 6 + [PIN]
    out = sm(bars, [{"at": 1, "val": 100.0, "ws": -50}, {"at": 4, "val": 100.1, "ws": 3}])
    assert out["state"][2] == INVALID and out["state"][4] == ARMED and out["need"][4]
    assert not out["need"][8] and out["state"][14] == TOUCH1 and not out["sig"].any()


def _brk_at(v, close_off=-1.0):
    return (v + 0.3, v + 0.4, v + close_off - 0.2, v + close_off, 200.0)


@pytest.mark.parametrize("ret", range(12))
def test_sm3_2_breakdown_memory_has_no_capacity_limit(ret):
    # 12 个不同支撑区（100, 98, ..., 78）依次被放量跌穿，所有箱体窗口都含跌穿前数据；之后回到第 ret 个支撑区：
    # 每一个都必须仍然失效。逐一检查才能拦下任何容量上限，无论溢出时覆盖最早的（b2fbdb7 的 8 槽）还是最新的记忆
    bars, boxes = [], []
    for k in range(12):
        v = 100.0 - 2.0 * k
        boxes.append({"at": 2 * k, "val": v, "ws": -100})
        bars += [(v + 0.6, v + 0.9, v + 0.5, v + 0.7), _brk_at(v)]
    r = 100.0 - 2.0 * ret + 0.2
    boxes.append({"at": 24, "val": r, "ws": -100})
    bars += [(77.0, r + 0.8, 76.9, r + 0.7), (r + 0.7, r + 1.6, r + 0.6, r + 1.3), (r + 0.8, r + 1.0, r, r + 0.4)] \
        + [(r + 0.8, r + 1.6, r + 0.6, r + 1.3)] * 6 + [(r + 0.7, r + 0.8, r - 0.5, r + 0.75)]
    out = sm(bars, boxes)
    assert int(out["brk"].sum()) == 12
    assert (out["state"][24:] == INVALID).all() and not out["sig"].any()


def test_sm3_3_block_uses_every_memory_entry():
    # 100（先写入）与 95（后写入）都被放量跌穿；与 95 同区、窗口含跌穿前数据的箱体必须失效（不能只看第一条记忆）
    bars = [ABOVE, TOUCH, BRK, (99.0, 99.2, 96.0, 96.2), (96.2, 96.5, 95.8, 96.0), (96.0, 96.1, 95.2, 95.4),
            (95.4, 95.5, 93.8, 94.0, 200.0), (94.0, 97.0, 93.9, 96.9), (96.9, 97.0, 96.0, 96.5), (96.5, 96.6, 95.6, 95.9)]
    boxes = [{"at": 0, "val": 100.0, "ws": -50}, {"at": 4, "val": 95.0, "ws": -10}, {"at": 8, "val": 95.2, "ws": -5}]
    out = sm(bars, boxes)
    assert out["state"][6] == INVALID and (out["state"][8:] == INVALID).all()
    # 95 先被跌穿、100 后被跌穿：回到 100 支撑区同样失效
    bars = [(96.0, 96.8, 95.8, 96.5), (96.0, 96.2, 95.2, 95.6), (95.3, 95.4, 93.8, 94.0, 200.0),
            (94.0, 97.0, 93.9, 96.9), (96.9, 99.0, 96.8, 98.9), ABOVE, TOUCH, BRK, BELOW, RECLAIM, ABOVE, TOUCH] \
        + [ABOVE] * 6 + [PIN]
    boxes = [{"at": 0, "val": 95.0, "ws": -50}, {"at": 5, "val": 100.0, "ws": -10}, {"at": 10, "val": 100.3, "ws": 4}]
    out = sm(bars, boxes)
    assert out["state"][7] == INVALID and (out["state"][10:] == INVALID).all() and not out["sig"].any()


def test_sm3_3_rebreakdown_moves_every_overlapping_entry():
    # 记忆 100 与 101.6 都与箱体 100.8 同区；一次收盘低于两者带下沿的再次跌穿把两条记忆的时刻都后移
    bars = [ABOVE, TOUCH, BRK, (99.0, 101.9, 98.9, 101.8), (101.9, 102.0, 101.7, 101.9), _brk_at(101.6, -0.8),
            (100.8, 101.0, 100.6, 100.9), (100.9, 101.0, 100.7, 100.8), (100.6, 100.7, 99.0, 99.2, 200.0),
            (99.2, 102.5, 99.1, 102.4), (102.4, 102.6, 102.2, 102.5)]
    boxes = [{"at": 0, "val": 100.0, "ws": -50}, {"at": 4, "val": 101.6, "ws": -40}, {"at": 7, "val": 100.8, "ws": -30},
             {"at": 10, "val": 101.9, "ws": 7}]
    out = sm(bars, boxes)
    assert out["state"][5] == INVALID and out["state"][7] == INVALID
    assert out["state"][10] == INVALID                                    # 101.6 的时刻已后移到 bar 8 收盘 > 窗口起点


def test_sm3_3_rebreakdown_time_is_the_bar_close():
    def run(ws):
        bars = [ABOVE, TOUCH, BRK, BELOW, (100.8, 101.5, 100.7, 100.8), (100.8, 100.85, 99.2, 99.3, 200.0), BELOW, BELOW]
        boxes = [{"at": 0, "val": 100.0, "ws": -50}, {"at": 4, "val": 100.9, "ws": -10}, {"at": 7, "val": 100.3, "ws": ws}]
        return sm(bars, boxes)
    assert run(5)["state"][7] == INVALID                                  # 窗口起于再次跌穿 K 线开盘：仍含跌穿
    assert run(6)["state"][7] == ARMED                                    # 起于其收盘：全新结构


def test_sm3_3_expired_entry_is_not_revived():
    bars = [ABOVE, TOUCH, BRK, BELOW, (100.8, 101.6, 100.7, 101.5), _brk_at(100.9, -0.6),
            (100.2, 100.3, 99.0, 99.1, 200.0), (99.1, 99.6, 99.0, 99.5), (99.5, 99.7, 99.3, 99.6)]
    boxes = [{"at": 0, "val": 100.0, "ws": -50}, {"at": 4, "val": 100.9, "ws": 3}, {"at": 8, "val": 99.2, "ws": 5}]
    out = sm(bars, boxes)
    assert out["state"][4] == ARMED and out["state"][5] == INVALID
    assert out["state"][8] != INVALID                                     # 只剩 100.9 一条有效记忆，99.2 与之不同区


def test_sm3_3_different_zone_switch_recomputes_requirement():
    bars = [ABOVE, TOUCH] + [ABOVE] * 6 + [PIN, (100.9, 102.9, 100.4, 102.8), (102.8, 103.0, 102.4, 102.6)]
    out = sm(bars, [{"at": 0, "val": 100.0}, {"at": 10, "val": 102.0}])
    assert out["sig"][8] and out["need"][9] and not out["need"][10] and out["state"][10] == TOUCH1


def test_sm3_3_requirement_anchor_stays_where_it_was_set():
    # 100 → 100.8（同区，认领）→ 101.6：与锚点 100 相距 1.6 ATR → 不认领，按新箱体重新判定
    bars = [ABOVE, TOUCH] + [ABOVE] * 6 + [PIN, TOUCH, TOUCH, (101.6, 101.9, 101.3, 101.8), (101.8, 101.9, 101.5, 101.7)]
    out = sm(bars, [{"at": 0, "val": 100.0}, {"at": 10, "val": 100.8}, {"at": 12, "val": 101.6}])
    assert out["need"][10] and out["state"][10] == ARMED
    assert not out["need"][12] and out["state"][12] == TOUCH1


def test_sm3_3_signal_anchors_requirement_at_signal_box():
    # 测试进度 100 → 100.8（carry）；信号在 100.8 箱体上发出；下一个箱体 101.7 与 100.8 同区（0.9 ATR）：要求保留
    up = (101.6, 102.0, 101.4, 101.8)
    bars = [ABOVE, TOUCH, ABOVE] + [up] * 6 + [(101.4, 101.5, 100.5, 101.45), (101.4, 101.5, 101.0, 101.3),
                                               (101.3, 101.5, 101.1, 101.4)]
    out = sm(bars, [{"at": 0, "val": 100.0}, {"at": 3, "val": 100.8}, {"at": 11, "val": 101.7}])
    assert out["sig"][9] and out["need"][11] and out["state"][11] == ARMED


def test_sm3_4_value_area_one_sided_expansion_does_not_wrap():
    for h, exp in (([5.0, 1.0, 0.5, 2.0], (0, 0, 1)), ([2.0, 0.5, 1.0, 5.0], (3, 2, 3))):
        h = np.array(h)
        assert _value_area.py_func(h, 0.7) == exp and _value_area(h, 0.7) == exp


def test_sm3_5_invalid_breakdown_overlapping_older_entry_is_not_recorded():
    # 已记录的设计选择：INVALID 期间与旧记忆同区的箱体再次放量跌穿，收盘未低于旧记忆的带下沿时不写入任何记忆，
    # 因此之后不与旧记忆同区、但与该箱体同区的箱体可以重新 ARMED（对照：无旧记忆时同一根 K 线在 TOUCH1 中被记录）
    rebrk = (101.2, 101.25, 99.7, 99.8, 200.0)
    up2, t2, pin2 = (102.2, 102.9, 102.1, 102.6), (102.3, 102.4, 101.7, 102.1), (102.4, 102.5, 101.3, 102.45)
    bars = [ABOVE, TOUCH, BRK, (99.0, 101.3, 98.9, 101.2), (101.2, 101.4, 101.0, 101.3), rebrk,
            (99.8, 101.0, 99.7, 100.9), up2, up2, t2] + [up2] * 6 + [pin2]
    boxes = [{"at": 0, "val": 100.0, "ws": -50}, {"at": 4, "val": 100.9, "ws": -40}, {"at": 7, "val": 101.5, "ws": -30}]
    out = sm(bars, boxes)
    assert out["brk"][5] and out["state"][4] == INVALID and out["state"][7] == ARMED and out["sig"][16]
    ctrl = list(bars)
    ctrl[2] = (100.3, 100.4, 99.6, 100.45)
    out = sm(ctrl, boxes)
    assert out["state"][5] == INVALID and out["state"][7] == INVALID and not out["sig"].any()


def test_sm3_6_window_start_must_be_non_decreasing():
    from strategy_state_machine import state_machine_args
    df = _mk_df([[100, 100.5, 99.5, 100.0]] * 30)
    t = df.index
    boxes = pd.DataFrame({"available_time": [t[2], t[5]], "VAL": [100.0, 100.0], "VAH": [101.0, 101.0],
                          "window_start": [t[1], t[0]]})
    with pytest.raises(ValueError, match="window_start"):
        state_machine_args(df, StrategyConfig(), boxes)


def test_doc3_10_activation_reject_is_skipped_when_carry_keeps_progress():
    bars = [ABOVE, TOUCH, (100.2, 100.3, 99.6, 99.7), (99.9, 100.6, 99.85, 100.4), ABOVE]
    boxes = [{"at": 0, "val": 100.0, "ws": -50}, {"at": 3, "val": 100.3, "ws": -10}]
    keep = sm(bars, boxes, StrategyConfig(reject_broken_at_activation=True))
    reset = sm(bars, boxes, StrategyConfig(reject_broken_at_activation=True, carry_touches_on_box_update=False))
    assert keep["state"][3] == TOUCH1 and reset["state"][3] == INVALID


def test_doc3_10_breakdown_of_rejected_box_writes_zone_memory():
    cfg = StrategyConfig(reject_broken_at_activation=True)
    tail = [(99.0, 101.2, 98.9, 101.0), ABOVE, ABOVE, TOUCH] + [ABOVE] * 6 + [PIN]
    boxes = [{"at": 1, "val": 100.0, "ws": -20}, {"at": 5, "val": 100.2, "ws": -10}]
    with_brk = sm([BELOW, (99.2, 100.0, 99.1, 99.9), BRK] + tail, boxes, cfg)
    no_brk = sm([BELOW, (99.2, 100.0, 99.1, 99.9), (99.9, 100.2, 99.6, 99.8)] + tail, boxes, cfg)
    assert with_brk["state"][1] == INVALID and with_brk["brk"][2]
    assert with_brk["state"][5] == INVALID and not with_brk["sig"].any()
    assert no_brk["state"][5] == ARMED and no_brk["sig"].any()



# ============================================================================
# 第三轮复核确认问题的回归测试：DOC3-1、EX3-4 … EX3-8、DOC3-6（执行 / 指标 / 管道）
# ============================================================================
@pytest.mark.parametrize("days", [30, 120, 2000])
def test_doc3_1_end_settlement_flag_on_long_windows(days):
    # np.arange 以 float64 估算长度：窗口 > 约 104 天时会丢掉终点网格点，最后收盘时刻缺失的结算就不再被标记
    n = 24 * days
    rows = [[100.0, 100.5, 99.5, 100.0]] * n
    idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)
    df["volume"], df["quote_volume"] = 100.0, 1e6
    sig = pd.DataFrame({"signal": False, "sl": np.nan, "tp": np.nan, "poc_tp": np.nan}, index=idx)
    sig.loc[idx[n - 9], ["signal", "sl", "tp", "poc_tp"]] = [True, 50.0, 500.0, 400.0]
    t = pd.date_range(idx[0], periods=3 * days + 1, freq="8h")             # 00:00 起，止于最后收盘（次日 00:00）
    assert t[-1] == idx[-1] + pd.Timedelta(hours=1)
    full = pd.DataFrame({"funding_rate": 1e-4, "funding_interval_hours": 8}, index=t)
    cfg = BacktestConfig(time_stop_bars=8)
    ok = run_backtest(df, sig, cfg, full)
    tr = ok.trades.iloc[0]
    assert tr["exit_bar"] == n - 1 and tr["exit_reason"] == "TIME" and tr["n_funding"] == 1 and tr["funding_complete"]
    miss = run_backtest(df, sig, cfg, full.iloc[:-1])                     # 缺少恰在最后收盘（00:00）的结算
    assert miss.funding_end_missing and not miss.trades.iloc[0]["funding_complete"]
    assert miss.trades.iloc[0]["n_funding"] == 0


def _fund_case_df():
    df = _mk_df([[100.0, 100.5, 99.5, 100.0]] * 24)
    sig = pd.DataFrame({"signal": False, "sl": np.nan, "tp": np.nan, "poc_tp": np.nan}, index=df.index)
    sig.loc[sig.index[1], ["signal", "sl", "tp", "poc_tp"]] = [True, 50.0, 500.0, 400.0]
    return df, sig


def test_ex3_4_off_grid_or_nan_settlements_raise():
    df, sig = _fund_case_df()
    cfg = BacktestConfig(time_stop_bars=20)
    stamps = ["2024-01-01 08:00", "2024-01-01 16:00", "2024-01-02 00:00"]
    base = pd.DataFrame({"funding_rate": 1e-4, "funding_interval_hours": 8}, index=pd.DatetimeIndex(stamps, tz="UTC"))
    run_backtest(df, sig, cfg, base)
    off_nan = pd.concat([base, pd.DataFrame({"funding_rate": [np.nan], "funding_interval_hours": [8]},
                                            index=pd.DatetimeIndex(["2024-01-01 12:00"], tz="UTC"))]).sort_index()
    with pytest.raises(ValueError, match="off the 8h settlement grid"):
        run_backtest(df, sig, cfg, off_nan)                               # 旧实现：静默按 0 计（跳过 12:00）
    near_dup = pd.concat([base, pd.DataFrame({"funding_rate": [1e-4], "funding_interval_hours": [8]},
                                             index=pd.DatetimeIndex(["2024-01-01 08:01"], tz="UTC"))]).sort_index()
    with pytest.raises(ValueError, match="off the 8h settlement grid"):
        run_backtest(df, sig, cfg, near_dup)                              # 旧实现：08:00 附近收两次
    on_nan = base.copy()
    on_nan.iloc[1, 0] = np.nan
    with pytest.raises(ValueError, match="NaN rates"):
        run_backtest(df, sig, cfg, on_nan)


def test_ex3_4_settlement_interval_must_be_known():
    df, sig = _fund_case_df()
    cfg = BacktestConfig(time_stop_bars=20)
    four_h = pd.DataFrame({"funding_rate": 1e-4},
                          index=pd.DatetimeIndex(["2024-01-01 08:00", "2024-01-01 16:00", "2024-01-01 20:00",
                                                  "2024-01-02 00:00"], tz="UTC"))
    with pytest.raises(ValueError, match="funding_interval_hours"):
        run_backtest(df, sig, cfg, four_h)                                # 旧实现：默认按 8h，4h 表缺点不报错
    with pytest.raises(ValueError, match="misses the settlement"):
        run_backtest(df, sig, cfg, four_h.assign(funding_interval_hours=4))
    with pytest.raises(ValueError, match="missing near"):
        run_backtest(df, sig, cfg, four_h.assign(funding_interval_hours=np.nan))


def _mk_res_trades(start, end, trades):
    res = _mk_res(start, end)
    return BacktestResult(equity=res.equity, position=res.position, trades=trades, config=res.config,
                          initial_capital=res.initial_capital, start_time=res.start_time)


def _one_trade(entry, exit_, capped=False, reason="TP"):
    return pd.DataFrame({"entry_time": pd.to_datetime([entry], utc=True),
                         "exit_time": pd.to_datetime([exit_], utc=True),
                         "exit_reason": pd.Series([reason], dtype="str"), "pnl": [100.0], "r_multiple": [1.0],
                         "hold_bars": [10], "fees": [1.0], "funding": [0.0], "ret_on_base": [0.001],
                         "capped": [capped], "funding_complete": [True]})


def test_ex3_5_trade_stats_and_daily_equity_are_locked_past_oos_start():
    from config import VAL_END, VAL_START
    tr = _one_trade("2025-06-20", "2025-07-10")
    with pytest.raises(OOSLockedError):
        trade_stats(tr)                                                   # VAL 入场、OOS 离场
    assert trade_stats(tr, allow_oos=True)["trades"] == 1
    full = _mk_res_trades(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-08-01", tz="UTC"), tr)
    with pytest.raises(OOSLockedError):
        daily_equity(full)
    # 把净值截到 OOS_START、交易表不截：仍然被锁（交易离场在 OOS 期间）
    cut = _mk_res_trades(pd.Timestamp("2024-01-01", tz="UTC"), OOS_START, tr)
    with pytest.raises(OOSLockedError):
        segment_metrics(cut, VAL_START, VAL_END)
    # 研究结果最后一根 K 线的信号：入场时刻恰为 OOS_START、期末未平仓 → 不含 OOS 信息，不锁
    last = _mk_res_trades(pd.Timestamp("2024-01-01", tz="UTC"), OOS_START, _one_trade("2025-07-01", None, reason="OPEN"))
    segment_metrics(last, VAL_START, VAL_END)


def test_ex3_6_capped_position_straddling_segment_start_is_reported():
    s, m, e = (pd.Timestamp(x, tz="UTC") for x in ("2024-01-01", "2024-02-01", "2024-03-01"))
    for capped, exp in ((True, 1), (False, 0)):
        res = _mk_res_trades(s, e, _one_trade("2024-01-25", "2024-02-05", capped=capped))
        assert segment_metrics(res, m, e)["capped_open_at_start"] == exp
        assert segment_metrics(res, m, e)["capped_entries"] == 0          # 段内没有入场


def test_ex3_8_first_day_regime_and_short_4h_inputs():
    from strategy_state_machine import generate_signals
    from ppvp_engine import build_boxes, shift_known_at_open
    assert shift_known_at_open(np.zeros(0)).shape == (0,)
    df, fund, spot = pipeline.load_market("BTCUSDT")
    a = StrategyConfig(regime_filter="daily_sma")
    full = generate_signals(df.iloc[:200], a, *build_boxes(df.iloc[:200], a), spot)
    for k in (1, 3, 6, 23):
        cut = df.iloc[:k]
        sig = generate_signals(cut, a, *build_boxes(cut, a), spot)
        assert np.array_equal(sig["regime_raw"].to_numpy(), full["regime_raw"].to_numpy()[:k])
        b = StrategyConfig(regime_filter="crash_cooldown")
        generate_signals(cut, b, *build_boxes(cut, b), spot)               # 不足一根 4H 时不报错


def test_doc3_6_cache_isolated_from_index_writes():
    pipeline.clear_cache()
    df, _, _ = pipeline.load_market("BTCUSDT")
    t0 = df.index[0]
    df.index.asi8[0] += 3600 * 10**9                                      # DatetimeIndex.asi8 是可写视图
    assert pipeline.load_market("BTCUSDT")[0].index[0] == t0
    boxes, df4h = pipeline.get_boxes("BTCUSDT", StrategyConfig())
    t4 = df4h.index[0]
    df4h.index.asi8[0] += 3600 * 10**9
    assert pipeline.get_boxes("BTCUSDT", StrategyConfig())[1].index[0] == t4
    pipeline.clear_cache()


def test_ex3_7_backtest_result_repr_shows_no_values():
    df = _mk_df([[100.0, 100.5, 99.5, 100.0 + 0.123456 * k] for k in range(5)])
    sig = pd.DataFrame({"signal": [True] + [False] * 4, "sl": [90.0] + [np.nan] * 4, "tp": [120.0] + [np.nan] * 4,
                        "poc_tp": [110.0] + [np.nan] * 4}, index=df.index)
    res = run_backtest(df, sig, BacktestConfig(include_funding=False))
    text = repr(res)
    assert text.startswith("BacktestResult(bars=5") and "100" not in text.replace("bars=5", "")
    assert "trades" not in text                                           # 全量结果的交易笔数也是 OOS 信息（第四轮 EX4-3）


# ============================================================================
# 第四轮复核：独立参考实现对拍、清除阈值与簿记细节（SM4-1 … SM4-6）
# ============================================================================
def reference_state_machine(o, h, l, c, v, atr, vsma, open_ns, box_idx, box_val, box_vah, box_wstart, pin, eng,
                            k1, k2, min_bars, min_bounce, carry, roll, brk_body, brk_vol, sl_buf, reject_broken,
                            max_lag, sl_pattern, events=None):
    """按 strategy_state_machine 文档规则独立编写的参考实现，只用于对拍测试。

    与 Numba 内核的区别只在数据结构：放量跌穿记忆与“须先到带上方”要求都是保持顺序的 Python 列表，
    失效与清除用列表推导整体重建，不做交换删除。浮点表达式与内核逐项相同，因此输出必须逐位一致。
    events（可选 dict）累计各类事件的次数，供测试检查对拍是否真的覆盖了这些路径。"""
    n = len(o)
    ev = events if events is not None else {}

    def bump(key):
        ev[key] = ev.get(key, 0) + 1

    state = np.zeros(n, np.int8)
    zlo, zhi, sl, tp = (np.full(n, np.nan) for _ in range(4))
    brk, sig = np.zeros(n, np.bool_), np.zeros(n, np.bool_)
    pat = np.zeros(n, np.int8)
    o_t1_low, o_t2_low = np.full(n, np.nan), np.full(n, np.nan)
    o_t1_bar, o_t2_bar, o_t2_start, o_box = (np.full(n, -1, np.int64) for _ in range(4))
    o_need = np.zeros(n, np.bool_)
    hour = np.int64(3600) * np.int64(1000000000)
    st, cur = 0, -1
    seq = dict(t1_low=np.nan, t1_bar=-1, t1_last=-1, max_high=np.nan, t2_low=np.nan, t2_bar=-1, t2_start=-1,
               seq_val=np.nan, t2_val=np.nan)
    fresh_seq = dict(seq)
    memory = []                       # [VAL, 跌穿收盘时刻]
    reqs = []                         # {"val": 锚点, "t": 写入时刻, "claimed": 是否被当前箱体认领}
    need = True
    for i in range(n):
        b, a = box_idx[i], atr[i]
        if b != cur:
            if b < 0:
                st = 0
            else:
                ov = (k1 + k2) * a
                kept = [m for m in memory if not box_wstart[b] >= m[1]]
                if len(kept) < len(memory):
                    bump("memory_expired")
                memory = kept
                blocked = (not np.isnan(a)) and any(abs(box_val[b] - m[0]) <= ov for m in memory)
                keep = ((not blocked) and carry and cur >= 0 and st in (2, 3, 4) and not np.isnan(a)
                        and abs(box_val[b] - seq["seq_val"]) <= ov)
                if not keep:
                    seq = dict(fresh_seq)
                    for r in reqs:
                        r["claimed"] = bool((not np.isnan(a)) and abs(box_val[b] - r["val"]) <= ov)
                    need = any(r["claimed"] for r in reqs)
                    if need:
                        bump("claimed_at_switch")
                else:
                    bump("carry")
                kept = [r for r in reqs if r["claimed"] or not box_wstart[b] >= r["t"]]
                if len(kept) < len(reqs):
                    bump("requirement_expired")
                reqs = kept
                if blocked:
                    st = 5
                    bump("memory_block")
                elif not keep:
                    st = 1
                    zl_new = box_val[b] - k1 * a
                    below = not (i > 0 and c[i - 1] >= zl_new)
                    if below and not need:
                        reqs.append({"val": box_val[b], "t": open_ns[i], "claimed": True})
                        need = True
                        bump("write_a")
                    if reject_broken and i > 0 and not np.isnan(a) and c[i - 1] < zl_new:
                        st = 5
                        bump("reject")
            cur = b
        o_box[i] = cur
        if st == 0 or np.isnan(a):
            state[i], o_need[i] = st, need
            continue
        zl, zh = box_val[cur] - k1 * a, box_val[cur] + k2 * a
        zlo[i], zhi[i] = zl, zh
        if reqs:
            kept = [r for r in reqs if not l[i] > (zh if r["claimed"] else r["val"] + k2 * a)]
            if len(kept) < len(reqs):
                bump("cleared_in_state_%d" % st)
                if any(not r["claimed"] for r in reqs if r not in kept):
                    bump("cleared_unclaimed")
            reqs = kept
            need = any(r["claimed"] for r in reqs)
        is_brk = (c[i] < o[i]) and (c[i] < zl) and (o[i] - c[i] >= brk_body * a) and (v[i] >= brk_vol * vsma[i])
        brk[i] = is_brk
        if st in (5, 6):
            if is_brk:
                same = False
                if st == 5:
                    for m in memory:
                        if abs(box_val[cur] - m[0]) <= (k1 + k2) * a:
                            same = True
                            if c[i] < m[0] - k1 * a:
                                m[1] = open_ns[i] + hour
                                bump("memory_moved")
                if not same:
                    memory.append([box_val[cur], open_ns[i] + hour])
                    st = 5
                    bump("memory_write_inactive")
            state[i], o_need[i] = st, need
            continue
        if is_brk:
            st = 5
            memory.append([box_val[cur], open_ns[i] + hour])
            bump("memory_write_active")
            state[i], o_need[i] = st, need
            continue
        touched = l[i] <= zh
        if st == 1:
            if (not need) and touched:
                st = 2
                seq.update(t1_low=l[i], t1_bar=i, t1_last=i, seq_val=box_val[cur])
        elif st == 2:
            if touched:
                if l[i] < seq["t1_low"]:
                    seq.update(t1_low=l[i], t1_bar=i)
                seq["t1_last"] = i
            else:
                st = 3
                seq["max_high"] = h[i]
        elif st == 3:
            if touched:
                if (seq["max_high"] - zh >= min_bounce * a) or (i - seq["t1_last"] >= min_bars):
                    st = 4
                    seq.update(t2_low=l[i], t2_bar=i, t2_start=i, t2_val=box_val[cur])
                else:
                    st = 2
                    if l[i] < seq["t1_low"]:
                        seq.update(t1_low=l[i], t1_bar=i)
                    seq["t1_last"] = i
            elif h[i] > seq["max_high"]:
                seq["max_high"] = h[i]
        elif st == 4:
            if touched:
                if l[i] < seq["t2_low"]:
                    seq.update(t2_low=l[i], t2_bar=i)
            elif roll:
                st = 3
                seq.update(t1_low=seq["t2_low"], t1_bar=seq["t2_bar"], t1_last=i - 1, seq_val=seq["t2_val"],
                           max_high=h[i], t2_low=np.nan, t2_bar=-1, t2_start=-1, t2_val=np.nan)
                bump("roll")
            else:
                st = 6
                bump("exhausted")
        if st == 4 and touched and c[i] >= zl and (max_lag < 0 or i - seq["t2_start"] <= max_lag):
            p = (1 if pin[i] else 0) | (2 if eng[i] else 0)
            if p > 0:
                lowest = min(seq["t1_low"], l[i]) if sl_pattern else min(seq["t1_low"], seq["t2_low"])
                if (p & 2) and i > 0 and l[i - 1] < lowest:
                    lowest = l[i - 1]
                sig[i], pat[i] = True, p
                sl[i], tp[i] = lowest - sl_buf * a, box_vah[cur]
                o_t1_low[i], o_t1_bar[i] = seq["t1_low"], seq["t1_bar"]
                o_t2_low[i], o_t2_bar[i], o_t2_start[i] = seq["t2_low"], seq["t2_bar"], seq["t2_start"]
                st = 1
                reqs.append({"val": box_val[cur], "t": open_ns[i] + hour, "claimed": True})
                need = True
                seq = dict(fresh_seq)
                bump("signal")
        state[i], o_need[i] = st, need
    return (state, zlo, zhi, brk, sig, pat, sl, tp, o_t1_low, o_t1_bar, o_t2_low, o_t2_bar, o_t2_start, o_box, o_need)


def _fuzz_case(rng):
    """随机合成行情与箱体序列：价格在若干支撑区之间游走，含放量跌穿、Pinbar、带内震荡、同区/异区箱体切换、
    窗口起点早于或晚于近期事件的箱体、ATR 起伏与开头的 NaN ATR。"""
    n = int(rng.integers(150, 260))
    o, h, l, c = (np.empty(n) for _ in range(4))
    v = np.full(n, 100.0)
    px = 100.0
    for i in range(n):
        op = px + rng.normal(0.0, 0.15)
        r = rng.random()
        if r < 0.08:                                         # 放量实体阴线
            cl = op - rng.uniform(0.8, 2.0)
            v[i] = rng.uniform(160.0, 260.0)
        elif r < 0.18:                                       # 看涨 Pinbar
            cl = op + rng.uniform(0.0, 0.08)
        else:
            cl = op + rng.normal(0.0, 0.55)
        hi = max(op, cl) + abs(rng.normal(0.0, 0.25))
        lo = min(op, cl) - (rng.uniform(0.9, 1.6) if r >= 0.08 and r < 0.18 else abs(rng.normal(0.0, 0.25)))
        if rng.random() < 0.05:
            v[i] = rng.uniform(150.0, 200.0)
        o[i], h[i], l[i], c[i] = op, hi, lo, cl
        px = cl + (100.0 - cl) * 0.03                          # 弱均值回归，价格反复回到若干支撑区
    atr = 1.0 + 0.35 * np.sin(np.arange(n) / rng.uniform(5.0, 25.0) + rng.uniform(0.0, 6.0))
    if rng.random() < 0.3:
        atr[: int(rng.integers(1, 8))] = np.nan
    vsma = np.full(n, 100.0)
    open_ns = BASE + np.arange(n, dtype=np.int64) * HOUR
    box_idx = np.full(n, -1, dtype=np.int64)
    vals, vahs, wss = [], [], []
    at = int(rng.integers(0, 12))
    ws = -60
    while at < n:
        k = len(vals)
        box_idx[at:] = k
        ref = c[at - 1] if at > 0 else c[0]
        val = round(ref + rng.uniform(-2.6, 1.6), 1) if rng.random() < 0.7 else (vals[-1] + rng.choice([-1.0, 0.0, 1.0]) if vals else 100.0)
        vals.append(val)
        vahs.append(val + rng.uniform(1.0, 6.0))
        ws = max(ws, at - int(rng.integers(0, 40)))
        wss.append(BASE + np.int64(ws) * HOUR)
        at += int(rng.integers(4, 28))
    cfg = StrategyConfig(zone_k1=float(rng.choice([0.3, 0.5, 0.7])), zone_k2=float(rng.choice([0.3, 0.5, 0.7])),
                         min_bars_between_touches=int(rng.choice([3, 6])), min_bounce_atr=float(rng.choice([0.5, 1.0])),
                         carry_touches_on_box_update=bool(rng.random() < 0.7), roll_failed_second_touch=bool(rng.random() < 0.6),
                         reject_broken_at_activation=bool(rng.random() < 0.4),
                         max_pattern_lag_bars=[None, 0, 2][int(rng.integers(0, 3))],
                         sl_anchor=["test", "pattern"][int(rng.integers(0, 2))])
    pin, eng = bullish_pinbar(o, h, l, c, cfg), bullish_engulfing(o, h, l, c, cfg)
    args = (o, h, l, c, v, atr, vsma, open_ns, box_idx, np.array(vals), np.array(vahs), np.array(wss, dtype=np.int64),
            pin, eng) + state_machine_params(cfg)
    return args


def test_sm4_kernel_matches_independent_reference_on_random_markets():
    """SM4-1/2/3：用独立编写、保持顺序的参考实现与 Numba 内核对拍 600 个随机合成场景，15 个输出逐位一致；
    同时要求对拍真的覆盖了各类事件（否则“一致”可能是空洞的）。交换删除、认领、失效、清除的任何簿记错误都会在这里暴露。"""
    rng = np.random.default_rng(20261104)
    events = {}
    for case in range(600):
        args = _fuzz_case(rng)
        got = _run_state_machine(*args)
        exp = reference_state_machine(*args, events=events)
        for name, x, y in zip(SM_OUT, got, exp):
            if not np.array_equal(np.asarray(x).view(np.uint8), np.asarray(y).view(np.uint8)):
                raise AssertionError(f"case {case}: kernel output '{name}' differs from the reference implementation")
    for key in ("write_a", "signal", "claimed_at_switch", "requirement_expired", "cleared_unclaimed", "memory_block",
                "memory_expired", "memory_moved", "memory_write_inactive", "memory_write_active", "reject", "carry",
                "roll", "exhausted", "cleared_in_state_5"):
        assert events.get(key, 0) >= 5, f"random cases exercised '{key}' only {events.get(key, 0)} times"


SIG8 = [ABOVE, TOUCH] + [ABOVE] * 6 + [PIN]          # signal at bar 8 on a VAL-100 box -> requirement anchored at 100


def test_sm4_claimed_requirement_cleared_at_current_box_band_top():
    # N1: a claimed requirement is measured against the CLAIMING box's band top, not its anchor's band top.
    # (i) higher claiming box 100.8 (band top 101.3): low 100.6 > anchor top 100.5 must NOT clear it
    out = sm(SIG8 + [IN, (100.9, 101.2, 100.6, 101.0)], [{"at": 0, "val": 100.0}, {"at": 10, "val": 100.8}])
    assert out["sig"][8] and out["need"][10] and out["state"][10] == ARMED
    # (ii) lower claiming box 99.2 (band top 99.7): low 99.8 clears it; the next touch is a first test
    out = sm(SIG8 + [IN, (100.0, 100.2, 99.8, 100.1), (100.0, 100.1, 99.6, 99.9)],
             [{"at": 0, "val": 100.0}, {"at": 10, "val": 99.2}])
    assert not out["need"][10] and out["state"][11] == TOUCH1


def test_sm4_unclaimed_requirement_cleared_at_anchor_band_top_in_any_state():
    # N2 / N3 / N21: while box B (101.2, other zone) is current, the A-zone requirement (anchor 100) is unclaimed and
    # is cleared by a bar with low > 100.5 even though that bar is inside B's band and the state is TOUCH1
    bars = SIG8 + [(100.9, 101.1, 100.4, 101.0), (101.0, 101.2, 100.4, 101.1), (101.4, 101.6, 100.9, 101.3),
                   (100.6, 100.7, 100.3, 100.5)]
    out = sm(bars, [{"at": 0, "val": 100.0}, {"at": 10, "val": 101.2}, {"at": 12, "val": 100.0}])
    assert out["state"][10] == TOUCH1 and out["state"][11] == TOUCH1
    assert not out["need"][12] and out["state"][12] == TOUCH1          # back in zone A: nothing left to claim


@pytest.mark.parametrize("ws_b, need_at_return", [(2, True), (3, False)])
def test_sm4_activation_requirement_write_time_is_activation_open(ws_b, need_at_return):
    # N12 / N13: (a) requirement of box A (activated at bar 3 from below) carries t = open of bar 3. A box with
    # window start = bar 3 is fresh structure for it (expires while unclaimed); window start = bar 2 is not.
    bars = [(98.0, 98.4, 97.8, 98.2), (98.2, 98.6, 98.0, 98.4), (98.5, 99.2, 98.4, 99.0)] + [IN] * 5 \
        + [(100.4, 100.5, 100.0, 100.3)]
    out = sm(bars, [{"at": 3, "val": 100.0, "ws": -10}, {"at": 6, "val": 102.0, "ws": ws_b},
                    {"at": 8, "val": 100.0, "ws": ws_b}])
    assert out["need"][3] and out["state"][3] == ARMED
    assert out["need"][8] == need_at_return and out["state"][8] == (ARMED if need_at_return else TOUCH1)


def test_sm4_requirement_cleared_on_the_activation_bar():
    # N40: activation from below writes a requirement; the activation bar itself is entirely above the band -> cleared
    out = sm([BELOW, (100.8, 101.5, 100.7, 101.2), TOUCH], [{"at": 1, "val": 100.0}])
    assert not out["need"][1] and out["state"][2] == TOUCH1


def test_sm4_rejected_box_writes_activation_requirement():
    # N18: a rejected box (prev close below its band low) carries an (a) requirement that the next same-zone box claims
    bars = [IN, IN_LOW, IN, TOUCH]
    out = sm(bars, [{"at": 2, "val": 100.3}, {"at": 3, "val": 100.0}], StrategyConfig(reject_broken_at_activation=True))
    assert out["state"][2] == INVALID and out["need"][3] and out["state"][3] == ARMED


def test_sm4_memory_swap_remove_keeps_surviving_entry_time():
    # N25: memory [100 @ close of bar 2, 95 @ close of bar 6]; box 97.5 with window start bar 3 expires only 100.
    # The surviving 95 entry must keep its own time: box 95.2 with window start bar 4 still contains pre-breakdown data
    bars = [ABOVE, TOUCH, BRK, (99.0, 99.2, 96.0, 96.2), (96.2, 96.5, 95.8, 96.0), (96.0, 96.1, 95.2, 95.4),
            (95.4, 95.5, 93.8, 94.0, 200.0), (94.0, 97.0, 93.9, 96.9), (96.9, 98.0, 96.8, 97.9), (97.9, 98.0, 97.0, 97.5),
            (97.0, 97.2, 95.6, 96.0)]
    boxes = [{"at": 0, "val": 100.0, "ws": -50}, {"at": 4, "val": 95.0, "ws": -10}, {"at": 8, "val": 97.5, "ws": 3},
             {"at": 10, "val": 95.2, "ws": 4}]
    out = sm(bars, boxes)
    assert out["state"][2] == INVALID and out["state"][6] == INVALID and out["state"][8] == ARMED
    assert out["state"][10] == INVALID


def test_sm4_requirement_swap_remove_keeps_claim_flag_on_clear():
    # N27: requirements [98.8 (signal, unclaimed after the switch), 100 ((a) at box 100, claimed)]; bar 10 clears only
    # the unclaimed 98.8 entry (low 99.6 > 99.3). The claimed 100 entry must stay claimed -> no first test at bar 10
    up = (99.8, 100.6, 99.6, 100.3)
    bars = [up, (99.8, 100.0, 99.0, 99.4)] + [up] * 6 + [(99.7, 99.8, 98.5, 99.75), (99.7, 99.9, 99.0, 99.2),
                                                         (99.3, 100.2, 99.6, 100.0)]
    out = sm(bars, [{"at": 0, "val": 98.8}, {"at": 10, "val": 100.0}])
    assert out["sig"][8] and out["need"][10] and out["state"][10] == ARMED


def test_sm4_requirement_swap_remove_keeps_claim_flag_on_expiry():
    # N28: same two requirements; box 100.1 (window start bar 9) expires the unclaimed 98.8 entry (fresh structure)
    # and claims the 100 entry, which must stay claimed
    up = (99.8, 100.6, 99.6, 100.3)
    bars = [up, (99.8, 100.0, 99.0, 99.4)] + [up] * 6 + [(99.7, 99.8, 98.5, 99.75), (99.7, 99.9, 99.0, 99.2),
                                                         (99.3, 100.2, 99.2, 100.0), (100.0, 100.3, 99.2, 99.9),
                                                         (99.9, 100.3, 99.7, 100.1)]
    out = sm(bars, [{"at": 0, "val": 98.8}, {"at": 10, "val": 100.0}, {"at": 12, "val": 100.1, "ws": 9}])
    assert out["sig"][8] and out["need"][12] and out["state"][12] == ARMED


def test_sm4_memory_expiry_checks_every_entry_after_swap():
    # N32: memory in array order [100 @ bar 3, 95 @ bar 15 (moved by a re-breakdown), 90 @ bar 11]; box 90.3 with window
    # start bar 12 must expire both 100 and 90 (only 95 survives) and therefore re-arm
    bars = [ABOVE, TOUCH, BRK, (99.0, 99.2, 96.0, 96.2), (96.2, 96.5, 95.8, 96.0), (96.0, 96.1, 95.2, 95.4),
            (95.4, 95.5, 93.8, 94.0, 200.0), (94.0, 94.2, 91.0, 91.2), (91.2, 91.5, 90.8, 91.0), (91.0, 91.1, 90.2, 90.4),
            (90.4, 90.5, 88.8, 89.0, 200.0), (89.0, 96.0, 88.9, 95.8), (95.8, 96.0, 95.4, 95.6),
            (95.6, 95.7, 95.0, 95.3), (95.3, 95.4, 93.9, 94.1, 200.0), (94.1, 94.2, 90.4, 90.6), (90.6, 90.9, 90.5, 90.8)]
    boxes = [{"at": 0, "val": 100.0, "ws": -50}, {"at": 4, "val": 95.0, "ws": -40}, {"at": 8, "val": 90.0, "ws": -30},
             {"at": 12, "val": 95.2, "ws": -20}, {"at": 16, "val": 90.3, "ws": 12}]
    out = sm(bars, boxes)
    assert out["state"][10] == INVALID and out["state"][12] == INVALID and out["brk"][14]
    assert out["state"][16] != INVALID                                  # re-armed (first touch on the same bar)


@pytest.mark.parametrize("ws_a", [8, 9])
def test_sm4_return_box_claims_before_expiry(ws_a):
    # N9: A(100) -> B(101.2, window start bar 8: not fresh) -> A'(100, window start 8 or 9). Claims are computed
    # before expiry, so A' claims the outstanding A requirement even when A' itself is fresh structure (ws 9)
    boxes = [{"at": 0, "val": 100.0}, {"at": 12, "val": 101.2, "ws": 8}, {"at": 13, "val": 100.0, "ws": ws_a}]
    out = sm(_post_signal_chop(), boxes)
    assert out["need"][13] and list(np.flatnonzero(out["sig"])) == [8]


def test_sm4_unclaimed_requirement_expires_at_carry_switch():
    # N10: signal at bar 8 (anchor 100); box 98.8 (other zone, not fresh) gets a first test; the sequence is carried to
    # box 99.0 whose window starts at the signal close (fresh) -> the unclaimed 100 requirement expires there, even
    # though the switch is a carry. After the 99.0 signal and a departure above 99.5, box 100 re-arms with nothing to claim
    bars = SIG8 + [IN, (100.0, 100.1, 99.2, 99.4), (99.5, 100.2, 99.4, 100.0)] + [(100.0, 100.4, 99.6, 100.1)] * 5 \
        + [(99.9, 100.0, 98.8, 99.95), (100.0, 100.3, 99.7, 100.1), TOUCH]
    boxes = [{"at": 0, "val": 100.0}, {"at": 10, "val": 98.8}, {"at": 17, "val": 99.0, "ws": 9},
             {"at": 19, "val": 100.0, "ws": 9}]
    out = sm(bars, boxes)
    assert out["state"][10] == TOUCH1 and out["state"][16] == 3 and list(np.flatnonzero(out["sig"])) == [8, 17]
    assert not out["need"][19] and out["state"][19] == TOUCH1


def test_sm4_requirement_boundaries_are_strict_and_inclusive():
    # 清除：low 恰等于带上沿（100.5）不算“整根位于带上方”，要求保留（N5）
    out = sm(SIG8 + [(100.8, 101.2, 100.5, 101.0), TOUCH], [{"at": 0, "val": 100.0}])
    assert out["need"][9] and out["state"][10] == ARMED
    # 认领：新箱体 VAL 与锚点恰好相距 (k1+k2)*ATR = 1.0 仍属同区（N7）；上一根收盘不在带下沿之下，不触发激活条件 (a)
    out = sm(SIG8 + [(100.4, 100.9, 100.3, 100.7), (100.9, 101.3, 100.8, 101.1)],
             [{"at": 0, "val": 100.0}, {"at": 10, "val": 101.0}])
    assert out["need"][10] and out["state"][10] == ARMED


def test_sm4_signal_anchor_is_the_box_current_at_the_signal():
    # 第 2 次测试从 VAL 100 的箱体开始（t2_val = 100），K 线 9 carry 到 VAL 100.8 的箱体并在其上发出信号：
    # 要求的锚点是 100.8（信号时的箱体），因此与 100.8 同区、与 100 不同区的 VAL 101.7 箱体必须认领它（N15）
    bars = [ABOVE, TOUCH] + [ABOVE] * 6 + [(101.0, 101.2, 100.3, 100.6), (101.3, 101.4, 100.4, 101.35),
                                           (101.3, 101.5, 101.0, 101.3), (101.3, 101.6, 101.1, 101.4)]
    out = sm(bars, [{"at": 0, "val": 100.0}, {"at": 9, "val": 100.8}, {"at": 11, "val": 101.7}])
    assert out["state"][8] == TOUCH2 and out["sig"][9] and out["need"][11] and out["state"][11] == ARMED


def test_sm4_carry_switch_does_not_claim_requirements():
    # 已记录的设计选择（SM4-5）：carry 切换不重新认领。信号（K 线 8，锚点 100）后，第 1 次测试在不同区的 98.8 箱体上，
    # 序列 carry 到与 100 同区的 99.6 箱体并在 K 线 17 发出信号；100 区的要求在此期间保持未认领，
    # 按锚点带上沿 100.5 判断，K 线 18（low 100.2）只清除 99.6 区的要求，回到 VAL 100 的箱体时仍须先到带上方（N24）
    bars = SIG8 + [IN, (100.0, 100.1, 99.2, 99.4), (99.5, 100.2, 99.4, 100.0)] + [(100.0, 100.4, 99.6, 100.1)] * 5 \
        + [(100.0, 100.05, 99.5, 100.0), (100.3, 100.6, 100.2, 100.4), TOUCH]
    out = sm(bars, [{"at": 0, "val": 100.0}, {"at": 10, "val": 98.8}, {"at": 17, "val": 99.6}, {"at": 19, "val": 100.0}])
    assert list(np.flatnonzero(out["sig"])) == [8, 17] and not out["need"][16]
    assert not out["need"][18] and out["need"][19] and out["state"][19] == ARMED


def test_sm4_invalid_boxes_claim_requirements():
    # 被否决的箱体（N23）与被放量跌穿记忆阻断的箱体（N22）也认领同区的未完成要求（INVALID 期间按它的带上沿清除）
    out = sm(_post_signal_chop(), [{"at": 0, "val": 100.0}, {"at": 12, "val": 100.3}, {"at": 13, "val": 100.0}],
             StrategyConfig(reject_broken_at_activation=True))
    assert out["state"][12] == INVALID and out["need"][12]
    pre = [(99.6, 99.8, 99.2, 99.4), (99.4, 99.5, 97.9, 98.0, 200.0), (98.0, 101.2, 97.9, 101.0)]
    out = sm(pre + _post_signal_chop(), [{"at": 0, "val": 98.8, "ws": -50}, {"at": 2, "val": 100.0, "ws": -40},
                                         {"at": 15, "val": 99.5, "ws": -30}, {"at": 16, "val": 100.0, "ws": -20}])
    assert out["state"][15] == INVALID and out["need"][15]



# ============================================================================
# 第四轮复核：执行 / 指标 / 管道（EX4-4、EX4-7）
# ============================================================================
def test_ex4_4_cache_isolated_from_box_rangeindex_writes():
    # RangeIndex.copy(deep=True) 与原对象共享已物化的整数数组：经 boxes.index.array 的写入曾污染缓存
    pipeline.clear_cache()
    boxes, _ = pipeline.get_boxes("BTCUSDT", StrategyConfig())
    try:
        boxes.index.array[0] = 12345
    except (TypeError, ValueError):
        pass                                                              # 不可写同样满足隔离
    fresh = pipeline.get_boxes("BTCUSDT", StrategyConfig())[0]
    assert fresh.index.to_numpy()[0] == 0 and fresh.reset_index()["box_id"].iloc[0] == 0
    pipeline.clear_cache()


def test_ex4_7_series_helpers_are_locked_past_oos_start():
    res = _mk_res(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-08-01", tz="UTC"))
    for fn in (max_drawdown, drawdown_series):
        with pytest.raises(OOSLockedError):
            fn(res.equity)
        fn(res.equity, allow_oos=True)
    d = daily_equity(res, allow_oos=True)
    with pytest.raises(OOSLockedError):
        daily_returns(d)
    r = daily_returns(d, allow_oos=True)
    for fn in (sharpe, sortino):
        with pytest.raises(OOSLockedError):
            fn(r)
        fn(r, allow_oos=True)
    # 研究结果不被锁：逐小时净值恰好止于 OOS_START，日度标签止于 2025-06-30
    ok = _mk_res(pd.Timestamp("2024-01-01", tz="UTC"), OOS_START)
    assert daily_equity(ok).index[-1] == pd.Timestamp("2025-06-30", tz="UTC")
    sharpe(daily_returns(daily_equity(ok)))
    max_drawdown(ok.equity)
    drawdown_series(ok.equity)
