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
    tiny = df.iloc[1:3]                                                    # (01:00, 03:00] 内无结算网格点
    run_backtest(tiny, sig.iloc[1:3], BacktestConfig(), full.iloc[:0])


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
