"""
test_impulse_cp2.py — 脉冲研究 Checkpoint 2：多空双向撮合引擎、因果特征与版本输入的手算 / 合成测试。

撮合只使用合成数据（impulse_study.synthetic）；真实行情只在最后两个测试中用于因果特征回归（不进入撮合，不计算任何盈亏）。
期望值全部在测试中按公式手算给出。
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from impulse_study import daily as DY
from impulse_study import engine as E
from impulse_study import features as F
from impulse_study import synthetic as SY
from impulse_study import versions as VS

START = pd.Timestamp("2021-01-04 00:00", tz="UTC")       # 15 分钟整刻
P0 = 10_000.0
MAIN = VS.COSTS["main_17bp"]
GROSS = VS.COSTS["gross_0bp"]
RISK = VS.SIZINGS["risk_0.5pct"]
NOTIONAL = VS.SIZINGS["notional_1x"]
B1 = VS.VERSIONS["B1_pure_long"]
B3 = VS.VERSIONS["B3_pure_bi_mom"]
B4 = VS.VERSIONS["B4_pure_1bar"]
I0 = 1000                                                  # 默认触发分钟
E0 = I0 + 1                                                # 入场分钟


def bars(n=1500, price=P0, vol=10.0, spread=0.0005):
    O, H, L, C, V = SY.flat_bars(n, price, vol, spread)
    return O, H, L, C, V


def ctx_from(O, H, L, C, V, bad=None, start_idx=400, last_idx=None, funding=None, gaps=(), box=None):
    g = SY.grid_from_arrays(START, O, H, L, C, V, bad)
    last = len(C) - 40 if last_idx is None else last_idx
    return SY.synthetic_context(g, g.index[start_idx], g.index[last], box_ok=box, funding=funding, archive_gap_minutes=gaps)


def run(ctx, signals, spec=B1, cost=MAIN, sizing=RISK):
    return E.run_portfolio(ctx, SY.manual_inputs(ctx, spec, signals), cost, sizing)


def q_risk(eref, sl, eq=VS.INITIAL_CAPITAL, cost=MAIN):
    return min(VS.RISK_FRACTION * eq / (max(abs(eref - sl), VS.MIN_DIST_FRAC * eref) + cost.c_rt * eref),
               VS.LEVERAGE_CAP * eq / eref)


# ----------------------------------------------------------------------------
# 1. 多空镜像对称、滑点方向、资金费方向
# ----------------------------------------------------------------------------
def test_mirror_symmetry_long_vs_short_on_reflected_path():
    """价格按 K − P 镜像：原路径做多与镜像路径做空的盈亏、离场原因、离场分钟、v1 / k1 逐一相同（零成本，数量相同）。"""
    g = SY.random_grid(3000, start=str(START.tz_localize(None)), seed=7, impulse_rate=0.0)
    K = 2.0 * float(g["close"].mean())
    O, H, L, C, V = (g[c].to_numpy() for c in ("open", "high", "low", "close", "volume"))
    ctx_a = ctx_from(O, H, L, C, V, start_idx=400, last_idx=2900)
    ctx_b = ctx_from(K - O, K - L, K - H, K - C, V, start_idx=400, last_idx=2900)
    rng = np.random.default_rng(3)
    sig_a, sig_b = {}, {}
    for i in range(500, 2800, 97):
        d = float(rng.uniform(0.004, 0.008)) * C[i]          # 止损距离（远大于 0.15% 下限，2.0x 上限不触发）
        sig_a[i] = (1, C[i] - d)
        sig_b[i] = (-1, (K - C[i]) + d)
    ra = run(ctx_a, sig_a, cost=GROSS)
    rb = run(ctx_b, sig_b, cost=GROSS)
    ta, tb = ra.trades, rb.trades
    assert len(ta) == len(tb) > 10
    assert (ta["reason"].to_numpy() == tb["reason"].to_numpy()).all()
    assert (ta["exit_bar"].to_numpy() == tb["exit_bar"].to_numpy()).all()
    np.testing.assert_allclose(ta["qty"], tb["qty"], rtol=1e-12)
    np.testing.assert_allclose(ta["net_pnl"], tb["net_pnl"], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(ta["k1"].fillna(-9), tb["k1"].fillna(-9), rtol=1e-9)
    np.testing.assert_allclose(ta["v1"].fillna(-9), tb["v1"].fillna(-9), rtol=1e-12)
    np.testing.assert_allclose(ra.equity.dropna(), rb.equity.dropna(), rtol=1e-12)
    assert set(ta["reason_name"]) >= {"time_bar1"}


def test_fees_and_slippage_directions_long_and_short():
    O, H, L, C, V = bars()
    ctx = ctx_from(O, H, L, C, V)
    rl = run(ctx, {I0: (1, P0 - 100)})
    rs = run(ctx, {I0: (-1, P0 + 100)})
    q = 500.0 / 117.0
    tl, ts = rl.trades.iloc[0], rs.trades.iloc[0]
    assert tl["qty"] == pytest.approx(q, rel=1e-13) and ts["qty"] == pytest.approx(q, rel=1e-13)
    assert tl["entry_fill"] == pytest.approx(10_005.0) and tl["exit_fill"] == pytest.approx(9_998.0)
    assert ts["entry_fill"] == pytest.approx(9_995.0) and ts["exit_fill"] == pytest.approx(10_002.0)
    assert tl["fee_entry"] == pytest.approx(0.0005 * q * 10_005.0) and tl["fee_exit"] == pytest.approx(0.0005 * q * 9_998.0)
    assert ts["fee_entry"] == pytest.approx(0.0005 * q * 9_995.0) and ts["fee_exit"] == pytest.approx(0.0005 * q * 10_002.0)
    assert tl["net_pnl"] == pytest.approx(-17.0015 * q, rel=1e-12)
    assert ts["net_pnl"] == pytest.approx(-16.9985 * q, rel=1e-12)
    assert tl["gross_pnl"] == 0.0 and ts["gross_pnl"] == 0.0
    assert tl["reason_name"] == ts["reason_name"] == "time_bar1"
    assert rl.meta["final_equity"] == pytest.approx(VS.INITIAL_CAPITAL - 17.0015 * q, rel=1e-13)


@pytest.mark.parametrize("rate", [1e-4, -1e-4])
def test_funding_sign_long_pays_short_receives(rate):
    O, H, L, C, V = bars()
    tau = START + pd.Timedelta(minutes=E0 + 5)              # 结算时点 = 持仓中第 6 根 1m 的开盘
    fund = pd.Series([rate], index=[tau])
    ctx = ctx_from(O, H, L, C, V, funding=fund)
    rl = run(ctx, {I0: (1, P0 - 100)})
    rs = run(ctx, {I0: (-1, P0 + 100)})
    q = 500.0 / 117.0
    assert rl.trades.iloc[0]["funding"] == pytest.approx(-q * P0 * rate, rel=1e-12)   # 正费率多头付，负费率多头收
    assert rs.trades.iloc[0]["funding"] == pytest.approx(+q * P0 * rate, rel=1e-12)   # 正费率空头收，负费率空头付
    assert rl.trades.iloc[0]["n_funding"] == rs.trades.iloc[0]["n_funding"] == 1
    g = run(ctx, {I0: (1, P0 - 100)}, cost=GROSS)                                     # 零成本情景不计资金费
    assert g.trades.iloc[0]["funding"] == 0.0


def test_funding_timing_conventions():
    O, H, L, C, V = bars()
    q = 500.0 / 117.0
    r = 2e-4
    # (a) 结算时点恰为入场时刻（入场棒开盘）：入场不早于 τ → 不收付
    ctx = ctx_from(O, H, L, C, V, funding=pd.Series([r], index=[START + pd.Timedelta(minutes=E0)]))
    assert run(ctx, {I0: (1, P0 - 100)}).trades.iloc[0]["n_funding"] == 0
    # (b) 时间离场在第 1 根合成棒末分钟收盘、收盘时刻恰为 τ：按该分钟收盘价收付
    C2 = C.copy()
    C2[E0 + 14] = 10_002.0
    ctx = ctx_from(O, H, L, C2, V, funding=pd.Series([r], index=[START + pd.Timedelta(minutes=E0 + 15)]))
    t = run(ctx, {I0: (1, P0 - 100)}).trades.iloc[0]
    assert t["reason_name"] == "time_bar1" and t["funding"] == pytest.approx(-q * 10_002.0 * r, rel=1e-12)
    # (c) 盘中止损发生在收盘时刻 τ 之前：不收付 τ
    L3 = L.copy()
    L3[E0 + 4] = P0 - 150
    ctx = ctx_from(O, H, L3, C, V, funding=pd.Series([r], index=[START + pd.Timedelta(minutes=E0 + 5)]))
    t = run(ctx, {I0: (1, P0 - 100)}).trades.iloc[0]
    assert t["reason_name"] == "stop_intrabar" and t["n_funding"] == 0
    # (d) 跳空止损发生在 τ 开盘（τ 时仍持仓）：收付 τ，价格取该分钟开盘价
    O4, L4 = O.copy(), L.copy()
    O4[E0 + 5], L4[E0 + 5] = P0 - 120, P0 - 130
    ctx = ctx_from(O4, H, L4, C, V, funding=pd.Series([r], index=[START + pd.Timedelta(minutes=E0 + 5)]))
    t = run(ctx, {I0: (1, P0 - 100)}).trades.iloc[0]
    assert t["reason_name"] == "stop_gap_open" and t["funding"] == pytest.approx(-q * (P0 - 120) * r, rel=1e-12)


def test_short_mark_to_market_equity():
    O, H, L, C, V = bars()
    C2 = C.copy()
    C2[E0 + 3] = 9_950.0
    ctx = ctx_from(O, H, L, C2, V)
    res = run(ctx, {I0: (-1, P0 + 100)})
    q = 500.0 / 117.0
    entry_fill = P0 * (1 - 0.0005)
    cash_after_entry = VS.INITIAL_CAPITAL - 0.0005 * q * entry_fill
    assert res.equity.iloc[E0 + 3] == pytest.approx(cash_after_entry - q * (9_950.0 - entry_fill), rel=1e-13)
    assert res.equity.iloc[I0] == VS.INITIAL_CAPITAL                       # 触发分钟收盘：空仓
    assert res.equity.iloc[E0 + 14] == pytest.approx(res.meta["final_equity"], rel=1e-13)


# ----------------------------------------------------------------------------
# 2. ATR_15m(14)：自然整刻、初始化、部分缺失、整根维护缺失
# ----------------------------------------------------------------------------
def _bucket_bars(trs, closes=None, bad_buckets=(), partial=None):
    """每个 15m 桶内价格恒为 c_b，只在桶内第 3 分钟的高、第 7 分钟的低放出 ±TR/2。"""
    nb = len(trs)
    n = nb * 15
    c = np.full(nb, 100.0) if closes is None else np.asarray(closes, float)
    O = np.repeat(c, 15)
    C = O.copy()
    H = O.copy()
    L = O.copy()
    for b, tr in enumerate(trs):
        H[b * 15 + 3] = c[b] + tr / 2
        L[b * 15 + 7] = c[b] - tr / 2
    V = np.full(n, 5.0)
    bad = np.zeros(n, bool)
    for b in bad_buckets:
        bad[b * 15:(b + 1) * 15] = True
    if partial is not None:
        b, mins = partial
        for k in mins:
            bad[b * 15 + k] = True
    return O, H, L, C, V, bad


def test_atr15_initialization_and_wilder():
    trs = [float(k) for k in range(1, 15)] + [28.0, 7.0]
    O, H, L, C, V, bad = _bucket_bars(trs)
    a = F.atr15_natural(O, H, L, C, bad, int(START.value))
    assert np.isnan(a["atr"][:13]).all()
    assert a["atr"][13] == pytest.approx(7.5)                              # 前 14 个有效 TR 的算术均值
    assert a["atr"][14] == pytest.approx((13 * 7.5 + 28.0) / 14)
    assert a["atr"][15] == pytest.approx((13 * a["atr"][14] + 7.0) / 14)
    atr_at = a["atr_at"]
    assert np.isnan(atr_at[14 * 15 - 2])                                   # 第 14 根桶在 210 分钟收盘之前：未初始化
    assert atr_at[14 * 15 - 1] == pytest.approx(7.5)                       # 决策时刻 T+1 = 210 分钟：恰好可读
    assert atr_at[15 * 15 - 2] == pytest.approx(7.5) and atr_at[15 * 15 - 1] == pytest.approx(a["atr"][14])


def test_atr15_reads_only_closed_natural_buckets():
    trs = [2.0] * 20
    O, H, L, C, V, bad = _bucket_bars(trs)
    H[15 * 16 + 3] = 100 + 50.0                                            # 第 16 根桶内的大振幅
    a = F.atr15_natural(O, H, L, C, bad, int(START.value))
    assert a["atr"][15] == pytest.approx(2.0)
    assert a["atr"][16] == pytest.approx((13 * 2.0 + 51.0) / 14)
    t_in_bucket16 = 15 * 16 + 10                                           # 桶 16 尚未收盘
    assert a["atr_at"][t_in_bucket16] == pytest.approx(2.0)
    assert a["atr_at"][15 * 17 - 1] == pytest.approx(a["atr"][16])         # 桶 16 收盘时刻的决策
    with pytest.raises(ValueError):
        F.atr15_natural(O, H, L, C, bad, int(START.value) + 60_000_000_000)   # 非 15 分钟整刻起点


def test_atr15_partial_bucket_uses_valid_minutes_only():
    trs = [2.0] * 20
    O, H, L, C, V, bad = _bucket_bars(trs, partial=(16, [3, 4, 5]))
    H[16 * 15 + 4] = 999.0                                                # 维护占位分钟里的离谱价格必须被忽略
    a = F.atr15_natural(O, H, L, C, bad, int(START.value))
    assert a["n_valid"][16] == 12
    assert a["high"][16] == pytest.approx(100.0) and a["low"][16] == pytest.approx(99.0)
    assert a["atr"][16] == pytest.approx((13 * 2.0 + 1.0) / 14)            # TR = 1（高点所在分钟被剔除）


def test_atr15_fully_bad_bucket_holds_value_and_previous_close():
    trs = [2.0] * 20
    closes = [100.0] * 16 + [100.0, 500.0, 102.0, 102.0]
    O, H, L, C, V, bad = _bucket_bars(trs, closes=closes, bad_buckets=(17,))
    a = F.atr15_natural(O, H, L, C, bad, int(START.value))
    assert a["n_valid"][17] == 0
    assert a["atr"][17] == a["atr"][16] and np.isnan(a["tr"][17])         # 不计 TR、不衰减
    assert a["tr"][18] == pytest.approx(3.0)                               # max(2, |103 − 100|, |101 − 100|)：前值收盘 100 衔接
    assert a["atr"][18] == pytest.approx((13 * a["atr"][16] + 3.0) / 14)


def test_atr15_init_skips_fully_bad_buckets():
    trs = [float(k) for k in range(1, 16)]
    O, H, L, C, V, bad = _bucket_bars(trs, bad_buckets=(5,))
    a = F.atr15_natural(O, H, L, C, bad, int(START.value))
    assert np.isnan(a["atr"][13])                                          # 只有 13 个有效 TR
    valid_trs = [t for k, t in enumerate(trs) if k != 5][:14]
    assert a["atr"][14] == pytest.approx(np.mean(valid_trs))


def test_no_trade_before_atr15_initialized():
    O, H, L, C, V = bars(n=800)
    ctx = ctx_from(O, H, L, C, V, start_idx=30, last_idx=700)
    up = np.zeros(len(C), bool)
    up[[100, 500]] = True
    ctx.cond["pct"] = {"up1": up, "up2": np.zeros(len(C), bool), "dn1": np.zeros(len(C), bool), "dn2": np.zeros(len(C), bool)}
    vin = F.build_version_inputs(ctx, B1)
    assert np.isnan(ctx.atr["atr_at"][100]) and vin.entry_dir[100] == 0
    assert vin.entry_dir[500] == 1 and vin.skipped["atr_not_ready"] == 1


# ----------------------------------------------------------------------------
# 3. S6 二段支撑与日线 as_of 边界
# ----------------------------------------------------------------------------
def _daily(lows, highs, closes=None):
    idx = pd.date_range("2021-01-01", periods=len(lows), freq="1D", tz="UTC").as_unit("ns")
    lows, highs = np.asarray(lows, float), np.asarray(highs, float)
    c = (lows + highs) / 2 if closes is None else np.asarray(closes, float)
    d = pd.DataFrame({"open": c, "high": highs, "low": lows, "close": c, "volume": 1.0, "n_min": 1440}, index=idx)
    d["available_time"] = d.index + pd.Timedelta(days=1)
    return d


def test_s6_requires_exit_above_poc_after_touch():
    n = 40
    lo, hi = np.full(n, 112.0), np.full(n, 114.0)
    lo[n - 10], hi[n - 10] = 100.5, 101.5                      # 触及 [99, 101]
    lo[n - 9:], hi[n - 9:] = 101.2, 102.8                       # 反弹达到 102.6 但一直低于 POC 110
    d = _daily(lo, hi)
    atr = pd.Series(2.0, index=d.index)
    as_of = d.index[-1] + pd.Timedelta(days=1)
    prof = pd.DataFrame({"POC": 110.0, "VAL": 100.0, "VAH": 120.0, "P_low": 90.0, "P_high": 130.0, "n_bars": 1},
                        index=pd.DatetimeIndex([as_of]))
    assert bool(DY.box_state(d, prof, atr)["primed"].iloc[0])                      # 原状态主口径：满足
    assert not bool(DY.box_state(d, prof, atr, exit_above_poc=True)["primed"].iloc[0])  # S6：从未越过 POC
    hi2 = hi.copy()
    hi2[n - 4] = 110.5                                          # 触及之后某日最高价 ≥ 反弹线且 > POC
    assert bool(DY.box_state(_daily(lo, hi2), prof, atr, exit_above_poc=True)["primed"].iloc[0])
    hi3 = hi.copy()
    hi3[n - 12] = 115.0                                         # 越过 POC 发生在触及之前：不算
    assert not bool(DY.box_state(_daily(lo, hi3), prof, atr, exit_above_poc=True)["primed"].iloc[0])


def test_s6_state_used_by_box_versions_and_asof_boundary():
    days = pd.date_range("2021-03-01", periods=2, freq="1D", tz="UTC").as_unit("ns")
    st = pd.DataFrame({"ATR": 2.0, "band_lo": [99.0, 50.0], "band_hi": 101.0, "bounce_level": 102.6, "POC": [110.0, 60.0],
                       "VAL": 100.0, "primed": [True, True], "valid": True}, index=days)
    t = pd.DatetimeIndex(["2021-03-01 23:59", "2021-03-02 00:00"], tz="UTC").as_unit("ns")
    ok = DY.minute_box_ok(np.asarray(t.asi8), np.array([105.0, 105.0]), st)
    assert ok.tolist() == [True, False]                         # 23:59 用 3/1 的状态，00:00 才切换到 3/2
    O, H, L, C, V = bars(n=1500)
    box = np.zeros(1500, bool)
    box[700] = True
    ctx = ctx_from(O, H, L, C, V, box=box)
    up = np.zeros(1500, bool)
    up[[700, 900]] = True
    z = np.zeros(1500, bool)
    ctx.cond["pct"] = {"up1": up, "up2": z, "dn1": z, "dn2": z}
    vin = F.build_version_inputs(ctx, VS.VERSIONS["A1_box_opt"])
    assert vin.entry_dir[700] == 1 and vin.entry_dir[900] == 0 and vin.skipped["box_rejected"] == 1


# ----------------------------------------------------------------------------
# 4. W 选择与脉冲起点极值
# ----------------------------------------------------------------------------
def _cond_ctx():
    O, H, L, C, V = bars(n=1500)
    t = 1000
    C[t - 15], C[t - 5] = 9_970.0, 9_990.0
    L[t - 14:t + 1] = [9_960, 9_965, 9_968, 9_975, 9_980, 9_982, 9_984, 9_986, 9_988, 9_989, 9_985, 9_992, 9_995, 9_998, 10_040]
    H[t - 14:t + 1] = [10_090, 10_005, 10_005, 10_005, 10_005, 10_005, 10_005, 10_005, 10_005, 10_005, 10_005, 10_005,
                       10_010, 10_020, 10_060]
    C[t] = 10_050.0
    ctx = ctx_from(O, H, L, C, V)
    z = np.zeros(1500, bool)
    return ctx, t, z


def test_w_selection_uses_own_rule_and_direction():
    ctx, t, z = _cond_ctx()
    one = z.copy()
    one[t] = True
    atr = ctx.atr["atr_at"][t]
    ctx.cond["pct"] = {"up1": one, "up2": z, "dn1": z, "dn2": one}      # 只有 ① 向上；反方向 ② 成立但不得影响 W
    vin = F.build_version_inputs(ctx, VS.VERSIONS["B2_pure_long_sl"])
    assert vin.w[t] == 5 and vin.cat[t] == 1
    assert vin.sl[t] == pytest.approx(min(9_990.0, 9_985.0) - 0.1 * atr)   # L_origin = min(C_{t−5}, min L_{t−4..t})
    ctx.cond["pct"] = {"up1": one, "up2": one, "dn1": z, "dn2": z}      # ① + ② → W = 15
    vin = F.build_version_inputs(ctx, VS.VERSIONS["B2_pure_long_sl"])
    assert vin.w[t] == 15 and vin.cat[t] == 3
    assert vin.sl[t] == pytest.approx(min(9_970.0, 9_960.0) - 0.1 * atr)
    # Z 口径的版本只看 Z 条件：百分比口径的 ② 不影响 A5 的 W
    ctx.box_ok[:] = True
    ctx.cond["z"] = {"up1": one, "up2": z, "dn1": z, "dn2": z}
    vin_z = F.build_version_inputs(ctx, VS.VERSIONS["A5_box_origin_z"])
    assert vin_z.w[t] == 5 and vin_z.sl[t] == pytest.approx(9_985.0 - 0.1 * atr)


def test_sl_c_and_short_origin():
    ctx, t, z = _cond_ctx()
    one = z.copy()
    one[t] = True
    ctx.box_ok[:] = True
    ctx.cond["pct"] = {"up1": z, "up2": one, "dn1": z, "dn2": z}
    vin = F.build_version_inputs(ctx, VS.VERSIONS["A3_box_half"])
    assert vin.w[t] == 15 and vin.sl[t] == pytest.approx(10_050.0 - 0.5 * (10_050.0 - 9_960.0))
    # 空头：H_origin = max(C_{t−W}, max H_{t−W+1..t})，SL_B = H_origin + 0.1·ATR
    O, H, L, C, V = bars(n=1500)
    C[t - 5] = 10_030.0
    H[t - 4:t + 1] = [10_040.0, 10_010.0, 10_008.0, 10_006.0, 10_002.0]
    C[t] = 9_960.0
    L[t] = 9_950.0
    ctx2 = ctx_from(O, H, L, C, V)
    ctx2.cond["pct"] = {"up1": z, "up2": z, "dn1": one, "dn2": z}
    vin2 = F.build_version_inputs(ctx2, B3)
    atr = ctx2.atr["atr_at"][t]
    assert vin2.entry_dir[t] == -1 and vin2.w[t] == 5
    assert vin2.sl[t] == pytest.approx(10_040.0 + 0.1 * atr)


def test_sl_a_and_a4_flush_long():
    ctx, t, z = _cond_ctx()
    one = z.copy()
    one[t] = True
    ctx.box_ok[:] = True
    ctx.cond["pct"] = {"up1": z, "up2": z, "dn1": one, "dn2": one}
    vin = F.build_version_inputs(ctx, VS.VERSIONS["A4_box_flush"])
    atr = ctx.atr["atr_at"][t]
    assert vin.entry_dir[t] == 1 and vin.w[t] == 0 and vin.cat[t] == 3
    assert vin.sl[t] == pytest.approx(10_050.0 - 2.5 * atr)


# ----------------------------------------------------------------------------
# 5. 双向同分钟冲突跳过
# ----------------------------------------------------------------------------
def test_ambiguous_minutes_skipped_only_in_bidirectional_versions():
    ctx, t, z = _cond_ctx()
    one = z.copy()
    one[t] = True
    ctx.cond["pct"] = {"up1": one, "up2": z, "dn1": z, "dn2": one}
    assert F.build_version_inputs(ctx, B3).entry_dir[t] == 0
    assert F.build_version_inputs(ctx, B3).skipped["ambiguous"] == 1
    assert F.build_version_inputs(ctx, B1).entry_dir[t] == 1
    ctx.cond["z"] = {"up1": one, "up2": z, "dn1": z, "dn2": one}
    assert F.build_version_inputs(ctx, VS.VERSIONS["B5_pure_bi_mom_z"]).entry_dir[t] == 0
    ctx.cond["z"] = {"up1": z, "up2": z, "dn1": z, "dn2": one}
    assert F.build_version_inputs(ctx, VS.VERSIONS["B5_pure_bi_mom_z"]).entry_dir[t] == -1
    # 单方向版本只检查自身方向：方向不明分钟上 A4（只看向下）与 A1（只看向上）照常做多
    ctx.box_ok[:] = True
    assert F.build_version_inputs(ctx, VS.VERSIONS["A4_box_flush"]).entry_dir[t] == 1
    assert F.build_version_inputs(ctx, VS.VERSIONS["A1_box_opt"]).entry_dir[t] == 1
    assert F.build_version_inputs(ctx, VS.VERSIONS["A4_box_flush"]).cat[t] == 2           # 类别取向下脉冲的指标（仅 ②）


# ----------------------------------------------------------------------------
# 6. 开盘越过止损、盘中止损、同分钟止损优先
# ----------------------------------------------------------------------------
def test_gap_through_stop_at_entry_open_long_and_short():
    O, H, L, C, V = bars()
    O[E0], H[E0], L[E0], C[E0] = 9_880.0, 9_890.0, 9_870.0, 9_885.0
    t = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 100)}).trades.iloc[0]
    q = 500.0 / 117.0
    assert t["reason_name"] == "stop_gap_at_entry_open" and t["exit_bar"] == E0 and t["exit_time_idx"] == E0
    assert t["entry_fill"] == pytest.approx(9_880 * 1.0005) and t["exit_fill"] == pytest.approx(9_880 * 0.9998)
    assert t["fee_entry"] == pytest.approx(0.0005 * q * 9_880 * 1.0005)
    assert t["fee_exit"] == pytest.approx(0.0005 * q * 9_880 * 0.9998)
    assert t["net_pnl"] == pytest.approx(q * (9_880 * 0.9998 - 9_880 * 1.0005) - 0.0005 * q * 9_880 * (1.0005 + 0.9998))
    O2, H2, L2, C2, V2 = bars()
    O2[E0], H2[E0], L2[E0], C2[E0] = 10_120.0, 10_130.0, 10_110.0, 10_125.0
    ts = run(ctx_from(O2, H2, L2, C2, V2), {I0: (-1, P0 + 100)}).trades.iloc[0]
    assert ts["reason_name"] == "stop_gap_at_entry_open"
    assert ts["entry_fill"] == pytest.approx(10_120 * 0.9995) and ts["exit_fill"] == pytest.approx(10_120 * 1.0002)


def test_gap_and_intrabar_stops_during_holding():
    O, H, L, C, V = bars()
    O[E0 + 3], L[E0 + 3] = 9_850.0, 9_840.0
    t = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 100)}).trades.iloc[0]
    assert t["reason_name"] == "stop_gap_open" and t["exit_bar"] == E0 + 3 and t["exit_time_idx"] == E0 + 3
    assert t["exit_fill"] == pytest.approx(9_850 * 0.9998)
    O, H, L, C, V = bars()
    L[E0 + 3] = 9_890.0
    t = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 100)}).trades.iloc[0]
    assert t["reason_name"] == "stop_intrabar" and t["exit_time_idx"] == E0 + 4
    assert t["exit_ref"] == 9_900.0 and t["exit_fill"] == pytest.approx(9_900 * 0.9998)
    O, H, L, C, V = bars()
    H[E0 + 3] = 10_110.0
    t = run(ctx_from(O, H, L, C, V), {I0: (-1, P0 + 100)}).trades.iloc[0]
    assert t["reason_name"] == "stop_intrabar" and t["exit_fill"] == pytest.approx(10_100 * 1.0002)


def test_mae_mfe_stop_minute_counts_only_open_and_stop_price():
    """MAE / MFE 相对入场开盘价：止损分钟只计开盘价与止损价（分钟内止损之后的价格不计）；时间离场计入离场分钟的高低；
    入场开盘跳空离场记 0。"""
    O, H, L, C, V = bars()                                   # 平盘 10,000，高低 ±5
    H[E0 + 1] = 10_030.0                                     # 止损前的有利极值
    O[E0 + 3], H[E0 + 3], L[E0 + 3] = 9_995.0, 10_060.0, 9_700.0   # 盘中跌穿 9,900 止损，最低 9,700
    t = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 100)}).trades.iloc[0]
    assert t["reason_name"] == "stop_intrabar"
    assert t["mae_px"] == pytest.approx(-100.0) and t["mfe_px"] == pytest.approx(30.0)
    O, H, L, C, V = bars()
    H[E0 + 3], L[E0 + 3] = 10_045.0, 9_950.0                 # 未触及止损：持有期极值照常计入
    t = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 100)}).trades.iloc[0]
    assert t["reason_name"] == "time_bar1" and t["mae_px"] == pytest.approx(-50.0) and t["mfe_px"] == pytest.approx(45.0)
    O, H, L, C, V = bars()
    L[E0 + 14] = 9_955.0                                     # 第 1 根末分钟的低点：收盘离场时计入
    t = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 100)}).trades.iloc[0]
    assert t["reason_name"] == "time_bar1" and t["mae_px"] == pytest.approx(-45.0)
    O, H, L, C, V = bars()
    H[E0 + 2], L[E0 + 2] = 10_400.0, 9_990.0                # 空头：止损分钟开盘 10,000，盘中最高 10,400 与最低 9,990 都不计
    O[E0 + 2] = 10_000.0
    ts = run(ctx_from(O, H, L, C, V), {I0: (-1, P0 + 100)}).trades.iloc[0]
    assert ts["reason_name"] == "stop_intrabar" and ts["mae_px"] == pytest.approx(-100.0) and ts["mfe_px"] == pytest.approx(5.0)
    O, H, L, C, V = bars()
    O[E0], H[E0], L[E0], C[E0] = 9_880.0, 9_890.0, 9_870.0, 9_885.0
    tg = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 100)}).trades.iloc[0]
    assert tg["reason_name"] == "stop_gap_at_entry_open" and tg["mae_px"] == 0.0 and tg["mfe_px"] == 0.0


def _continuation_bars(strong=True, side=1):
    O, H, L, C, V = bars()
    V[E0:E0 + 15] = 20.0 if strong else 12.0                   # v1 = 300 / 150 = 2.0（或 180 / 150 = 1.2）
    C[E0 + 14] = 10_004.0 if side == 1 else 9_996.0            # k1 = 0.8
    return O, H, L, C, V


def test_stop_has_priority_over_time_exit_in_last_minute_of_bar1_and_bar2():
    O, H, L, C, V = _continuation_bars()
    L[E0 + 14] = 9_890.0
    t = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 100)}).trades.iloc[0]
    assert t["reason_name"] == "stop_intrabar" and t["exit_bar"] == E0 + 14 and t["exit_ref"] == 9_900.0
    O, H, L, C, V = _continuation_bars()
    L[E0 + 29] = 9_890.0
    t = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 100)}).trades.iloc[0]
    assert t["reason_name"] == "stop_intrabar" and t["exit_bar"] == E0 + 29 and t["cont"] == 1
    O, H, L, C, V = _continuation_bars()
    t = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 100)}).trades.iloc[0]
    assert t["reason_name"] == "time_bar2" and t["exit_bar"] == E0 + 29 and t["exit_ref"] == C[E0 + 29]


# ----------------------------------------------------------------------------
# 7. 数量冻结、下限与杠杆上限、等名义
# ----------------------------------------------------------------------------
def test_quantity_frozen_at_signal_close_not_entry_open():
    O, H, L, C, V = bars()
    O[E0] = 10_050.0                                           # 下一根开盘远离 C_t
    H[E0] = 10_060.0
    t = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 100)}).trades.iloc[0]
    assert t["qty"] == pytest.approx(500.0 / (100.0 + 17.0), rel=1e-13)   # 按 E_ref = C_t = 10,000
    assert t["entry_fill"] == pytest.approx(10_050 * 1.0005)


def test_min_distance_floor_limits_quantity_but_not_the_stop():
    O, H, L, C, V = bars(spread=0.0)
    L[E0 + 2] = P0 - 6.0                                       # 触及 SL = 9,995（距离 5 < 下限 15）
    t = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 5.0)}).trades.iloc[0]
    assert t["qty"] == pytest.approx(500.0 / (15.0 + 17.0), rel=1e-13)
    assert t["reason_name"] == "stop_intrabar" and t["exit_ref"] == P0 - 5.0


def test_leverage_cap_and_equal_notional():
    O, H, L, C, V = bars(spread=0.0)
    t = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 5.0)}, cost=GROSS).trades.iloc[0]
    assert t["qty"] == pytest.approx(20.0) and t["leverage_at_eref"] == pytest.approx(2.0)   # 500 / 15 = 33.3 被 2.0x 截断
    t = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 100.0)}, sizing=NOTIONAL).trades.iloc[0]
    assert t["qty"] == pytest.approx(10.0) and t["leverage_at_eref"] == pytest.approx(1.0)
    q17 = q_risk(P0, P0 - 5.0)
    assert q17 * P0 / VS.INITIAL_CAPITAL < 2.0                 # 主成本下下限 + 摩擦使名义杠杆 ≤ 1.5625x


def test_invalid_stop_side_is_dropped():
    ctx, t, z = _cond_ctx()
    one = z.copy()
    one[t] = True
    ctx.cond["pct"] = {"up1": z, "up2": z, "dn1": one, "dn2": z}
    ctx.C[t] = 9_900.0                                         # 空头止损 H_origin + 0.1·ATR 仍在 E_ref 之上；人为改到下方
    ctx.H[t - 4:t + 1] = 9_800.0
    ctx.C[t - 5] = 9_800.0
    vin = F.build_version_inputs(ctx, B3)
    assert vin.entry_dir[t] == 0 and vin.skipped["invalid_stop"] == 1


# ----------------------------------------------------------------------------
# 8. 维护停机强制离场、漏记分钟打标
# ----------------------------------------------------------------------------
def test_halt_during_holding_forces_exit_at_first_valid_open():
    O, H, L, C, V = bars()
    bad = np.zeros(len(C), bool)
    bad[E0 + 3:E0 + 8] = True
    O[E0 + 8], H[E0 + 8], L[E0 + 8], C[E0 + 8] = 10_020.0, 10_030.0, 10_015.0, 10_025.0
    fund = pd.Series([1e-4], index=[START + pd.Timedelta(minutes=E0 + 5)])     # 停机期间的结算
    ctx = ctx_from(O, H, L, C, V, bad=bad, funding=fund)
    res = run(ctx, {I0: (1, P0 - 100), E0 + 20: (1, P0 - 100), E0 + 22: (1, P0 - 100)})
    t = res.trades.iloc[0]
    q = 500.0 / 117.0
    assert t["reason_name"] == "halt_forced_exit" and t["exit_bar"] == E0 + 8 and t["exit_time_idx"] == E0 + 8
    assert t["exit_fill"] == pytest.approx(10_020 * 0.9998) and t["halt_in_holding"]
    assert t["funding"] == pytest.approx(-q * P0 * 1e-4, rel=1e-12)
    assert res.trades.iloc[1]["trigger_idx"] == E0 + 22                      # 冷却：下一触发 ≥ 离场时刻 + 14
    assert res.trades["trigger_idx"].tolist() == [I0, E0 + 22]


def test_archive_gap_flag_on_holding_window():
    O, H, L, C, V = bars()
    ctx = ctx_from(O, H, L, C, V, gaps=[START + pd.Timedelta(minutes=E0 + 7)])
    res = run(ctx, {I0: (1, P0 - 100), I0 + 200: (1, P0 - 100)})
    assert res.trades["has_archive_gap"].tolist() == [True, False]
    ctx = ctx_from(O, H, L, C, V, gaps=[START + pd.Timedelta(minutes=I0 - 3)])
    t = run(ctx, {I0: (1, P0 - 100)}).trades.iloc[0]
    assert not t["has_archive_gap"] and t["signal_lookback_has_archive_gap"]


# ----------------------------------------------------------------------------
# 9. 冷却、单仓互斥、窗口起点清零、样本末端、前缀截断
# ----------------------------------------------------------------------------
def test_cooldown_and_single_position():
    O, H, L, C, V = bars()
    ctx = ctx_from(O, H, L, C, V)
    res = run(ctx, {I0: (1, P0 - 100), I0 + 5: (-1, P0 + 100), I0 + 20: (1, P0 - 100), I0 + 29: (1, P0 - 100),
                    I0 + 30: (1, P0 - 100)})
    assert res.trades["trigger_idx"].tolist() == [I0, I0 + 30]          # 时间离场 t+16 → 下一触发 ≥ t+30
    O, H, L, C, V = bars()
    L[E0] = 9_890.0                                                     # 入场第 1 分钟盘中止损：离场时刻 t+2
    res = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 100), I0 + 15: (1, P0 - 100), I0 + 16: (1, P0 - 100)})
    assert res.trades["trigger_idx"].tolist() == [I0, I0 + 16]


def test_window_start_reset_and_end_boundary():
    O, H, L, C, V = bars()
    ctx = ctx_from(O, H, L, C, V, start_idx=600, last_idx=1300)
    res = run(ctx, {500: (1, P0 - 100), 600: (1, P0 - 100), 1300: (1, P0 - 100), 1301: (1, P0 - 100)})
    assert res.trades["trigger_idx"].tolist() == [600, 1300]
    assert np.isnan(res.equity.iloc[599]) and res.equity.iloc[600] == VS.INITIAL_CAPITAL
    with pytest.raises(ValueError):
        ctx_bad = ctx_from(O, H, L, C, V, start_idx=600, last_idx=len(C) - 30)
        run(ctx_bad, {600: (1, P0 - 100)})
    # IS 的几何恰好容纳最长持仓：最后触发 23:29 → 第 2 根合成棒最后一分钟 23:59 是网格的最后一分钟
    assert VS.LAST_TRIGGER + pd.Timedelta(minutes=30) == VS.MAIN_END - pd.Timedelta(minutes=1)
    ctx_edge = ctx_from(O, H, L, C, V, start_idx=600, last_idx=len(C) - 31)
    assert run(ctx_edge, {len(C) - 31: (1, P0 - 100)}).trades.iloc[0]["exit_bar"] <= len(C) - 1
    g = SY.grid_from_arrays(START, O, H, L, C, V)
    ctx_w = SY.synthetic_context(g, g.index[600], g.index[1300])
    assert ctx_w.window[600] and ctx_w.window[1300] and not ctx_w.window[1301] and not ctx_w.window[599]


def test_continuation_rules():
    for strong, side, want in ((True, 1, "time_bar2"), (False, 1, "time_bar1"), (True, -1, "time_bar2")):
        O, H, L, C, V = _continuation_bars(strong, side)
        t = run(ctx_from(O, H, L, C, V), {I0: (side, P0 - side * 100)}).trades.iloc[0]
        assert t["reason_name"] == want
        assert t["v1"] == pytest.approx(2.0 if strong else 1.2) and t["k1"] == pytest.approx(0.8)
    O, H, L, C, V = _continuation_bars()
    C[E0 + 14] = 10_000.0                                              # k1 = 0 < 0.10
    assert run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 100)}).trades.iloc[0]["reason_name"] == "time_bar1"
    O, H, L, C, V = _continuation_bars()
    V[:E0] = 0.0                                                       # 基准中位数 = 0 → 不满足延续
    t = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 100)}).trades.iloc[0]
    assert t["reason_name"] == "time_bar1" and t["cont"] == 0 and np.isnan(t["v1"])
    O, H, L, C, V = _continuation_bars()
    H[E0:E0 + 15] = L[E0:E0 + 15] = C[E0:E0 + 15] = O[E0:E0 + 15] = P0  # 第 1 根合成棒 H = L
    t = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 100)}).trades.iloc[0]
    assert t["reason_name"] == "time_bar1" and t["cont"] == 0
    O, H, L, C, V = _continuation_bars()
    assert run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 100)}, spec=B4).trades.iloc[0]["reason_name"] == "time_bar1"


def test_prefix_truncation_engine_synthetic():
    g = SY.random_grid(30_000, start=str(START.tz_localize(None)), seed=11, halts=((12_000, 25),))
    box = (np.arange(len(g)) // 700) % 2 == 0
    fund = pd.Series(np.random.default_rng(5).normal(1e-4, 3e-4, 40),
                     index=pd.date_range(START, periods=40, freq="8h"))
    full = SY.synthetic_context(g, g.index[2000], g.index[len(g) - 40], box_ok=box, funding=fund)
    for name in ("B3_pure_bi_mom", "A2_box_origin", "B4_pure_1bar"):
        spec = VS.VERSIONS[name]
        rf = E.run_portfolio(full, F.build_version_inputs(full, spec), MAIN, RISK)
        for cut in (17_000, 23_456):
            gt = g.iloc[:cut]
            tc = SY.synthetic_context(gt, gt.index[2000], gt.index[cut - 32], box_ok=box[:cut], funding=fund)
            rt = E.run_portfolio(tc, F.build_version_inputs(tc, spec), MAIN, RISK)
            a = rf.trades[rf.trades["trigger_idx"] <= cut - 32].reset_index(drop=True)
            b = rt.trades.reset_index(drop=True)
            assert len(a) == len(b) > 0
            pd.testing.assert_frame_equal(a.drop(columns=["exit_time"]), b.drop(columns=["exit_time"]))
            last_exit = int(b["exit_bar"].max())
            np.testing.assert_array_equal(rf.equity.to_numpy()[:last_exit + 1], rt.equity.to_numpy()[:last_exit + 1])


def test_trade_list_identical_across_costs_and_sizings_and_ledger_reconciles():
    """离场只取决于价格路径与冻结的 SL，因此 4 种成本 × 2 种仓位下成交清单逐笔相同；逐笔 bp 与 R 与仓位口径无关；
    台账净盈亏之和等于期末净值减初始资金（机会—成交—净值对账的单元版本）。"""
    g = SY.random_grid(40_000, start=str(START.tz_localize(None)), seed=33, halts=((20_000, 30),))
    fund = pd.Series(np.random.default_rng(9).normal(1e-4, 3e-4, 90), index=pd.date_range(START, periods=90, freq="8h"))
    ctx = SY.synthetic_context(g, g.index[2000], g.index[len(g) - 40], funding=fund)
    vin = F.build_version_inputs(ctx, B3)
    key = ["trigger_idx", "entry_idx", "exit_bar", "side", "reason", "sl", "e_ref", "exit_ref", "exit_time_idx"]
    ref = None
    for cost in VS.COSTS.values():
        per_sizing = []
        for sizing in VS.SIZINGS.values():
            r = E.run_portfolio(ctx, vin, cost, sizing)
            tr = r.trades
            assert len(tr) > 50
            if ref is None:
                ref = tr[key].copy()
            pd.testing.assert_frame_equal(tr[key], ref)
            assert tr["net_pnl"].sum() == pytest.approx(r.meta["final_equity"] - VS.INITIAL_CAPITAL, rel=1e-9, abs=1e-6)
            per_sizing.append(tr[["gross_bps", "net_bps", "net_R"]].to_numpy())
        np.testing.assert_allclose(per_sizing[0], per_sizing[1], rtol=1e-9, atol=1e-9)
        if not cost.include_funding:
            assert (E.run_portfolio(ctx, vin, cost, RISK).trades["funding"] == 0).all()


def test_funding_at_or_after_grid_end_is_not_read():
    """IS 右开：恰在 IS_END 的结算不在分钟网格上，不读取、不收付（只影响恰在网格末端收盘离场的交易）；
    网格之内而不在分钟开盘时刻上的结算时间戳直接报错。"""
    n = 1400
    i0 = n - 31                                                          # 持满 2 根恰在网格末端收盘离场
    O, H, L, C, V = bars(n=n)
    V[i0 + 1:i0 + 16] = 20.0                                             # v1 = 2.0
    C[i0 + 15] = 10_004.0                                                # k1 = 0.8 → 持有第 2 根
    g = SY.grid_from_arrays(START, O, H, L, C, V)
    end = START + pd.Timedelta(minutes=n)                                # 网格右端（不含）
    ctx = SY.synthetic_context(g, g.index[400], g.index[i0], funding=pd.Series([5e-4], index=[end]))
    vin = SY.manual_inputs(ctx, B1, {i0: (1, P0 - 100)})
    t = E.run_portfolio(ctx, vin, MAIN, RISK).trades.iloc[0]
    assert t["reason_name"] == "time_bar2" and t["exit_bar"] == n - 1 and t["n_funding"] == 0
    with pytest.raises(ValueError):
        SY.synthetic_context(g, g.index[400], g.index[i0], funding=pd.Series([5e-4], index=[START + pd.Timedelta(seconds=90)]))


def test_numba_and_python_paths_agree(monkeypatch):
    g = SY.random_grid(12_000, start=str(START.tz_localize(None)), seed=21, halts=((6_000, 12),))
    ctx = SY.synthetic_context(g, g.index[1500], g.index[len(g) - 40])
    vin = F.build_version_inputs(ctx, B3)
    r1 = E.run_portfolio(ctx, vin, MAIN, RISK)
    monkeypatch.setattr(E, "_trade_path", E._trade_path.py_func)
    monkeypatch.setattr(E, "_run_portfolio", E._run_portfolio.py_func)
    r2 = E.run_portfolio(ctx, vin, MAIN, RISK)
    assert len(r1.trades) == len(r2.trades) > 5
    pd.testing.assert_frame_equal(r1.trades, r2.trades)
    np.testing.assert_array_equal(r1.equity.to_numpy(), r2.equity.to_numpy())


def test_event_simulation_matches_portfolio_per_trade_returns():
    g = SY.random_grid(20_000, start=str(START.tz_localize(None)), seed=4)
    ctx = SY.synthetic_context(g, g.index[1500], g.index[len(g) - 40])
    vin = F.build_version_inputs(ctx, B3)
    res = E.run_portfolio(ctx, vin, MAIN, RISK)
    tr = res.trades
    ev = E.simulate_events(ctx, tr["trigger_idx"].to_numpy(), tr["side"].to_numpy(), tr["sl"].to_numpy(),
                           tr["e_ref"].to_numpy(), False, MAIN)
    np.testing.assert_allclose(ev["net_bps"], tr["net_bps"], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(ev["gross_bps"], tr["gross_bps"], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(ev["net_R"], tr["net_R"], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(ev["mae_bps"], tr["mae_px"] / tr["e_ref"] * 1e4, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(ev["mfe_bps"], tr["mfe_px"] / tr["e_ref"] * 1e4, rtol=1e-9, atol=1e-9)
    assert (ev["reason"].to_numpy().astype(int) == tr["reason"].to_numpy()).all()
    assert (ev["cont"].to_numpy().astype(int) == tr["cont"].to_numpy()).all()
    assert (ev["exit_bar"].to_numpy().astype(int) == tr["exit_bar"].to_numpy()).all()


# ----------------------------------------------------------------------------
# 10. 独立复核后补充：数量口径、基准中位数、起点极值、S6 反弹线、镜像止损、资格规则、等号边界、资金费与停机约定、
#     输入校验与台账结构（每个测试都对应复核中存活的违规变异或复核发现的缺陷）
# ----------------------------------------------------------------------------
def test_caps_and_notional_use_eref_not_entry_open_and_are_flagged():
    O, H, L, C, V = bars(spread=0.0)
    O[E0] = H[E0] = 10_050.0                                   # O_{t+1} ≠ E_ref = C_t = 10,000
    t = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 5.0)}, cost=GROSS).trades.iloc[0]
    assert t["qty"] == pytest.approx(20.0, rel=1e-13)         # 2.0·Equity / E_ref（按 O_{t+1} 会是 19.90）
    assert t["cap_bound"] and t["floor_bound"]
    t = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 5.0)}, sizing=NOTIONAL).trades.iloc[0]
    assert t["qty"] == pytest.approx(10.0, rel=1e-13) and not t["cap_bound"] and not t["floor_bound"]
    t = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 5.0)}).trades.iloc[0]
    assert t["qty"] == pytest.approx(500.0 / 32.0, rel=1e-13) and t["floor_bound"] and not t["cap_bound"]
    t = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 100.0)}).trades.iloc[0]
    assert not t["floor_bound"] and not t["cap_bound"]


def test_second_trade_sized_from_current_equity():
    O, H, L, C, V = bars()
    L[E0 + 3] = 9_890.0                                        # 第一笔盘中止损，亏损
    for sizing in (RISK, NOTIONAL):
        res = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 100), I0 + 200: (1, P0 - 100)}, sizing=sizing)
        t1, t2 = res.trades.iloc[0], res.trades.iloc[1]
        eq = VS.INITIAL_CAPITAL + t1["net_pnl"]
        assert t1["reason_name"] == "stop_intrabar" and eq < VS.INITIAL_CAPITAL
        assert t2["equity_at_signal"] == pytest.approx(eq, rel=1e-13)
        assert res.equity.iloc[I0 + 200] == pytest.approx(eq, rel=1e-13)
        want = 0.005 * eq / (100.0 + 0.0017 * P0) if sizing is RISK else eq / P0
        assert t2["qty"] == pytest.approx(want, rel=1e-13)


def _baseline_fixture(t=1000, n=1500, spikes=True):
    """20 个窗口和：k = 0（含 t）为 150，k = 19（含 t−299）为 165，k = 1..9 为 15，k = 10..18 为 3,000。
    中位数 = (150 + 165) / 2 = 157.5，均值 = 1,372.5；t+1 与 t−300 放极端成交量（都不在 [T−299, T+1) 内）。"""
    V = np.full(n, 10.0)
    sums = {0: 150.0, 19: 165.0}
    sums.update({k: 15.0 for k in range(1, 10)})
    sums.update({k: 3000.0 for k in range(10, 19)})
    for k, s in sums.items():
        V[t + 1 - 15 * (k + 1): t + 1 - 15 * k] = s / 15.0
    if spikes:
        V[t + 1] = 1e6
        V[t - 300] = 1e6
    return V


def test_baseline_window_count_and_median():
    t, n = 1000, 1500
    V = _baseline_fixture(t, n)
    nb = np.zeros(n, bool)
    assert F.baseline_median(V, nb)[t] == pytest.approx(157.5, rel=1e-13)
    for idx, moved in ((t + 1, False), (t - 300, False), (t, True), (t - 299, True)):
        V2 = V.copy()
        V2[idx] += 5.0
        assert (F.baseline_median(V2, nb)[t] != pytest.approx(157.5, rel=1e-13)) == moved, idx
    # 引擎层面：第 1 根成交量 210 → v1 = 210 / 157.5 = 1.33（按均值会是 0.15，按 19 个窗口会是另一数值）
    O, H, L, C, _ = bars()
    V = _baseline_fixture(I0, 1500, spikes=False)
    V[E0:E0 + 15] = 14.0
    C[E0 + 14] = 10_004.0
    tr = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 100)}).trades.iloc[0]
    assert tr["v1"] == pytest.approx(210.0 / 157.5, rel=1e-13) and tr["reason_name"] == "time_bar2"


def test_origin_close_term_binds():
    """L_origin = min(C_{t−W}, min L_{t−W+1..t})：跳空上行时 C_{t−W} 是更低的一项（空头镜像同理）。"""
    t = 1000
    z = np.zeros(1500, bool)
    one = z.copy()
    one[t] = True
    O, H, L, C, V = bars()
    C[t - 5] = 9_990.0
    L[t - 4:t + 1] = [9_995.0, 9_996.0, 9_997.0, 9_998.0, 10_000.0]
    C[t], H[t] = 10_050.0, 10_060.0
    ctx = ctx_from(O, H, L, C, V)
    ctx.cond["pct"] = {"up1": one, "up2": z, "dn1": z, "dn2": z}
    atr = ctx.atr["atr_at"][t]
    assert F.build_version_inputs(ctx, VS.VERSIONS["B2_pure_long_sl"]).sl[t] == pytest.approx(9_990.0 - 0.1 * atr)
    ctx.box_ok[:] = True
    assert F.build_version_inputs(ctx, VS.VERSIONS["A3_box_half"]).sl[t] == pytest.approx(10_050.0 - 0.5 * 60.0)
    O, H, L, C, V = bars()
    C[t - 5] = 10_030.0
    H[t - 4:t + 1] = [10_020.0, 10_010.0, 10_008.0, 10_006.0, 10_002.0]
    C[t], L[t] = 9_960.0, 9_950.0
    ctx2 = ctx_from(O, H, L, C, V)
    ctx2.cond["pct"] = {"up1": z, "up2": z, "dn1": one, "dn2": z}
    assert F.build_version_inputs(ctx2, B3).sl[t] == pytest.approx(10_030.0 + 0.1 * ctx2.atr["atr_at"][t])


def test_s6_bounce_level_and_strict_poc():
    """S6 的 j2 必须同时满足 High ≥ VAL + 1.3·ATR 与 High > POC（严格大于）；分别在 POC 低于与高于反弹线时检验。"""
    nd = 40
    lo, hi = np.full(nd, 112.0), np.full(nd, 114.0)
    lo[nd - 10], hi[nd - 10] = 100.5, 101.5                     # 触及 [99, 101]
    lo[nd - 9:], hi[nd - 9:] = 100.8, 101.6
    atr = pd.Series(2.0, index=_daily(lo, hi).index)
    as_of = _daily(lo, hi).index[-1] + pd.Timedelta(days=1)

    def primed(hi_j, poc):
        h = hi.copy()
        h[nd - 4] = hi_j
        prof = pd.DataFrame({"POC": poc, "VAL": 100.0, "VAH": 120.0, "P_low": 90.0, "P_high": 130.0, "n_bars": 1},
                            index=pd.DatetimeIndex([as_of]))
        return bool(DY.box_state(_daily(lo, h), prof, atr, exit_above_poc=True)["primed"].iloc[0])

    assert not primed(102.0, 101.8)          # 高于 POC 但低于反弹线 102.6
    assert primed(102.7, 101.8)
    assert not primed(110.0, 110.0)          # 恰等于 POC：不满足严格大于
    assert primed(110.5, 110.0)


def test_short_gap_stop_during_holding():
    O, H, L, C, V = bars()
    O[E0 + 3], H[E0 + 3] = 10_150.0, 10_160.0
    t = run(ctx_from(O, H, L, C, V), {I0: (-1, P0 + 100)}).trades.iloc[0]
    assert t["reason_name"] == "stop_gap_open" and t["exit_time_idx"] == E0 + 3 and t["exit_bar"] == E0 + 3
    assert t["exit_fill"] == pytest.approx(10_150.0 * 1.0002)


def test_mirror_symmetry_with_gap_and_intrabar_stops():
    """镜像路径上的多空对称，且覆盖持仓中跳空止损与盘中止损（止损距离 0.30%–0.45%，零成本下 0.15% 下限与 2.0x 上限都不触发）。"""
    g = SY.random_grid(3000, start=str(START.tz_localize(None)), seed=7, impulse_rate=0.0)
    O, H, L, C, V = (g[c].to_numpy().copy() for c in ("open", "high", "low", "close", "volume"))
    rng = np.random.default_rng(5)
    sig_a, sig_b = {}, {}
    K = 2.0 * float(C.mean())
    for k, i in enumerate(range(500, 2800, 97)):
        d = float(rng.uniform(0.0030, 0.0045)) * C[i]
        sl = C[i] - d
        e = i + 1
        if k % 3 == 0:                                         # 持仓第 3 分钟开盘跳空越过止损
            O[e + 2] = sl - 0.5 * d
            H[e + 2], L[e + 2] = max(H[e + 2], O[e + 2]), min(L[e + 2], O[e + 2])
        elif k % 3 == 1:                                       # 持仓第 5 分钟盘中刺穿止损
            L[e + 4] = min(L[e + 4], sl - 0.5 * d)
        sig_a[i] = (1, sl)
        sig_b[i] = (-1, K - sl)
    ra = run(ctx_from(O, H, L, C, V, start_idx=400, last_idx=2900), sig_a, cost=GROSS)
    rb = run(ctx_from(K - O, K - L, K - H, K - C, V, start_idx=400, last_idx=2900), sig_b, cost=GROSS)
    ta, tb = ra.trades, rb.trades
    assert len(ta) == len(tb) > 10
    assert {"stop_gap_open", "stop_intrabar"} <= set(ta["reason_name"])
    assert (ta["reason"].to_numpy() == tb["reason"].to_numpy()).all()
    assert (ta["exit_bar"].to_numpy() == tb["exit_bar"].to_numpy()).all()
    assert not ta["cap_bound"].any() and not ta["floor_bound"].any() and not tb["cap_bound"].any()
    np.testing.assert_allclose(ta["qty"], tb["qty"], rtol=1e-12)
    np.testing.assert_allclose(ta["net_pnl"], tb["net_pnl"], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(ra.equity.dropna(), rb.equity.dropna(), rtol=1e-12)


@pytest.mark.parametrize("bad_at,counter", [(-7, "contaminated"), (-15, "contaminated"), (1, "entry_bar_bad"),
                                            (-16, None)])
def test_contamination_and_bad_entry_bar_void_the_signal(bad_at, counter):
    t = 1000
    z = np.zeros(1500, bool)
    one = z.copy()
    one[t] = True
    O, H, L, C, V = bars()
    bad = np.zeros(1500, bool)
    bad[t + bad_at] = True
    ctx = ctx_from(O, H, L, C, V, bad=bad)
    ctx.cond["pct"] = {"up1": one, "up2": z, "dn1": z, "dn2": z}
    vin = F.build_version_inputs(ctx, B1)
    if counter is None:                                        # T−16 不在回看窗口 [T−15, T] 内
        assert vin.entry_dir[t] == 1
    else:
        assert vin.entry_dir[t] == 0 and vin.skipped[counter] == 1 and vin.skipped_exclusive[counter] == 1
        assert (vin.entry_dir[t + 1:t + 3] == 0).all()        # 作废，不顺延


def test_baseline_undefined_excludes_and_is_counted():
    O, H, L, C, V = bars(n=800)
    ctx = ctx_from(O, H, L, C, V, start_idx=200, last_idx=700)
    up = np.zeros(800, bool)
    up[[250, 500]] = True
    z = np.zeros(800, bool)
    ctx.cond["pct"] = {"up1": up, "up2": z, "dn1": z, "dn2": z}
    vin = F.build_version_inputs(ctx, B1)
    assert np.isfinite(ctx.atr["atr_at"][250]) and np.isnan(ctx.base_med[250])     # ATR 已就绪、基准尚无定义
    assert vin.entry_dir[250] == 0 and vin.entry_dir[500] == 1
    assert vin.skipped["baseline_undefined"] == 1 and vin.skipped_exclusive["baseline_undefined"] == 1


def test_exclusive_exclusion_counts_reconcile_with_signals():
    # 同一分钟既方向不明又维护污染：逐项计数各记一次，互斥计数只记排在前面的“方向不明”
    t = 1000
    z = np.zeros(1500, bool)
    one = z.copy()
    one[t] = True
    O, H, L, C, V = bars()
    bad = np.zeros(1500, bool)
    bad[t - 7] = True
    ctx = ctx_from(O, H, L, C, V, bad=bad)
    ctx.cond["pct"] = {"up1": one, "up2": z, "dn1": z, "dn2": one}
    vin = F.build_version_inputs(ctx, B3)
    assert vin.skipped["ambiguous"] == vin.skipped["contaminated"] == 1
    assert vin.skipped_exclusive["ambiguous"] == 1 and vin.skipped_exclusive["contaminated"] == 0
    # 随机网格上的 10 个版本：窗口内自身方向触发分钟 = 入场信号 + 互斥排除之和
    g = SY.random_grid(30_000, start=str(START.tz_localize(None)), seed=2, impulse_rate=3e-3,
                       halts=((5_000, 20), (17_000, 59)))
    box = (np.arange(len(g)) // 500) % 3 == 0
    ctx = SY.synthetic_context(g, g.index[250], g.index[len(g) - 40], box_ok=box)
    for spec in VS.VERSIONS.values():
        c = ctx.cond[spec.trigger_rule]
        up, dn = c["up1"] | c["up2"], c["dn1"] | c["dn2"]
        trig = (up | dn) if spec.direction == "bi" else (up if spec.direction == "up_long" else dn)
        vin = F.build_version_inputs(ctx, spec)
        assert int((ctx.window & trig).sum()) == int((vin.entry_dir != 0).sum()) + sum(vin.skipped_exclusive.values())
        assert all(vin.skipped_exclusive[k] <= vin.skipped[k] for k in vin.skipped)


def test_equality_boundaries_of_stops_continuation_and_flags():
    # 入场开盘恰等于 SL → 入场开盘跳空离场
    O, H, L, C, V = bars()
    O[E0] = L[E0] = P0 - 100
    assert run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 100)}).trades.iloc[0]["reason_name"] == "stop_gap_at_entry_open"
    # 持仓中开盘恰等于 SL → 跳空止损
    O, H, L, C, V = bars()
    O[E0 + 3] = L[E0 + 3] = P0 - 100
    t = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 100)}).trades.iloc[0]
    assert t["reason_name"] == "stop_gap_open" and t["exit_bar"] == E0 + 3
    # 盘中最低恰等于 SL（多头）/ 最高恰等于 SL（空头）→ 盘中止损
    O, H, L, C, V = bars()
    L[E0 + 3] = P0 - 100
    assert run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 100)}).trades.iloc[0]["reason_name"] == "stop_intrabar"
    O, H, L, C, V = bars()
    H[E0 + 3] = P0 + 100
    assert run(ctx_from(O, H, L, C, V), {I0: (-1, P0 + 100)}).trades.iloc[0]["reason_name"] == "stop_intrabar"
    # v1 恰为 1.3、k1 恰为 0.10 → 继续持有第 2 根
    O, H, L, C, V = bars()
    V[E0:E0 + 15] = 13.0                                       # v1 = 195 / 150
    C[E0 + 14] = 10_000.5                                      # k1 = (2·10,000.5 − 10,005 − 9,995) / 10
    t = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 100)}).trades.iloc[0]
    assert t["v1"] == VS.V1_MIN and t["k1"] == VS.K1_MIN and t["cont"] == 1 and t["reason_name"] == "time_bar2"
    # 漏记分钟恰为离场分钟 → 打标；在离场分钟之后 → 不打标
    O, H, L, C, V = bars()
    t = run(ctx_from(O, H, L, C, V, gaps=[START + pd.Timedelta(minutes=E0 + 14)]), {I0: (1, P0 - 100)}).trades.iloc[0]
    assert t["reason_name"] == "time_bar1" and t["has_archive_gap"]
    t = run(ctx_from(O, H, L, C, V, gaps=[START + pd.Timedelta(minutes=E0 + 15)]), {I0: (1, P0 - 100)}).trades.iloc[0]
    assert not t["has_archive_gap"]


def test_atr15_partial_bucket_close_is_last_valid_close():
    """部分有效的桶：收盘价取最后一根有效分钟的收盘，桶末尾的维护占位价格（即使离谱）不进入下一根桶的 TR。"""
    trs = [2.0] * 20
    O, H, L, C, V, bad = _bucket_bars(trs, partial=(16, [12, 13, 14]))
    C[16 * 15 + 11] = H[16 * 15 + 11] = 103.0
    for a in (O, H, L, C):
        a[16 * 15 + 12:16 * 15 + 15] = 777.0
    a = F.atr15_natural(O, H, L, C, bad, int(START.value))
    assert a["n_valid"][16] == 12 and a["close"][16] == 103.0
    assert a["tr"][17] == pytest.approx(4.0)                    # max(101 − 99, |101 − 103|, |99 − 103|)


def test_cont_records_bar1_test_outcome_and_v1_k1_independently():
    O, H, L, C, V = _continuation_bars()
    t = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 100)}, spec=B4).trades.iloc[0]
    assert t["reason_name"] == "time_bar1" and t["cont"] == 1                # B4 照常检验，只作诊断
    assert t["v1"] == pytest.approx(2.0) and t["k1"] == pytest.approx(0.8)
    O, H, L, C, V = _continuation_bars(strong=False)
    assert run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 100)}, spec=B4).trades.iloc[0]["cont"] == 0
    O, H, L, C, V = _continuation_bars()
    V[:E0] = 0.0                                                             # B_t = 0：v1 无定义，k1 照常记录
    t = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 100)}).trades.iloc[0]
    assert t["cont"] == 0 and np.isnan(t["v1"]) and t["k1"] == pytest.approx(0.8)
    O, H, L, C, V = _continuation_bars()
    H[E0:E0 + 15] = L[E0:E0 + 15] = C[E0:E0 + 15] = O[E0:E0 + 15] = P0     # H = L：k1 无定义，v1 照常记录
    t = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 100)}).trades.iloc[0]
    assert t["cont"] == 0 and np.isnan(t["k1"]) and t["v1"] == pytest.approx(2.0)
    O, H, L, C, V = _continuation_bars()
    L[E0 + 14] = 9_890.0                                                     # t+15 分钟内止损：未检验
    t = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 100)}).trades.iloc[0]
    assert t["reason_name"] == "stop_intrabar" and t["cont"] == -1 and np.isnan(t["v1"]) and np.isnan(t["k1"])


def test_funding_any_number_of_settlements_per_trade():
    """结算次数不设上限（每分钟一次结算的合成情形）：持满 2 根收付 30 次；停机路径逐次收付；下一笔的净值与数量随之正确。"""
    rates = 1e-5 * (1 + np.arange(40))
    fund = pd.Series(rates, index=START + pd.to_timedelta(np.arange(E0, E0 + 40), unit="min"))   # τ_k = 入场时刻 + k 分钟
    q = 500.0 / 117.0
    O, H, L, C, V = _continuation_bars()                       # 持满 2 根：time_bar2，离场收盘时刻 = τ_30
    ctx = ctx_from(O, H, L, C, V, funding=fund)
    res = run(ctx, {I0: (1, P0 - 100), I0 + 200: (1, P0 - 100)})
    t1, t2 = res.trades.iloc[0], res.trades.iloc[1]
    unit = sum(-O[E0 + k] * rates[k] for k in range(1, 30)) - C[E0 + 29] * rates[30]
    assert t1["reason_name"] == "time_bar2" and t1["n_funding"] == 30
    assert t1["funding"] == pytest.approx(q * unit, rel=1e-12)
    assert t2["equity_at_signal"] == pytest.approx(VS.INITIAL_CAPITAL + t1["net_pnl"], rel=1e-13)
    assert t2["qty"] == pytest.approx(0.005 * t2["equity_at_signal"] / 117.0, rel=1e-13)
    ev = E.simulate_events(ctx, np.array([I0]), np.array([1.0]), np.array([P0 - 100]), np.array([P0]), False, MAIN)
    assert ev["net_bps"].iloc[0] == pytest.approx(t1["net_bps"], rel=1e-12)
    # 停机路径：t+5 起停机 26 分钟（越过 t+30），停机占位分钟按其开盘价（停机前最后收盘价）收付，恢复开盘按该开盘价
    O, H, L, C, V = bars()
    bad = np.zeros(len(C), bool)
    bad[E0 + 5:E0 + 31] = True
    C[E0 + 4] = 10_003.0
    for a in (O, H, L, C):
        a[E0 + 5:E0 + 31] = 10_003.0
    O[E0 + 31], H[E0 + 31], L[E0 + 31], C[E0 + 31] = 10_020.0, 10_030.0, 10_015.0, 10_025.0
    t = run(ctx_from(O, H, L, C, V, bad=bad, funding=fund), {I0: (1, P0 - 100)}).trades.iloc[0]
    assert t["reason_name"] == "halt_forced_exit" and t["exit_bar"] == E0 + 31 and t["n_funding"] == 31
    assert t["funding"] == pytest.approx(q * sum(-O[E0 + k] * rates[k] for k in range(1, 32)), rel=1e-12)
    assert t["minutes_held"] == 31                             # 停机使持仓超过 30 分钟


def test_settlement_enters_equity_of_the_minute_it_opens():
    O, H, L, C, V = bars()
    r = 2e-4
    res = run(ctx_from(O, H, L, C, V, funding=pd.Series([r], index=[START + pd.Timedelta(minutes=E0 + 5)])),
              {I0: (1, P0 - 100)})
    q = 500.0 / 117.0
    fill = P0 * 1.0005
    cash0 = VS.INITIAL_CAPITAL - 0.0005 * q * fill
    assert res.equity.iloc[E0 + 4] == pytest.approx(cash0 + q * (P0 - fill), rel=1e-13)
    assert res.equity.iloc[E0 + 5] == pytest.approx(cash0 - q * P0 * r + q * (P0 - fill), rel=1e-13)


def test_halt_and_time_exit_conventions():
    q = 500.0 / 117.0
    # 停机恢复开盘已越过 SL：仍记 halt_forced_exit；停机占位分钟（即使价格离谱）不计入 MAE / MFE
    O, H, L, C, V = bars()
    bad = np.zeros(len(C), bool)
    bad[E0 + 3:E0 + 8] = True
    for a in (O, H, L, C):
        a[E0 + 3:E0 + 8] = 10_050.0
    O[E0 + 8], H[E0 + 8], L[E0 + 8], C[E0 + 8] = 9_880.0, 9_890.0, 9_870.0, 9_885.0
    t = run(ctx_from(O, H, L, C, V, bad=bad), {I0: (1, P0 - 100)}).trades.iloc[0]
    assert t["reason_name"] == "halt_forced_exit" and t["exit_ref"] == 9_880.0
    assert t["mfe_px"] == pytest.approx(5.0) and t["mae_px"] == pytest.approx(-120.0)
    # 时间离场的收盘时刻恰为 τ、下一分钟是价格不同的占位分钟：按离场分钟收盘价收付
    O, H, L, C, V = bars()
    C[E0 + 14] = 10_002.0
    bad = np.zeros(len(C), bool)
    bad[E0 + 15:E0 + 20] = True
    for a in (O, H, L, C):
        a[E0 + 15:E0 + 20] = 10_050.0
    fund = pd.Series([2e-4], index=[START + pd.Timedelta(minutes=E0 + 15)])
    t = run(ctx_from(O, H, L, C, V, bad=bad, funding=fund), {I0: (1, P0 - 100)}).trades.iloc[0]
    assert t["reason_name"] == "time_bar1" and t["funding"] == pytest.approx(-q * 10_002.0 * 2e-4, rel=1e-12)


def test_bars_held_and_minutes_held():
    cases = []
    O, H, L, C, V = bars()
    O[E0], L[E0] = 9_880.0, 9_870.0
    cases.append((O, H, L, C, V, None, "stop_gap_at_entry_open", 1, 0))
    O, H, L, C, V = bars()
    O[E0 + 3], L[E0 + 3] = 9_850.0, 9_840.0
    cases.append((O, H, L, C, V, None, "stop_gap_open", 4, 3))
    O, H, L, C, V = bars()
    L[E0 + 3] = 9_890.0
    cases.append((O, H, L, C, V, None, "stop_intrabar", 4, 4))
    O, H, L, C, V = bars()
    cases.append((O, H, L, C, V, None, "time_bar1", 15, 15))
    O, H, L, C, V = bars()
    bad = np.zeros(len(C), bool)
    bad[E0 + 3:E0 + 8] = True
    cases.append((O, H, L, C, V, bad, "halt_forced_exit", 9, 8))
    for O, H, L, C, V, bad, reason, bars_held, minutes_held in cases:
        t = run(ctx_from(O, H, L, C, V, bad=bad), {I0: (1, P0 - 100)}).trades.iloc[0]
        assert (t["reason_name"], t["bars_held"], t["minutes_held"]) == (reason, bars_held, minutes_held)


def test_zero_trade_run_returns_full_ledger_schema():
    O, H, L, C, V = bars()
    ctx = ctx_from(O, H, L, C, V)
    full = run(ctx, {I0: (1, P0 - 100)}).trades
    empty = run(ctx, {}).trades
    assert len(empty) == 0 and list(empty.columns) == list(full.columns)
    assert (empty.dtypes == full.dtypes).all()
    assert np.isnan(empty["net_bps"].mean())


def test_run_portfolio_and_event_simulation_reject_invalid_inputs():
    O, H, L, C, V = bars()
    bad = np.zeros(len(C), bool)
    bad[1200:1203] = True
    ctx = ctx_from(O, H, L, C, V, bad=bad)
    n = len(C)

    def sim(i, side=1.0, sl=P0 - 100, eref=P0):
        return E.simulate_events(ctx, np.array([i]), np.array([side]), np.array([sl]), np.array([eref]), False, MAIN)

    assert len(sim(I0)) == 1 and len(E.simulate_events(ctx, np.array([], np.int64), [], [], [], False, MAIN)) == 0
    for call in (lambda: sim(n - 1), lambda: sim(n - 35), lambda: sim(300),        # 网格末端 / 窗口之外
                 lambda: sim(1199), lambda: sim(1205),                                # 入场 K 线为 bad / 维护污染
                 lambda: sim(I0, sl=P0 + 100), lambda: sim(I0, sl=np.nan), lambda: sim(I0, side=0.0),
                 lambda: E.simulate_events(ctx, np.array([1000.0]), [1.0], [P0 - 100], [P0], False, MAIN),
                 lambda: E.simulate_events(ctx, np.array([I0, I0 + 50]), [1.0], [P0 - 100], [P0], False, MAIN)):
        with pytest.raises(ValueError):
            call()
    with pytest.raises(ValueError):                                                   # 入场 K 线为 bad 的手工信号
        run(ctx, {1199: (1, P0 - 100)})
    with pytest.raises(ValueError):                                                   # 止损在盈利一侧
        run(ctx, {I0: (1, P0 + 100)})
    with pytest.raises(ValueError):
        run(ctx, {I0: (2, P0 - 100)})


def test_trade_reaching_end_of_data_raises():
    """计划没有定义“数据末端”离场：停机延续到网格末端时引擎报错（IS 内最后一次停机在 2023-09-12，不会发生）。"""
    n = 1400
    O, H, L, C, V = bars(n=n)
    bad = np.zeros(n, bool)
    bad[n - 20:] = True
    g = SY.grid_from_arrays(START, O, H, L, C, V, bad)
    ctx = SY.synthetic_context(g, g.index[400], g.index[n - 31])
    with pytest.raises(ValueError):
        E.run_portfolio(ctx, SY.manual_inputs(ctx, B1, {n - 31: (1, P0 - 100)}), MAIN, RISK)
    with pytest.raises(ValueError):
        E.simulate_events(ctx, np.array([n - 31]), [1.0], [P0 - 100], [P0], False, MAIN)


CP2_ONLY =pytest.mark.skipif(E.REAL_DATA_UNLOCKED, reason="Checkpoint 2 lock test; the approved CP3 unlock commit "
                                                              "makes it inapplicable (see experiment plan section 10)")


@CP2_ONLY
def test_engine_refuses_real_data_in_checkpoint2():
    O, H, L, C, V = bars()
    ctx0 = ctx_from(O, H, L, C, V)
    with pytest.raises(dataclasses.FrozenInstanceError):         # 上下文标签不能被就地改写
        ctx0.source = "synthetic"
    ctx = dataclasses.replace(ctx0, source="binance_is")
    with pytest.raises(E.EngineLockedError):
        run(ctx, {I0: (1, P0 - 100)})
    with pytest.raises(E.EngineLockedError):
        E.simulate_events(ctx, np.array([I0]), np.array([1.0]), np.array([P0 - 100]), np.array([P0]), False, MAIN)
    assert E.REAL_DATA_UNLOCKED is False


def test_synthetic_label_requires_synthetic_grid():
    """source = 'synthetic' 只接受 synthetic.grid_from_arrays 构造（带合成数据标记）的网格；切片后标记保留。"""
    O, H, L, C, V = bars()
    g = SY.grid_from_arrays(START, O, H, L, C, V)
    assert g.attrs["provenance"] == "synthetic" and g.iloc[:900].attrs["provenance"] == "synthetic"
    untagged = g.copy()
    untagged.attrs.clear()
    with pytest.raises(ValueError):
        SY.synthetic_context(untagged, g.index[400], g.index[1400])
    with pytest.raises(ValueError):
        F.make_context(untagged, np.zeros(len(g), bool), "synthetic")


def test_unlocked_real_context_requires_funding_and_frozen_gap_minutes():
    """解锁之后的输入完整性检查（不翻转锁：直接调用检查函数；数据是合成的）。网格覆盖表 5b 的 2021-01-12 06:10。"""
    n = 1500
    O, H, L, C, V = bars(n=n)
    g = SY.grid_from_arrays(pd.Timestamp("2021-01-12 00:00", tz="UTC"), O, H, L, C, V)
    fund = pd.Series([1e-4, 1e-4], index=pd.DatetimeIndex(["2021-01-12 08:00", "2021-01-12 16:00"], tz="UTC"))
    box = np.zeros(n, bool)
    no_fund = F.make_context(g, box, "binance_is", g.index[400], g.index[n - 40], VS.ARCHIVE_GAP_MINUTES)
    with pytest.raises(ValueError):
        E._require_real_inputs(no_fund)
    wrong_gaps = F.make_context(g, box, "binance_is", g.index[400], g.index[n - 40], (), fund)
    with pytest.raises(ValueError):
        E._require_real_inputs(wrong_gaps)
    ok = F.make_context(g, box, "binance_is", g.index[400], g.index[n - 40], VS.ARCHIVE_GAP_MINUTES, fund)
    E._require_real_inputs(ok)
    assert ok.funding_supplied and ok.n_funding_mapped == 2 and int(ok.archive_gap.sum()) == 1
    assert ok.index[ok.archive_gap][0] == pd.Timestamp("2021-01-12 06:10", tz="UTC")


def test_context_rejects_bad_funding_and_off_grid_gap_minutes():
    O, H, L, C, V = bars()
    t1, t2 = START + pd.Timedelta(minutes=500), START + pd.Timedelta(minutes=980)
    with pytest.raises(ValueError):                                       # 重复的结算时间戳
        ctx_from(O, H, L, C, V, funding=pd.Series([1e-4, 3e-4], index=pd.DatetimeIndex([t1, t1])))
    with pytest.raises(ValueError):                                       # 非有限费率
        ctx_from(O, H, L, C, V, funding=pd.Series([1e-4, np.nan], index=pd.DatetimeIndex([t1, t2])))
    with pytest.raises(ValueError):                                       # 网格范围内、但不在分钟开盘时刻上的漏记分钟
        ctx_from(O, H, L, C, V, gaps=[START + pd.Timedelta(seconds=90)])
    ctx = ctx_from(O, H, L, C, V, funding=pd.Series([1e-4, 3e-4], index=pd.DatetimeIndex([t1, t2])))
    assert ctx.funding_supplied and ctx.n_funding_mapped == 2


CP1_RESULTS = Path(__file__).resolve().parent / "reports" / "impulse_vs_box_round" / "cp1" / "cp1_results.json"
FROZEN_KEY = "fc0d37b9d198ff85"                     # 与实验计划第 13 节一致；任何版本、成本、仓位或常数的改动都会改变它
FROZEN_VERSION_KEYS = {"A1_box_opt": "95c0b1a6d42b785e", "A2_box_origin": "011121615297f5a8",
                       "A3_box_half": "d5268a55772d6fea", "A4_box_flush": "716a23042e451fbd",
                       "A5_box_origin_z": "db69c6f1e7545906", "B1_pure_long": "c0db58c7ec8ea580",
                       "B2_pure_long_sl": "a9952888c80053d9", "B3_pure_bi_mom": "afb2d6775f44d7b6",
                       "B4_pure_1bar": "68866f7f8bfbf3fa", "B5_pure_bi_mom_z": "88eeb9c392542a21"}


def test_frozen_identifiers_and_constants():
    assert VS.frozen_key() == FROZEN_KEY
    assert VS.version_keys() == FROZEN_VERSION_KEYS
    assert VS.MAIN_START == pd.Timestamp("2020-03-01", tz="UTC")
    assert VS.LAST_TRIGGER == pd.Timestamp("2023-12-31 23:29", tz="UTC")
    assert (VS.BASELINE_WINDOWS, VS.BAR_MIN, VS.ATR15_LEN, VS.COOLDOWN_MIN) == (20, 15, 14, 15)
    assert (VS.V1_MIN, VS.K1_MIN, VS.SL_A_MULT, VS.SL_B_BUFFER, VS.SL_C_FRAC) == (1.3, 0.10, 2.5, 0.1, 0.5)
    assert (VS.RISK_FRACTION, VS.MIN_DIST_FRAC, VS.LEVERAGE_CAP, VS.EQUAL_NOTIONAL) == (0.005, 0.0015, 2.0, 1.0)
    assert VS.BOX_STATE_VARIANT == "S6_reentry_above_poc" and VS.SEED_BASE == 20261106
    if CP1_RESULTS.exists():                                       # 冻结的漏记分钟 = Checkpoint 1 表 5b
        r1 = json.loads(CP1_RESULTS.read_text())
        t5b = sorted(pd.Timestamp(m["minute"]) for m in r1["qa_volume_discrepancies_localized"]["minutes"]
                     if m["class"] == "tail_omission")
        assert t5b == sorted(pd.Timestamp(m, tz="UTC") for m in VS.ARCHIVE_GAP_MINUTES) and len(t5b) == 10


def test_frozen_versions_inheritance():
    v = VS.VERSIONS
    assert len(v) == 10
    a2, a5 = v["A2_box_origin"], v["A5_box_origin_z"]
    assert {k: x for k, x in a2.__dict__.items() if k not in ("name", "trigger_rule", "inherits", "note")} == \
        {k: x for k, x in a5.__dict__.items() if k not in ("name", "trigger_rule", "inherits", "note")}
    b3, b5 = v["B3_pure_bi_mom"], v["B5_pure_bi_mom_z"]
    assert {k: x for k, x in b3.__dict__.items() if k not in ("name", "trigger_rule", "inherits", "note")} == \
        {k: x for k, x in b5.__dict__.items() if k not in ("name", "trigger_rule", "inherits", "note")}
    assert a5.trigger_rule == b5.trigger_rule == "z" and VS.SIGMA_REF == 0.0008088465199541001
    assert {k: x.c_rt for k, x in VS.COSTS.items()} == pytest.approx(
        {"gross_0bp": 0.0, "conventional_14bp": 0.0014, "main_17bp": 0.0017, "harsh_34bp": 0.0034})


# ----------------------------------------------------------------------------
# 真实数据：因果特征回归（不进入撮合，不计算盈亏）
# ----------------------------------------------------------------------------
from impulse_study import data_1m as D  # noqa: E402

CACHE = D.CACHE_DIR / D.CACHE_FILES["perp_1m"]


def _real_ctx(df1m, s1d, cut=None):
    from impulse_study.scope import IS_END, IS_START
    end = IS_END if cut is None else cut
    df = df1m[df1m.index < end]
    grid = D.minute_grid(df, IS_START, end)
    box, _ = F.s6_box_ok(df, s1d, grid)                                  # 冻结的 S6 口径（versions.BOX_STATE_VARIANT）
    last = VS.LAST_TRIGGER if cut is None else min(VS.LAST_TRIGGER, cut - pd.Timedelta(minutes=31))
    return F.make_context(grid, box, "binance_is", last_trigger=last, archive_gap_minutes=VS.ARCHIVE_GAP_MINUTES)


@pytest.mark.skipif(not CACHE.exists(), reason="1m cache not built (python -m impulse_study.data_1m --rebuild)")
def test_real_data_features_are_causal_under_prefix_truncation():
    df = D.load_perp_1m()
    s1d = D.load_spot_warmup("1d")
    full = _real_ctx(df, s1d)
    for cut in (pd.Timestamp("2021-06-26 13:37", tz="UTC"), pd.Timestamp("2022-11-09 00:00", tz="UTC")):
        tc = _real_ctx(df, s1d, cut)
        n = len(tc.C)
        k = n - 32                                                     # 截断数据中仍可交易的最后一分钟
        np.testing.assert_array_equal(tc.atr["atr_at"], full.atr["atr_at"][:n])
        np.testing.assert_array_equal(tc.base_med, full.base_med[:n])
        np.testing.assert_array_equal(tc.box_ok, full.box_ok[:n])
        for spec in VS.VERSIONS.values():
            a, b = F.build_version_inputs(full, spec), F.build_version_inputs(tc, spec)
            for f in ("entry_dir", "w", "cat"):
                np.testing.assert_array_equal(getattr(b, f)[:k], getattr(a, f)[:k])
            np.testing.assert_array_equal(b.sl[:k], a.sl[:k])


@CP2_ONLY
@pytest.mark.skipif(not CACHE.exists(), reason="1m cache not built")
def test_real_data_context_is_locked_out_of_the_engine():
    df = D.load_perp_1m()
    ctx = _real_ctx(df, D.load_spot_warmup("1d"), cut=pd.Timestamp("2020-04-01", tz="UTC"))
    vin = F.build_version_inputs(ctx, B1)
    with pytest.raises(E.EngineLockedError):
        E.run_portfolio(ctx, vin, MAIN, RISK)
