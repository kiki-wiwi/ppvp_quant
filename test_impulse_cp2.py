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
    """截断点（17,000、23,456）离停机足够远：若截断落在某笔持仓所遇的停机之内，截断后的上下文按第 5.5 节报错
    （停机延续到数据末端），而全量上下文有定义的停机离场，因此引擎层面的截断核对须避开停机。"""
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
    """numba 与纯 Python 路径逐位一致：每分钟一次的资金费结算（变长结算数组，单笔可达 30 次）、覆盖结算的停机
    （停机强制离场）与独立事件模拟都在比较之内。"""
    halts = ((6_000, 12), (9_000, 40))
    g = SY.random_grid(12_000, start=str(START.tz_localize(None)), seed=21, halts=halts)
    fund = pd.Series(np.random.default_rng(2).normal(1e-5, 3e-5, len(g)), index=g.index)
    ctx = SY.synthetic_context(g, g.index[1500], g.index[len(g) - 40], funding=fund)
    vin = F.build_version_inputs(ctx, B3)
    tr0 = E.run_portfolio(ctx, vin, MAIN, RISK).trades
    idx = np.r_[tr0["trigger_idx"].to_numpy(), [h0 - 5 for h0, _ in halts]]           # 末两个事件持仓中遇到停机
    side = np.r_[tr0["side"].to_numpy(), [1, -1]].astype(float)
    eref = ctx.C[idx]
    sl = eref * (1 - 0.01 * side)

    def both():
        return E.run_portfolio(ctx, vin, MAIN, RISK), E.simulate_events(ctx, idx, side, sl, eref, False, MAIN)

    r1, e1 = both()
    monkeypatch.setattr(E, "_trade_path", E._trade_path.py_func)
    monkeypatch.setattr(E, "_run_portfolio", E._run_portfolio.py_func)
    monkeypatch.setattr(E, "_simulate_events", E._simulate_events.py_func)
    r2, e2 = both()
    assert len(r1.trades) == len(r2.trades) > 5 and (r1.trades["n_funding"] > 8).any()
    assert (e1["reason"].to_numpy()[-2:] == E.HALT).all()
    pd.testing.assert_frame_equal(r1.trades, r2.trades, check_exact=True)
    np.testing.assert_array_equal(r1.equity.to_numpy(), r2.equity.to_numpy())
    pd.testing.assert_frame_equal(e1, e2, check_exact=True)


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
    cases = ((lambda: sim(n - 1), "at least 30 minutes"),                            # 网格末端
             (lambda: sim(n - 35), "outside the context's trading window"),          # 窗口止于 n − 40
             (lambda: sim(300), "outside the context's trading window"),             # 窗口始于 400
             (lambda: sim(1199), "maintenance placeholder"),                        # 入场 K 线为 bad
             (lambda: sim(1205), "maintenance-contaminated"),
             (lambda: sim(I0, sl=P0 + 100), "losing side"), (lambda: sim(I0, sl=np.nan), "losing side"),
             (lambda: sim(I0, side=0.0), "side must be"),
             (lambda: E.simulate_events(ctx, np.array([1000.0]), [1.0], [P0 - 100], [P0], False, MAIN), "integers"),
             (lambda: E.simulate_events(ctx, np.array([I0, I0 + 50]), [1.0], [P0 - 100], [P0], False, MAIN),
              "same length"))
    for call, msg in cases:                                                           # 每个用例只由它自己的检查拒绝
        with pytest.raises(ValueError, match=msg):
            call()
    with pytest.raises(ValueError, match="maintenance placeholder"):                  # 入场 K 线为 bad 的手工信号
        run(ctx, {1199: (1, P0 - 100)})
    with pytest.raises(ValueError, match="losing side"):                              # 止损在盈利一侧
        run(ctx, {I0: (1, P0 + 100)})
    with pytest.raises(ValueError, match="entry_dir must be"):
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
    """解锁之后的输入完整性检查（不翻转锁：直接调用检查函数；数据是合成的）。网格覆盖表 5b 的 2021-01-12 06:10。
    映射到网格上的结算必须恰为网格内全部 UTC 00:00 / 08:00 / 16:00 分钟（IS 全网格即 4,383 次），不多不少。
    这里的拒绝样例都改变了映射次数；次数相同、分钟不对的样例见下一个测试。"""
    n = 1500
    O, H, L, C, V = bars(n=n)
    g = SY.grid_from_arrays(pd.Timestamp("2021-01-12 00:00", tz="UTC"), O, H, L, C, V)
    sched = pd.DatetimeIndex(["2021-01-12 00:00", "2021-01-12 08:00", "2021-01-12 16:00", "2021-01-13 00:00"], tz="UTC")
    fund = pd.Series(1e-4, index=sched)
    box = np.zeros(n, bool)
    no_fund = F.make_context(g, box, "binance_is", g.index[400], g.index[n - 40], VS.ARCHIVE_GAP_MINUTES)
    with pytest.raises(ValueError, match="funding settlement"):
        E._require_real_inputs(no_fund)
    extra = pd.Series([1e-4], index=[sched[0] + pd.Timedelta(hours=4)])                # 计划外的 04:00 结算
    for f in (fund.iloc[:0], fund.iloc[1:], pd.Series(1e-4, index=sched - pd.Timedelta(days=400)), pd.concat([fund, extra])):
        partial = F.make_context(g, box, "binance_is", g.index[400], g.index[n - 40], VS.ARCHIVE_GAP_MINUTES, f)
        with pytest.raises(ValueError, match="funding settlement"):
            E._require_real_inputs(partial)
    wrong_gaps = F.make_context(g, box, "binance_is", g.index[400], g.index[n - 40], (), fund)
    with pytest.raises(ValueError, match="archive-gap"):
        E._require_real_inputs(wrong_gaps)
    ok = F.make_context(g, box, "binance_is", g.index[400], g.index[n - 40], VS.ARCHIVE_GAP_MINUTES, fund)
    E._require_real_inputs(ok)
    assert ok.funding_supplied and ok.n_funding_mapped == 4 and int(ok.archive_gap.sum()) == 1
    assert ok.index[ok.archive_gap][0] == pd.Timestamp("2021-01-12 06:10", tz="UTC")
    full_is = pd.date_range("2020-01-01 00:00", "2023-12-31 23:59", freq="1min", tz="UTC").as_unit("ns")
    assert int((full_is.asi8 % (8 * 3_600_000_000_000) == 0).sum()) == 4_383      # IS 全网格的结算分钟数


def test_unlocked_funding_check_compares_settlement_minutes_not_only_their_count():
    """结算次数相同、分钟不对的资金费序列同样被拒绝（第 10 节运行脚本只断言 4,383 次，逐分钟核对只在引擎里）：
    整体偏移 1 小时（时区或取整失误），或 08:00 被 09:00 取代。网格为 2021-01-12 全天，网格内的结算分钟恰为
    00:00 / 08:00 / 16:00。"""
    n = 1440
    g = SY.grid_from_arrays(pd.Timestamp("2021-01-12 00:00", tz="UTC"), *bars(n=n))
    sched = pd.DatetimeIndex(["2021-01-12 00:00", "2021-01-12 08:00", "2021-01-12 16:00"], tz="UTC")
    moved = pd.DatetimeIndex(["2021-01-12 00:00", "2021-01-12 09:00", "2021-01-12 16:00"], tz="UTC")
    box = np.zeros(n, bool)

    def ctx_with(idx):
        return F.make_context(g, box, "binance_is", g.index[400], g.index[n - 40], VS.ARCHIVE_GAP_MINUTES,
                              pd.Series(1e-4, index=idx))

    E._require_real_inputs(ctx_with(sched))
    for idx in (sched + pd.Timedelta(hours=1), moved):
        ctx = ctx_with(idx)
        assert ctx.n_funding_mapped == len(sched) == 3                    # 次数相同：只有逐分钟核对能拒绝
        with pytest.raises(ValueError, match="funding settlement"):
            E._require_real_inputs(ctx)


def test_unlocked_entry_points_run_the_real_input_checks(monkeypatch):
    """解锁之后 run_portfolio 与 simulate_events 都经 _guard 调用 _require_real_inputs（第 10 节：引擎在解锁后也核对
    资金费结算与漏记分钟）。锁只在本测试进程内用 monkeypatch 翻转，提交中的锁值不变；数据是带 'binance_is' 标签的
    合成网格，正确的代码在任何撮合之前报错。第 3 阶段的翻转提交之后本测试照常通过。"""
    n = 1500
    g = SY.grid_from_arrays(pd.Timestamp("2021-01-12 00:00", tz="UTC"), *bars(n=n))
    sched = pd.DatetimeIndex(["2021-01-12 00:00", "2021-01-12 08:00", "2021-01-12 16:00", "2021-01-13 00:00"], tz="UTC")
    box = np.zeros(n, bool)
    no_fund = F.make_context(g, box, "binance_is", g.index[400], g.index[n - 40], VS.ARCHIVE_GAP_MINUTES)
    no_gaps = F.make_context(g, box, "binance_is", g.index[400], g.index[n - 40], (), pd.Series(1e-4, index=sched))
    monkeypatch.setattr(E, "REAL_DATA_UNLOCKED", True)
    for ctx, msg in ((no_fund, "funding settlement"), (no_gaps, "archive-gap")):
        with pytest.raises(ValueError, match=msg):
            E.run_portfolio(ctx, SY.manual_inputs(ctx, B1, {I0: (1, P0 - 100)}), MAIN, RISK)
        with pytest.raises(ValueError, match=msg):
            E.simulate_events(ctx, np.array([I0]), np.array([1.0]), np.array([P0 - 100]), np.array([P0]), False, MAIN)


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
# 11. 第二轮独立复核后补充：每个测试对应第二轮变异测试中存活的违规变异或复核发现的校验缺口
# ----------------------------------------------------------------------------
def sim_event(ctx, i, side=1.0, sl=P0 - 100, eref=P0, cost=MAIN):
    return E.simulate_events(ctx, np.array([i]), np.array([side]), np.array([sl]), np.array([eref]), False, cost)


@pytest.mark.parametrize("side", [1, -1])
def test_settlement_inside_halt_has_the_side_sign(side):
    """结算时点落在持仓中的停机内：按占位分钟的开盘价（停机前最后收盘价）收付，正费率多头付、空头收。"""
    O, H, L, C, V = bars()
    bad = np.zeros(len(C), bool)
    bad[E0 + 3:E0 + 8] = True                                  # 停机占位分钟保持平盘价 10,000（= 停机前最后收盘价）
    O[E0 + 8], H[E0 + 8], L[E0 + 8], C[E0 + 8] = 10_020.0, 10_030.0, 10_015.0, 10_025.0
    r = 1e-4
    fund = pd.Series([r], index=[START + pd.Timedelta(minutes=E0 + 5)])     # τ 落在停机中
    res = run(ctx_from(O, H, L, C, V, bad=bad, funding=fund), {I0: (side, P0 - side * 100)})
    t = res.trades.iloc[0]
    q = 500.0 / 117.0                                          # 0.5%·100,000 / (100 + 0.0017·10,000)
    assert t["reason_name"] == "halt_forced_exit" and t["exit_bar"] == E0 + 8 and t["n_funding"] == 1
    assert t["funding"] == pytest.approx(-side * q * P0 * r, rel=1e-12)    # 多头付 4.27，空头收 4.27
    fill = P0 * (1 + side * 0.0005)
    cash0 = VS.INITIAL_CAPITAL - 0.0005 * q * fill
    assert res.equity.iloc[E0 + 5] == pytest.approx(cash0 - side * q * P0 * r + side * q * (P0 - fill), rel=1e-13)


def test_event_simulation_rejects_side_other_than_plus_or_minus_one():
    ctx = ctx_from(*bars())
    for side, sl in ((2.0, P0 - 100), (0.5, P0 - 100), (-2.0, P0 + 100)):     # 止损都在该“方向”的亏损一侧：只有方向检查能拒绝
        with pytest.raises(ValueError, match="side must be"):
            sim_event(ctx, I0, side=side, sl=sl)
    assert int(sim_event(ctx, I0, side=-1.0, sl=P0 + 100)["reason"].iloc[0]) == E.TIME1


def test_engine_rejects_signal_with_undefined_baseline():
    """交易窗口早于基准就绪（t < 299）：run_portfolio 与 simulate_events 都报错，第一个基准有定义的分钟照常成交。"""
    ctx = ctx_from(*bars(), start_idx=100)
    t = 250                                                    # 窗口内、未污染、ATR 已就绪（t ≥ 209）、基准尚无定义
    assert np.isnan(ctx.base_med[t]) and np.isnan(ctx.base_med[298]) and ctx.base_med[299] == pytest.approx(150.0)
    assert np.isfinite(ctx.atr["atr_at"][t]) and not ctx.frame["contaminated"].to_numpy()[t]
    with pytest.raises(ValueError, match="baseline volume undefined"):
        run(ctx, {t: (1, P0 - 100)})
    with pytest.raises(ValueError, match="baseline volume undefined"):
        sim_event(ctx, t)
    t = run(ctx, {299: (1, P0 - 100)}).trades.iloc[0]          # v1 = 150 / 150
    assert t["v1"] == pytest.approx(1.0) and t["reason_name"] == "time_bar1"


def test_event_simulation_requires_30_minutes_after_the_event():
    n = 1500
    ctx = ctx_from(*bars(n=n), last_idx=n - 10)                # 交易窗口延伸到 n − 10（run_portfolio 会拒绝这种上下文）
    assert ctx.window[n - 20] and ctx.window[n - 31]
    with pytest.raises(ValueError, match="at least 30 minutes"):
        sim_event(ctx, n - 20)                                 # 平盘 v1 = 1.0 会在 n − 5 收盘离场，但仍须报错
    with pytest.raises(ValueError, match="at least 30 minutes"):
        sim_event(ctx, n - 30)                                 # t + 30 = n 不在网格内：恰好差 1 分钟
    ok = sim_event(ctx, n - 31)                                # 恰好留出 30 分钟：接受
    assert int(ok["reason"].iloc[0]) == E.TIME1 and int(ok["exit_bar"].iloc[0]) == n - 31 + 15
    with pytest.raises(ValueError, match="at least 30 minutes beyond the last eligible trigger"):
        run(ctx, {I0: (1, P0 - 100)})                          # run_portfolio：窗口末端之后不足 30 分钟


def test_cap_and_floor_flags_are_strict_inequalities():
    """恰好相等时不记为约束：以损定仓数量恰等于 2.0x 上限、止损距离恰等于 0.15% 下限。"""
    O, H, L, C, V = bars(spread=0.0)
    t = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 25.0)}, cost=GROSS).trades.iloc[0]
    assert 0.005 * 100_000.0 / 25.0 == 2.0 * 100_000.0 / P0 == 20.0
    assert t["qty"] == 20.0 and not t["cap_bound"] and not t["floor_bound"]
    t = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 15.0)}).trades.iloc[0]
    assert 0.0015 * P0 == 15.0
    assert t["qty"] == pytest.approx(500.0 / 32.0, rel=1e-13) and not t["floor_bound"] and not t["cap_bound"]


def test_n_funding_mapped_counts_only_settlements_on_the_grid():
    O, H, L, C, V = bars()
    n = len(C)
    ts = pd.DatetimeIndex([START - pd.Timedelta(hours=8), START + pd.Timedelta(minutes=500), START + pd.Timedelta(minutes=980),
                           START + pd.Timedelta(minutes=n), START + pd.Timedelta(minutes=n + 480)])
    ctx = ctx_from(O, H, L, C, V, funding=pd.Series([1e-4] * 5, index=ts))   # 网格之前 / 网格内 ×2 / 网格右端（不含）/ 之后
    assert ctx.funding_supplied and ctx.n_funding_mapped == 2
    assert np.flatnonzero(np.isfinite(ctx.funding_rate_open)).tolist() == [500, 980]


def test_s6_box_ok_uses_the_frozen_s6_variant():
    """features.s6_box_ok 的接线：冻结的 S6 口径、网格分钟与收盘价（spot_1d = None 不读任何文件）。"""
    g = SY.random_grid(130 * 1440, start="2021-01-01", seed=8, sigma=1.2e-3, impulse_rate=1e-3)
    box, st = F.s6_box_ok(g, None, g)
    states = DY.build_box_states(g, None, variants=("main", VS.BOX_STATE_VARIANT))
    ns = np.asarray(g.index.as_unit("ns").asi8)
    s6 = DY.minute_box_ok(ns, g["close"].to_numpy(), states[VS.BOX_STATE_VARIANT])
    main = DY.minute_box_ok(ns, g["close"].to_numpy(), states["main"])
    pd.testing.assert_frame_equal(st, states[VS.BOX_STATE_VARIANT])
    np.testing.assert_array_equal(box, s6)
    assert 0 < box.sum() < main.sum() and not (box & ~main).any()      # S6 比 CP1 原口径多一个条件：只会更少


def test_s6_box_ok_forwards_the_spot_warmup():
    """features.s6_box_ok 把现货日线预热原样传给 ATR_1D（计划第 1.1、3 节），并按 grid（而非 df1m）的分钟与收盘价判定。"""
    g = SY.random_grid(130 * 1440, start="2021-01-01", seed=8, sigma=1.2e-3, impulse_rate=1e-3)
    days = pd.date_range("2020-11-01", "2020-12-31", freq="1D", tz="UTC").as_unit("ns")
    px = np.full(len(days), 10_000.0)
    spot = pd.DataFrame({"open": px, "high": px * 1.5, "low": px * 0.6, "close": px, "volume": 1.0}, index=days)
    grid = g.iloc[30 * 1440:]                                          # 与 df1m 不同的网格：钉住网格分钟与收盘价的接线
    box, st = F.s6_box_ok(g, spot, grid)
    want = DY.build_box_states(g, spot, variants=(VS.BOX_STATE_VARIANT,))[VS.BOX_STATE_VARIANT]
    pd.testing.assert_frame_equal(st, want)
    ns = np.asarray(grid.index.as_unit("ns").asi8)
    np.testing.assert_array_equal(box, DY.minute_box_ok(ns, grid["close"].to_numpy(), want))
    assert (box != F.s6_box_ok(g, None, grid)[0]).any()               # 夹具能区分是否传入了现货日线预热


@pytest.mark.parametrize("offset,lookback,holding", [(-1, False, True), (0, True, False), (15, True, False),
                                                    (16, False, False)])
def test_archive_gap_flags_boundaries(offset, lookback, holding):
    """漏记分钟在 t − offset：触发回看 [t−15, t] 与持仓 [t+1, 离场分钟] 两个标记的四条边界（offset = −1 即入场分钟 t+1）。"""
    ctx = ctx_from(*bars(), gaps=[START + pd.Timedelta(minutes=I0 - offset)])
    t = run(ctx, {I0: (1, P0 - 100)}).trades.iloc[0]
    assert bool(t["signal_lookback_has_archive_gap"]) is lookback and bool(t["has_archive_gap"]) is holding


def test_ledger_times_follow_the_exit_time_definition():
    O, H, L, C, V = bars()
    O[E0 + 3], L[E0 + 3] = 9_850.0, 9_840.0
    t = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 100)}).trades.iloc[0]
    assert t["reason_name"] == "stop_gap_open" and t["exit_time"] == START + pd.Timedelta(minutes=E0 + 3)
    O, H, L, C, V = bars()
    L[E0 + 3] = 9_890.0
    t = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 100)}).trades.iloc[0]
    assert t["reason_name"] == "stop_intrabar" and t["exit_time"] == START + pd.Timedelta(minutes=E0 + 4)
    t = run(ctx_from(*bars()), {I0: (1, P0 - 100)}).trades.iloc[0]
    assert t["reason_name"] == "time_bar1" and t["exit_time"] == START + pd.Timedelta(minutes=E0 + 15)
    assert t["trigger_time"] == START + pd.Timedelta(minutes=I0) and t["entry_time"] == START + pd.Timedelta(minutes=E0)


def test_daily_equity_is_the_last_minute_of_each_utc_day():
    res = run(ctx_from(*bars()), {I0: (1, P0 - 100)})          # 1,500 分钟：2021-01-04 全天 + 01-05 前 60 分钟
    want_day1 = VS.INITIAL_CAPITAL - 17.0015 * 500.0 / 117.0   # 01-04 内一笔 time_bar1 亏损：费 10.0015·q + 滑点 7·q
    d = res.daily_equity
    assert list(d.index) == [pd.Timestamp("2021-01-04", tz="UTC"), pd.Timestamp("2021-01-05", tz="UTC")]
    assert d.iloc[0] == pytest.approx(want_day1, rel=1e-13) and d.iloc[0] == res.equity.iloc[1439]
    assert d.iloc[1] == pytest.approx(want_day1, rel=1e-13) and res.equity.iloc[400] == VS.INITIAL_CAPITAL


def test_ledger_category_w_and_atr_are_read_at_the_trigger_minute():
    t0 = 1003                                                  # t0 % 15 = 13：t0 与 t0 + 1 的决策读取不同的 ATR 桶
    O, H, L, C, V = bars()
    H[t0 - 8] = 10_050.0                                       # 只放大桶 [990, 1005) 的 TR：t0 + 1 决策时才收盘
    ctx = ctx_from(O, H, L, C, V)
    atr_at = ctx.atr["atr_at"]
    assert atr_at[t0] == pytest.approx(10.0) and atr_at[t0 + 1] == pytest.approx((13 * 10.0 + 55.0) / 14)
    z = np.zeros(len(C), bool)
    one = z.copy()
    one[t0] = True
    ctx.cond["pct"] = {"up1": one, "up2": one, "dn1": z, "dn2": z}                 # ① + ② → 类别 3、W = 15
    vin = F.build_version_inputs(ctx, VS.VERSIONS["B2_pure_long_sl"])
    t = E.run_portfolio(ctx, vin, MAIN, RISK).trades.iloc[0]
    assert t["trigger_idx"] == t0 and t["category"] == 3 and t["w"] == 15
    assert t["atr15"] == atr_at[t0] == pytest.approx(10.0)                         # 平盘每桶 TR = 10,005 − 9,995
    assert t["sl"] == pytest.approx(9_995.0 - 0.1 * 10.0)                           # SL_B 用同一个 t 时 ATR


def test_exclusive_exclusion_order_for_each_adjacent_pair():
    n = 800
    O, H, L, C, V = bars(n=n)
    bad = np.zeros(n, bool)
    bad[[100, 151, 251]] = True
    box = np.ones(n, bool)
    box[270] = False
    ctx = ctx_from(O, H, L, C, V, bad=bad, start_idx=30, last_idx=700, box=box)
    up = np.zeros(n, bool)
    up[[105, 150, 250, 270, 500]] = True
    z = np.zeros(n, bool)
    ctx.cond["pct"] = {"up1": up, "up2": z, "dn1": z, "dn2": z}
    contam, atr_nan, base_nan = ctx.frame["contaminated"].to_numpy(), np.isnan(ctx.atr["atr_at"]), np.isnan(ctx.base_med)
    # 105：污染 + ATR 未就绪 + 基准无定义；150：ATR 未就绪 + 入场 K 线 bad + 基准无定义；
    # 250：入场 K 线 bad + 基准无定义（ATR 已就绪）；270：基准无定义 + 箱体不成立（仅 A 组）；500：合格
    assert contam[105] and atr_nan[105] and not bad[106] and base_nan[105]
    assert not contam[150] and atr_nan[150] and bad[151] and base_nan[150]
    assert not contam[250] and not atr_nan[250] and bad[251] and base_nan[250]
    assert not contam[270] and not atr_nan[270] and not bad[271] and base_nan[270] and not box[270]
    order = ["ambiguous", "contaminated", "atr_not_ready", "entry_bar_bad", "baseline_undefined", "box_rejected", "invalid_stop"]
    want = {"ambiguous": 0, "contaminated": 1, "atr_not_ready": 1, "entry_bar_bad": 1, "baseline_undefined": 1,
            "box_rejected": 0, "invalid_stop": 0}
    for name in ("A1_box_opt", "B1_pure_long"):
        vin = F.build_version_inputs(ctx, VS.VERSIONS[name])
        assert list(vin.skipped_exclusive) == order
        assert vin.skipped_exclusive == want, name
        assert np.flatnonzero(vin.entry_dir).tolist() == [500]
    a1 = F.build_version_inputs(ctx, VS.VERSIONS["A1_box_opt"])
    assert a1.skipped == {"ambiguous": 0, "contaminated": 1, "atr_not_ready": 2, "entry_bar_bad": 2, "baseline_undefined": 4,
                          "box_rejected": 1, "invalid_stop": 0}


def test_run_portfolio_rejects_version_inputs_not_aligned_with_the_grid(monkeypatch):
    """版本输入的每个数组都必须是一维且与分钟网格等长；撮合核按分钟索引直接读取，长度不符时报错而不是越界读写。"""
    ctx = ctx_from(*bars())                                    # n = 1500，交易窗口 400..1460
    vin = SY.manual_inputs(ctx, B1, {I0: (1, P0 - 100), 1300: (1, P0 - 100)})
    assert len(E.run_portfolio(ctx, vin, MAIN, RISK).trades) == 2
    monkeypatch.setattr(E, "_trade_path", E._trade_path.py_func)     # 检查一旦缺失，纯 Python 核以 IndexError 失败，
    monkeypatch.setattr(E, "_run_portfolio", E._run_portfolio.py_func)   # 而不是破坏测试进程的内存
    for f in ("entry_dir", "sl", "eref", "w", "cat", "atr_at"):
        a = getattr(vin, f)
        for wrong in (a[:1100].copy(), np.r_[a, a[:5]], a.reshape(1, -1), a.reshape(-1, 1)):   # 短 / 长 / 二维 ×2
            with pytest.raises(ValueError, match="one entry per grid minute"):
                E.run_portfolio(ctx, dataclasses.replace(vin, **{f: wrong}), MAIN, RISK)


def test_run_portfolio_rejects_inputs_built_on_another_context():
    """版本输入只能与构造它的那个上下文对象一起撮合：另一个交易窗口、另一组触发条件、另一个网格长度或缺少来源都报错。"""
    g = SY.random_grid(6_000, start=str(START.tz_localize(None)), seed=3, impulse_rate=3e-3)
    annex = SY.synthetic_context(g, g.index[400], g.index[len(g) - 40])
    main = SY.synthetic_context(g, g.index[1000], g.index[len(g) - 40])
    assert len(E.run_portfolio(annex, F.build_version_inputs(annex, B3), MAIN, RISK).trades) > 0
    alt = dataclasses.replace(main, cond={"pct": {k: ~v for k, v in main.cond["pct"].items()}, "z": main.cond["z"]})
    short = SY.synthetic_context(g.iloc[:4_000], g.index[400], g.index[3_960])
    for ctx, vin in ((annex, F.build_version_inputs(main, B3)), (alt, F.build_version_inputs(main, B3)),
                     (short, F.build_version_inputs(annex, B3)),
                     (annex, dataclasses.replace(F.build_version_inputs(annex, B3), context=None))):
        with pytest.raises(ValueError, match="different context"):
            E.run_portfolio(ctx, vin, MAIN, RISK)


def test_run_portfolio_rejects_inputs_after_in_place_context_changes():
    """版本输入构造之后，同一个上下文对象的交易窗口、该版本口径的触发条件或（A 组）箱体被原地改动时报错；
    与该版本无关的改动（B3 的 Z 口径条件、B 组的箱体）不影响结果。"""
    g = SY.random_grid(6_000, start=str(START.tz_localize(None)), seed=3, impulse_rate=3e-3)

    def fresh(spec):
        ctx = SY.synthetic_context(g, g.index[1_000], g.index[len(g) - 40])
        return ctx, F.build_version_inputs(ctx, spec)

    changes = {
        "window widened": (B3, lambda c: c.window.__setitem__(slice(400, 1_000), True)),
        "pct element flipped": (B3, lambda c: c.cond["pct"]["up1"].__setitem__(3_000, not c.cond["pct"]["up1"][3_000])),
        "pct dict replaced": (B3, lambda c: c.cond.__setitem__("pct", {k: ~v for k, v in c.cond["pct"].items()})),
        "z dict replaced": (VS.VERSIONS["B5_pure_bi_mom_z"],
                            lambda c: c.cond.__setitem__("z", {k: ~v for k, v in c.cond["z"].items()})),
        "box edited": (VS.VERSIONS["A1_box_opt"], lambda c: c.box_ok.__setitem__(slice(None), True)),
    }
    for name, (spec, change) in changes.items():
        ctx, vin = fresh(spec)
        E.run_portfolio(ctx, vin, MAIN, RISK)
        change(ctx)
        with pytest.raises(ValueError, match="changed in place"):
            E.run_portfolio(ctx, vin, MAIN, RISK)
        E.run_portfolio(ctx, F.build_version_inputs(ctx, spec), MAIN, RISK)
    for spec, change in ((B3, changes["z dict replaced"][1]), (B3, changes["box edited"][1])):
        ctx, vin = fresh(spec)
        ref = E.run_portfolio(ctx, vin, MAIN, RISK).trades
        change(ctx)
        assert E.run_portfolio(ctx, vin, MAIN, RISK).trades.equals(ref)
    ctx, vin = fresh(B3)
    ctx.window[400:1_000] = True
    assert (E.run_portfolio(ctx, F.build_version_inputs(ctx, B3), MAIN, RISK).trades["trigger_idx"] < 1_000).any()


def test_make_context_requires_a_complete_unique_minute_grid():
    g = SY.grid_from_arrays(START, *bars())
    for bad_grid in (g.drop(g.index[700:705]), pd.concat([g.iloc[:800], g.iloc[797:]]), g.iloc[::-1], g.iloc[:0]):
        assert bad_grid.attrs["provenance"] == "synthetic"
        with pytest.raises(ValueError, match="complete minute grid"):
            SY.synthetic_context(bad_grid, g.index[400], g.index[1400])
    for box in (True, np.zeros(len(g) - 1, bool), np.zeros((len(g), 1), bool)):
        with pytest.raises(ValueError, match="one value per grid minute"):
            F.make_context(g, box, "synthetic", g.index[400], g.index[1400])
    with pytest.raises(ValueError, match="complete minute grid"):                 # 与数据标签无关
        F.make_context(g.drop(g.index[700:705]), np.zeros(len(g) - 5, bool), "binance_is", g.index[400], g.index[1400])
    with pytest.raises(ValueError, match="one value per grid minute"):
        F.make_context(g, np.zeros(len(g) - 1, bool), "binance_is", g.index[400], g.index[1400])


def test_make_context_copies_box_ok():
    O, H, L, C, V = bars()
    box = np.zeros(len(C), bool)
    ctx = ctx_from(O, H, L, C, V, box=box)
    box[:] = True                                              # 调用方之后改写自己的数组，不影响上下文
    assert not ctx.box_ok.any() and ctx.box_ok is not box


def test_make_context_requires_a_timezone_aware_index_and_uses_utc():
    """无时区的网格报错；其他时区的网格统一转换为 UTC，日净值按 UTC 日归属。"""
    g = SY.grid_from_arrays(START, *bars())
    naive = g.tz_localize(None)
    assert naive.attrs["provenance"] == "synthetic"
    with pytest.raises(ValueError, match="timezone-aware"):
        SY.synthetic_context(naive, g.index[400], g.index[1400])
    local = g.tz_convert("Asia/Shanghai")
    assert local.attrs["provenance"] == "synthetic"
    ctx = SY.synthetic_context(local, g.index[400], g.index[1400])
    assert str(ctx.index.tz) == "UTC" and (ctx.index == g.index).all()
    res = run(ctx, {I0: (1, P0 - 100)})
    assert list(res.daily_equity.index) == [pd.Timestamp("2021-01-04", tz="UTC"), pd.Timestamp("2021-01-05", tz="UTC")]


def test_n_signals_in_window_counts_only_in_window_signals():
    ctx = ctx_from(*bars(), start_idx=600, last_idx=1300)
    sig = {500: (1, P0 - 100), 600: (1, P0 - 100), 1300: (1, P0 - 100), 1301: (1, P0 - 100)}
    res = E.run_portfolio(ctx, SY.manual_inputs(ctx, B1, sig), MAIN, RISK)
    assert res.trades["trigger_idx"].tolist() == [600, 1300] and res.meta["n_signals_in_window"] == 2


def test_daily_equity_last_minute_differs_from_min_max_first():
    """日净值取当日最后一分钟：夹具中当日最后一分钟的净值既不是当日最低、最高，也不是当日第一分钟的值。"""
    O, H, L, C, V = bars()
    C[1315], H[1315] = 10_030.0, 10_035.0                      # 第二笔 time_bar1 小幅盈利，日末值介于当日最低与最高之间
    res = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 100), 1300: (1, P0 - 100)})
    day1 = res.equity.iloc[400:1440]
    assert day1.min() < day1.iloc[-1] < day1.max() and day1.iloc[-1] != day1.iloc[0]
    assert res.daily_equity.iloc[0] == res.equity.iloc[1439]


def test_funding_supplied_flag_and_grid_without_settlement_minutes():
    """funding_supplied 只在传入资金费序列时为真；网格内没有任何结算分钟时，未传入资金费的真实标签上下文仍被拒绝。"""
    assert not ctx_from(*bars()).funding_supplied
    n = 400                                                    # 01:00–07:39：网格内没有任何结算分钟
    g = SY.grid_from_arrays(pd.Timestamp("2021-01-12 01:00", tz="UTC"), *bars(n=n))
    no_fund = F.make_context(g, np.zeros(n, bool), "binance_is", g.index[0], g.index[-1], VS.ARCHIVE_GAP_MINUTES)
    with pytest.raises(ValueError, match="funding settlement"):
        E._require_real_inputs(no_fund)


def test_ledger_column_types():
    t = run(ctx_from(*bars()), {I0: (1, P0 - 100)}).trades
    for c in ("trigger_idx", "entry_idx", "exit_bar", "side", "reason", "cont", "n_funding", "exit_time_idx",
              "funding_last_boundary"):                         # 显式列出，不引用 engine._INT_COLS 本身
        assert t[c].dtype == np.int64, c
    assert t["cap_bound"].dtype == bool and t["floor_bound"].dtype == bool


def test_halt_covering_t15_records_no_continuation_test():
    """停机覆盖 t+15：不做延续判别（cont = −1，v1、k1 缺失），在停机后第一根有效开盘离场。"""
    O, H, L, C, V = bars()
    bad = np.zeros(len(C), bool)
    bad[E0 + 5:E0 + 31] = True
    t = run(ctx_from(O, H, L, C, V, bad=bad), {I0: (1, P0 - 100)}).trades.iloc[0]
    assert t["reason_name"] == "halt_forced_exit" and t["exit_bar"] == E0 + 31
    assert t["cont"] == -1 and np.isnan(t["v1"]) and np.isnan(t["k1"])


def test_funding_last_boundary_column():
    """诊断列 funding_last_boundary：时间离场等于离场时刻索引，盘中止损为离场时刻索引减 1。"""
    O, H, L, C, V = bars()
    L[E0 + 3] = 9_890.0
    t = run(ctx_from(O, H, L, C, V), {I0: (1, P0 - 100)}).trades.iloc[0]
    assert t["reason_name"] == "stop_intrabar" and t["funding_last_boundary"] == t["exit_time_idx"] - 1 == E0 + 3
    t = run(ctx_from(*bars()), {I0: (1, P0 - 100)}).trades.iloc[0]
    assert t["reason_name"] == "time_bar1" and t["funding_last_boundary"] == t["exit_time_idx"] == E0 + 15


def test_defensive_checks_of_private_helpers():
    """公开入口的其他检查使这两处在正常调用中不可达；直接调用私有函数固定它们的行为。"""
    ctx = ctx_from(*bars())
    n = len(ctx.C)
    with pytest.raises(ValueError, match="outside the grid"):
        E._check_signals(ctx, np.array([n - 1]), np.array([1.0]), np.array([P0 - 100]), np.array([P0]))
    m, i = 1020, 1000                                          # 触发后只有 19 分钟数据
    O, H, L, C, V = SY.flat_bars(m, P0, 10.0, 0.0005)
    V[i + 1:i + 16] = 20.0                                     # v1 = 300 / 150 = 2.0
    C[i + 15] = 10_004.0                                       # k1 = 0.8 → 继续持有第 2 根，走到网格末端
    with pytest.raises(ValueError, match="end of the minute grid"):
        E._trade_path(i, 1.0, P0 - 100, O, H, L, C, V, np.zeros(m, np.bool_), np.full(m, np.nan), 150.0, False,
                      VS.V1_MIN, VS.K1_MIN, m)


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
