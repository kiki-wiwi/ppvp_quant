"""
test_impulse_cp1.py — 脉冲研究 Checkpoint 1 的单元测试（数据层、触发器、日线箱体状态、无未来函数）。

合成数据测试不需要网络与缓存；``test_real_data_*`` 在 data/processed_1m 缓存存在时运行（否则跳过），
把 Checkpoint 1 报告中的核心计数钉住，供后续阶段检测意外改动。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from impulse_study import daily as DY
from impulse_study import data_1m as D
from impulse_study import signals as S
from impulse_study.scope import IS_END, IS_START, ScopeError, assert_before_is, assert_within_is
from ppvp_engine import volume_profile

ROOT = Path(__file__).resolve().parent
UTC = "UTC"


def _bars(start: str, closes, vol=1.0, spread=0.0005) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    idx = pd.date_range(start, periods=len(closes), freq="1min", tz=UTC).as_unit("ns")
    opens = np.concatenate([[closes[0]], closes[:-1]])
    hi = np.maximum(opens, closes) * (1 + spread)
    lo = np.minimum(opens, closes) * (1 - spread)
    v = np.full(len(closes), vol, dtype=float) if np.isscalar(vol) else np.asarray(vol, dtype=float)
    df = pd.DataFrame({"open": opens, "high": hi, "low": lo, "close": closes, "volume": v,
                       "quote_volume": v * closes, "count": np.where(v > 0, 10.0, 0.0), "taker_buy_volume": v / 2},
                      index=idx)
    df.index.name = "open_time"
    return df


def _random_walk_1m(days: int, seed: int, start: str = "2020-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = days * 1440
    r = rng.normal(0, 0.0012, n)
    r[rng.integers(0, n, 40)] += rng.choice([-1, 1], 40) * 0.02      # 稀疏大脉冲
    closes = 10_000 * np.exp(np.cumsum(r))
    vol = rng.lognormal(0, 0.5, n)
    return _bars(start, closes, vol)


# ----------------------------------------------------------------------------
# 范围断言
# ----------------------------------------------------------------------------
def test_scope_guards():
    ok = pd.DatetimeIndex([IS_START, IS_END - pd.Timedelta(minutes=1)])
    assert_within_is(ok, "ok")
    with pytest.raises(ScopeError):
        assert_within_is(pd.DatetimeIndex([IS_END]), "val bar")
    with pytest.raises(ScopeError):
        assert_within_is(pd.DatetimeIndex([IS_START - pd.Timedelta(minutes=1)]), "pre-IS")
    assert_before_is(pd.DatetimeIndex([IS_START - pd.Timedelta(days=1)]), "warm-up ok")
    with pytest.raises(ScopeError):
        assert_before_is(pd.DatetimeIndex([IS_START]), "warm-up reaching IS")


# ----------------------------------------------------------------------------
# 分钟网格与维护标记
# ----------------------------------------------------------------------------
def test_minute_grid_flags():
    df = _bars("2020-01-01", np.linspace(100, 101, 10))
    df.loc[df.index[4], ["open", "high", "low", "close", "volume", "count"]] = [df["close"].iloc[3]] * 4 + [0.0, 0.0]
    df = df.drop(df.index[7])
    g = D.minute_grid(df, pd.Timestamp("2020-01-01", tz=UTC), pd.Timestamp("2020-01-01 00:10", tz=UTC))
    assert len(g) == 10
    assert g["missing"].tolist() == [False] * 7 + [True] + [False] * 2
    assert g["placeholder"].tolist() == [False] * 4 + [True] + [False] * 5
    assert g["zero_volume"].sum() == 1
    assert g["bad"].tolist() == [False] * 4 + [True, False, False, True, False, False]


# ----------------------------------------------------------------------------
# 触发器
# ----------------------------------------------------------------------------
def test_returns_hand_computed_and_nan_at_bad_endpoints():
    closes = [100.0] * 20 + [100.2, 100.5, 100.7, 100.9, 101.0, 101.3]
    df = _bars("2020-01-01", closes)
    g = D.minute_grid(df, df.index[0], df.index[-1] + pd.Timedelta(minutes=1))
    f = S.impulse_frame(g)
    T = 25
    assert f["r1"].iloc[T] == pytest.approx(101.3 / 101.0 - 1, abs=0)
    assert f["r5"].iloc[T] == pytest.approx(101.3 / 100.2 - 1, abs=0)
    assert f["r15"].iloc[T] == pytest.approx(101.3 / 100.0 - 1, abs=0)
    assert np.isnan(f["r15"].iloc[14]) and not np.isnan(f["r15"].iloc[15])
    # 端点为占位分钟 → 收益缺失
    df2 = df.copy()
    df2.loc[df2.index[20], ["open", "high", "low", "close", "volume"]] = [100.0, 100.0, 100.0, 100.0, 0.0]
    f2 = S.impulse_frame(D.minute_grid(df2, df2.index[0], df2.index[-1] + pd.Timedelta(minutes=1)))
    assert np.isnan(f2["r1"].iloc[20]) and np.isnan(f2["r1"].iloc[21]) and np.isnan(f2["r5"].iloc[25])
    assert f2["r15"].iloc[25] == f["r15"].iloc[25]               # 端点 10 与 25 均有效：收益有定义……
    assert bool(f2["contaminated"].iloc[25])                      # ……但 [T−15, T] 跨过占位分钟，记为维护污染
    assert bool(f2["contaminated"].iloc[20]) and not bool(f2["contaminated"].iloc[19])


def test_conditions_thresholds_and_split():
    idx = pd.date_range("2020-01-01", periods=6, freq="1min", tz=UTC)
    f = pd.DataFrame({"r1": [0.0018, 0.0017, 0.0019, -0.0018, -0.0019, 0.0],
                      "r5": [0.0047, 0.0050, 0.0046, -0.0047, -0.0040, 0.0],
                      "r15": [0.0, 0.0118, 0.0119, -0.0118, -0.0300, 0.01179]}, index=idx)
    # 使用精确可表示的比较：构造值本身即阈值（同一个 float 字面量）
    c = S.conditions(f)
    assert c["up1"].tolist() == [True, False, False, False, False, False]
    assert c["up2"].tolist() == [False, True, True, False, False, False]
    assert c["dn1"].tolist() == [False, False, False, True, False, False]
    assert c["dn2"].tolist() == [False, False, False, True, True, False]
    sel = np.ones(6, dtype=bool)
    assert S.split_counts(c["up1"], c["up2"], sel) == {"only_1": 1, "only_2": 2, "both": 0, "total": 3}
    assert S.split_counts(c["dn1"], c["dn2"], sel) == {"only_1": 0, "only_2": 1, "both": 1, "total": 2}
    cz = S.conditions(f, 2.0)                     # Z 口径：阈值等比缩放
    assert not cz["up1"].any() and not cz["up2"].any() and cz["dn2"].tolist() == [False] * 4 + [True, False]


def test_onset_and_schedule_semantics():
    m = np.zeros(200, dtype=bool)
    m[[10, 11, 12, 30, 39, 40, 41, 56, 100, 129, 130, 131, 175]] = True
    assert np.flatnonzero(S.onset(m)).tolist() == [10, 30, 39, 56, 100, 129, 175]
    # sched30：触发 T → T+1 入场，T+16 离场（1 根合成 15m 棒），冷却 15 分钟 → 下一触发须 ≥ T+30
    assert np.flatnonzero(S.schedule(m, 30)).tolist() == [10, 40, 100, 130, 175]
    # sched45：持满 2 根于 T+31 离场 → 下一触发须 ≥ T+45
    assert np.flatnonzero(S.schedule(m, 45)).tolist() == [10, 56, 129, 175]
    entry, exit1, exit2 = 1, 16, 31
    assert exit1 + 15 - entry == 30 and exit2 + 15 - entry == 45


def test_eligible_window_bounds():
    idx = pd.DatetimeIndex([IS_START + pd.Timedelta(minutes=14), IS_START + pd.Timedelta(minutes=15),
                            IS_END - pd.Timedelta(minutes=31), IS_END - pd.Timedelta(minutes=30)])
    assert S.eligible_window(idx).tolist() == [False, True, True, False]


def test_sigma7d_excludes_impulse_window():
    df = _random_walk_1m(9, seed=3)
    f = S.impulse_frame(D.minute_grid(df, df.index[0], df.index[-1] + pd.Timedelta(minutes=1)))
    T = len(f) - 1
    r1 = f["r1"].to_numpy()
    w = r1[T - 15 - 10_079:T - 15 + 1]
    assert f["sigma7d"].iloc[T] == pytest.approx(np.sqrt(np.nanmean(w ** 2)), rel=1e-12)
    assert np.isnan(f["sigma7d"].iloc[9_000])            # 不足 95% 的 7 日窗口


# ----------------------------------------------------------------------------
# 成交量分布与日线状态
# ----------------------------------------------------------------------------
def test_profile_levels_match_ppvp_engine():
    rng = np.random.default_rng(7)
    for _ in range(60):
        n = int(rng.integers(5, 3000))
        mid = 100 + np.cumsum(rng.normal(0, 0.3, n))
        half = np.abs(rng.normal(0, 0.2, n))
        half[rng.integers(0, n, max(1, n // 50))] = 0.0          # 一字线
        lo, hi = mid - half, mid + half
        v = rng.lognormal(0, 1, n)
        ref = volume_profile(lo, hi, v, DY.N_BINS, DY.VA_PCT)
        poc, val, vah, pl, ph = DY.profile_levels(lo, hi, v)
        assert (poc, val, vah, pl, ph) == (ref["poc"], ref["val"], ref["vah"], ref["p_low"], ref["p_high"])
        edges = np.linspace(pl, ph, DY.N_BINS + 1)
        h_nb = DY._hist_uniform(lo, hi, v, edges)
        h_py = DY._hist_uniform.py_func(lo, hi, v, edges)
        assert np.array_equal(h_nb, h_py)
        assert np.allclose(h_nb, ref["hist"], rtol=1e-12, atol=1e-12)


def _daily_frame(lows, highs, start="2021-01-01"):
    idx = pd.date_range(start, periods=len(lows), freq="1D", tz=UTC).as_unit("ns")
    lows, highs = np.asarray(lows, float), np.asarray(highs, float)
    d = pd.DataFrame({"open": (lows + highs) / 2, "high": highs, "low": lows, "close": (lows + highs) / 2,
                      "volume": 1.0, "n_min": 1440}, index=idx)
    d["available_time"] = d.index + pd.Timedelta(days=1)
    return d


def _flat_prof(as_of, val=100.0, poc=110.0):
    return pd.DataFrame({"POC": poc, "VAL": val, "VAH": 120.0, "P_low": 90.0, "P_high": 130.0, "n_bars": 1},
                        index=pd.DatetimeIndex(as_of))


def test_box_state_touch_then_bounce_logic():
    n = 40
    lows, highs = np.full(n, 112.0), np.full(n, 114.0)          # 平时远离支撑带 [VAL±0.5ATR] = [99, 101]
    d = _daily_frame(lows, highs)
    atr = pd.Series(2.0, index=d.index)                        # band = [99, 101]，bounce = 101 + 1.6 = 102.6
    as_of = d.index[-1] + pd.Timedelta(days=1)

    def primed(lw, hg, **kw):
        dd = _daily_frame(lw, hg)
        st = DY.box_state(dd, _flat_prof([as_of]), atr, **kw)
        return bool(st["primed"].iloc[0])

    lw, hg = lows.copy(), highs.copy()
    assert not primed(lw, hg)                                  # 从未触及
    lw2 = lw.copy()
    lw2[n - 10] = 100.5                                        # 第 n-10 日触及（其最高价 114 ≥ 102.6，但同日不算反弹）
    hg2 = hg.copy()
    hg2[n - 10] = 101.5
    hg2[n - 9:] = 101.8                                        # 之后各日最高 101.8 < 102.6：没有反弹
    lw2[n - 9:] = 100.8
    assert not primed(lw2, hg2)
    hg3 = hg2.copy()
    hg3[n - 4] = 102.7                                         # 触及之后的某日最高价达到 bounce → primed
    assert primed(lw2, hg3)
    hg4 = hg2.copy()
    hg4[n - 10] = 103.0                                        # 只有触及当日同时达到 bounce
    assert not primed(lw2, hg4) and primed(lw2, hg4, same_day_bounce=True)
    lw5, hg5 = lows.copy(), highs.copy()                       # 触及发生在 31 天前（窗口外），之后有反弹
    lw5[n - 31], hg5[n - 31] = 100.0, 101.0
    assert not primed(lw5, hg5)
    lw6, hg6 = lows.copy(), highs.copy()                       # 触及恰在 30 天窗口的第一天
    lw6[n - 30], hg6[n - 30] = 100.0, 101.0
    assert primed(lw6, hg6)


def test_minute_box_ok_uses_floor_day_and_zone():
    days = pd.date_range("2021-03-01", periods=3, freq="1D", tz=UTC).as_unit("ns")
    st = pd.DataFrame({"ATR": 2.0, "band_lo": [99.0, 50.0, 99.0], "band_hi": 101.0, "bounce_level": 102.6,
                       "POC": [110.0, 60.0, 110.0], "VAL": 100.0, "primed": [True, True, False], "valid": True},
                      index=days)
    t = pd.DatetimeIndex(["2021-03-01 23:59", "2021-03-02 00:00", "2021-03-01 12:00", "2021-03-01 12:01",
                          "2021-03-03 10:00", "2021-03-04 00:00"], tz=UTC).as_unit("ns")
    px = np.array([105.0, 105.0, 98.9, 110.0, 105.0, 105.0])
    ok = DY.minute_box_ok(np.asarray(t.asi8), px, st)
    # 23:59 用 3/1 的状态（区间 [99,110] 内）；00:00 用 3/2 的状态（区间 [50,60]，105 不在内）；低于 band_lo / 恰为 POC；
    # 3/3 未 primed；3/4 无状态
    assert ok.tolist() == [True, False, False, True, False, False]


def test_daily_from_1m_available_time_and_partial_day():
    df = _bars("2020-01-01", 100 + np.arange(1440 * 2 + 30) * 0.001)
    d = DY.daily_from_1m(df)
    assert len(d) == 2                                          # 第 3 天只有 30 分钟：未收盘，剔除
    assert (d["available_time"] - d.index == pd.Timedelta(days=1)).all()
    assert (d["n_min"] == 1440).all()
    assert d["high"].iloc[0] == df["high"].iloc[:1440].max() and d["close"].iloc[1] == df["close"].iloc[2879]


def test_prefix_truncation_synthetic():
    """只用 open_time < τ 的数据重算，τ 之前每一分钟的特征、触发条件与箱体过滤都与全量计算逐位相同。"""
    df = _random_walk_1m(100, seed=11)
    s1d = None
    full_st = DY.build_box_states(df, s1d, variants=("main",))["main"]
    t0, t1 = df.index[0], df.index[-1] + pd.Timedelta(minutes=1)
    f = S.impulse_frame(D.minute_grid(df, t0, t1))
    tns = np.asarray(f.index.as_unit("ns").asi8)
    c = S.conditions(f)
    box = DY.minute_box_ok(tns, f["close"].to_numpy(), full_st)
    assert full_st["valid"].sum() > 0
    rng = np.random.default_rng(5)
    cuts = [df.index[int(i)] for i in rng.integers(61 * 1440, len(df) - 1, 4)] + [df.index[61 * 1440]]
    for tau in cuts:
        dft = df[df.index < tau]
        st = DY.build_box_states(dft, s1d, variants=("main",))["main"]
        ft = S.impulse_frame(D.minute_grid(dft, t0, tau))
        n = len(ft)
        assert np.array_equal(ft[["r1", "r5", "r15", "sigma7d", "sigma60"]].to_numpy(),
                              f[["r1", "r5", "r15", "sigma7d", "sigma60"]].to_numpy()[:n], equal_nan=True)
        ct = S.conditions(ft)
        assert all(np.array_equal(ct[k], c[k][:n]) for k in ct)
        assert np.array_equal(DY.minute_box_ok(tns[:n], ft["close"].to_numpy(), st), box[:n])


def test_future_shock_does_not_change_past_box_state():
    """把截断点之后的价格整体改写（未来冲击），截断点之前的箱体状态与过滤结果不变。"""
    df = _random_walk_1m(90, seed=21)
    st0 = DY.build_box_states(df, None, variants=("main",))["main"]
    cut = df.index[75 * 1440 + 333]
    df2 = df.copy()
    later = df2.index >= cut
    df2.loc[later, ["open", "high", "low", "close"]] *= 1.5
    df2.loc[later, "volume"] *= 9.0
    st1 = DY.build_box_states(df2, None, variants=("main",))["main"]
    keep = st1.index <= cut.floor("D")
    pd.testing.assert_frame_equal(st0[keep], st1[keep])


# ----------------------------------------------------------------------------
# 裁决规则（不联网：替换下载函数）
# ----------------------------------------------------------------------------
def test_adjudication_rule_v2(monkeypatch):
    day = pd.Timestamp("2021-06-01", tz=UTC)
    good = _bars("2021-06-01", 100 + np.sin(np.arange(1440) / 50.0), vol=2.0)
    bad = good.copy()                                             # 月度归档：40 分钟被记成零成交占位
    hole = bad.index[600:640]
    bad.loc[hole, ["open", "high", "low", "close"]] = float(bad["close"].iloc[599])
    bad.loc[hole, ["volume", "quote_volume", "count", "taker_buy_volume"]] = 0.0
    agg = good.copy()                                             # 逐笔聚合 = 正确的分钟数据
    h1 = D.agg_ohlcv(good, "1h")
    monkeypatch.setattr(D, "fetch_daily_1m", lambda d, market="futures/um": (good.copy(), {"url": "daily", "sha256": "x"}))
    monkeypatch.setattr(D, "fetch_aggtrades_1m", lambda d: (agg.copy(), {"url": "agg", "sha256": "y"}))
    rec = D.adjudicate_day(day, bad, h1)
    assert rec["verdict"] == "replace_with_daily_archive"
    assert rec["silent_traded_minutes"]["monthly"] == 40 and rec["silent_traded_minutes"]["daily"] == 0
    # 日度归档自身也有“零成交却有逐笔成交”的分钟 → 不替换
    monkeypatch.setattr(D, "fetch_daily_1m", lambda d, market="futures/um": (bad.copy(), {"url": "daily", "sha256": "x"}))
    rec2 = D.adjudicate_day(day, bad, h1)
    assert rec2["verdict"] == "unresolved_keep_monthly"


# ----------------------------------------------------------------------------
# 真实数据（缓存存在时）：钉住 Checkpoint 1 报告的核心数字
# ----------------------------------------------------------------------------
CACHE = ROOT / "data" / "processed_1m" / D.CACHE_FILES["perp_1m"]
RESULTS = ROOT / "reports" / "impulse_vs_box_round" / "cp1" / "cp1_results.json"


@pytest.mark.skipif(not CACHE.exists(), reason="1m cache not built (python -m impulse_study.data_1m --build)")
def test_real_data_cache_hash_and_headline_counts():
    assert all(v is True for v in D.verify().values())
    df = D.load_perp_1m()
    assert len(df) == 2_103_840 and int((df["src"] == 1).sum()) == 1440
    f = S.impulse_frame(D.minute_grid(df))
    c = S.conditions(f)
    base = S.eligible_window(f.index) & ~f["contaminated"].to_numpy()
    up, dn = (c["up1"] | c["up2"]) & base, (c["dn1"] | c["dn2"]) & base
    got = {"up_raw": int(up.sum()), "dn_raw": int(dn.sum()), "up_sched45": int(S.schedule(up, 45).sum()),
           "dn_sched45": int(S.schedule(dn, 45).sum())}
    if RESULTS.exists():
        r = json.loads(RESULTS.read_text())["counts"]["pct"]
        want = {"up_raw": r["up"]["IS"]["raw"]["total"], "dn_raw": r["dn"]["IS"]["raw"]["total"],
                "up_sched45": r["up"]["IS"]["sched45"]["total"], "dn_sched45": r["dn"]["IS"]["sched45"]["total"]}
        assert got == want
    assert got == {"up_raw": 27_835, "dn_raw": 27_466, "up_sched45": 5_930, "dn_sched45": 5_789}
