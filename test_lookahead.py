"""
test_lookahead.py — 防未来函数与撮合真实性单元测试（pytest）。

核心测试 ``test_no_lookahead``：随机选取 5 个历史截断时点 T_cut，分别在“仅含 open_time ≤ T_cut 的截断数据”
与“2020-2026 全量数据”上运行 4H PPVP 箱体、1H 状态机信号与回测，断言 t ≤ T_cut 的全部
VAL/POC/VAH/支撑带/状态/中间量（形态、均量、宏观状态掩码、破位条件）/信号/止损止盈价位、
净值曲线与已平仓交易逐位（bitwise）一致。覆盖 BTC 与 ETH、基线配置与两套开启全部可选模块的配置。
``test_no_lookahead_event_cuts`` 在最容易泄露的时点（信号 K 线及其前一根、箱体确认前 1 小时/确认当时、
入场 K 线、持仓中途、宏观状态翻转、4H 未走完）追加截断，并在全量结果上校验执行不变量
（入场价 = 信号 K 线收盘 × (1+滑点)、入场 K 线不评估离场、资金费可由结算事件独立重算）。

OOS 隔离：截断点落在 OOS（≥ 2025-07-01）时只输出“通过/失败”，不输出任何信号/交易计数；
事件型截断点只取自研究区间（< 2025-07-01）。本文件不计算任何绩效指标。

运行：``python -m pytest test_lookahead.py -v -s``
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pipeline
from backtester import run_backtest
from config import OOS_END, OOS_START, RESEARCH_DATA_END, BacktestConfig, StrategyConfig
from data_loader import load_funding, load_klines, resample_ohlcv
from metrics import daily_equity, daily_returns, sharpe
from numba_compat import HAVE_NUMBA
from pipeline import spot_daily_warmup
from ppvp_engine import build_4h, build_boxes, detect_pivots, map_boxes_to_1h, to_ns
from strategy_state_machine import _run_state_machine, generate_signals, state_machine_args

ROOT = Path(__file__).resolve().parent
SEED = 20261104
SYMBOLS = ("BTCUSDT", "ETHUSDT")
FULL_END = pd.Timestamp("2026-11-01", tz="UTC")

CONFIGS = {
    "baseline": (StrategyConfig(), BacktestConfig()),
    "all_features_A": (
        StrategyConfig(pivot_left=8, pivot_right=3, profile_pivots=3, regime_filter="daily_sma", min_rr=1.0,
                       max_box_width_atr=8.0, max_box_width_pct=0.05, reject_broken_at_activation=True,
                       max_pattern_lag_bars=2, sl_anchor="pattern", pin_require_bull_close=True),
        BacktestConfig(sizing="risk", tp_mode="poc_partial", time_stop_bars=48, exit_on_breakdown=True)),
    "all_features_B": (
        StrategyConfig(regime_filter="crash_cooldown", roll_failed_second_touch=False,
                       carry_touches_on_box_update=False, va_pct=0.65, n_bins=40, engulf_ratio=0.8),
        BacktestConfig(sizing="inv_vol", venue="spot", compounding=False)),
}
_LOG: dict = {}


# ----------------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def markets():
    out = {}
    for sym in SYMBOLS:
        df = load_klines(sym, "1h", "perp")
        df = df[df.index < FULL_END]
        out[sym] = (df, load_funding(sym), spot_daily_warmup(sym))
    return out


@pytest.fixture(scope="module")
def market(markets):
    return markets["BTCUSDT"]


@pytest.fixture(scope="module", autouse=True)
def _write_log():
    yield
    if _LOG:
        out = ROOT / "reports" / "lookahead_test_log.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(_LOG, indent=2, ensure_ascii=False))


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


def _record(sym, name, t_cut, rec, tag):
    """OOS 截断点只记录通过/失败；研究区间截断点记录比较规模（均为计数，非绩效）。"""
    key = f"{tag}:{sym}"
    if t_cut >= OOS_START:
        _LOG.setdefault(key, []).append({"t_cut": str(t_cut), "config": name, "passed": True, "counts": "redacted (OOS)"})
        print(f"PASS {sym} {name:15s} T_cut={t_cut}  (OOS cut: counts redacted)")
    else:
        _LOG.setdefault(key, []).append(dict(rec, config=name, passed=True))
        print(f"PASS {sym} {name:15s} T_cut={t_cut}  boxes={rec['boxes_compared']:5d} bars={rec['bars_compared']:6d} "
              f"signals={rec['signals_compared']:4d} closed_trades={rec['closed_trades_compared']:4d}")


def recompute_funding(df1h, funding, res, bcfg) -> tuple[np.ndarray, np.ndarray]:
    """由结算事件独立重算每笔交易的资金费与结算次数（不调用回测器内部函数）。

    窗口：入场时刻 < T ≤ 结束时刻；SL/TP/BE/POC 结束于离场 K 线开盘，TIME/BREAKDOWN 结束于离场 K 线收盘，
    OPEN 结束于最后一根 K 线开盘。价格 = 第一根 open ≥ T 的开盘价（收盘类离场恰在 T 时为离场 K 线收盘价）；
    数量 = 入场数量，POC 分批之后（结算所在 K 线晚于分批 K 线）乘以 (1 - partial_frac)。
    """
    tr = res.trades
    out_f, out_n = np.zeros(len(tr)), np.zeros(len(tr), dtype=np.int64)
    if not bcfg.funding_on or len(tr) == 0:
        return out_f, out_n
    o_ns = to_ns(df1h.index)
    c_ns = o_ns + np.int64(3600 * 10**9)
    opens = df1h["open"].to_numpy()
    closes = df1h["close"].to_numpy()
    ft = to_ns(funding.index)
    fr = funding["funding_rate"].to_numpy()
    for k, t in enumerate(tr.itertuples(index=False)):
        e_t = c_ns[t.entry_bar]
        if t.exit_bar < 0:
            end_t, close_exit = o_ns[len(df1h) - 1], False
        elif t.exit_reason in ("TIME", "BREAKDOWN"):
            end_t, close_exit = c_ns[t.exit_bar], True
        else:
            end_t, close_exit = o_ns[t.exit_bar], False
        sel = np.flatnonzero((ft > e_t) & (ft <= end_t))
        for j in sel:
            T = ft[j]
            if close_exit and T == c_ns[t.exit_bar]:
                bar, px = t.exit_bar, closes[t.exit_bar]
            else:
                bar = int(np.searchsorted(o_ns, T, side="left"))
                px = opens[bar]
            q = t.qty * (1.0 - bcfg.partial_frac) if (t.partial_bar >= 0 and bar > t.partial_bar) else t.qty
            out_f[k] += q * px * fr[j]
            out_n[k] += 1
    return out_f, out_n


def assert_execution_invariants(df1h, funding, sig, res, bcfg, ctx):
    tr = res.trades
    if len(tr) == 0:
        return
    eb = tr["entry_bar"].to_numpy()
    exp = sig["entry_ref"].to_numpy()[eb] * (1.0 + bcfg.slippage)
    assert np.array_equal(tr["entry_px"].to_numpy().view(np.int64), exp.view(np.int64)), \
        f"{ctx}: entry fill must equal signal-bar close * (1 + slippage)"
    assert (sig["signal"].to_numpy()[eb]).all(), f"{ctx}: every entry must sit on a tradable signal bar"
    closed = tr["exit_bar"].to_numpy() >= 0
    assert (tr["exit_bar"].to_numpy()[closed] > eb[closed]).all(), f"{ctx}: exit evaluated on the entry bar"
    f, nf = recompute_funding(df1h, funding, res, bcfg)
    assert np.array_equal(nf, tr["n_funding"].to_numpy()), f"{ctx}: funding settlement count mismatch"
    assert np.allclose(f, tr["funding"].to_numpy(), rtol=1e-12, atol=1e-9), f"{ctx}: funding amount mismatch"


# ----------------------------------------------------------------------------
# 1. 核心：随机截断防未来函数测试
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("symbol", SYMBOLS)
def test_no_lookahead(markets, symbol):
    df, fund, spot_d = markets[symbol]
    rng = np.random.default_rng(SEED)
    pool = df.index[(df.index >= pd.Timestamp("2020-01-02", tz="UTC")) & (df.index < pd.Timestamp("2026-10-25", tz="UTC"))]
    picked = np.sort(rng.choice(to_ns(pool), size=5, replace=False))
    cuts = [pd.Timestamp(int(x), unit="ns", tz="UTC") for x in picked]
    _LOG[f"seeded_cuts:{symbol}"] = {"seed": SEED, "cuts": [str(c) for c in cuts]}
    for name, (scfg, bcfg) in CONFIGS.items():
        full_out = _pipeline(df, fund, spot_d, scfg, bcfg)
        assert_execution_invariants(df, fund, full_out[1], full_out[2], bcfg, f"[{symbol} {name} full]")
        for t_cut in cuts:
            trunc = df[df.index <= t_cut]
            trunc_out = _pipeline(trunc, fund, spot_d, scfg, bcfg)
            rec = compare_truncated_vs_full(full_out, trunc_out, df, t_cut, f"[{symbol} {name} @ {t_cut}]")
            _record(symbol, name, t_cut, rec, "random")


def _event_cuts(df, boxes, sig, res, scfg, n_sig=12, n_box=4, n_trade=4, n_flip=4, n_rand=10, seed=SEED):
    """研究区间（< OOS_START）内的事件型截断点：信号 K 线及其前一根、箱体确认前 1 小时与确认当时、
    入场 K 线、持仓中途、宏观状态翻转点及其前一根、一个 4H 未走完的时点，以及额外的随机点。"""
    h1 = pd.Timedelta(hours=1)
    lim = OOS_START - h1
    first_ok = df.index[0] + pd.Timedelta(days=2)

    def spaced(seq, k, when=None):
        when = seq if when is None else when
        seq = [x for x, t in zip(seq, when) if first_ok <= t <= lim]
        if not seq:
            return []
        idx = np.unique(np.linspace(0, len(seq) - 1, num=min(k, len(seq))).round().astype(int))
        return [seq[i] for i in idx]

    cuts = []
    for t in spaced(list(sig.index[sig["sig_raw"].to_numpy()]), n_sig):
        cuts += [t - h1, t]
    for t in spaced(list(boxes["available_time"]), n_box):
        cuts += [t - h1, t]                                    # 截断后最后收盘 = 确认前 1 小时 / 恰为确认时刻
    tr = res.trades
    tr = tr[tr["entry_time"] < OOS_START]
    for k in spaced(list(range(len(tr))), n_trade, when=list(tr["entry_time"])):
        eb = int(tr["entry_bar"].iloc[k])
        xb = int(tr["exit_bar"].iloc[k]) if tr["exit_bar"].iloc[k] >= 0 else eb + 2
        cuts += [df.index[eb], df.index[min((eb + max(xb, eb + 1)) // 2, len(df) - 1)]]
    if scfg.regime_filter is not None:
        rr = sig["regime_raw"].to_numpy()
        flips = list(sig.index[1:][rr[1:] != rr[:-1]])
        for t in spaced(flips, n_flip):
            cuts += [t - h1, t]
    cuts.append(pd.Timestamp("2023-06-14 13:00", tz="UTC"))     # 4H 只走完 2 根 1H
    cuts += [pd.Timestamp("2020-01-05 12:00", tz="UTC"),          # 三套配置均零交易（A 套在 01-05 还零箱体）：空表 schema 必须一致
             pd.Timestamp("2020-01-19 20:00", tz="UTC")]
    rng = np.random.default_rng(seed + 1)
    pool = df.index[(df.index >= first_ok) & (df.index <= lim)]
    cuts += [pd.Timestamp(int(x), unit="ns", tz="UTC") for x in rng.choice(to_ns(pool), size=n_rand, replace=False)]
    return sorted({t for t in cuts if first_ok <= t <= lim})


@pytest.mark.parametrize("symbol", SYMBOLS)
@pytest.mark.parametrize("name", list(CONFIGS))
def test_no_lookahead_event_cuts(markets, symbol, name):
    df, fund, spot_d = markets[symbol]
    scfg, bcfg = CONFIGS[name]
    full_out = _pipeline(df, fund, spot_d, scfg, bcfg)
    boxes_f, sig_f, res_f = full_out
    big = symbol == "BTCUSDT"
    cuts = _event_cuts(df, boxes_f, sig_f, res_f, scfg, n_sig=12 if big else 6, n_box=4 if big else 2,
                       n_trade=4 if big else 2, n_flip=4 if big else 2, n_rand=10 if big else 4)
    for t_cut in cuts:
        trunc_out = _pipeline(df[df.index <= t_cut], fund, spot_d, scfg, bcfg)
        rec = compare_truncated_vs_full(full_out, trunc_out, df, t_cut, f"[event {symbol} {name} @ {t_cut}]")
        _record(symbol, name, t_cut, rec, "event")
    print(f"PASS event cuts {symbol} {name}: {len(cuts)} cuts")


def test_research_pipeline_matches_full_period():
    """生产路径（pipeline.load_market/get_boxes/run，研究截断 + 资金费 ≤ end）与全量运行在研究末端之前逐位一致。只输出通过/失败。"""
    pipeline.clear_cache()
    for sym in SYMBOLS:
        for name in ("baseline", "all_features_A"):
            scfg, bcfg = CONFIGS[name]
            r = pipeline.run(sym, scfg, bcfg, end=RESEARCH_DATA_END, segments=())
            f = pipeline.run(sym, scfg, bcfg, end=OOS_END, segments=(), allow_oos=True)
            t_cut = r["df1h"].index[-1]
            compare_truncated_vs_full((f["boxes"], f["signals"], f["result"]), (r["boxes"], r["signals"], r["result"]),
                                      f["df1h"], t_cut, f"[pipeline {sym} {name}]")
            fund_r = pipeline.load_market(sym)[1]
            assert fund_r.index[-1] == RESEARCH_DATA_END, "research funding must include the settlement at end"
            print(f"PASS pipeline research-vs-full {sym} {name}")
    pipeline.clear_cache()


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
    # 现货不融资：名义 ≤ 净值 / (1 + taker)，买入后现金 = 净值 - 名义 - 手续费 ≥ 0
    assert t["leverage"] == pytest.approx(1.0 / 1.001, rel=1e-12)
    assert t["equity_at_entry"] - t["notional"] * 1.001 >= -1e-9
    exp_fees = t["notional"] * 0.001 + t["qty"] * t["exit_px"] * 0.001
    assert t["fees"] == pytest.approx(exp_fees, rel=1e-12)


def test_entry_fill_is_signal_bar_close_not_open():
    df = _mk_df([[100, 103, 99, 102], [102, 102.5, 101.5, 102.2], [102, 102.5, 101.5, 102.2]])
    res = run_backtest(df, _mk_sig(df, {0: (95.0, 110.0, 105.0)}), BacktestConfig(include_funding=False))
    assert res.trades.iloc[0]["entry_px"] == 102.0 * (1 + 0.0002)


def test_tp_gap_up_fills_at_limit_price():
    df = _mk_df([[100, 100.5, 99.5, 100], [107, 108, 106.5, 107.5]])
    res = run_backtest(df, _mk_sig(df, {0: (95.0, 105.0, 102.0)}), BacktestConfig(include_funding=False))
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "TP" and t["exit_px"] == pytest.approx(105.0 * (1 - 0.0002), rel=1e-15)


def _ramp_df(n=24, drop=None):
    rows = [[100.0 + k, 100.8 + k, 99.2 + k, 100.5 + k] for k in range(n)]      # open != close
    df = _mk_df(rows)
    if drop is not None:
        df = df.drop(df.index[drop])
    return df


def _fund3(rates):
    return pd.DataFrame({"funding_rate": list(rates), "funding_interval_hours": 8},
                        index=pd.DatetimeIndex(["2024-01-01 08:00", "2024-01-01 16:00", "2024-01-02 00:00"], tz="UTC"))


def test_funding_priced_at_settlement_bar_open_and_close_exit():
    r = (0.0001, -0.0002, 0.0003)
    df = _ramp_df()
    cfg = BacktestConfig(time_stop_bars=10)
    res = run_backtest(df, _mk_sig(df, {5: (50.0, 500.0, 400.0)}), cfg, _fund3(r))
    t = res.trades.iloc[0]
    # 08:00 结算按 08:00 那根的开盘价 108；15:00 K 线收盘(16:00)时间止损 → 承担 16:00 结算，按该根收盘价 115.5
    assert t["exit_reason"] == "TIME" and t["n_funding"] == 2
    assert t["funding"] == pytest.approx(t["qty"] * 108.0 * r[0] + t["qty"] * 115.5 * r[1], rel=1e-12)


def test_funding_settlement_inside_missing_bar_is_still_charged():
    r = (0.0001, -0.0002, 0.0003)
    df = _ramp_df(drop=8)                                         # 删除 08:00 的 1H（结算时点所在 K 线缺失）
    res = run_backtest(df, _mk_sig(df, {5: (50.0, 500.0, 400.0)}), BacktestConfig(time_stop_bars=6), _fund3(r))
    t = res.trades.iloc[0]
    assert t["n_funding"] == 1                                    # 顺延到 09:00 那根，按其开盘价 109 结算
    assert t["funding"] == pytest.approx(t["qty"] * 109.0 * r[0], rel=1e-12)


def test_funding_table_is_mandatory_and_must_cover_window():
    df = _ramp_df()
    sig = _mk_sig(df, {5: (50.0, 500.0, 400.0)})
    with pytest.raises(ValueError):
        run_backtest(df, sig, BacktestConfig())                   # 开启资金费却未传入资金费表
    gap = _fund3((0.0001, -0.0002, 0.0003)).iloc[[0, 2]]          # 缺少 16:00 结算
    with pytest.raises(ValueError):
        run_backtest(df, sig, BacktestConfig(), gap)
    run_backtest(df, sig, BacktestConfig(include_funding=False))  # 关闭资金费时不需要


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
        assert len(a) == 15
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
