"""
test_lookahead.py — 防未来函数与撮合真实性单元测试（pytest）。

核心测试 ``test_no_lookahead``：随机选取 5 个历史截断时点 T_cut，分别在“仅含 open_time ≤ T_cut 的截断数据”
与“2020-2026 全量数据”上运行 4H PPVP 箱体、1H 状态机信号与回测，断言 t ≤ T_cut 的全部
VAL/POC/VAH/支撑带/状态/中间量（形态、均量、宏观状态掩码、破位条件）/信号/止损止盈价位、
净值曲线、逐根持仓离场价位与已平仓交易逐位（bitwise）一致。覆盖 BTC 与 ETH、基线配置与四套开启可选模块的配置。
``test_no_lookahead_event_cuts`` 在最容易泄露的时点（信号 K 线及其前一根、箱体确认前 1 小时/确认当时、
入场 K 线、持仓中途、按类别抽取的离场 K 线及其前一根（TIME / BREAKDOWN / 结算时刻收盘离场 / 分批 K 线上的收盘离场 /
SL / TP / BE / POC）、POC 分批 K 线、持仓期放量实体 K 线、宏观状态翻转、4H 未走完、数据开头的若干时点）追加截断，
并在全量结果上校验执行不变量：入场价 = 信号 K 线收盘 × (1+滑点)；交易表的止损/止盈/POC = 信号表在入场 K 线上的值；
资金费可由结算事件独立重算；逐根离场价位等于其文档定义的来源（信号止损 / 保本价 / VAL_入场 - k1*ATR_i）；
每笔交易的离场 K 线、原因与分批 K 线可由记录的价位与当根 OHLC 独立重推；箱体满足 P_low ≤ VAL ≤ POC ≤ VAH ≤ P_high。
截断比较看不见“第 i 根的价位读取了第 i 根自身的数据”（截断到 i 时两次运行都含第 i 根）：价位来源不变量在每根持仓 K 线上
覆盖这一类，``test_exit_levels_ignore_their_own_bar`` 再在抽样的持仓 K 线上把最后一根的高低收改为开盘价、成交量改为 0，
做与实现无关的交叉检验。只在箱体切换 K 线上做出的状态机决策，由 ``test_state_machine_prefix_at_every_box_switch``
在每个切换 K 线及其后一根上做内核级截断。

OOS 隔离：截断点落在 OOS（≥ 2025-07-01）时只输出“通过/失败”，不输出任何计数；事件型截断点只取自研究区间。
所有接收全量数据（含 OOS）的测试都经 ``_sealed`` 封装：失败时只抛出经过清理的 GateFailure（无局部变量与参数、
无断言展开、非断言异常不带消息），Ctrl-C 以不带回溯的新异常重新抛出，因此 pytest 的 -l / -vv / -vvv / --tb=long /
--fulltrace 都无法渲染持有 OOS 信号、交易、净值或原始行情的帧；行情 fixture 与 BacktestResult 的 repr 不显示数值。
本文件不计算任何绩效指标。

运行：``python -m pytest test_lookahead.py -v -s``
"""
from __future__ import annotations

import functools
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pipeline
from backtester import run_backtest
from config import HOLDOUT_SYMBOLS, OOS_END, OOS_START, RESEARCH_DATA_END, BacktestConfig, StrategyConfig
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
    # 专门覆盖收盘类离场路径（时间止损 / 入场箱体破位 / 结算时刻收盘离场的资金费 / POC 分批）
    "exits_C": (StrategyConfig(),
                BacktestConfig(exit_on_breakdown=True, time_stop_bars=12, tp_mode="poc_partial", move_to_breakeven=False)),
    # 全部仓位在 POC 平仓（partial_frac = 1，离场原因 POC）
    "exits_D": (StrategyConfig(),
                BacktestConfig(exit_on_breakdown=True, time_stop_bars=12, tp_mode="poc_partial", partial_frac=1.0)),
}
# 这些配置在研究区间内必须真的产生下列离场，且事件型截断必须在每类离场的 K 线及其前一根上各截断至少一次，
# 否则闸门对这些路径的“通过”就是空洞的
REQUIRED_EXIT_COVERAGE = {"exits_C": ("TIME", "BREAKDOWN", "settlement_close_exit", "poc_partial"),
                          "exits_D": ("POC", "TIME", "BREAKDOWN", "settlement_close_exit"),
                          "all_features_A": ("BE", "poc_partial")}
EXIT_CATEGORIES = ("TIME", "BREAKDOWN", "settlement_close_exit", "close_exit_on_partial_bar", "SL", "TP", "BE", "POC",
                   "poc_partial")
_LOG: dict = {}


class GateFailure(AssertionError):
    """闸门失败：只携带消息与不含局部变量的回溯文本。"""


def _sanitized(exc: BaseException) -> str:
    """失败文本：异常类型 + 无局部变量的回溯（只有源码行）+ 经过清理的消息。
    AssertionError 只保留 pytest 断言展开之前的部分（_fail 的上下文消息完整保留；裸 assert 对全量数组的展开会被去掉）；
    其他异常（pandas/numpy 报错可能含数据值）只保留类型，不保留消息。"""
    frames = "".join(traceback.format_tb(exc.__traceback__))
    if isinstance(exc, AssertionError):
        msg = str(exc).split("\nassert ")[0]
        msg = "" if msg.startswith("assert ") else msg
    else:
        msg = "(message suppressed: non-assertion error inside a full-data test)"
    return f"[{type(exc).__name__}] {msg}\n{frames}"


def _sealed(fn):
    """接触全量（含 OOS）数据的测试：任何失败都改为抛出 GateFailure(经过清理的文本)，并在包装器之外重新抛出，
    使 pytest 的 -l / -vv / -vvv / --tb=long / --fulltrace 都无法渲染持有 OOS 信号、交易或净值的帧。
    pytest 的 skip / xfail 原样抛出；Ctrl-C 与 SystemExit 以不带回溯的新异常重新抛出。"""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        kind, text = None, ""
        try:
            return fn(*args, **kwargs)
        except (pytest.skip.Exception, pytest.xfail.Exception):
            raise
        except KeyboardInterrupt:
            kind = "interrupt"
        except SystemExit as exc:
            kind, text = "exit", str(exc.code)
        except BaseException as exc:                   # noqa: BLE001 — 含 pytest 的 Failed
            kind, text = "fail", _sanitized(exc)
        args = kwargs = None                           # noqa: F841 — 不让本帧持有 fixture（全量行情）
        if kind == "interrupt":
            raise KeyboardInterrupt()
        if kind == "exit":
            raise SystemExit(text)
        raise GateFailure(text)
    return wrapper


class _SealedMarkets(dict):
    """全量行情（含 OOS）：repr 不显示内容。"""
    def __repr__(self) -> str:
        return f"<sealed market data: {sorted(self)}>"


class _SealedTuple(tuple):
    """单个品种的 (1H 行情, 资金费, 现货日线) 三元组：repr 不显示内容（--fulltrace 时 pytest 会渲染 fixture 值）。"""
    def __repr__(self) -> str:
        return "<sealed market data>"


# ----------------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def markets():
    out = _SealedMarkets()
    for sym in SYMBOLS:
        df = load_klines(sym, "1h", "perp")
        df = df[df.index < FULL_END]
        out[sym] = _SealedTuple((df, load_funding(sym), spot_daily_warmup(sym)))
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


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def assert_frames_bitwise(a: pd.DataFrame, b: pd.DataFrame, ctx: str) -> None:
    if list(a.columns) != list(b.columns):
        _fail(f"{ctx}: columns differ")
    if list(a.dtypes.astype(str)) != list(b.dtypes.astype(str)):
        _fail(f"{ctx}: dtypes differ")
    if len(a) != len(b):
        _fail(f"{ctx}: length {len(a)} vs {len(b)}")
    if not a.index.equals(b.index):
        _fail(f"{ctx}: index differs")
    for col in a.columns:
        eq = _bitwise_equal(_col_array(a[col]), _col_array(b[col]))
        if not eq.all():
            j = int(np.flatnonzero(~eq)[0])
            raise AssertionError(f"{ctx}: column '{col}' differs at row {j} ({a.index[j]}): {a[col].iloc[j]!r} vs {b[col].iloc[j]!r}")


def compare_truncated_vs_full(full_out, trunc_out, df_full, t_cut, ctx, redact: bool = False) -> dict:
    """redact=True（OOS 截断点）：任何异常都只报告“不一致”，不输出任何数值；返回值也只含截断时点（不含计数）。"""
    if not redact:
        return _compare(full_out, trunc_out, t_cut, ctx)
    try:
        _compare(full_out, trunc_out, t_cut, ctx)
        ok = True
    except Exception:                                  # noqa: BLE001 — 任何异常的消息都可能含 OOS 数值
        ok = False
    if not ok:
        raise AssertionError(f"{ctx}: mismatch (OOS cut, details redacted)")
    return {"t_cut": str(t_cut)}


def _compare(full_out, trunc_out, t_cut, ctx) -> dict:
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
    if not np.array_equal(res_f.equity.to_numpy()[:n].view(np.int64), res_t.equity.to_numpy().view(np.int64)):
        _fail(f"{ctx} equity")
    if not np.array_equal(res_f.position.to_numpy()[:n].view(np.int64), res_t.position.to_numpy().view(np.int64)):
        _fail(f"{ctx} position")
    # 逐根持仓离场价位（止损 / 破位参照位）：读取更晚 K 线的价位在任何持仓中的截断点都可见，而不只是恰好改变离场时。
    # 截断比较看不见“第 i 根的价位读取了第 i 根自身的数据”（截断到 i 时两边都含第 i 根），
    # 这由 test_exit_levels_ignore_their_own_bar 与全量运行上的价位来源不变量覆盖。
    assert_frames_bitwise(res_f.levels.iloc[:n], res_t.levels, f"{ctx} exit levels")
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
    """OOS 截断点只记录通过/失败（rec 只含截断时点）；研究区间截断点记录比较规模（均为计数，非绩效）。"""
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
    数量 = 入场数量；POC 分批在 K 线内成交，因此结算 K 线晚于分批 K 线、或收盘类离场恰在分批 K 线收盘时刻结算时，
    数量乘以 (1 - partial_frac)。
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
            at_close = close_exit and T == c_ns[t.exit_bar]
            if at_close:
                bar, px = t.exit_bar, closes[t.exit_bar]
            else:
                bar = int(np.searchsorted(o_ns, T, side="left"))
                px = opens[bar]
            after = t.partial_bar >= 0 and (bar > t.partial_bar or (at_close and bar == t.partial_bar))
            q = t.qty * (1.0 - bcfg.partial_frac) if after else t.qty
            out_f[k] += q * px * fr[j]
            out_n[k] += 1
    return out_f, out_n


def assert_execution_invariants(df1h, funding, sig, res, bcfg, ctx):
    tr = res.trades
    if len(tr) == 0:
        return
    eb = tr["entry_bar"].to_numpy()
    exp = sig["entry_ref"].to_numpy()[eb] * (1.0 + bcfg.slippage)
    # 显式 raise（不用裸 assert）：失败信息只含上下文，不含数值（全量运行含 OOS 数据）
    if not np.array_equal(tr["entry_px"].to_numpy().view(np.int64), exp.view(np.int64)):
        _fail(f"{ctx}: entry fill must equal signal-bar close * (1 + slippage)")
    if not (sig["signal"].to_numpy()[eb]).all():
        _fail(f"{ctx}: every entry must sit on a tradable signal bar")
    # 入场时冻结的止损 / 止盈 / POC 必须逐位等于信号表在入场 K 线上的值：否则用未来数据改写这些价位的泄露
    # 会被交易表“自洽地”带进价位来源与离场决策两条不变量
    for col_t, col_s in (("sl", "sl"), ("tp", "tp"), ("poc", "poc_tp")):
        exp_v = sig[col_s].to_numpy(dtype=np.float64)[eb]
        if not np.array_equal(tr[col_t].to_numpy(dtype=np.float64).view(np.int64), exp_v.view(np.int64)):
            _fail(f"{ctx}: trade '{col_t}' must equal the signal table's '{col_s}' on the entry bar")
    closed = tr["exit_bar"].to_numpy() >= 0
    if not (tr["exit_bar"].to_numpy()[closed] > eb[closed]).all():
        _fail(f"{ctx}: exit evaluated on the entry bar")
    f, nf = recompute_funding(df1h, funding, res, bcfg)
    if not np.array_equal(nf, tr["n_funding"].to_numpy()):
        _fail(f"{ctx}: funding settlement count mismatch")
    if not np.allclose(f, tr["funding"].to_numpy(), rtol=1e-12, atol=1e-9):
        _fail(f"{ctx}: funding amount mismatch")
    assert_level_provenance(sig, res, bcfg, ctx)
    assert_exit_decisions(df1h, sig, res, bcfg, ctx)


def expected_levels(sig, res, bcfg) -> tuple[np.ndarray, np.ndarray]:
    """逐根离场价位的文档定义（只用入场时字段与第 i 根开盘时已知的 k1*ATR_i）：
    stop = 入场止损，POC 分批之后的 K 线（move_to_breakeven）= 入场价；brk_level = VAL_入场 - k1*ATR_i（仅 exit_on_breakdown）；
    本根开始时空仓的 K 线为 NaN。持仓期间允许的价位变化只有这两种，新增动态（如移动止损）必须同时更新本函数。"""
    n = len(sig)
    stop, brk = np.full(n, np.nan), np.full(n, np.nan)
    sl_sig = sig["sl"].to_numpy(dtype=np.float64)
    if bcfg.exit_on_breakdown:
        val, k1 = sig["VAL"].to_numpy(), sig["k1_atr"].to_numpy()
    for t in res.trades.itertuples(index=False):
        e, p = int(t.entry_bar), int(t.partial_bar)
        last = int(t.exit_bar) if t.exit_bar >= 0 else n - 1
        for i in range(e + 1, last + 1):
            stop[i] = t.entry_px if (bcfg.move_to_breakeven and 0 <= p < i) else sl_sig[e]
            if bcfg.exit_on_breakdown:
                brk[i] = val[e] - k1[i]
    return stop, brk


def assert_level_provenance(sig, res, bcfg, ctx):
    stop, brk = expected_levels(sig, res, bcfg)
    for name, exp in (("stop", stop), ("brk_level", brk)):
        if not np.array_equal(res.levels[name].to_numpy().view(np.int64), exp.view(np.int64)):
            _fail(f"{ctx}: recorded exit level '{name}' differs from its documented source")


def assert_exit_decisions(df1h, sig, res, bcfg, ctx):
    """按引擎顺序（止损 → POC 分批 → 同根保本 → VAH 止盈 → 收盘类离场）由记录的价位与当根 OHLC/放量实体条件
    独立重推每笔交易的 (离场 K 线, 原因, 分批 K 线)。离场判定若读取了记录值以外的信息（例如更晚的 ATR），即使价位
    记录正确也会在这里暴露。"""
    o, h, l, c = (df1h[k].to_numpy() for k in ("open", "high", "low", "close"))
    n = len(df1h)
    st, bk = res.levels["stop"].to_numpy(), res.levels["brk_level"].to_numpy()
    bv = sig["brk_bodyvol"].to_numpy(dtype=bool) if "brk_bodyvol" in sig.columns else np.zeros(n, dtype=bool)
    ts = bcfg.time_stop_bars or -1
    poc_mode = bcfg.tp_mode == "poc_partial"
    tp_sig, poc_sig = sig["tp"].to_numpy(dtype=np.float64), sig["poc_tp"].to_numpy(dtype=np.float64)
    for t in res.trades.itertuples(index=False):
        e = int(t.entry_bar)
        tp_e, poc_e = tp_sig[e], poc_sig[e]
        partial, got = -1, (-1, "OPEN")
        poc_ok = poc_mode and poc_e > t.entry_px and poc_e < tp_e
        for i in range(e + 1, n):
            if l[i] <= st[i]:
                got = (i, "BE" if (partial >= 0 and bcfg.move_to_breakeven) else "SL")
                break
            if poc_ok and partial < 0 and h[i] >= poc_e:
                partial = i
                if bcfg.partial_frac >= 1.0:
                    got = (i, "POC")
                    break
                if bcfg.move_to_breakeven and l[i] <= t.entry_px:
                    got = (i, "BE")
                    break
            if h[i] >= tp_e:
                got = (i, "TP")
                break
            ts_hit = ts > 0 and (i - e) >= ts
            if ts_hit or (bcfg.exit_on_breakdown and bv[i] and c[i] < bk[i]):
                got = (i, "TIME" if ts_hit else "BREAKDOWN")
                break
        if got != (int(t.exit_bar), t.exit_reason) or partial != int(t.partial_bar):
            _fail(f"{ctx}: an exit decision cannot be re-derived from the recorded levels and that bar's own data")


def assert_box_invariants(boxes, ctx):
    ok = (boxes["P_low"] <= boxes["VAL"]) & (boxes["VAL"] <= boxes["POC"]) & (boxes["POC"] <= boxes["VAH"]) \
        & (boxes["VAH"] <= boxes["P_high"])
    if not bool(ok.all()):
        _fail(f"{ctx}: box violates P_low <= VAL <= POC <= VAH <= P_high")


def exit_event_bars(df, res) -> dict:
    """研究区间（事件 K 线收盘 ≤ OOS_START）内各类离场事件所在的 K 线（按时间排序）。"""
    tr = res.trades
    tr = tr[(tr["exit_bar"] >= 0) & (tr["exit_time"] <= OOS_START)]
    reason = tr["exit_reason"]
    close_ex = reason.isin(["TIME", "BREAKDOWN"])
    masks = {"TIME": reason == "TIME", "BREAKDOWN": reason == "BREAKDOWN",
             "settlement_close_exit": close_ex & tr["exit_time"].dt.hour.isin([0, 8, 16]),
             "close_exit_on_partial_bar": close_ex & (tr["partial_bar"] == tr["exit_bar"]),
             "SL": reason == "SL", "TP": reason == "TP", "BE": reason == "BE", "POC": reason == "POC"}
    out = {k: sorted(int(x) for x in tr.loc[m, "exit_bar"]) for k, m in masks.items()}
    pb = res.trades["partial_bar"].to_numpy()
    out["poc_partial"] = sorted(int(x) for x in pb[pb >= 0] if df.index[int(x)] + pd.Timedelta(hours=1) <= OOS_START)
    return out


def assert_exit_coverage(df1h, res, name):
    """研究区间内必须真的出现所要求的离场类型（只统计计数是否 ≥ 1，不输出任何数值）。"""
    need = REQUIRED_EXIT_COVERAGE.get(name)
    if not need:
        return
    ev = exit_event_bars(df1h, res)
    missing = [k for k in need if not ev[k]]
    if missing:
        _fail(f"[{name}] research window lacks required exit paths: {missing}")


# ----------------------------------------------------------------------------
# 1. 核心：随机截断防未来函数测试
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("symbol", SYMBOLS)
@_sealed
def test_no_lookahead(markets, symbol):
    df, fund, spot_d = markets[symbol]
    seed = SEED + SYMBOLS.index(symbol)
    rng = np.random.default_rng(seed)
    pool = df.index[(df.index >= pd.Timestamp("2020-01-02", tz="UTC")) & (df.index < pd.Timestamp("2026-10-25", tz="UTC"))]
    picked = np.sort(rng.choice(to_ns(pool), size=5, replace=False))
    cuts = [pd.Timestamp(int(x), unit="ns", tz="UTC") for x in picked]
    _LOG[f"seeded_cuts:{symbol}"] = {"seed": seed, "cuts": [str(c) for c in cuts]}
    for name, (scfg, bcfg) in CONFIGS.items():
        full_out = _pipeline(df, fund, spot_d, scfg, bcfg)
        assert_box_invariants(full_out[0], f"[{symbol} {name} full]")
        assert_execution_invariants(df, fund, full_out[1], full_out[2], bcfg, f"[{symbol} {name} full]")
        for t_cut in cuts:
            trunc = df[df.index <= t_cut]
            trunc_out = _pipeline(trunc, fund, spot_d, scfg, bcfg)
            rec = compare_truncated_vs_full(full_out, trunc_out, df, t_cut, f"[{symbol} {name} @ {t_cut}]",
                                            redact=t_cut >= OOS_START)
            _record(symbol, name, t_cut, rec, "random")


def _event_cuts(df, boxes, sig, res, scfg, n_sig=12, n_box=4, n_trade=4, n_flip=4, n_rand=10, n_cat=2, n_brk=6,
                seed=SEED):
    """研究区间（< OOS_START）内的事件型截断点：信号 K 线及其前一根、箱体确认前 1 小时与确认当时、
    入场 K 线、持仓中途、按类别抽取的离场 K 线及其前一根（每类至多 n_cat 个事件）、POC 分批 K 线及其前一根、
    持仓期间满足放量实体条件的 K 线、宏观状态翻转点及其前一根、一个 4H 未走完的时点、零交易的早期时点、
    数据首日的若干时点（预热与不足一根 4H 的边界），以及额外的随机点。
    （只在实际离场的 K 线截断，查不出“压制离场”的泄露，因此还要在持仓中的放量实体 K 线截断。）
    返回 (cuts, coverage)，coverage[类别] = (在事件 K 线截断的次数, 在其前一根截断的次数)。"""
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
    events = exit_event_bars(df, res)
    picked = {}
    for cat in EXIT_CATEGORIES:
        bars = [b for b in events[cat] if b >= 1]
        picked[cat] = spaced(bars, n_cat, when=[df.index[b] for b in bars])
        for b in picked[cat]:
            cuts += [df.index[b - 1], df.index[b]]
    pos = res.position.to_numpy() > 0
    held = np.zeros(len(df), dtype=bool)
    held[1:] = pos[:-1]                                         # 第 i 根开始时持仓
    brk_bars = list(df.index[held & sig["brk_bodyvol"].to_numpy()])
    for t in spaced(brk_bars, n_brk):
        cuts += [t - h1, t]
    if scfg.regime_filter is not None:
        rr = sig["regime_raw"].to_numpy()
        flips = list(sig.index[1:][rr[1:] != rr[:-1]])
        for t in spaced(flips, n_flip):
            cuts += [t - h1, t]
    cuts.append(pd.Timestamp("2023-06-14 13:00", tz="UTC"))     # 4H 只走完 2 根 1H
    cuts += [pd.Timestamp("2020-01-05 12:00", tz="UTC"),          # 三套配置均零交易（A 套在 01-05 还零箱体）：空表 schema 必须一致
             pd.Timestamp("2020-01-19 20:00", tz="UTC")]
    rng = np.random.default_rng(seed + 101)
    pool = df.index[(df.index >= first_ok) & (df.index <= lim)]
    cuts += [pd.Timestamp(int(x), unit="ns", tz="UTC") for x in rng.choice(to_ns(pool), size=n_rand, replace=False)]
    out = {t for t in cuts if first_ok <= t <= lim}
    # 数据首日：日线预热（尚无完整永续日线）、不足一根 4H、刚好走完第一根 4H 等边界
    out |= {df.index[0] + pd.Timedelta(hours=k) for k in (0, 2, 3, 5, 22, 23, 26)}
    cut_set = out
    coverage = {cat: (sum(df.index[b] in cut_set for b in picked[cat]),
                      sum(df.index[b - 1] in cut_set for b in picked[cat])) for cat in EXIT_CATEGORIES}
    return sorted(out), coverage


@pytest.mark.parametrize("symbol", SYMBOLS)
@pytest.mark.parametrize("name", list(CONFIGS))
@_sealed
def test_no_lookahead_event_cuts(markets, symbol, name):
    df, fund, spot_d = markets[symbol]
    scfg, bcfg = CONFIGS[name]
    full_out = _pipeline(df, fund, spot_d, scfg, bcfg)
    boxes_f, sig_f, res_f = full_out
    assert_exit_coverage(df, res_f, name)
    big = symbol == "BTCUSDT"
    cuts, coverage = _event_cuts(df, boxes_f, sig_f, res_f, scfg, n_sig=12 if big else 6, n_box=4 if big else 2,
                                 n_trade=4 if big else 2, n_flip=4 if big else 2, n_rand=10 if big else 4,
                                 n_cat=2, n_brk=6 if big else 3, seed=SEED + SYMBOLS.index(symbol))
    lacking = [c for c in REQUIRED_EXIT_COVERAGE.get(name, ()) if min(coverage[c]) < 1]
    if lacking:
        _fail(f"[{symbol} {name}] event cuts miss the bar or the bar before for required exit paths: {lacking}")
    _LOG.setdefault(f"event_exit_coverage:{symbol}", {})[name] = {k: list(v) for k, v in coverage.items()}
    for t_cut in cuts:
        trunc_out = _pipeline(df[df.index <= t_cut], fund, spot_d, scfg, bcfg)
        rec = compare_truncated_vs_full(full_out, trunc_out, df, t_cut, f"[event {symbol} {name} @ {t_cut}]")
        _record(symbol, name, t_cut, rec, "event")
    print(f"PASS event cuts {symbol} {name}: {len(cuts)} cuts")


@_sealed
def test_research_pipeline_matches_full_period():
    """生产路径（pipeline.load_market/get_boxes/run，研究截断 + 资金费 ≤ end）与全量运行在研究末端之前逐位一致。只输出通过/失败。"""
    pipeline.clear_cache()
    for sym in SYMBOLS:
        for name in ("baseline", "all_features_A"):
            scfg, bcfg = CONFIGS[name]
            # ETH 是留出标的：不带 allow_oos 的结构性运行不出回测结果；闸门是只输出通过/失败的既定例外
            r = pipeline.run(sym, scfg, bcfg, end=RESEARCH_DATA_END, segments=(), allow_oos=sym in HOLDOUT_SYMBOLS)
            f = pipeline.run(sym, scfg, bcfg, end=OOS_END, segments=(), allow_oos=True)
            t_cut = r["df1h"].index[-1]
            compare_truncated_vs_full((f["boxes"], f["signals"], f["result"]), (r["boxes"], r["signals"], r["result"]),
                                      f["df1h"], t_cut, f"[pipeline {sym} {name}]")
            fund_r = pipeline.load_market(sym)[1]
            assert fund_r.index[-1] == RESEARCH_DATA_END, "research funding must include the settlement at end"
            print(f"PASS pipeline research-vs-full {sym} {name}")
    pipeline.clear_cache()


@pytest.mark.parametrize("symbol", SYMBOLS)
@pytest.mark.parametrize("name", ["all_features_A", "exits_C"])
@_sealed
def test_exit_levels_ignore_their_own_bar(markets, symbol, name):
    """第 i 根的离场价位在第 i 根开盘时记录，不得依赖第 i 根自身的 OHLCV：在持仓中的截断点 j，把第 j 根的
    高/低/收改为开盘价、成交量改为 0 后重跑，第 j 根的价位必须与原值逐位一致。截断点取自研究区间：
    最高价触及 POC 的持仓 K 线（保本上移的触发点）、POC 分批 K 线、放量实体 K 线、入场后第一根与均匀抽取的持仓 K 线。
    这是抽样检查：入场时冻结的止盈 / POC 不在逐根价位中，由全量运行上“交易表价位 = 信号表价位”的不变量覆盖。只用研究区间数据。"""
    df_all, fund_all, spot_d = markets[symbol]
    df = df_all[df_all.index < OOS_START]
    fund = fund_all[fund_all.index <= OOS_START]
    scfg, bcfg = CONFIGS[name]
    _, sig, res = _pipeline(df, fund, spot_d, scfg, bcfg)
    n = len(df)
    hi = df["high"].to_numpy()
    held = np.isfinite(res.levels["stop"].to_numpy())
    poc_touch = np.zeros(n, dtype=bool)
    for t in res.trades.itertuples(index=False):
        last = int(t.exit_bar) if t.exit_bar >= 0 else n - 1
        for i in range(int(t.entry_bar) + 1, last + 1):
            poc_touch[i] |= hi[i] >= t.poc
    pb = res.trades["partial_bar"].to_numpy()
    first_held = res.trades["entry_bar"].to_numpy() + 1                 # 入场后第一根：入场时冻结价位的泄露在这里出现
    groups = (np.flatnonzero(held & poc_touch), pb[pb >= 0],
              np.flatnonzero(held & sig["brk_bodyvol"].to_numpy(dtype=bool)), np.flatnonzero(held),
              first_held[first_held < n])
    cuts = set()
    for g, k in zip(groups, (10, 6, 6, 8, 8)):
        g = np.asarray(sorted(int(x) for x in g if x > 48))
        if len(g):
            cuts |= {int(x) for x in g[np.unique(np.linspace(0, len(g) - 1, num=min(k, len(g))).round().astype(int))]}
    assert cuts, "no in-position bars to test"
    ref = res.levels.to_numpy()
    for j in sorted(cuts):
        m = df.iloc[: j + 1].copy()
        o = m["open"].iloc[-1]
        for col in ("high", "low", "close"):
            m.iloc[-1, m.columns.get_loc(col)] = o
        m.iloc[-1, m.columns.get_loc("volume")] = 0.0
        m.iloc[-1, m.columns.get_loc("quote_volume")] = 0.0
        got = _pipeline(m, fund, spot_d, scfg, bcfg)[2].levels.to_numpy()[-1]
        if not np.array_equal(got.view(np.int64), ref[j].view(np.int64)):
            _fail(f"[{symbol} {name} @ {df.index[j]}] exit levels of a bar depend on that bar's own data")
    _LOG.setdefault(f"masked_last_bar:{symbol}", {})[name] = len(cuts)
    print(f"PASS masked-last-bar {symbol} {name}: {len(cuts)} in-position cuts")


@pytest.mark.parametrize("symbol", SYMBOLS)
@pytest.mark.parametrize("name", ["baseline", "all_features_A", "all_features_B"])
@_sealed
def test_state_machine_prefix_at_every_box_switch(markets, symbol, name):
    """只在箱体切换时做出的决策（认领与失效“须先到带上方”要求、激活条件 (a)、放量跌穿记忆的阻断与失效、carry、否决）
    只有截断在切换 K 线或其后一根时才可能暴露泄露，抽样截断点几乎不会落在那里。这里在研究区间的每一个切换 K 线 t
    及 t+1 上做状态机内核级截断：逐根输入截到 t、箱体只保留第 t 根收盘时已确认的，15 个输出必须与全长运行逐位一致。"""
    df_all, _, _ = markets[symbol]
    df = df_all[df_all.index < OOS_START]
    scfg, _ = CONFIGS[name]
    boxes, _ = build_boxes(df, scfg)
    args = list(state_machine_args(df, scfg, boxes))
    full = _run_state_machine(*args)
    n = len(df)
    open_ns, bidx = args[7], args[8]
    avail = to_ns(boxes["available_time"])
    per_bar = (0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13)                 # o h l c v atr vsma open_ns box_idx pin eng
    per_box = (9, 10, 11)                                          # VAL VAH window_start
    switches = np.flatnonzero(np.diff(bidx) != 0) + 1
    cuts = sorted({int(x) for s in switches for x in (s, s + 1) if x < n - 1})
    assert len(cuts) > 1000, "too few box switches to make the check meaningful"
    for t in cuts:
        nb_t = int(np.searchsorted(avail, open_ns[t] + np.int64(3600 * 10**9), side="right"))
        a = list(args)
        for k in per_bar:
            a[k] = args[k][: t + 1]
        for k in per_box:
            a[k] = args[k][:nb_t]
        part = _run_state_machine(*a)
        for x, y in zip(full, part):
            if not np.array_equal(np.asarray(x)[: t + 1].view(np.uint8), np.asarray(y).view(np.uint8)):
                _fail(f"[{symbol} {name}] state-machine output at the box switch {df.index[t]} depends on later data")
    _LOG.setdefault(f"switch_prefix_cuts:{symbol}", {})[name] = len(cuts)
    print(f"PASS switch-prefix {symbol} {name}: {len(cuts)} kernel-level cuts")


# ----------------------------------------------------------------------------
# 2. 时间对齐：4H 绝对 UTC 边界、维护缺口、可见时间
# ----------------------------------------------------------------------------
@_sealed
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


@_sealed
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


@_sealed
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


def test_funding_recompute_close_exit_on_partial_bar():
    """收盘类离场与 POC 分批发生在同一根 K 线、且该 K 线收盘恰为结算时刻：分批在 K 线内先成交，只按剩余数量计资金费。
    同时验证闸门的独立重算（recompute_funding）在这一组合下与引擎一致。"""
    rows = [[100.0, 100.5, 99.5, 100.0]] * 7 + [[100.0, 102.5, 99.9, 100.2]] + [[100.2, 100.5, 99.9, 100.2]] * 4
    df = _mk_df(rows)
    sig = _mk_sig(df, {5: (90.0, 110.0, 102.0)})
    sig["entry_ref"] = np.where(sig["signal"], df["close"], np.nan)
    fund = pd.DataFrame({"funding_rate": [0.002, 0.001], "funding_interval_hours": 8},
                        index=pd.DatetimeIndex(["2024-01-01 08:00", "2024-01-01 16:00"], tz="UTC"))
    cfg = BacktestConfig(tp_mode="poc_partial", partial_frac=0.5, move_to_breakeven=False, time_stop_bars=2)
    res = run_backtest(df, sig, cfg, fund)
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "TIME" and t["exit_bar"] == 7 and t["partial_bar"] == 7 and t["n_funding"] == 1
    assert t["funding"] == pytest.approx(t["qty"] * 0.5 * 100.2 * 0.002, rel=1e-12)
    assert_execution_invariants(df, fund, sig, res, cfg, "[synthetic partial+close exit]")


def test_risk_sizing_leverage_cap():
    df = _mk_df([[100, 100.5, 99.5, 100]] * 5)
    res = run_backtest(df, _mk_sig(df, {0: (99.9, 110.0, 105.0)}),
                       BacktestConfig(sizing="risk", risk_per_trade=0.01, max_leverage=1.5, include_funding=False))
    assert res.trades.iloc[0]["leverage"] == pytest.approx(1.5, rel=1e-12)


# ----------------------------------------------------------------------------
# 4. JIT 与纯 Python 路径逐位一致；日度夏普口径
# ----------------------------------------------------------------------------
@pytest.mark.skipif(not HAVE_NUMBA, reason="numba not installed")
@_sealed
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


def test_sealed_failures_carry_no_frames_or_values():
    """第四轮 EX4-3 / EX4-5：被封装的测试失败时，抛出的异常不含内层帧、局部变量、断言展开或非断言异常的消息；
    Ctrl-C 以不带回溯的新异常重新抛出；pytest.skip 原样通过。"""
    def frames_of(tb):
        names = []
        while tb is not None:
            names.append(tb.tb_frame.f_code.co_name)
            tb = tb.tb_next
        return names

    @_sealed
    def bare_assert(arg):
        secret = np.arange(5) + 123456
        assert secret.sum() < 0, "context only"

    with pytest.raises(GateFailure) as ei:
        bare_assert(_SealedTuple((1, 2)))
    text = str(ei.value)
    assert text.startswith("[AssertionError] context only") and "123456" not in text and "bare_assert" not in frames_of(ei.tb)

    @_sealed
    def value_error():
        raise ValueError("2026-07-01 00:00 value " + str(40 + 2.123))     # 数值只出现在运行时消息里，不在源码行里

    with pytest.raises(GateFailure) as ei:
        value_error()
    assert str(ei.value).startswith("[ValueError]") and "42.123" not in str(ei.value)

    @_sealed
    def interrupted():
        secret = np.arange(3) + 777                                        # noqa: F841
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt) as ei:
        interrupted()
    assert ei.value.__context__ is None and "interrupted" not in frames_of(ei.tb)

    @_sealed
    def skipped():
        pytest.skip("not applicable")

    with pytest.raises(pytest.skip.Exception):
        skipped()
    assert repr(_SealedTuple((1, 2))) == "<sealed market data>"


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
@_sealed
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
