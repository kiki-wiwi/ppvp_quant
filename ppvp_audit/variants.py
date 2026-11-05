"""
ppvp_audit/variants.py — 预注册的 9 个实验版本（A1、A2、A3、B1、B2、C1、D1、D2、E1）的实现。定义见
reports/audit_round/experiment_plan.md，参数在这里与计划逐字对应，运行时不得改动。

* A1 / A2 / A3 / E1：信号接受层的过滤（与基线已有的 min_rr / 宽度 / 宏观过滤器相同的语义：状态机照常发出并消费序列）。
* B1：状态机内的附加条件。为不改动生产状态机，由 strategy_state_machine._run_state_machine 的源码派生一个副本，
  只在“发出信号”的判定处多一个可开关的条件；关闭时与生产版本逐位一致（tests 钉住）。
* B2：Pinbar 三个阈值取 0、关闭吞没：第 2 次测试中第一根触及支撑带且收盘 ≥ 带下沿的 K 线即发信号
  （IS 内没有振幅为 0 的 1H K 线，与“去掉形态筛选”完全等价）。
* C1：同一批父信号的回撤限价单，由 ppvp_audit.engine_path 撮合。
* D1 / D2：backtester 的 poc_partial 出场（partial_frac = 0.5，D2 另开 move_to_breakeven）。
"""
from __future__ import annotations

import hashlib
import inspect
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

import strategy_state_machine as ssm
from config import BacktestConfig, StrategyConfig

from .engine_path import tick_size

RR_MIN_A1 = 1.5
POS_MAX_A2 = 0.5
ER_MAX_A3 = 0.35
C1_VALID_BARS = 3
D_PARTIAL_FRAC = 0.5
E_PRED_SLIP = 0.0002          # 预计成交价 = 信号收盘 × (1 + 2bp)；过滤是决策规则，成本压力情景下不随滑点改变
VERSIONS = ("A1", "A2", "A3", "B1", "B2", "C1", "D1", "D2", "E1")


@dataclass(frozen=True)
class Spec:
    version: str
    base_strategy_key: str
    base_backtest_key: str
    params: dict = field(default_factory=dict)

    def key(self) -> str:
        d = asdict(self)
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def spec_for(version: str, scfg: StrategyConfig, bcfg: BacktestConfig) -> Spec:
    p = {"A1": {"rr_pred_min": RR_MIN_A1, "entry_pred": "close*(1+slippage)"},
         "A2": {"pos_min": 0.0, "pos_max": POS_MAX_A2, "entry_pred": "close*(1+slippage)"},
         "A3": {"er_max": ER_MAX_A3, "er_window": "4H closes of pivot-a bar .. pivot-b bar inclusive", "zero_denominator": "ER=0"},
         "B1": {"low_below_first_test_low_by_ticks": 1, "close_ge": "max(VAL, first_test_low)", "tick": "0.01 before 2022-02-28 05:00 UTC, else 0.1"},
         "B2": {"pin_wick_frac": 0.0, "pinbar_ratio": 0.0, "pin_close_pos": 0.0, "use_engulfing": False},
         "C1": {"limit": "(high+low)/2 of the confirmation bar", "valid_bars": C1_VALID_BARS, "fill_rule": "through",
                "sensitivity": "touch", "fill_price": "L*(1+slippage)", "fee": "maker"},
         "D1": {"tp_mode": "poc_partial", "partial_frac": D_PARTIAL_FRAC, "move_to_breakeven": False},
         "D2": {"tp_mode": "poc_partial", "partial_frac": D_PARTIAL_FRAC, "move_to_breakeven": True},
         "E1": {"A1": True, "A2": True, "A3": True}}[version]
    return Spec(version, scfg.key(), bcfg.key(), p)


# ---------------------------------------------------------------------------
# 过滤类（A1 / A2 / A3 / E1）
# ---------------------------------------------------------------------------
def box_efficiency_ratio(boxes: pd.DataFrame, df4h: pd.DataFrame) -> np.ndarray:
    """每个箱体冻结分布窗口内 4H 收盘的效率比：枢轴 a 所在 4H 到枢轴 b 所在 4H（含两端）。分母为 0 时记 0。"""
    c4 = df4h["close"].to_numpy(dtype=np.float64)
    a = boxes["pivot_a_idx"].to_numpy(dtype=np.int64)
    b = boxes["pivot_b_idx"].to_numpy(dtype=np.int64)
    er = np.empty(len(boxes))
    for k in range(len(boxes)):
        seg = c4[a[k]: b[k] + 1]
        den = np.abs(np.diff(seg)).sum()
        er[k] = 0.0 if den == 0 else abs(seg[-1] - seg[0]) / den
    return er


def filter_masks(out: dict, bcfg: BacktestConfig) -> pd.DataFrame:
    """在每个信号 K 线上计算 A1 / A2 / A3 的通过与否（决策时已知的量）。"""
    sig, boxes, df4h = out["signals"], out["boxes"], out["df4h"]
    e = sig["entry_ref"].to_numpy(dtype=np.float64) * (1.0 + E_PRED_SLIP)
    vah, val, sl = (sig[k].to_numpy(dtype=np.float64) for k in ("VAH", "VAL", "sl"))
    with np.errstate(invalid="ignore", divide="ignore"):
        rr_pred = (vah - e) / (e - sl)
        pos = (e - val) / (vah - val)
    er_box = box_efficiency_ratio(boxes, df4h)
    bid = sig["box_id"].to_numpy(dtype=np.int64)
    er = np.where(bid >= 0, er_box[np.clip(bid, 0, None)], np.nan)
    with np.errstate(invalid="ignore"):
        m = pd.DataFrame({"rr_pred": rr_pred, "pos_pred": pos, "er": er,
                          "pass_A1": rr_pred >= RR_MIN_A1, "pass_A2": (pos >= 0.0) & (pos <= POS_MAX_A2),
                          "pass_A3": er <= ER_MAX_A3}, index=sig.index)
    m["pass_E1"] = m["pass_A1"] & m["pass_A2"] & m["pass_A3"]
    return m


def apply_filter(out: dict, bcfg: BacktestConfig, version: str) -> pd.DataFrame:
    m = filter_masks(out, bcfg)
    sig = out["signals"].copy()
    col = {"A1": "pass_A1", "A2": "pass_A2", "A3": "pass_A3", "E1": "pass_E1"}[version]
    blocked = sig["signal"].to_numpy() & ~m[col].to_numpy()
    sig[f"block_{version}"] = blocked
    sig["signal"] = sig["signal"].to_numpy() & m[col].to_numpy()
    for k in ("rr_pred", "pos_pred", "er"):
        sig[k] = m[k]
    return sig


# ---------------------------------------------------------------------------
# B1：状态机副本（只在“发出信号”的判定处多一个条件）
# ---------------------------------------------------------------------------
_B1_FN = None
_SIG_OLD = """def _run_state_machine(o, h, l, c, v, atr, vsma, open_ns, box_idx, box_val, box_vah, box_wstart,
                       pin, eng, k1, k2, min_bars, min_bounce, carry, roll,
                       brk_body, brk_vol, sl_buf, reject_broken, max_lag, sl_pattern):"""
_SIG_NEW = """def _run_state_machine_b1(o, h, l, c, v, atr, vsma, open_ns, box_idx, box_val, box_vah, box_wstart,
                       pin, eng, k1, k2, min_bars, min_bounce, carry, roll,
                       brk_body, brk_vol, sl_buf, reject_broken, max_lag, sl_pattern, b1_on, b1_tick):"""
_COND_OLD = """            if p > 0:
                if sl_pattern:"""
_COND_NEW = """            if p > 0 and ((not b1_on) or (l[i] <= t1_low - b1_tick[i] and c[i] >= max(box_val[cur], t1_low))):
                if sl_pattern:"""


def b1_state_machine():
    """返回 B1 状态机副本（numba 编译）。源码取自生产函数，两处文本替换必须各出现恰好一次。"""
    global _B1_FN
    if _B1_FN is None:
        from numba_compat import njit
        src = inspect.getsource(ssm._run_state_machine.py_func)
        assert src.startswith("@njit(cache=True)\n")
        src = src[len("@njit(cache=True)\n"):]
        for old, new in ((_SIG_OLD, _SIG_NEW), (_COND_OLD, _COND_NEW)):
            if src.count(old) != 1:
                raise RuntimeError("state machine source changed; the B1 patch site is not unique")
            src = src.replace(old, new)
        ns = {"np": np}
        exec(compile(src, "<b1_state_machine>", "exec"), ns)
        _B1_FN = njit(cache=False)(ns["_run_state_machine_b1"])
    return _B1_FN


@contextmanager
def b1_patched(df1h: pd.DataFrame, on: bool = True):
    """在 with 块内把 strategy_state_machine 的状态机替换为 B1 副本（on=False 时副本应与生产版本逐位一致）。"""
    fn = b1_state_machine()
    tick = tick_size(df1h.index).astype(np.float64)
    orig = ssm._run_state_machine

    def wrapped(*args):
        return fn(*args, bool(on), tick)
    wrapped.py_func = lambda *args: fn.py_func(*args, bool(on), tick)
    ssm._run_state_machine = wrapped
    try:
        yield
    finally:
        ssm._run_state_machine = orig


# ---------------------------------------------------------------------------
# B2 / D1 / D2 配置，C1 限价
# ---------------------------------------------------------------------------
def b2_strategy(scfg: StrategyConfig) -> StrategyConfig:
    return scfg.with_(pin_wick_frac=0.0, pinbar_ratio=0.0, pin_close_pos=0.0, pin_require_bull_close=False,
                      use_pinbar=True, use_engulfing=False)


def d_backtest(bcfg: BacktestConfig, move_be: bool) -> BacktestConfig:
    return bcfg.with_(tp_mode="poc_partial", partial_frac=D_PARTIAL_FRAC, move_to_breakeven=move_be)


def c1_limit_prices(df1h: pd.DataFrame, signals: pd.DataFrame) -> np.ndarray:
    h = df1h["high"].to_numpy(dtype=np.float64)
    l = df1h["low"].to_numpy(dtype=np.float64)
    return np.where(signals["signal"].to_numpy(), 0.5 * (h + l), np.nan)
