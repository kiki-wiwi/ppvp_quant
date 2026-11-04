"""
config.py — 策略 / 回测配置、样本切分与 OOS 隔离常量。

所有可调参数集中在两个不可变 dataclass 中：
* ``StrategyConfig``：4H PPVP 箱体、1H 状态机、K 线形态、止损止盈价位、可选过滤器（信号层）。
* ``BacktestConfig``：资金、费率、滑点、资金费率、仓位管理、出场管理（执行层）。

默认值即 Phase 2 的基线（Baseline）配置；可选模块默认关闭，留给 Phase 3 做单维度实验。
"""
from __future__ import annotations

import hashlib
import json
import math
import numbers
from dataclasses import asdict, dataclass, replace

import pandas as pd

# ---------------------------------------------------------------------------
# 样本切分（全部为 UTC，右开区间）
# ---------------------------------------------------------------------------
RESEARCH_START = pd.Timestamp("2020-01-01", tz="UTC")
IS_START, IS_END = RESEARCH_START, pd.Timestamp("2024-01-01", tz="UTC")       # 训练集 2020-01-01 ~ 2023-12-31
VAL_START, VAL_END = IS_END, pd.Timestamp("2025-07-01", tz="UTC")             # 验证集 2024-01-01 ~ 2025-06-30
OOS_START, OOS_END = VAL_END, pd.Timestamp("2026-11-01", tz="UTC")            # 盲测集 2025-07-01 ~ 2026-10-31
SEGMENTS = {"IS": (IS_START, IS_END), "VAL": (VAL_START, VAL_END), "OOS": (OOS_START, OOS_END)}

# 研究阶段（Phase 2/3）默认只把数据截断到 VAL_END 之前送入引擎（pipeline.run 是唯一受支持的研究入口），
# OOS K 线不参与任何绩效计算。唯一例外是 Phase 1 的防未来函数闸门（test_lookahead.py）：它按用户规格在
# 2020-2026 全量数据上做“截断 vs 全量”逐位相等检验与运行耗时基准，只输出通过/失败，不输出任何 OOS 统计量。
RESEARCH_DATA_END = VAL_END


@dataclass(frozen=True)
class StrategyConfig:
    # ---- 4H 枢轴点与成交量分布（PPVP）----
    pivot_left: int = 5                 # 枢轴左侧 K 线数 L
    pivot_right: int = 5                # 枢轴右侧 K 线数 R（确认滞后 = R 根 4H）
    profile_pivots: int = 2             # 箱体窗口覆盖的相邻已确认枢轴个数（2 = 最近一段摆动）
    n_bins: int = 50                    # 价格分桶数
    va_pct: float = 0.70                # Value Area 成交量占比

    # ---- 1H 指标 ----
    atr_len: int = 14                   # Wilder ATR 周期（1H）
    vol_sma_len: int = 20               # 放量判定的成交量均线周期（1H）

    # ---- 底部支撑带与两次测试判定 ----
    zone_k1: float = 0.5                # 支撑带下沿 = VAL - k1 * ATR_1H
    zone_k2: float = 0.5                # 支撑带上沿 = VAL + k2 * ATR_1H
    min_bars_between_touches: int = 6   # 两次测试最少间隔（1H 根数）
    min_bounce_atr: float = 1.0         # 或：第 1 次测试后最小反弹幅度（相对带上沿，ATR 倍数）
    carry_touches_on_box_update: bool = True   # 新箱体 VAL 与“本轮第 1 次测试所在箱体的 VAL”相距 ≤ (k1+k2)*ATR 时保留测试进度
    roll_failed_second_touch: bool = True      # 第 2 次测试无形态即离开时，将其滚动为新的“第 1 次测试”

    # ---- 箱体失效（放量实体跌穿）----
    break_body_atr: float = 0.6         # 阴线实体 ≥ x * ATR
    break_vol_mult: float = 1.5         # 成交量 ≥ x * 成交量均线

    # ---- 看涨 K 线形态 ----
    use_pinbar: bool = True
    pin_wick_frac: float = 0.60         # 下影线 ≥ 振幅 * x
    pinbar_ratio: float = 2.0           # 下影线 ≥ 实体 * x
    pin_close_pos: float = 0.50         # 收盘位置 (close-low)/range ≥ x（注意：x ≤ pin_wick_frac 时由下影线条件蕴含，不起约束作用）
    pin_require_bull_close: bool = False  # True：Pinbar 还须收阳（close > open）；规格未要求实体颜色，基线关闭
    use_engulfing: bool = True
    engulf_ratio: float = 1.0           # 阳线实体覆盖前一根阴线实体的比例（1.0 = 完全吞没）
    engulf_gap_tol: float = 0.0002      # 连续市场开盘价与前收盘价的 2bp 容差

    max_pattern_lag_bars: int | None = None  # 确认形态距第 2 次测试首根 K 线的最大根数（0 = 必须是重新进入支撑带的那一根；None = 第 2 次测试连续在带内期间任意一根）

    # ---- 止损 / 止盈价位 ----
    sl_buffer_atr: float = 0.2          # SL = min(第1次测试低点, 第2次测试锚点低点[, 吞没前一根低点]) - x * ATR
    sl_anchor: str = "test"             # 'test'：第 2 次测试全部 K 线的最低点（基线，不把止损放在本轮已打出的低点之上）；'pattern'：仅确认形态 K 线的低点（规格字面）
    min_rr: float = 0.0                 # 最小初始盈亏比 (VAH-Entry)/(Entry-SL)；基线仅要求 VAH > Entry

    # ---- 可选：箱体有效性过滤（基线关闭）----
    max_box_width_atr: float | None = None    # (VAH-VAL)/ATR_1H 上限
    max_box_width_pct: float | None = None    # (VAH-VAL)/VAL 上限
    reject_broken_at_activation: bool = False  # 箱体激活时价格已收在支撑带下方 → 直接视为失效

    # ---- 可选：宏观状态过滤（基线关闭）----
    regime_filter: str | None = None    # None | 'daily_sma' | 'crash_cooldown'
    regime_sma_days: int = 200
    regime_slope_days: int = 20
    crash_body_atr4: float = 2.0        # 4H 阴线实体 ≥ x * ATR_4H
    crash_vol_mult4: float = 2.0        # 且 4H 成交量 ≥ x * 均量
    crash_cooldown_hours: int = 48

    def __post_init__(self):
        floats = (self.va_pct, self.zone_k1, self.zone_k2, self.min_bounce_atr, self.break_body_atr, self.break_vol_mult,
                  self.pin_wick_frac, self.pinbar_ratio, self.pin_close_pos, self.engulf_ratio, self.engulf_gap_tol,
                  self.sl_buffer_atr, self.min_rr, self.crash_body_atr4, self.crash_vol_mult4)
        _require(all(_finite(x) for x in floats), "strategy parameters must be finite numbers")
        _require(self.pivot_left >= 0 and self.pivot_right >= 0, "pivot_left / pivot_right must be >= 0")
        _require(self.profile_pivots >= 1, "profile_pivots must be >= 1")
        _require(self.n_bins >= 2, "n_bins must be >= 2")
        _require(0.0 < self.va_pct <= 1.0, "va_pct must be in (0, 1]")
        _require(self.atr_len >= 1 and self.vol_sma_len >= 1, "atr_len / vol_sma_len must be >= 1")
        _require(self.zone_k1 >= 0.0 and self.zone_k2 >= 0.0, "zone_k1 / zone_k2 must be >= 0")
        _require(self.min_bars_between_touches >= 0 and self.min_bounce_atr >= 0.0, "touch separation must be >= 0")
        _require(self.break_body_atr >= 0.0 and self.break_vol_mult >= 0.0, "breakdown thresholds must be >= 0")
        _require(0.0 <= self.pin_wick_frac <= 1.0 and 0.0 <= self.pin_close_pos <= 1.0, "pinbar fractions must be in [0, 1]")
        _require(self.pinbar_ratio >= 0.0 and self.engulf_ratio > 0.0 and self.engulf_gap_tol >= 0.0, "pattern thresholds")
        _require(self.max_pattern_lag_bars is None or self.max_pattern_lag_bars >= 0, "max_pattern_lag_bars must be None or >= 0")
        _require(self.sl_anchor in ("test", "pattern"), "sl_anchor must be 'test' or 'pattern'")
        _require(self.sl_buffer_atr >= 0.0 and self.min_rr >= 0.0, "sl_buffer_atr / min_rr must be >= 0")
        _require(self.regime_filter in (None, "daily_sma", "crash_cooldown"), "regime_filter must be None/'daily_sma'/'crash_cooldown'")
        _require(self.regime_sma_days >= 1 and self.regime_slope_days >= 1, "regime_sma_days / regime_slope_days must be >= 1")
        _require(self.crash_body_atr4 >= 0.0 and self.crash_vol_mult4 >= 0.0 and self.crash_cooldown_hours >= 0,
                 "crash_* parameters must be >= 0")
        for w in (self.max_box_width_atr, self.max_box_width_pct):
            _require(w is None or (_finite(w) and w > 0.0), "max_box_width_* must be None or a finite number > 0")

    def key(self) -> str:
        return _hash(asdict(self))

    def structural_key(self) -> str:
        """决定箱体表的参数子集（用于缓存 PPVP 计算结果）。"""
        return _hash({k: getattr(self, k) for k in ("pivot_left", "pivot_right", "profile_pivots", "n_bins", "va_pct")})

    def with_(self, **kw) -> "StrategyConfig":
        return replace(self, **kw)


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 100_000.0

    # ---- 交易场所与成本 ----
    venue: str = "perp"                 # 'perp'：永续（期货费率 + 资金费率）；'spot'：现货等效 1x（无资金费）
    include_funding: bool = True        # 仅 venue='perp' 时生效：逐个结算时点按真实历史费率扣减/补贴
    perp_taker_fee: float = 0.0005      # 5 bp：市价入场、止损、市价离场
    perp_maker_fee: float = 0.0002      # 2 bp：限价止盈
    spot_taker_fee: float = 0.0010      # 现货 VIP0 默认 10 bp（可按实际等级修改）
    spot_maker_fee: float = 0.0010
    slippage: float = 0.0002            # 每笔成交单边 2 bp（对所有成交方向不利）

    # ---- 仓位管理 ----
    sizing: str = "fixed"               # 'fixed' | 'risk' | 'inv_vol'
    fixed_leverage: float = 1.0         # fixed：名义仓位 = 净值 * x
    risk_per_trade: float = 0.01        # risk：单笔止损风险占净值比例
    target_vol: float = 0.40            # inv_vol：年化目标波动率
    vol_lookback_bars: int = 720        # inv_vol：实现波动率回看 1H 根数（30 天）
    max_leverage: float = 1.5           # 名义杠杆硬上限（spot 模式强制 ≤ 1.0）
    compounding: bool = True            # False：以初始资金而非当前净值计算仓位

    # ---- 出场管理 ----
    tp_mode: str = "vah"                # 'vah'：VAH 全平；'poc_partial'：POC 分批 + 保本 + VAH 尾仓
    partial_frac: float = 0.5           # POC 平仓比例（1.0 = 全部在 POC 平仓，离场原因记为 'POC'）
    move_to_breakeven: bool = True      # POC 成交后剩余仓位止损上移至入场价
    time_stop_bars: int | None = None   # 持仓满 N 根 1H 仍未离场 → 收盘市价离场
    exit_on_breakdown: bool = False     # 持仓期间出现放量实体跌穿“入场时箱体”的支撑带下沿 → 收盘市价离场

    def __post_init__(self):
        floats = (self.initial_capital, self.perp_taker_fee, self.perp_maker_fee, self.spot_taker_fee, self.spot_maker_fee,
                  self.slippage, self.fixed_leverage, self.risk_per_trade, self.target_vol, self.max_leverage,
                  self.partial_frac)
        _require(all(_finite(x) for x in floats), "backtest parameters must be finite numbers")
        _require(self.venue in ("perp", "spot"), "venue must be 'perp' or 'spot'")
        _require(self.sizing in ("fixed", "risk", "inv_vol"), "sizing must be 'fixed', 'risk' or 'inv_vol'")
        _require(self.tp_mode in ("vah", "poc_partial"), "tp_mode must be 'vah' or 'poc_partial'")
        _require(0.0 < self.partial_frac <= 1.0, "partial_frac must be in (0, 1]")
        _require(self.initial_capital > 0.0, "initial_capital must be > 0")
        _require(min(self.perp_taker_fee, self.perp_maker_fee, self.spot_taker_fee, self.spot_maker_fee, self.slippage) >= 0.0,
                 "fees and slippage must be >= 0")
        _require(self.fixed_leverage > 0.0 and self.max_leverage > 0.0, "leverage parameters must be > 0")
        _require(self.risk_per_trade > 0.0 and self.target_vol > 0.0, "risk_per_trade / target_vol must be > 0")
        _require(self.vol_lookback_bars >= 2, "vol_lookback_bars must be >= 2")
        _require(self.time_stop_bars is None or self.time_stop_bars >= 1, "time_stop_bars must be None or >= 1")

    @property
    def taker_fee(self) -> float:
        return self.perp_taker_fee if self.venue == "perp" else self.spot_taker_fee

    @property
    def maker_fee(self) -> float:
        return self.perp_maker_fee if self.venue == "perp" else self.spot_maker_fee

    @property
    def funding_on(self) -> bool:
        return self.venue == "perp" and self.include_funding

    @property
    def leverage_cap(self) -> float:
        return self.max_leverage if self.venue == "perp" else min(1.0, self.max_leverage)

    def key(self) -> str:
        return _hash(asdict(self))

    def with_(self, **kw) -> "BacktestConfig":
        return replace(self, **kw)


def _finite(x) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, bool) and math.isfinite(float(x))


def _require(ok: bool, msg: str) -> None:
    if not ok:
        raise ValueError(msg)


def _hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


BASELINE_STRATEGY = StrategyConfig()
BASELINE_BACKTEST = BacktestConfig()
