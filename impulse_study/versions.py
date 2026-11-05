"""
impulse_study/versions.py — Checkpoint 2 冻结的 10 个预注册版本、成本情景、仓位口径与全部常数。

本文件与 reports/impulse_vs_box_round/experiment_plan_impulse.md 一一对应；任何改动都会改变 ``frozen_key()``，
实验计划记录了冻结时的键值。第 3 阶段只按这里的定义运行，不得增删版本或改动参数。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import pandas as pd

from config import IS_END

# ----------------------------------------------------------------------------
# 样本窗口与初始状态
# ----------------------------------------------------------------------------
MAIN_START = pd.Timestamp("2020-03-01", tz="UTC")              # 主比较窗口起点（统一空仓、零冷却、初始资金）
MAIN_END = IS_END                                              # 右开
LAST_TRIGGER = IS_END - pd.Timedelta(minutes=31)               # 触发 K 线 T ≤ IS_END − 31min（最长 30 分钟持仓在窗口内）
INITIAL_CAPITAL = 100_000.0

# ----------------------------------------------------------------------------
# 冻结常数
# ----------------------------------------------------------------------------
SIGMA_REF = 0.0008088465199541001     # CP1 冻结：IS 可交易分钟 σ7d 的中位数（显示为 0.0008088465）；Z 口径为全 IS 波动标定
ATR15_LEN = 14                        # ATR_15m(14)：UTC 自然整刻 15m 已收盘棒，前 14 个有效 TR 的算术均值初始化，其后 Wilder
SL_A_MULT = 2.5                       # SL_A = E_ref ∓ 2.5·ATR_15m
SL_B_BUFFER = 0.1                     # SL_B = 脉冲起点极值 ∓ 0.1·ATR_15m
SL_C_FRAC = 0.5                       # SL_C = C_t ∓ 0.5·|C_t − 脉冲起点极值|
W_IND1, W_IND2 = 5, 15                # 只满足指标 ① → W = 5；满足指标 ②（含 ① + ②）→ W = 15
RISK_FRACTION = 0.005                 # 以损定仓：单笔风险预算 = 0.5% 净值
MIN_DIST_FRAC = 0.0015                # 仓位分母中止损距离的下限（只限制数量，不移动 SL）
LEVERAGE_CAP = 2.0                    # 以损定仓的名义上限（按 E_ref 计）
EQUAL_NOTIONAL = 1.0                  # 等名义对照口径
V1_MIN, K1_MIN = 1.3, 0.10            # 第 1 根合成 15m 棒满时的量能延续条件
BASELINE_WINDOWS, BAR_MIN = 20, 15    # 基准成交量：入场前紧邻的 20 个不重叠 15 分钟窗口 [T−299min, T+1min) 的中位数
MAX_BARS = 2                          # 合成 15m 持仓棒上限（30 分钟）
COOLDOWN_MIN = 15                     # 平仓后冷却 15 分钟：下一次入场时刻 ≥ 离场时刻 + 15min
SEED_BASE = 20261106                  # 第 3 / 4 阶段全部随机抽样的主种子（用途偏移见实验计划第 8 节）
BOX_STATE_VARIANT = "S6_reentry_above_poc"   # A 组主口径：daily.build_box_states 的 S6（features.s6_box_ok 只认这一口径）

# Checkpoint 1 表 5b：官方 1m K 线把该分钟末尾成交漏记的 10 个分钟（UTC 开盘时刻）。只用于打标 has_archive_gap /
# signal_lookback_has_archive_gap，不修补价格；解锁后真实上下文必须恰好标记这 10 个分钟（engine._require_real_inputs）。
ARCHIVE_GAP_MINUTES = ("2021-01-12 06:10", "2021-05-15 00:23", "2022-06-18 20:07", "2022-06-22 11:28",
                       "2022-06-23 19:09", "2022-06-24 11:31", "2022-07-04 11:54", "2022-07-16 05:59",
                       "2022-08-19 00:21", "2022-08-21 08:10")


@dataclass(frozen=True)
class CostScenario:
    name: str
    fee_taker: float          # 单边 taker 手续费
    slip_entry: float         # 入场不利滑点
    slip_exit: float          # 离场不利滑点
    include_funding: bool     # 零成本毛收益情景不计资金费

    @property
    def c_rt(self) -> float:
        """预计往返摩擦率 = 2 × taker + 入场滑点 + 离场滑点（仓位分母用）。"""
        return 2.0 * self.fee_taker + self.slip_entry + self.slip_exit


COSTS = {
    "gross_0bp": CostScenario("gross_0bp", 0.0, 0.0, 0.0, False),
    "conventional_14bp": CostScenario("conventional_14bp", 0.0005, 0.0002, 0.0002, True),
    "main_17bp": CostScenario("main_17bp", 0.0005, 0.0005, 0.0002, True),
    "harsh_34bp": CostScenario("harsh_34bp", 0.0010, 0.0010, 0.0004, True),
}
MAIN_COST = "main_17bp"


@dataclass(frozen=True)
class Sizing:
    name: str
    mode: str                 # 'risk'：以损定仓；'notional'：等名义
    risk_fraction: float = RISK_FRACTION
    min_dist_frac: float = MIN_DIST_FRAC
    leverage_cap: float = LEVERAGE_CAP
    notional_mult: float = EQUAL_NOTIONAL


SIZINGS = {"risk_0.5pct": Sizing("risk_0.5pct", "risk"), "notional_1x": Sizing("notional_1x", "notional")}
MAIN_SIZING = "risk_0.5pct"


@dataclass(frozen=True)
class VersionSpec:
    name: str
    group: str                # 'A'（S6 日线箱体）/ 'B'（无箱体）
    trigger_rule: str         # 'pct'（0.18% / 0.47% / 1.18%）/ 'z'（阈值 × σ7d / σ_ref）
    direction: str            # 'up_long'：向上脉冲做多；'dn_long'：向下脉冲做多（急跌反抽）；'bi'：涨做多 + 跌做空
    box: bool                 # True：要求 S6 二段支撑区
    stop: str                 # 'A' / 'B' / 'C'
    hold: str                 # 'cont'：第 1 根满时按量能判别，最长 2 根；'fixed1'：固定 1 根即走
    ambiguous_skip: bool      # 双向版本：同一分钟上下条件同时成立时不入场
    inherits: str = ""
    note: str = ""


VERSIONS = {
    "A1_box_opt": VersionSpec("A1_box_opt", "A", "pct", "up_long", True, "A", "cont", False),
    "A2_box_origin": VersionSpec("A2_box_origin", "A", "pct", "up_long", True, "B", "cont", False),
    "A3_box_half": VersionSpec("A3_box_half", "A", "pct", "up_long", True, "C", "cont", False),
    "A4_box_flush": VersionSpec("A4_box_flush", "A", "pct", "dn_long", True, "A", "cont", False),
    "A5_box_origin_z": VersionSpec("A5_box_origin_z", "A", "z", "up_long", True, "B", "cont", False,
                                   inherits="A2_box_origin", note="全 IS 波动标定的稳健性版本（σ_ref 用全 IS 中位数，IS 内非完全无前视）"),
    "B1_pure_long": VersionSpec("B1_pure_long", "B", "pct", "up_long", False, "A", "cont", False),
    "B2_pure_long_sl": VersionSpec("B2_pure_long_sl", "B", "pct", "up_long", False, "B", "cont", False),
    "B3_pure_bi_mom": VersionSpec("B3_pure_bi_mom", "B", "pct", "bi", False, "B", "cont", True),
    "B4_pure_1bar": VersionSpec("B4_pure_1bar", "B", "pct", "up_long", False, "A", "fixed1", False),
    "B5_pure_bi_mom_z": VersionSpec("B5_pure_bi_mom_z", "B", "z", "bi", False, "B", "cont", True,
                                    inherits="B3_pure_bi_mom", note="全 IS 波动标定的稳健性版本（σ_ref 用全 IS 中位数，IS 内非完全无前视）"),
}
BOX_SINGLE_VARIABLE_PAIRS = (("A1_box_opt", "B1_pure_long"), ("A2_box_origin", "B2_pure_long_sl"))
STOP_FAMILIES = {"A_up_box": ("A1_box_opt", "A2_box_origin", "A3_box_half"), "B_up": ("B1_pure_long", "B2_pure_long_sl")}


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def frozen_key() -> str:
    """全部版本、成本、仓位与常数的哈希（实验计划记录冻结时的值）。"""
    consts = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str))}
    consts.update({"MAIN_START": str(MAIN_START), "MAIN_END": str(MAIN_END), "LAST_TRIGGER": str(LAST_TRIGGER),
                   "ARCHIVE_GAP_MINUTES": list(ARCHIVE_GAP_MINUTES)})
    return _hash({"versions": {k: asdict(v) for k, v in VERSIONS.items()},
                  "costs": {k: asdict(v) for k, v in COSTS.items()},
                  "sizings": {k: asdict(v) for k, v in SIZINGS.items()}, "constants": consts})


def version_keys() -> dict:
    return {k: _hash(asdict(v)) for k, v in VERSIONS.items()}

