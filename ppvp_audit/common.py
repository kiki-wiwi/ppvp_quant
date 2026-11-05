"""
ppvp_audit/common.py — 本轮唯一的数据与回测入口：BTCUSDT 永续、IS 区间 [2020-01-01, 2024-01-01)。

``load_is_market`` / ``is_run`` 把数据截断在 IS_END 之前再送入引擎（pipeline.run(..., end=IS_END)），
并断言返回的 1H、4H、箱体、信号、净值与交易表都不晚于 IS_END。资金费表取结算时刻 ≤ IS_END
（IS 最后一根 1H 收盘恰为 IS_END，该时刻的结算由 IS 内的溢价决定，与 pipeline.load_market 的约定一致）。
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import pipeline
from config import BASELINE_BACKTEST, BASELINE_STRATEGY, IS_END, IS_START, BacktestConfig, StrategyConfig

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "audit_round"
FIG = OUT / "figures"
SYMBOL = "BTCUSDT"
SEED = 20261105                      # 本轮全部随机抽样的主种子（各用途在其上加固定偏移，见 experiment_plan.md）


class ISBoundaryError(RuntimeError):
    """本轮只允许使用 IS 数据；任何对象延伸到 IS_END 之后都视为越界。"""


def _assert_index_before(idx, end: pd.Timestamp, what: str, inclusive: bool = False) -> None:
    if len(idx) == 0:
        return
    last = pd.DatetimeIndex(idx).max()
    if (last > end) if inclusive else (last >= end):
        raise ISBoundaryError(f"{what} extends to {last}, beyond the IS boundary {end}")


def assert_is_only(out: dict) -> None:
    """检查 pipeline.run 的返回值没有任何 IS_END 之后的内容。"""
    _assert_index_before(out["df1h"].index, IS_END, "1H bars")
    _assert_index_before(out["df4h"].index, IS_END, "4H bars")
    if len(out["df4h"]):
        _assert_index_before(out["df4h"]["available_time"], IS_END, "4H available_time", inclusive=True)
    if len(out["boxes"]):
        _assert_index_before(out["boxes"]["available_time"], IS_END, "box available_time", inclusive=True)
    _assert_index_before(out["signals"].index, IS_END, "signals")
    res = out.get("result")
    if res is not None:
        _assert_index_before(res.equity.index, IS_END, "equity", inclusive=True)
        tr = res.trades
        if len(tr):
            _assert_index_before(tr["entry_time"], IS_END, "trade entries", inclusive=True)
            closed = tr["exit_time"].dropna()
            _assert_index_before(closed, IS_END, "trade exits", inclusive=True)


def is_run(scfg: StrategyConfig = BASELINE_STRATEGY, bcfg: BacktestConfig = BASELINE_BACKTEST,
           signals_hook=None) -> dict:
    """在 IS 截断数据上运行完整流水线，只计算 IS 分段指标。

    signals_hook(out_without_backtest) -> signals：实验版本可在回测前替换信号表（同一批 1H / 箱体）。
    """
    out = pipeline.run(SYMBOL, scfg, bcfg, end=IS_END, segments=("IS",))
    if signals_hook is not None:
        from backtester import run_backtest
        from metrics import segment_report
        sig = signals_hook(out)
        _, fund, _ = pipeline.load_market(SYMBOL, IS_END)
        res = run_backtest(out["df1h"], sig, bcfg, fund)
        out = dict(out, signals=sig, result=res, report=segment_report(res, ("IS",)))
    assert_is_only(out)
    return out


def load_is_market():
    """(df1h, funding, spot_daily_warmup)，均截断在 IS 之内（资金费含恰在 IS_END 的结算）。"""
    df1h, fund, spot = pipeline.load_market(SYMBOL, IS_END)
    _assert_index_before(df1h.index, IS_END, "1H bars")
    _assert_index_before(fund.index, IS_END, "funding", inclusive=True)
    _assert_index_before(spot.index, IS_START, "spot warm-up")
    return df1h, fund, spot


def sha256_file(p: Path, n: int | None = None) -> str:
    h = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    return h if n is None else h[:n]


def git_state() -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "describe": run("describe", "--always", "--dirty"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"), "status_porcelain": run("status", "--porcelain"),
            "diff_stat_vs_head": run("diff", "--stat", "HEAD")}


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    if isinstance(x, pd.DataFrame):
        return jsonable(x.to_dict(orient="records"))
    if isinstance(x, pd.Series):
        return jsonable(x.to_dict())
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    return x


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), ensure_ascii=False, indent=1))
