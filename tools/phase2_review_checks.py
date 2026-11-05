#!/usr/bin/env python3
"""
tools/phase2_review_checks.py — Phase 2 复核阶段在 IS 上试跑过的变体，按固定参数可复现地重跑并写入 JSON。

用法（仓库根目录）：python tools/phase2_review_checks.py
输出：reports/phase2_review_checks.json

只用 IS：每个变体都以 pipeline.run(..., end=IS_END, segments=("IS",)) 运行，数据截断在 2024-01-01 之前，
VAL、OOS 与 ETH 都不参与。这些结果只是复核者核对 Phase 2 结论时看过的 IS 数字，不作为采纳依据；
Phase 3 会按 PHASE2_baseline.md 第 11 节的规则重跑这些取值并同时报告 VAL，台账会标注它们的 IS 结果已被看过。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import pipeline  # noqa: E402
from config import BASELINE_BACKTEST, BASELINE_STRATEGY, IS_END  # noqa: E402

SYMBOL = "BTCUSDT"

# (组, 名称, 策略参数改动, 回测参数改动)
CHECKS = (
    [("baseline", "baseline", {}, {})]
    + [("daily_sma", f"sma{d}_slope{w}", {"regime_filter": "daily_sma", "regime_sma_days": d, "regime_slope_days": w}, {})
       for d in (170, 200, 230) for w in (17, 20, 23)]
    + [("crash_cooldown", "crash_cooldown_48h", {"regime_filter": "crash_cooldown"}, {})]
    + [("time_stop", f"time_stop_{h}h", {}, {"time_stop_bars": h})
       for h in (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 20, 22, 24, 26, 28, 30, 41, 48, 55, 72)]
    + [("sizing", f"risk_{r:.4f}", {}, {"sizing": "risk", "risk_per_trade": r}) for r in (0.005, 0.0075, 0.0085, 0.01, 0.0115, 0.0165)]
    + [("sizing", f"inv_vol_{v:.2f}", {}, {"sizing": "inv_vol", "target_vol": v}) for v in (0.34, 0.40, 0.46)]
    + [("box_width", f"max_width_{w}", {"max_box_width_atr": w}, {})
       for w in (2.55, 2.7, 2.85, 3.0, 3.15, 3.3, 3.45, 3.5, 4.0, 4.1, 4.25, 4.6, 5.0, 5.75)]
    + [("pinbar", "pin_require_bull_close", {"pin_require_bull_close": True}, {})]
    + [("pinbar", f"pin_wick_frac_{x}", {"pin_wick_frac": x}, {}) for x in (0.51, 0.69)]
    + [("min_rr", f"min_rr_{x}", {"min_rr": x}, {}) for x in (0.5, 1.0)]
    + [("sl_buffer", f"sl_buffer_{x}", {"sl_buffer_atr": x}, {}) for x in (0.5, 1.0)]
    + [("breakdown_exit", "exit_on_breakdown", {}, {"exit_on_breakdown": True})]
    + [("value_area", f"va_pct_{x}", {"va_pct": x}, {}) for x in (0.60, 0.65, 0.75, 0.80)]
    + [("pivots", f"L{a}_R{b}", {"pivot_left": a, "pivot_right": b}, {}) for a, b in ((4, 4), (6, 6), (5, 4))]
    + [("breakdown_thresholds", f"break_body_{x}", {"break_body_atr": x}, {}) for x in (0.51, 0.69)]
    + [("breakdown_thresholds", f"break_vol_{x}", {"break_vol_mult": x}, {}) for x in (1.275, 1.725)]
)


def main() -> None:
    t0 = time.time()
    rows = []
    ref = None
    for group, name, skw, bkw in CHECKS:
        out = pipeline.run(SYMBOL, BASELINE_STRATEGY.with_(**skw), BASELINE_BACKTEST.with_(**bkw), end=IS_END, segments=("IS",))
        rep = out["report"]["IS"]
        row = {"group": group, "name": name, "strategy": skw, "backtest": bkw, "is_sharpe": rep["sharpe"],
               "is_trades": rep["trades"], "is_total_return": rep["total_return"], "is_max_dd": rep["max_dd"],
               "is_win_rate": rep["win_rate"], "is_avg_r": rep["avg_r"]}
        if ref is None:
            ref = row
        n_ratio = rep["trades"] / ref["is_trades"]
        row["formula_benchmark"] = ref["is_sharpe"] * np.sqrt(n_ratio) if n_ratio < 1 else ref["is_sharpe"]
        row["margin_vs_benchmark"] = rep["sharpe"] - row["formula_benchmark"]
        row["delta_vs_baseline"] = rep["sharpe"] - ref["is_sharpe"]
        rows.append(row)
    assert abs(ref["is_sharpe"] - (-1.174067431234681)) < 1e-9 and ref["is_trades"] == 109   # IS 截断运行复现基线
    payload = {"symbol": SYMBOL, "segment": "IS only (data truncated at 2024-01-01)", "n_checks": len(rows),
               "note": "IS-only checks run during the Phase 2 review; not used for adoption decisions",
               "benchmark": "formula S_ref*sqrt(n/n_ref) when the variant has fewer IS trades, else S_ref",
               "rows": rows, "runtime_s": time.time() - t0}

    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [clean(v) for v in x]
        if isinstance(x, (np.floating, float)):
            return None if not np.isfinite(x) else round(float(x), 6)
        if isinstance(x, np.integer):
            return int(x)
        return x

    (ROOT / "reports" / "phase2_review_checks.json").write_text(json.dumps(clean(payload), ensure_ascii=False, indent=1))
    print(f"{len(rows)} IS checks done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
