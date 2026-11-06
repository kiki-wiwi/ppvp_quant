#!/usr/bin/env python3
"""
tools/bench_impulse_engine.py — Checkpoint 2：在 2,103,840 根合成 1m（与 IS 四年相同的分钟网格、主窗口与末端几何）上测引擎耗时。

只用合成数据（impulse_study.synthetic.random_grid）；真实行情在本阶段不进入撮合。输出
reports/impulse_vs_box_round/cp2/bench_engine.json。
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from impulse_study import engine as E  # noqa: E402
from impulse_study import features as F  # noqa: E402
from impulse_study import synthetic as SY  # noqa: E402
from impulse_study import versions as VS  # noqa: E402
from numba_compat import HAVE_NUMBA  # noqa: E402

OUT = ROOT / "reports" / "impulse_vs_box_round" / "cp2"
N = 2_103_840


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    t = {}
    t0 = time.perf_counter()
    halts = ((300_000, 59), (900_000, 2), (1_500_000, 35))
    g = SY.random_grid(N, start="2020-01-01", seed=20261106, halts=halts)
    t["synthetic_grid_s"] = time.perf_counter() - t0
    fund = pd.Series(np.random.default_rng(1).normal(1e-4, 2e-4, N // 480 + 1),
                     index=pd.date_range("2020-01-01", periods=N // 480 + 1, freq="8h", tz="UTC"))
    box = ((np.arange(N) // 1440) % 5 == 0) & ((np.arange(N) // 37) % 3 == 0)      # 约 7% 的分钟处在“箱体内”
    t0 = time.perf_counter()
    assert g.index[-1] == VS.MAIN_END - pd.Timedelta(minutes=1)      # 与 IS 相同的分钟网格与主窗口几何
    ctx = SY.synthetic_context(g, VS.MAIN_START, VS.LAST_TRIGGER, box_ok=box, funding=fund)
    t["features_context_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    vins = {k: F.build_version_inputs(ctx, v) for k, v in VS.VERSIONS.items()}
    t["version_inputs_10_s"] = time.perf_counter() - t0
    # 首次调用包含 numba 编译（cache=True 时后续进程直接读缓存）
    t0 = time.perf_counter()
    E.run_portfolio(ctx, vins["B1_pure_long"], VS.COSTS["main_17bp"], VS.SIZINGS["risk_0.5pct"])
    t["first_run_incl_compile_s"] = time.perf_counter() - t0
    per_version = {}
    for k, vin in vins.items():
        t0 = time.perf_counter()
        r = E.run_portfolio(ctx, vin, VS.COSTS["main_17bp"], VS.SIZINGS["risk_0.5pct"])
        per_version[k] = {"seconds": time.perf_counter() - t0, "signals": int((vin.entry_dir != 0).sum()),
                          "trades": int(len(r.trades))}
    t0 = time.perf_counter()
    n_runs = 0
    for k, vin in vins.items():
        for c in VS.COSTS.values():
            for s in VS.SIZINGS.values():
                E.run_portfolio(ctx, vin, c, s)
                n_runs += 1
    t["all_80_runs_s"] = time.perf_counter() - t0
    tr = E.run_portfolio(ctx, vins["B3_pure_bi_mom"], VS.COSTS["main_17bp"], VS.SIZINGS["risk_0.5pct"]).trades
    rng = np.random.default_rng(2)
    reps = 100
    # 候选池按实验计划第 8.3 节对 B3（百分比口径、无箱体）的资格构造：窗口内、无维护污染、ATR_15m 已就绪且 > 0、
    # 入场 K 线有效、基准有定义，且不是自身口径下任一方向的触发分钟。只用于计时；第 3 阶段按第 8.3 节逐版本构造。
    c = ctx.cond["pct"]
    entry_ok = np.r_[~ctx.bad[1:], False]
    trig = c["up1"] | c["up2"] | c["dn1"] | c["dn2"]
    atr = ctx.atr["atr_at"]
    with np.errstate(invalid="ignore"):
        atr_pos = np.isfinite(atr) & (atr > 0)
    cand = np.flatnonzero(ctx.window & ~ctx.frame["contaminated"].to_numpy() & atr_pos & entry_ok
                          & np.isfinite(ctx.base_med) & ~trig)
    idx = rng.choice(cand, size=len(tr) * reps)
    side = np.repeat(tr["side"].to_numpy(), reps).astype(float)
    eref = ctx.C[idx]
    dist = np.repeat(((tr["e_ref"] - tr["sl"]).abs() / tr["atr15"]).to_numpy(), reps) * ctx.atr["atr_at"][idx]
    t0 = time.perf_counter()
    ev = E.simulate_events(ctx, idx, side, eref - side * dist, eref, False, VS.COSTS["main_17bp"])
    t["matched_random_events_s"] = time.perf_counter() - t0
    res = {"minutes": N, "numba": HAVE_NUMBA, "python": sys.version.split()[0], "cpu_count": os.cpu_count(),
           "machine": platform.machine(), "timing_s": t, "per_version_main_config": per_version, "runs_in_grid": n_runs,
           "matched_random_events": int(len(ev)), "note": "synthetic data only; real market data is locked out of the "
                                                         "matching engine at Checkpoint 2"}
    (OUT / "bench_engine.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


if __name__ == "__main__":
    print(json.dumps(main(), indent=1, ensure_ascii=False))
