#!/usr/bin/env python3
"""
tools/run_impulse_cp2_features.py — Checkpoint 2：真实 IS 数据上的因果特征回归（不进入撮合，不计算任何盈亏）。

只统计触发时刻及之前已知的量：ATR_15m(14) 的初始化与自然整刻棒缺失情况、10 个版本在主比较窗口内的入场信号数与各项
资格排除计数、按持仓日程推算的可成交笔数区间（只用触发时间戳）、冻结的止损距离与由此决定的计划名义杠杆、0.15% 下限与
2.0x 上限的生效比例，以及前缀截断下这些特征逐位不变。输出 reports/impulse_vs_box_round/cp2/cp2_feature_checks.json。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from impulse_study import daily as DY  # noqa: E402
from impulse_study import data_1m as D  # noqa: E402
from impulse_study import features as F  # noqa: E402
from impulse_study import signals as S  # noqa: E402
from impulse_study import versions as VS  # noqa: E402
from impulse_study.scope import IS_END, IS_START  # noqa: E402

OUT = ROOT / "reports" / "impulse_vs_box_round" / "cp2"
CUTS = ("2020-03-12 13:37", "2021-06-26 13:37", "2022-11-09 00:00", "2023-12-31 23:59")


def real_context(df1m, s1d, cut=None):
    end = IS_END if cut is None else pd.Timestamp(cut, tz="UTC")
    df = df1m[df1m.index < end]
    grid = D.minute_grid(df, IS_START, end)
    st = DY.build_box_states(df, s1d, variants=("S6_reentry_above_poc",))["S6_reentry_above_poc"]
    box = DY.minute_box_ok(np.asarray(grid.index.as_unit("ns").asi8), grid["close"].to_numpy(), st)
    last = VS.LAST_TRIGGER if cut is None else min(VS.LAST_TRIGGER, end - pd.Timedelta(minutes=31))
    r1 = json.loads((ROOT / "reports" / "impulse_vs_box_round" / "cp1" / "cp1_results.json").read_text())
    gaps = [m["minute"] for m in r1["qa_volume_discrepancies_localized"]["minutes"] if m["class"] == "tail_omission"]
    return F.make_context(grid, box, "binance_is", last_trigger=last, archive_gap_minutes=pd.DatetimeIndex(gaps)), st


def q(x, ps=(0.05, 0.25, 0.5, 0.75, 0.95)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {str(p): float(np.quantile(x, p)) for p in ps} if len(x) else {}


def main() -> dict:
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    res = {"git": {"commit": run("rev-parse", "HEAD"), "dirty_files": run("status", "--porcelain").splitlines()},
           "scope": "BTCUSDT perp IS features only; the matching engine is not called; no prices after any trigger",
           "frozen_key": VS.frozen_key(), "version_keys": VS.version_keys()}
    df = D.load_perp_1m()
    s1d = D.load_spot_warmup("1d")
    ctx, st6 = real_context(df, s1d)
    a = ctx.atr
    nv = a["n_valid"]
    first_atr = int(np.flatnonzero(np.isfinite(a["atr"]))[0])
    res["atr15"] = {"buckets": int(len(nv)), "buckets_n_valid_0": int((nv == 0).sum()),
                    "buckets_partial_1_to_14": int(((nv >= 1) & (nv < 15)).sum()),
                    "first_initialized_bucket_close": str(pd.Timestamp(a["bucket_start_ns"][first_atr], tz="UTC")
                                                          + pd.Timedelta(minutes=15)),
                    "window_minutes_with_nan_atr": int((ctx.window & ~np.isfinite(a["atr_at"])).sum()),
                    "atr15_bps_quantiles_in_window": q(a["atr_at"][ctx.window] / ctx.C[ctx.window] * 1e4)}
    res["s6_state"] = {"valid_days": int(st6["valid"].sum()), "primed_days": int(st6["primed"].sum()),
                       "box_ok_minute_share_in_window": float(ctx.box_ok[ctx.window].mean())}
    res["window"] = {"first_trigger_minute": str(ctx.index[np.flatnonzero(ctx.window)[0]]),
                     "last_trigger_minute": str(ctx.index[np.flatnonzero(ctx.window)[-1]]),
                     "minutes": int(ctx.window.sum())}
    yrs = S.years_of(ctx.index)
    per = {}
    for name, spec in VS.VERSIONS.items():
        vin = F.build_version_inputs(ctx, spec)
        m = vin.entry_dir != 0
        dist = np.abs(vin.eref - vin.sl) / vin.eref
        rec = {"entry_signals": int(m.sum()), "long": int((vin.entry_dir == 1).sum()), "short": int((vin.entry_dir == -1).sum()),
               "skipped_in_window": vin.skipped,
               "category_share": {k: float((vin.cat[m] == c).mean()) for k, c in (("only1", 1), ("only2", 2), ("both", 3))},
               "w_share": {"5": float((vin.w[m] == 5).mean()), "15": float((vin.w[m] == 15).mean())} if spec.stop != "A" else {},
               "stop_distance_bps_quantiles": q(dist[m] * 1e4),
               "min_distance_floor_binds_share": float((dist[m] < VS.MIN_DIST_FRAC).mean())}
        for cname, c in VS.COSTS.items():
            lev = VS.RISK_FRACTION / (np.maximum(dist[m], VS.MIN_DIST_FRAC) + c.c_rt)
            rec[f"planned_leverage_{cname}"] = {"quantiles": q(np.minimum(lev, VS.LEVERAGE_CAP)),
                                                "cap_2x_binds_share": float((lev > VS.LEVERAGE_CAP).mean())}
        sched = {}
        for nm, blk in VS_BLOCKS.items():
            acc = S.schedule(m, blk)
            sched[nm] = {"IS": int(acc.sum()), **{str(y): int((acc & (yrs == y)).sum()) for y in (2020, 2021, 2022, 2023)}}
        rec["schedule_bounded_trade_counts"] = sched
        per[name] = rec
    res["versions"] = per
    # 前缀截断：特征与版本输入逐位不变
    pref = []
    for cut in CUTS:
        tc, _ = real_context(df, s1d, cut)
        n = len(tc.C)
        k = n - 32
        same = {"atr_at": bool(np.array_equal(tc.atr["atr_at"], ctx.atr["atr_at"][:n], equal_nan=True)),
                "base_med": bool(np.array_equal(tc.base_med, ctx.base_med[:n], equal_nan=True)),
                "box_ok": bool(np.array_equal(tc.box_ok, ctx.box_ok[:n]))}
        for name, spec in VS.VERSIONS.items():
            va, vb = F.build_version_inputs(ctx, spec), F.build_version_inputs(tc, spec)
            same[name] = bool(all(np.array_equal(getattr(vb, f)[:k], getattr(va, f)[:k]) for f in ("entry_dir", "w", "cat"))
                              and np.array_equal(vb.sl[:k], va.sl[:k], equal_nan=True))
        pref.append({"cut": cut, "minutes_compared": int(k), "all_equal": all(same.values()), "detail": same})
    res["prefix_truncation"] = pref
    res["runtime_s"] = time.time() - t_start
    (OUT / "cp2_feature_checks.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


VS_BLOCKS = {"sched45": 45, "sched30": 30, "sched16": 16}

if __name__ == "__main__":
    r = main()
    print(json.dumps({k: r[k] for k in ("atr15", "s6_state", "window", "prefix_truncation", "runtime_s")}, indent=1,
                     ensure_ascii=False)[:5000])
