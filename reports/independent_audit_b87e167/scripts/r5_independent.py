"""Independent re-implementation of the R5 daily trend-hold study from approved_plan.md.

Written from the plan text only; imports nothing from the audited repository.
Usage: python -I r5_independent.py <btc_1m_parquet> <funding_parquet> <out_dir> [--monthly-2023-11-10 <zip>]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

IS_START = pd.Timestamp("2020-01-01", tz="UTC")
MAIN_START = pd.Timestamp("2020-03-01", tz="UTC")
IS_END = pd.Timestamp("2024-01-01", tz="UTC")
COSTS = {
    "gross_0": dict(fee=0.0, buy=0.0, sell=0.0, funding=False),
    "main_17": dict(fee=0.0005, buy=0.0005, sell=0.0002, funding=True),
    "harsh_34": dict(fee=0.001, buy=0.001, sell=0.0004, funding=True),
}
CAPITAL = 100_000.0


def load(minute_path, funding_path):
    df = pd.read_parquet(minute_path)
    grid = pd.date_range(IS_START, IS_END, freq="1min", inclusive="left", tz="UTC").as_unit("ns")
    df = df.reindex(grid)
    o, h, l, c, v = (df[k].to_numpy(np.float64) for k in ("open", "high", "low", "close", "volume"))
    bad = np.isnan(c) | ((v == 0) & (o == h) & (h == l) & (l == c))
    fund = pd.read_parquet(funding_path)
    t = fund.index.round("min")
    keep = (t >= IS_START) & (t < IS_END)
    rates = pd.Series(fund["funding_rate"].to_numpy()[keep], index=t[keep].as_unit("ns"))
    assert len(rates) == 4383 and rates.index.equals(
        pd.date_range(IS_START, IS_END, freq="8h", inclusive="left", tz="UTC").as_unit("ns"))
    return grid, o, c, bad, rates


def features(grid, c, bad):
    n_days = len(grid) // 1440
    days_all = grid[::1440]
    C = np.empty(n_days)
    for k in range(n_days):
        seg_c, seg_b = c[k * 1440:(k + 1) * 1440], bad[k * 1440:(k + 1) * 1440]
        valid = np.flatnonzero(~seg_b)
        assert len(valid), days_all[k]
        C[k] = seg_c[valid[-1]]
    first = int(np.flatnonzero(days_all == MAIN_START)[0])
    rows = []
    L = None
    for k in range(first, n_days):
        d = days_all[k]
        prev = C[k - 60:k]                     # C_{d-60} .. C_{d-1}
        M = prev.mean()
        eref = C[k - 1]
        u = C[k - 20:k] / C[k - 21:k - 1] - 1  # u_{d-20} .. u_{d-1}
        sigma = np.sqrt(365.0) * u.std(ddof=1)
        update = (d == MAIN_START) or d.weekday() == 0
        if update:
            L = 1.0 if sigma == 0 else min(1.0, 0.20 / sigma)
        rows.append(dict(decision_time=d, eref=eref, sma60=M, sigma=sigma, budget=L,
                         g=bool(eref > M), budget_update=update))
    return pd.DataFrame(rows).set_index("decision_time"), C


def run(grid, o, c, bad, rates, feat, version, cost_name):
    cost = COSTS[cost_name]
    start = int(np.flatnonzero(grid == MAIN_START)[0])
    n = len(grid)
    # last-valid mark for every minute (bad minutes carry the previous valid close)
    idx = np.where(bad, -1, np.arange(n))
    idx = np.maximum.accumulate(idx)
    assert idx[start - 1] >= 0
    mark = c[idx]
    rate_at = np.full(n, np.nan)
    pos = grid.get_indexer(rates.index)
    rate_at[pos] = rates.to_numpy()
    events = sorted(set(range(start, n, 1440)) | set(p for p in pos if p >= start) | {n - 1})
    cash, q = CAPITAL, 0.0
    pending, prev_g = False, None
    eq = np.empty(n - start)
    qty_path = np.empty(n - start)
    orders, fund_rows, decisions = [], [], []
    last = start  # first minute not yet written into eq
    day_rows = feat.to_dict("index")
    for m in events:
        # write marks for minutes strictly before event m with the current (cash, q)
        if m > last:
            eq[last - start:m - start] = cash + q * mark[last:m]
            qty_path[last - start:m - start] = q
        t = grid[m]
        terminal = m == n - 1
        ref = mark[m] if terminal else (mark[m - 1] if bad[m] else o[m])
        if not np.isnan(rate_at[m]):
            flow = -q * ref * rate_at[m] if cost["funding"] else 0.0
            cash += flow
            fund_rows.append(dict(time=t, rate=rate_at[m], qty_before=q, ref=ref, flow=flow))
        if (m - start) % 1440 == 0:
            row = day_rows[t]
            g = True if version == "P" else row["g"]
            e_minus = (cash - (fund_rows[-1]["flow"] if fund_rows and fund_rows[-1]["time"] == t else 0.0)) \
                + q * mark[m - 1]
            initial = m == start
            flip = (not initial) and (g != prev_g)
            trig = initial or row["budget_update"] or flip or pending
            status = "hold"
            if trig:
                target = (1.0 if g else 0.0) * row["budget"] * e_minus / row["eref"]
                pending = False
                if target == q:
                    status = "zero_delta"
                elif bad[m]:
                    status, pending = "canceled_bad", True
                else:
                    delta = target - q
                    fill = ref * (1 + cost["buy"]) if delta > 0 else ref * (1 - cost["sell"])
                    fee = abs(delta) * fill * cost["fee"]
                    cash -= delta * fill + fee
                    orders.append(dict(time=t, terminal=False, e_minus=e_minus, q_before=q, q_target=target,
                                       delta=delta, ref=ref, fill=fill, fee=fee, slip=delta * (fill - ref)))
                    q = target
                    status = "executed"
            decisions.append(dict(time=t, g=g, trig=trig, status=status, e_minus=e_minus))
            prev_g = g
        if terminal and q != 0:
            delta = -q
            fill = ref * (1 - cost["sell"])
            fee = abs(delta) * fill * cost["fee"]
            cash -= delta * fill + fee
            orders.append(dict(time=t, terminal=True, e_minus=cash + q * mark[m], q_before=q, q_target=0.0,
                               delta=delta, ref=ref, fill=fill, fee=fee, slip=delta * (fill - ref)))
            q = 0.0
        eq[m - start] = cash + q * mark[m]
        qty_path[m - start] = q
        last = m + 1
    daily = eq[1439::1440]
    rets = daily / np.r_[CAPITAL, daily[:-1]] - 1
    return dict(eq=eq, qty=qty_path, mark=mark[start:], daily=daily, rets=rets,
                orders=pd.DataFrame(orders), funding=pd.DataFrame(fund_rows), decisions=pd.DataFrame(decisions))


def summarize(res, feat, version):
    eq, rets = res["eq"], res["rets"]
    orders, fund = res["orders"], res["funding"]
    peak = np.maximum.accumulate(eq)
    lev = res["qty"] * res["mark"] / eq
    return dict(
        total_return=eq[-1] / CAPITAL - 1, daily_mean=rets.mean(),
        sharpe=np.sqrt(365) * rets.mean() / rets.std(ddof=1),
        max_drawdown=(eq / peak - 1).min(), realized_volatility=np.sqrt(365) * rets.std(ddof=1),
        fees=orders["fee"].sum() if len(orders) else 0.0,
        slippage_cost=orders["slip"].sum() if len(orders) else 0.0,
        funding_cashflow=fund["flow"].sum(), order_count=len(orders),
        adjustment_count=int((~orders["terminal"]).sum()) if len(orders) else 0,
        mean_actual_leverage=lev.mean(), actual_leverage_max=lev.max(),
        minutes_above_leverage_cap=int((lev > 1).sum()),
        held_days=int((res["decisions"]["g"]).sum()) if version == "T" else len(rets),
        final_equity=eq[-1],
    )


def main():
    minute_path, funding_path, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    grid, o, c, bad, rates = load(minute_path, funding_path)
    feat, C = features(grid, c, bad)
    feat.to_csv(out / "features.csv")
    summary, daily = {}, {}
    for v in ("T", "P"):
        for k in COSTS:
            res = run(grid, o, c, bad, rates, feat, v, k)
            summary[f"{v}_{k}"] = summarize(res, feat, v)
            daily[f"{v}_{k}"] = res["rets"]
            res["orders"].to_csv(out / f"orders_{v}_{k}.csv", index=False)
            np.save(out / f"eq_{v}_{k}.npy", res["eq"])
    pd.DataFrame(daily, index=feat.index).to_csv(out / "daily_returns.csv")
    json.dump(summary, open(out / "summary.json", "w"), indent=1, default=float)
    print(pd.DataFrame(summary).T.to_string())


if __name__ == "__main__":
    main()
