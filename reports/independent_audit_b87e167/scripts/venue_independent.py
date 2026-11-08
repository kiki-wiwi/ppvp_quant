"""Independent re-implementation of the cross-venue accounts (approved_spec.md + boundary_addendum.md).
Imports nothing from the audited repository. Primary canonical-UM-signal accounts only.
Usage: python -I venue_independent.py <config.json> <out_dir>
config: {"asset", "window": [start, end], "um", "um_funding", "spot", "cm", "cm_funding", "signal_origin", "archived_dir", "prefix"}
"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

CAP = 100_000.0
MMR = 0.005
COSTS = {"gross_0": (0.0, 0.0, 0.0, False), "main_17": (0.0005, 0.0005, 0.0002, True), "harsh_34": (0.001, 0.001, 0.0004, True)}


def grid_frame(path, start, end, funding_path=None):
    df = pd.read_parquet(path)
    g = pd.date_range(start, end, freq="1min", inclusive="left", tz="UTC").as_unit("ns")
    df = df.reindex(g)
    o, h, l, c, v = (df[k].to_numpy(np.float64) for k in ("open", "high", "low", "close", "volume"))
    bad = np.isnan(c) | ((v == 0) & (o == h) & (h == l) & (l == c))
    if "bad" in df:  # registered flags (missing or placeholder) must agree with the definition
        assert np.array_equal(bad, df["bad"].fillna(True).to_numpy(bool)), path
    idx = np.maximum.accumulate(np.where(bad, -1, np.arange(len(g))))
    mark = np.where(idx >= 0, c[np.maximum(idx, 0)], np.nan)
    rate = np.full(len(g), np.nan)
    if funding_path:
        f = pd.read_parquet(funding_path)["funding_rate"]
        f.index = f.index.round("min")
        f = f[(f.index >= start) & (f.index < end)]
        pos = g.get_indexer(f.index)
        assert (pos >= 0).all()
        rate[pos] = f.to_numpy()
    return dict(index=g, o=o, h=h, l=l, c=c, bad=bad, mark=mark, rate=rate)


def daily_close(fr):
    n = len(fr["index"]) // 1440
    out = np.full(n, np.nan)
    for k in range(n):
        b = fr["bad"][k * 1440:(k + 1) * 1440]
        valid = np.flatnonzero(~b)
        if len(valid):
            out[k] = fr["c"][k * 1440 + valid[-1]]
    return out


def canonical_features(um, signal_origin, start):
    """g/L schedule on the canonical UM series; budget updated at signal_origin and every Monday."""
    C = daily_close(um)
    days = um["index"][::1440]
    rows, L = {}, None
    k0 = int(np.flatnonzero(days == signal_origin)[0])
    for k in range(k0, len(days)):
        d = days[k]
        M = C[k - 60:k].mean()
        u = C[k - 20:k] / C[k - 21:k - 1] - 1
        sigma = math.sqrt(365.0) * u.std(ddof=1)
        if d == signal_origin or d.weekday() == 0:
            L = 1.0 if sigma == 0 else min(1.0, 0.20 / sigma)
        rows[d] = dict(g=bool(C[k - 1] > M), budget=L, update=(d == signal_origin or d.weekday() == 0), eref=C[k - 1], sma=M, sigma=sigma)
    feat = pd.DataFrame.from_dict(rows, orient="index")
    return feat[feat.index >= start]


def run_linear(um, feat, version, cost, start):
    fee, bs, ss, fund_on = COSTS[cost]
    s = int(np.flatnonzero(um["index"] == start)[0]); n = len(um["index"])
    days = feat.index; rows = feat.to_dict("index")
    cash, q, pending, prev_g = CAP, 0.0, False, None
    eq = np.empty(n - s); last = s
    events = sorted(set(range(s, n, 1440)) | set(np.flatnonzero(np.isfinite(um["rate"])) [np.flatnonzero(np.isfinite(um["rate"])) >= s]) | {n - 1})
    funding_total = 0.0
    for m in events:
        if m > last:
            eq[last - s:m - s] = cash + q * um["mark"][last:m]
        term = m == n - 1
        ref = um["mark"][m] if term else (um["mark"][m - 1] if um["bad"][m] else um["o"][m])
        e_minus = cash + q * um["mark"][m - 1]
        if np.isfinite(um["rate"][m]):
            flow = -q * ref * um["rate"][m] if fund_on else 0.0
            cash += flow; funding_total += flow
        if (m - s) % 1440 == 0:
            r = rows[um["index"][m]]
            g = True if version.startswith("P_") else r["g"]
            first = m == s
            trig = first or r["update"] or (g != prev_g and not first) or pending
            if trig:
                target = (1.0 if g else 0.0) * r["budget"] * e_minus / r["eref"]
                pending = False
                if target != q:
                    if um["bad"][m]:
                        pending = True
                    else:
                        d = target - q; fill = ref * (1 + bs) if d > 0 else ref * (1 - ss)
                        cash -= d * fill + abs(d) * fill * fee; q = target
            prev_g = g
        if term and q:
            fill = ref * (1 - ss); cash += q * fill - q * fill * fee; q = 0.0
        eq[m - s] = cash + q * um["mark"][m]; last = m + 1
    return eq, dict(funding=funding_total)


def run_spot(sp, feat, version, cost, start):
    fee, bs, ss, _ = COSTS[cost]
    s = int(np.flatnonzero(sp["index"] == start)[0]); n = len(sp["index"])
    ref_close = daily_close(sp); dk0 = s // 1440
    rows = feat.to_dict("index")
    B, q, pending, prev_g, clips = CAP, 0.0, False, None, 0
    eq = np.empty(n - s); last = s
    for m in list(range(s, n, 1440)) + [n - 1]:
        if m > last:
            eq[last - s:m - s] = B + q * sp["mark"][last:m]
        term = m == n - 1
        if (m - s) % 1440 == 0 and not (term and (m - s) % 1440 != 0):
            r = rows[sp["index"][m]]
            g = True if version.startswith("P_") else r["g"]
            first = m == s
            e_minus = B + q * sp["mark"][m - 1]
            trig = first or r["update"] or (g != prev_g and not first) or pending
            if trig:
                target = (1.0 if g else 0.0) * r["budget"] * e_minus / ref_close[m // 1440 - 1]
                pending = False
                if target != q:
                    if sp["bad"][m]:
                        pending = True
                    else:
                        o = sp["o"][m]
                        if target > q:
                            F = o * (1 + bs); dq = min(target - q, B / (F * (1 + fee)))
                            clips += dq < target - q
                            B = B - dq * F * (1 + fee)
                            assert B >= -1e-9
                            B = max(B, 0.0)
                        else:
                            F = o * (1 - ss); dq = max(target - q, -q)
                            B = B + abs(dq) * F * (1 - fee)
                        q = q + dq
            prev_g = g
        if term and q:
            F = sp["mark"][m] * (1 - ss); B += q * F * (1 - fee); q = 0.0
        eq[m - s] = B + q * sp["mark"][m]; last = m + 1
    return eq, dict(clips=int(clips))


def run_coin(cm, sp, feat, version, cost, start):
    fee, bs, ss, fund_on = COSTS[cost]
    s = int(np.flatnonzero(cm["index"] == start)[0]); n = len(cm["index"])
    ss_ = int(np.flatnonzero(sp["index"] == start)[0]); assert n - s == len(sp["index"]) - ss_
    cm_close = daily_close(cm); sp_close = daily_close(sp)
    rows = feat.to_dict("index")
    C0 = CAP / (sp["o"][ss_] * (1 + bs) * (1 + fee))
    C0_hat = CAP / (sp_close[ss_ // 1440 - 1] * (1 + bs) * (1 + fee))
    C = C0; N = 0.0; P = cm_close[s // 1440 - 1]  # last CM mark
    pending, prev_g, fund_coin = False, None, 0.0
    eq = np.empty(n - s); coin_path = np.empty(n - s)
    spot_mark = sp["mark"][ss_:]
    events = set(range(s, n, 1440)) | set(np.flatnonzero(np.isfinite(cm["rate"]))[np.flatnonzero(np.isfinite(cm["rate"])) >= s])
    events = sorted(events)
    pressure_hits = 0
    def span(a, b):  # minutes a..b-1, no event: pressure check then close mark; N fixed
        nonlocal C, P, pressure_hits
        if b <= a:
            return
        closes = cm["mark"][a:b]
        if N != 0:
            valid = ~cm["bad"][a:b]
            prevP = np.r_[P, closes[:-1]]
            Cprev = C + N * (1 / P - 1 / prevP)  # coin equity at start of each minute
            coef = Cprev + N / prevP; bound = N + MMR * abs(N)
            ext = np.where(N > 0, cm["l"][a:b], cm["h"][a:b])
            hit = valid & ((coef * cm["o"][a:b] <= bound) | (coef * ext <= bound))
            pressure_hits += int(hit.sum())
            path = C + N * (1 / P - 1 / closes)
        else:
            path = np.full(b - a, C)
        coin_path[a - s:b - s] = path
        eq[a - s:b - s] = path * spot_mark[a - s:b - s]
        C, P = path[-1], closes[-1]
    last = s
    for m in events:
        span(last, m)
        # event minute: mark to open, funding on old face, then decision/trade, then close mark
        o = P if cm["bad"][m] else cm["o"][m]
        C_before = C
        C = C + N * (1 / P - 1 / o); P = o
        if np.isfinite(cm["rate"][m]) and fund_on and N:
            f = -N / o * cm["rate"][m]; C += f; fund_coin += f
        if N and not cm["bad"][m]:  # old position opening pressure: C*o <= mmr*|N|
            pressure_hits += int(C * o <= MMR * abs(N))
        if (m - s) % 1440 == 0 and version != "HODL_1x_Coin":
            r = rows[cm["index"][m]]
            g = True if version == "P_CM_DeltaMatched" else r["g"]
            first = m == s
            trig = first or r["update"] or (g != prev_g and not first) or pending
            if trig:
                base = C0_hat if first else C_before
                mult = (1.0 if g else 0.0) * r["budget"] if version == "V4_CM_CoinAccum" else (1.0 if g else 0.0) * r["budget"] - 1.0
                target = mult * base * cm_close[m // 1440 - 1]
                pending = False
                if target != N:
                    if cm["bad"][m]:
                        pending = True
                    else:
                        dN = target - N; F = o * (1 + bs) if dN > 0 else o * (1 - ss)
                        C += dN * (1 / F - 1 / o) - fee * abs(dN) / F; N = target
            prev_g = g
        if N and not cm["bad"][m]:  # new position: opening then adverse extreme
            coef = C + N / P; bound = N + MMR * abs(N)
            ext = cm["l"][m] if N > 0 else cm["h"][m]
            pressure_hits += int(coef * P <= bound) + int(coef * ext <= bound)
        close = cm["mark"][m]
        C = C + N * (1 / P - 1 / close); P = close
        coin_path[m - s] = C; eq[m - s] = C * spot_mark[m - s]
        last = m + 1
    span(last, n)
    # terminal: close contract at CM close with sell/buy slippage and fee, then sell coin at spot close
    m = n - 1
    if N:
        dN = -N; F = P * (1 + bs) if dN > 0 else P * (1 - ss)
        C += dN * (1 / F - 1 / P) - fee * abs(dN) / F; N = 0.0
    coin_path[-1] = C
    eq[-1] = C * spot_mark[-1] * (1 - ss) * (1 - fee)
    return eq, dict(C0=C0, C0_hat=C0_hat, final_coin=C, coin_growth=C / C0 - 1, funding_coin=fund_coin, pressure_hits=pressure_hits)


def main():
    cfg = json.load(open(sys.argv[1])); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    start, end = (pd.Timestamp(x, tz="UTC") for x in cfg["window"])
    load_start = pd.Timestamp(cfg["load_start"], tz="UTC")
    um = grid_frame(cfg["um"], load_start, end, cfg["um_funding"])
    sp = grid_frame(cfg["spot"], load_start, end)
    cm = grid_frame(cfg["cm"], load_start, end, cfg["cm_funding"])
    feat = canonical_features(um, pd.Timestamp(cfg["signal_origin"], tz="UTC"), start)
    feat.to_csv(out / f"{cfg['prefix']}features.csv")
    res = {}
    for cost in COSTS:
        for v in ("V1_USDM", "P_USDM"):
            res[(v, cost)] = run_linear(um, feat, v, cost, start)
        for v in ("V2_SPOT", "P_SPOT"):
            res[(v, cost)] = run_spot(sp, feat, v, cost, start)
        for v in ("V3_CM_DeltaMatched", "P_CM_DeltaMatched", "V4_CM_CoinAccum", "HODL_1x_Coin"):
            res[(v, cost)] = run_coin(cm, sp, feat, v, cost, start)
    rep = {}
    for (v, cost), (eq, meta) in res.items():
        daily = eq[1439::1440]
        a = pd.read_csv(Path(cfg["archived_dir"]) / f"{cfg['prefix']}{v}__{cost}" / "daily.csv", index_col=0)
        ae = a["equity"].to_numpy()
        rel = np.max(np.abs(daily - ae) / np.abs(ae))
        rets = daily / np.r_[CAP, daily[:-1]] - 1
        rmax = np.max(np.abs(rets - a["return"].to_numpy()))
        mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
        rep[f"{v}__{cost}"] = dict(final=float(daily[-1]), archived_final=float(ae[-1]), max_rel_daily_equity_diff=float(rel), max_abs_return_diff=float(rmax), mdd=mdd, **{k: float(x) for k, x in meta.items()})
    json.dump(rep, open(out / f"{cfg['prefix']}report.json", "w"), indent=1)
    df = pd.DataFrame(rep).T
    print(df[["final", "archived_final", "max_rel_daily_equity_diff", "max_abs_return_diff", "mdd"] + [c for c in ("coin_growth", "pressure_hits", "clips") if c in df]].to_string())


if __name__ == "__main__":
    main()
