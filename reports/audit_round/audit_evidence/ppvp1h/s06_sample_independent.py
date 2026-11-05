"""s06: independent verification on a sample of IS trades (no engine code): own Wilder ATR, own boxes (indep.py),
own as-of box mapping, own zone/touch/departure/pattern checks at the first-test start, first-test low bar,
second-test start and signal bar. Sample = 15 seeded-random trades + every trade whose box changed in-sequence."""
import sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
d = pickle.load(open(HERE / "is_objects.pkl", "rb"))
df1h, trades = d["df1h"], d["trades"]
bi = pd.read_pickle(HERE / "boxes_indep.pkl")
pv = pd.read_csv(HERE / "profile_versions_per_signal.csv")
opp = pd.read_csv("/home/user/ppvp_quant/reports/audit_round/signal_opportunities_B_original.csv")
o, h, l, c, v = (df1h[k].to_numpy(np.float64) for k in ("open", "high", "low", "close", "volume"))
n = len(df1h)

# own Wilder ATR(14), known at the open of bar i = value at close of i-1
tr = np.empty(n)
tr[0] = h[0] - l[0]
tr[1:] = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
atr = np.full(n, np.nan)
atr[13] = tr[:14].mean()
for i in range(14, n):
    atr[i] = (atr[i - 1] * 13 + tr[i]) / 14
atr_prev = np.r_[np.nan, atr[:-1]]
sig = d["sig"]
print("own ATR == engine atr column (max abs diff):", np.nanmax(np.abs(atr_prev - sig["atr"].to_numpy())))

avail = pd.DatetimeIndex(bi["available_time"]).as_unit("ns").asi8
open_ns = df1h.index.as_unit("ns").asi8


def box_asof(i):                 # latest box whose available_time <= open of bar i
    return int(np.searchsorted(avail, open_ns[i], side="right") - 1)


def pinbar(i):
    rng = h[i] - l[i]
    body = abs(c[i] - o[i])
    lower = min(o[i], c[i]) - l[i]
    return rng > 0 and lower >= 0.6 * rng and lower >= 2.0 * body and (c[i] - l[i]) >= 0.5 * rng


def engulf(i):
    po, pc = o[i - 1], c[i - 1]
    pb = po - pc
    if not (pb > 0 and c[i] > o[i]):
        return False
    lo_cur = min(o[i], pc) if abs(o[i] - pc) <= 0.0002 * pc else o[i]
    return (min(c[i], po) - max(lo_cur, pc)) / pb >= 1.0 - 1e-12


filled = pv[pv["status"] == "filled"].reset_index(drop=True)
rng = np.random.default_rng(20261105 + 31)
pick = set(rng.choice(len(filled), 15, replace=False).tolist())
pick |= set(np.flatnonzero(filled["n_box_switches_in_sequence"] > 0).tolist())
rows = []
for r in sorted(pick):
    t = filled.iloc[r]
    s, ss, t1, t2 = int(t.signal_bar), int(t.seq_start_bar), int(t.t1_bar), int(t.t2_start_bar)
    ks = {name: box_asof(b) for name, b in (("start", ss), ("low", t1), ("t2", t2), ("sig", s))}
    eng = {"start": t.box_at_seq_start, "low": t.box_at_t1_low, "t2": t.box_at_t2_start, "sig": t.box_at_signal}
    ids_ok = all(ks[k] == eng[k] for k in ks)
    zone = lambda b, k: (bi.VAL[k] - 0.5 * atr_prev[b], bi.VAL[k] + 0.5 * atr_prev[b])
    zl_s, zh_s = zone(s, ks["sig"])
    zl_t2, zh_t2 = zone(t2, ks["t2"])
    zl_ss, zh_ss = zone(ss, ks["start"])
    # departure: some bar strictly between first test and t2_start entirely above its own band
    dep = any(l[j] > bi.VAL[box_asof(j)] + 0.5 * atr_prev[j] for j in range(ss + 1, t2))
    trow = trades[trades["entry_bar"] == s].iloc[0]
    rows.append({"signal_open": df1h.index[s], "boxes(start,low,t2,sig)": (ks["start"], ks["low"], ks["t2"], ks["sig"]),
                 "ids_match_engine": ids_ok,
                 "start_touch": l[ss] <= zh_ss, "t2_touch": l[t2] <= zh_t2, "departure_above_band": dep,
                 "sig_touch": l[s] <= zh_s, "sig_close>=zone_lo": c[s] >= zl_s,
                 "pattern": ("P" if pinbar(s) else "") + ("E" if engulf(s) else ""),
                 "TP==indep VAH": float(trow.tp) == float(bi.VAH[ks["sig"]]),
                 "box_avail<=bar_open": bool(avail[ks["sig"]] <= open_ns[s])})
res = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(res.to_string())
print("sample size", len(res), "| all ids match:", bool(res.ids_match_engine.all()),
      "| all touch/departure/close/pattern checks pass:",
      bool((res.start_touch & res.t2_touch & res.departure_above_band & res.sig_touch & res["sig_close>=zone_lo"]).all()
           and (res.pattern != "").all()), "| TP==VAH all:", bool(res["TP==indep VAH"].all()))
