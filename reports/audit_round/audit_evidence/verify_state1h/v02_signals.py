"""v02: per-signal chain diagnostics from my own trace (v01), BASELINE, IS only. Recomputes the S-1..S-13 counts."""
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/user/ppvp_quant")
import numpy as np
import pandas as pd

D = Path(__file__).resolve().parent
O = pickle.load(open(D / "v00_objs.pkl", "rb"))
T = pickle.load(open(D / "v01_trace.pkl", "rb"))
df1h, sig, boxes, trades, args = O["df1h"], O["signals"], O["boxes"], O["trades"], O["args"]
o, h, l, c = (df1h[k].to_numpy(float) for k in ("open", "high", "low", "close"))
atr = args[5]; box_val = args[9]; box_vah = args[10]
k1 = k2 = 0.5
idx = df1h.index
st, zlo, zhi, box = T["state"], T["zlo"], T["zhi"], T["box"]
chains, chain_id, sk = T["chains"], T["chain_id"], T["switch_kind"]
touched = l <= zhi
filled_bars = set(trades["entry_bar"].to_numpy().tolist())
led = pd.read_csv("/home/user/ppvp_quant/reports/audit_round/signal_opportunities_B_original.csv")
led_status = dict(zip(led["signal_bar"], led["status"]))

rows = []
for s in np.flatnonzero(T["sig"]):
    cid = chain_id[s]; ch = chains[cid]; cs = ch["start"]
    t2s = int(T["t2start"][s])
    d0 = t2s
    while st[d0 - 1] == 3:
        d0 -= 1
    assert d0 < t2s and all(st[j] == 3 for j in range(d0, t2s))
    t1_last = d0 - 1
    assert T["t1last_at"][t2s - 1] == t1_last
    mh = h[d0:t2s].max()
    assert mh == T["maxhigh_at"][t2s - 1]
    bounce = (mh - zhi[t2s]) / atr[t2s]
    gap = t2s - t1_last
    bounce_dep_own = (mh - zhi[d0]) / atr[d0]           # departure-bar band top and departure-bar ATR
    bounce_dep_sameatr = (mh - zhi[d0]) / atr[t2s]      # departure-bar band top, touching-bar ATR
    bars = np.arange(cs, s + 1)
    tb = bars[touched[bars]]
    # every chain bar is either touched (in a test) or DEPARTED
    assert all((touched[j] or st[j] == 3) for j in bars[:-1])
    ev = ch["events"]
    rolls = [e for e in ev if e[0] == "roll"]
    merges = [e for e in ev if e[0] == "merge"]
    last_roll_bar = rolls[-1][1] if rolls else cs
    merges_final_t1 = [e for e in merges if last_roll_bar <= e[1] < t2s]
    chain_min = l[tb].min()
    anchor = min(T["t1low"][s], T["t2low"][s])
    # final first-test bars: touched bars from final seq start to t1_last
    if rolls:
        # final seq start = start of the failed test 2 that rolled last = first touched bar after the DEPARTED run before it
        rb = rolls[-1][1]
        fs = rb - 1
        while touched[fs - 1] and fs - 1 >= cs and st[fs - 1] == 4:
            fs -= 1
        final_seq_start = fs
    else:
        final_seq_start = cs
    t1bars = np.array([j for j in range(final_seq_start, t1_last + 1) if touched[j]])
    t2bars = np.arange(t2s, s + 1)
    carries = ch["carries"]
    sb, eb = ch["start_box"], box[s]
    fsb = box[final_seq_start]
    prev = cs - 1
    zl_cs, zh_cs = zlo[cs], zhi[cs]
    if l[prev] > zh_cs:
        prev_cls = "prev bar fully above band"
    elif c[prev] > zh_cs:
        prev_cls = "prev close above band, low in/below"
    elif c[prev] >= zl_cs:
        prev_cls = "prev close inside band"
    else:
        prev_cls = "prev close below band"
    rows.append({
        "s": s, "signal_time": idx[s] + pd.Timedelta(hours=1), "status": led_status[s], "filled": s in filled_bars,
        "cs": cs, "final_seq_start": final_seq_start, "t1_last": t1_last, "d0": d0, "t2s": t2s,
        "bounce": bounce, "gap": gap, "by_bounce": bounce >= 1.0, "by_bars": gap >= 6,
        "bounce_dep_own": bounce_dep_own, "bounce_dep_sameatr": bounce_dep_sameatr,
        "n_dep_bars": t2s - d0, "n_merge_chain": len(merges), "n_merge_final_t1": len(merges_final_t1),
        "n_roll": len(rolls), "n_carry": len(carries), "start_box": sb, "final_seq_box": fsb, "sig_box": eb,
        "dVAL_atr": abs(box_val[eb] - box_val[sb]) / atr[s], "dVAH_pct": box_vah[eb] / box_vah[sb] - 1.0,
        "t2s_touch_startbox": l[t2s] <= box_val[sb] + k2 * atr[t2s],
        "t2s_touch_finalseqbox": l[t2s] <= box_val[fsb] + k2 * atr[t2s],
        "dep_only_moved": l[d0] <= zhi[d0 - 1], "dep_switch": box[d0] != box[d0 - 1],
        "t2_only_moved": l[t2s] > zhi[t2s - 1], "t2_switch": box[t2s] != box[t2s - 1],
        "t1low": T["t1low"][s], "t2low": T["t2low"][s], "chain_min": chain_min, "anchor": anchor,
        "earlier_lower": chain_min < anchor, "sl_gap_atr": (anchor - chain_min) / atr[s],
        "sl_gap_pct": (anchor - chain_min) / c[s],
        "t1low_above_zh_sig": T["t1low"][s] > zhi[s], "t1low_below_zl_sig": T["t1low"][s] < zlo[s],
        "t1low_below_zl_atr": (zlo[s] - T["t1low"][s]) / atr[s],
        "stop_dist_atr": (c[s] - T["sl"][s]) / atr[s],
        "start_at_activation": sk[cs] == "arm", "prev_cls": prev_cls,
        "lag": s - t2s, "close_lt_val": c[s] < box_val[eb], "close_lt_t1low": c[s] < T["t1low"][s],
        "t1bar_wholly_below": bool(len(t1bars) and (h[t1bars] < zlo[t1bars]).any()),
        "t2bar_wholly_below": bool((h[t2bars] < zlo[t2bars]).any()),
        "chainbar_wholly_below": bool((h[tb] < zlo[tb]).any()),
        "eng": bool(T["pat"][s] & 2), "eng_prev_used": ch["eng_prev_used"],
        "flag_ledger": bool(led.set_index("signal_bar").loc[s, "profile_changed_between_tests"]),
        "switch_inside_final_t1": any(final_seq_start < cb <= t1_last for cb, _, _ in carries),
        "switch_before_final_seq": any(cb <= final_seq_start for cb, _, _ in carries),
        "carry_bars": [cb for cb, _, _ in carries],
    })
R = pd.DataFrame(rows)
R.to_csv(D / "v02_signals.csv", index=False)
F = R[R["filled"]]
print("raw", len(R), "filled", len(F), Counter(R["status"]))

print("\n== S-1 separation ==")
print("filled: by_bounce", int(F.by_bounce.sum()), "| by_bars", int(F.by_bars.sum()),
      "| bars-only", int((F.by_bars & ~F.by_bounce).sum()), "| bounce-only", int((F.by_bounce & ~F.by_bars).sum()))
print("all 120: bars-only", int((R.by_bars & ~R.by_bounce).sum()), R.loc[R.by_bars & ~R.by_bounce, ["signal_time", "status", "bounce", "gap"]].to_string())
print("filled min bounce %.4f | <1.10: %d | <1.05: %d" % (F.bounce.min(), (F.bounce < 1.10).sum(), (F.bounce < 1.05).sum()))
print("filled gap==2 (single departure bar):", int((F.gap == 2).sum()), "| n_dep_bars==1:", int((F.n_dep_bars == 1).sum()))
print("filled gap quantiles", F.gap.quantile([0, .25, .5]).to_dict())
print("filled: bounce vs departure-bar band (own ATR) < 1:", int((F.bounce_dep_own < 1).sum()),
      "| (touch-bar ATR) < 1:", int((F.bounce_dep_sameatr < 1).sum()))
print("   of those (own ATR) also by_bars:", int(((F.bounce_dep_own < 1) & F.by_bars).sum()),
      "| (touch ATR) also by_bars:", int(((F.bounce_dep_sameatr < 1) & F.by_bars).sum()))
print(F.loc[F.bounce_dep_own < 1, ["signal_time", "bounce", "bounce_dep_own", "bounce_dep_sameatr", "gap"]].to_string())
print("filled with >=1 merge in chain:", int((F.n_merge_chain >= 1).sum()), dict(Counter(F.n_merge_chain)),
      "| with >=1 merge in final t1:", int((F.n_merge_final_t1 >= 1).sum()))
# ledger comparison
L = led.set_index("signal_bar")
okdep = all(L.loc[r.s, "departure_bar"] == r.d0 and L.loc[r.s, "t1_last_bar"] == r.t1_last and
            L.loc[r.s, "touch_gap_bars_from_t1_last"] == r.gap and abs(L.loc[r.s, "departure_height_atr"] - r.bounce) < 1e-9
            for r in R.itertuples())
lab = {True: {True: "bounce+bars", False: "bounce"}, False: {True: "bars_only", False: "unexplained"}}
oklab = all(L.loc[r.s, "second_test_qualified_by"] == lab[r.by_bounce][r.by_bars] for r in R.itertuples())
print("ledger departure columns match my trace (120):", okdep, "| qualified_by labels:", oklab)

print("\n== S-2 carry ==")
print("filled with >=1 carry in chain:", int((F.n_carry >= 1).sum()), dict(Counter(F.n_carry)),
      "| start box != signal box:", int((F.start_box != F.sig_box).sum()))
C2 = F[F.n_carry >= 1]
print("|dVAL|/ATR(sig) max %.3f median %.3f | dVAH%% min %.2f max %.2f" % (C2.dVAL_atr.max(), C2.dVAL_atr.median(),
      100 * C2.dVAH_pct.min(), 100 * C2.dVAH_pct.max()))
print("t2 start not a touch of chain-start box band:", int((~F.t2s_touch_startbox).sum()),
      "| of final-seq-start box band:", int((~F.t2s_touch_finalseqbox).sum()))
mv = F.dep_only_moved | F.t2_only_moved
print("departure or second touch exists only vs moved band:", int(mv.sum()), "| dep-only", int(F.dep_only_moved.sum()),
      "| t2-only", int(F.t2_only_moved.sum()),
      "| with box switch at that bar:", int(((F.dep_only_moved & F.dep_switch) | (F.t2_only_moved & F.t2_switch)).sum()))
print(F.loc[mv, ["signal_time", "d0", "t2s", "dep_only_moved", "dep_switch", "t2_only_moved", "t2_switch"]].assign(
    d0_t=lambda x: idx[x.d0], t2s_t=lambda x: idx[x.t2s]).to_string())
print("t1low above signal-bar band top:", int(F.t1low_above_zh_sig.sum()), "| below signal-bar band low:", int(F.t1low_below_zl_sig.sum()))

print("\n== S-3 roll ==")
print("filled roll dist", dict(Counter(F.n_roll)), "| >=1:", int((F.n_roll >= 1).sum()))
E = F[F.earlier_lower]
print("earlier lower pullback low in chain:", len(E), "| all have rolls:", bool((E.n_roll >= 1).all()),
      "| SL gap ATR %.3f-%.3f | %% of close %.3f-%.3f" % (E.sl_gap_atr.min(), E.sl_gap_atr.max(), 100 * E.sl_gap_pct.min(), 100 * E.sl_gap_pct.max()))

print("\n== S-4 first test at activation ==")
print(pd.crosstab(F.prev_cls, F.start_at_activation))
A4 = F[F.start_at_activation & (F.prev_cls == "prev close inside band")]
print("filled: activation start with prior close inside band:", len(A4), "| of which no roll:", int((A4.n_roll == 0).sum()))

print("\n== S-5 lag ==")
print("filled lag dist", dict(sorted(Counter(F.lag).items())), "| lag>=1:", int((F.lag >= 1).sum()), "| all-120 max lag", int(R.lag.max()))

print("\n== S-6 reclaim ==")
print("filled close<VAL", int(F.close_lt_val.sum()), "| close<t1low", int(F.close_lt_t1low.sum()),
      "| either", int((F.close_lt_val | F.close_lt_t1low).sum()), "| both", int((F.close_lt_val & F.close_lt_t1low).sum()))

print("\n== S-7 touch geometry ==")
print("filled: final-t1 bar wholly below band", int(F.t1bar_wholly_below.sum()), "| t2 bar wholly below", int(F.t2bar_wholly_below.sum()),
      "| any chain test bar wholly below", int(F.chainbar_wholly_below.sum()))
print("t1low below signal band low: n", int(F.t1low_below_zl_sig.sum()), "max ATR %.3f" % F.loc[F.t1low_below_zl_sig, "t1low_below_zl_atr"].max(),
      "| stop dist ATR max %.3f" % F.stop_dist_atr.max())

print("\n== S-10 ==")
print("signals with engulfing", int(R.eng.sum()), "| prior low used", int(R.eng_prev_used.sum()))

print("\n== S-13 ledger flag ==")
print("filled ledger flag True:", int(F.flag_ledger.sum()), "| filled with >=1 carry:", int((F.n_carry >= 1).sum()))
M = F[(F.n_carry >= 1) & ~F.flag_ledger]
print("missed:", len(M), "| involve rolls:", int((M.n_roll >= 1).sum()), "| switch inside final t1 (no roll):",
      int(((M.n_roll == 0) & M.switch_inside_final_t1).sum()))
print("flag True but no carry:", int((F.flag_ledger & (F.n_carry == 0)).sum()))
print(M[["signal_time", "n_roll", "cs", "final_seq_start", "t1_last", "t2s", "carry_bars", "start_box", "final_seq_box", "sig_box"]].to_string())
