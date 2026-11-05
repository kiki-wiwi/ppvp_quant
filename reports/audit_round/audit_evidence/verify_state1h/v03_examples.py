"""v03: bar-level checks of every worked example in S-1..S-13 + S-8 requirement fates + S-9 state stats. BASELINE, IS only."""
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
df1h, sig, boxes, args = O["df1h"], O["signals"], O["boxes"], O["args"]
o, h, l, c, v = (df1h[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
atr, vsma = args[5], args[6]
box_val, box_vah, box_ws = args[9], args[10], boxes["window_start"]
idx = df1h.index
st, zlo, zhi, box, ev = T["state"], T["zlo"], T["zhi"], T["box"], T["ev"]
SN = {0: "NO_BOX", 1: "ARMED", 2: "TOUCH1", 3: "DEPART", 4: "TOUCH2", 5: "INVALID", 6: "EXH"}
R = pd.read_csv(D / "v02_signals.csv", parse_dates=["signal_time"])


def bar(t):
    return int(idx.get_loc(pd.Timestamp(t, tz="UTC")))


def show(i0, i1, note=""):
    print(f"--- {note} bars {idx[i0]} .. {idx[i1]}")
    for j in range(i0, i1 + 1):
        print(f"{j} {idx[j]:%Y-%m-%d %H:%M} O{o[j]:.2f} H{h[j]:.2f} L{l[j]:.2f} C{c[j]:.2f} box{box[j]} VAL{box_val[box[j]]:.2f} "
              f"zl{zlo[j]:.3f} zh{zhi[j]:.3f} atr{atr[j]:.4f} pre={SN[T['pre_state'][j]]} post={SN[st[j]]} "
              f"t1last={T['t1last_at'][j]} mh={T['maxhigh_at'][j]:.2f} {ev[j]} {T['switch_kind'][j]}")


def srow(ts):
    r = R[R.signal_time == pd.Timestamp(ts, tz="UTC")]
    assert len(r) == 1, ts
    return r.iloc[0]

# ---------------- S-1 example ----------------
r = srow("2020-01-22 20:00")
s = int(r.s); show(int(r.cs), s, "S-1 2020-01-22 20:00")
mh = h[int(r.d0):int(r.t2s)].max(); t2s = int(r.t2s); d0 = int(r.d0)
print("S-1 bounce (touch band, touch ATR) %.4f | (dep band, touch ATR) %.4f | (dep band, dep ATR) %.4f | gap %d | lag %d" % (
    (mh - zhi[t2s]) / atr[t2s], (mh - zhi[d0]) / atr[t2s], (mh - zhi[d0]) / atr[d0], t2s - (d0 - 1), s - t2s))
print("pin", bool(sig["pin"].iloc[s]), "eng", bool(sig["eng"].iloc[s]))

# ---------------- S-2 example ----------------
r = srow("2023-03-07 01:00")
s = int(r.s); cs = int(r.cs)
print("\nS-2 chain start", idx[cs], "t1_last", idx[int(r.t1_last)], "d0", idx[int(r.d0)], "t2s", idx[int(r.t2s)], "signal bar", idx[s])
show(cs, int(r.d0), "S-2 start..departure")
show(s - 1, s, "S-2 last two bars")
print("departure bars", int(r.t2s) - int(r.d0), "all fully above own band:",
      bool((l[int(r.d0):int(r.t2s)] > zhi[int(r.d0):int(r.t2s)]).all()))
b0, b1 = box[s - 1], box[s]
print("boxes", b0, b1, "VAL", box_val[b0], box_val[b1], "dVAL/ATR(s)", (box_val[b1] - box_val[b0]) / atr[s],
      "seqval", T["seqval_at"][s - 1], "VAH", box_vah[b0], box_vah[b1])
print("signal bar low", l[s], "zh(s)", zhi[s], "zh(s-1)", zhi[s - 1], "old-box band top with ATR(s)", box_val[b0] + 0.5 * atr[s],
      "low - old top", l[s] - (box_val[b0] + 0.5 * atr[s]))
print("SL", T["sl"][s], "= t1low", T["t1low"][s], "t2low", T["t2low"][s], "- 0.2*", atr[s], "->", min(T["t1low"][s], T["t2low"][s]) - 0.2 * atr[s],
      "| TP", T["tp"][s], "pattern", T["pat"][s])

# ---------------- S-3 example ----------------
r = srow("2021-03-13 05:00")
s = int(r.s); cs = int(r.cs)
print("\nS-3 rolls", int(r.n_roll), "chain_min", r.chain_min, "t1low", r.t1low, "t2low", r.t2low, "ATR", atr[s], "SL", T["sl"][s],
      "all-tests stop", r.chain_min - 0.2 * atr[s], "diff ATR", (T["sl"][s] - (r.chain_min - 0.2 * atr[s])) / atr[s],
      "pct of close", (T["sl"][s] - (r.chain_min - 0.2 * atr[s])) / c[s])
for e in T["chains"][T["chain_id"][s]]["events"]:
    print("   chain event", e[0], idx[e[1]], e[2:])

# ---------------- S-4 example ----------------
r = srow("2020-03-22 10:00")
s = int(r.s); cs = int(r.cs)
print("\nS-4 chain start", idx[cs], "switch", T["switch_kind"][cs], "box", box[cs], "prev box", box[cs - 1], "VAL", box_val[box[cs]],
      "band [%.2f, %.2f]" % (zlo[cs], zhi[cs]), "prev bar H/L/C", h[cs - 1], l[cs - 1], c[cs - 1],
      "prev-box band [%.2f, %.2f]" % (zlo[cs - 1], zhi[cs - 1]), "activation bar low", l[cs], "n_roll", int(r.n_roll))
# where did price come from (relative to the new band, ATR at cs)?
zl, zh = zlo[cs], zhi[cs]
j = cs - 1
while j >= 0 and not (l[j] > zh or h[j] < zl):
    j -= 1
print("   last bar fully outside new band before activation:", idx[j], "above" if l[j] > zh else "below", "bars back", cs - j)
# all 22 inside-band activation trades: from above or below?
F = R[R.filled & R.start_at_activation & (R.prev_cls == "prev close inside band")]
came = []
for rr in F.itertuples():
    cs = rr.cs; zl, zh = zlo[cs], zhi[cs]; j = cs - 1
    while j >= 0 and not (l[j] > zh or h[j] < zl):
        j -= 1
    came.append("above" if l[j] > zh else "below")
print("   22 inside-band activation trades: last excursion outside the new band before activation:", Counter(came),
      "| no-roll subset:", Counter(cc for cc, nr in zip(came, F.n_roll) if nr == 0))

# ---------------- S-6 example ----------------
r = srow("2020-08-19 07:00")
s = int(r.s)
print("\nS-6", idx[s], "VAL", box_val[box[s]], "zl", zlo[s], "zh", zhi[s], "t1low", r.t1low, "low", l[s], "close", c[s],
      "t1low>zh(s)", r.t1low > zhi[s], "n_carry", int(r.n_carry), "n_roll", int(r.n_roll))
show(int(r.t1_last) - 1, int(r.t1_last), "S-6 last test-1 bars")

# ---------------- S-7 examples ----------------
j = bar("2020-07-02 16:00")
print("\nS-7a", idx[j], "OHLC", o[j], h[j], l[j], c[j], "band [%.2f, %.2f]" % (zlo[j], zhi[j]), "below band low by", zlo[j] - l[j],
      "body/ATR", abs(c[j] - o[j]) / atr[j], "brk", bool(T["brk"][j]), "state", SN[st[j]], ev[j])
r = srow("2020-07-05 08:00")
s = int(r.s)
print("   signal", idx[s], "t1low", r.t1low, "t1bar", idx[int(T['t1bar'][s])], "SL", T["sl"][s], "ATR", atr[s],
      "(close-SL)/ATR", (c[s] - T["sl"][s]) / atr[s], "t2low", r.t2low)
j = bar("2023-02-13 17:00")
show(j - 2, j + 1, "S-7b")
print("   VAL change at 16:00:", box_val[box[j - 1]] - box_val[box[j - 2]])

# ---------------- S-8 requirements ----------------
reqs = T["reqs"]
fate = Counter((q["kind"], q["fate"][0] if q["fate"] else "alive") for q in reqs)
print("\nS-8 requirement fates:", dict(fate))
sigreq = [q for q in reqs if q["kind"] == "signal"]
claimed_later = [q for q in sigreq if any(cb > q["bar"] for cb, _ in q["claims"])]
print("signal reqs", len(sigreq), "| claimed by >=1 later same-zone box switch:", len(claimed_later),
      "| bars to fate median/max", np.median([q["fate"][1] - q["bar"] for q in sigreq]), max(q["fate"][1] - q["bar"] for q in sigreq))
q = [q for q in sigreq if idx[q["bar"]] == pd.Timestamp("2020-02-03 11:00", tz="UTC")][0]
fb = q["fate"][1]
seg = st[q["bar"] + 1:fb]
print("example req:", idx[q["bar"]], "box", q["box"], "VAL", q["val"], "claims", [(idx[cb], b, box_val[b]) for cb, b in q["claims"]],
      "fate", q["fate"][0], idx[fb], "| states between (exclusive):", dict(Counter(SN[x] for x in seg)),
      "| fate-bar state", SN[st[fb]], "| incl fate bar:", dict(Counter(SN[x] for x in st[q["bar"] + 1:fb + 1])))

# ---------------- S-9 ----------------
occ = Counter(st.tolist())
print("\nS-9 occupancy", {SN[k]: round(vv / len(st), 4) for k, vv in sorted(occ.items())})
prev = np.r_[st[0], st[:-1]]
into = (st == 5) & np.isin(prev, [1, 2, 3, 4])
nb = into & T["brk"]
sw = into & ~T["brk"]
print("transitions into INVALID from 1-4 (state array):", int(into.sum()), "| on a breakdown bar:", int(nb.sum()),
      "| without breakdown (switch):", int(sw.sum()), Counter(T["switch_kind"][sw]))
endseq = (st == 5) & np.isin(prev, [2, 3, 4])
print("in-progress sequences ended in INVALID (state array):", int(endseq.sum()), "| breakdown bars:", int(T["brk"].sum()),
      "| breakdown with close>=zl:", int((T["brk"] & (c >= zlo)).sum()))
j = bar("2020-06-11 15:00")
print("S-9 example", idx[j], "box", box[j], "VAL", box_val[box[j]], "band [%.2f, %.2f]" % (zlo[j], zhi[j]), "OHLC", o[j], h[j], l[j], c[j],
      "body/ATR %.3f vol/sma %.3f" % ((o[j] - c[j]) / atr[j], v[j] / vsma[j]), "pre", SN[T["pre_state"][j]], "post", SN[st[j]], ev[j])
show(j + 1, j + 1, "S-9 next bar")
print("   box", box[j + 1], "window_start", box_ws.iloc[box[j + 1]], "VAL", box_val[box[j + 1]])
k = j + 1
while st[k] == 5:
    k += 1
print("   first non-INVALID bar after:", idx[k], SN[st[k]], "box", box[k], "VAL", box_val[box[k]], T["switch_kind"][k],
      "| INVALID bars", k - (j))
# ---------------- S-12 example ----------------
for t in ("2023-06-09 17:00", "2023-06-09 18:00", "2023-06-09 19:00", "2023-06-09 20:00"):
    j = bar(t)
    print("S-12", t, "box", box[j], "low", l[j], "zh %.2f" % zhi[j], "state", SN[st[j]], "pre", SN[T["pre_state"][j]], ev[j])
# running t1 low at those bars: reconstruct from the chain containing them
j = bar("2023-06-09 17:00")
cid = T["chain_id"][j]
ch = T["chains"][cid]
print("   chain start", idx[ch["start"]], "signal?", "signal" in ch, idx[ch["signal"]] if "signal" in ch else None)
tl = min(l[x] for x in range(ch["start"], j) if l[x] <= zhi[x] and st[x] in (2,))
print("   min low of TOUCH1 bars before 17:00 in this chain:", tl)
