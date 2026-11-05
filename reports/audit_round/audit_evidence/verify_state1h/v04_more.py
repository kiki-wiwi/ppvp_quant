"""v04: remaining counts. S-11/S-12 wick-piercing population, S-4 in-band activation starts over all 428 chains,
S-7 depth of the first-test low below the band at its own bar, S-2 'never touched' over all test-2 bars. BASELINE, IS only."""
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/user/ppvp_quant")
import numpy as np
import pandas as pd
from ppvp_audit.engine_path import tick_size

D = Path(__file__).resolve().parent
O = pickle.load(open(D / "v00_objs.pkl", "rb"))
T = pickle.load(open(D / "v01_trace.pkl", "rb"))
df1h, args = O["df1h"], O["args"]
o, h, l, c = (df1h[k].to_numpy(float) for k in ("open", "high", "low", "close"))
atr, box_val = args[5], args[9]
idx = df1h.index
st, zlo, zhi, box = T["state"], T["zlo"], T["zhi"], T["box"]
st_mid, t1m = T["st_mid"], T["t1low_mid"]
tick = tick_size(idx)
R = pd.read_csv(D / "v02_signals.csv")
F = R[R.filled]

# ---- S-11 / S-12: DEPARTED/TOUCH2 bars whose low pierces the running first-test low (>=1 tick) or the band low,
#      with close >= max(VAL, first-test low)
val_cur = np.where(box >= 0, box_val[np.clip(box, 0, None)], np.nan)
ok = np.isin(st_mid, [3, 4]) & ~np.isnan(t1m) & ~np.isnan(zlo)
with np.errstate(invalid="ignore"):
    pierce_t1 = l <= t1m - tick
    pierce_zl = l < zlo
    reclaim = c >= np.maximum(val_cur, t1m)
pop = ok & (pierce_t1 | pierce_zl) & reclaim
print("S-11 population", int(pop.sum()), "| pierce t1 low", int((pop & pierce_t1).sum()), "| pierce band low", int((pop & pierce_zl).sum()))
print(pd.crosstab(pd.Series(st_mid[pop], name="state_mid"), pd.Series(st[pop], name="state_after")))
evs = [tuple(T["ev"][i]) for i in np.flatnonzero(pop)]
print("events on these bars:", Counter(e for t in evs for e in t))
print("-> INVALID:", int((pop & (st == 5)).sum()), "| breakdown flag:", int((pop & T["brk"]).sum()))
nt = pop & (l > zhi)
print("piercing but not a touch (low > band top):", int(nt.sum()), [str(idx[i]) for i in np.flatnonzero(nt)])

# ---- S-4: all chain starts
ch = T["chains"]
cnt = Counter()
for cid, cc in ch.items():
    cs = cc["start"]
    act = T["switch_kind"][cs] == "arm"
    p = cs - 1
    if l[p] > zhi[cs]:
        k = "prev fully above"
    elif c[p] > zhi[cs]:
        k = "prev close above"
    elif c[p] >= zlo[cs]:
        k = "prev close inside"
    else:
        k = "prev close below"
    cnt[(act, k)] += 1
print("\nS-4 all", len(ch), "chain starts:", dict(cnt))

# ---- S-7: depth of the first-test low below the band at the bar where it was made
t1b = F.s.map(lambda s: int(T["t1bar"][s])).to_numpy()
t1low = F.t1low.to_numpy()
below_own = t1low < zlo[t1b]
print("\nS-7 first-test low below the band low of its own bar:", int(below_own.sum()), "of", len(F),
      "| max depth ATR(own bar) %.3f" % ((zlo[t1b] - t1low) / atr[t1b])[below_own].max(),
      "| vs signal-bar band (from v02):", int(F.t1low_below_zl_sig.sum()))
depth = (zlo[t1b] - t1low) / atr[t1b]
print("   depth quantiles (own bar, ATR) among those below:", np.quantile(depth[below_own], [0.5, 0.9, 1.0]).round(3))

# ---- S-2: does ANY test-2 bar touch the chain-start box band?
never = 0
lst = []
for r in F.itertuples():
    if r.t2s_touch_startbox:
        continue
    bars = np.arange(r.t2s, r.s + 1)
    any_touch = bool((l[bars] <= box_val[r.start_box] + 0.5 * atr[bars]).any())
    lst.append((r.signal_time, any_touch))
    if not any_touch:
        never += 1
print("\nS-2 t2-start not touching chain-start band:", len(lst), "| no test-2 bar ever touched it:", never, lst)
