"""v06: P-4 effect of the POC tie-break on the Figure-1 'reached POC (strict)' diagnostic for the 6 tied trades;
P-7 nuance: of the boxes with an INVALID bar while current, how many were INVALID from their first bar (inherited)."""
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
V = Path(__file__).resolve().parent; sys.path.insert(0, str(V))
from vp_indep import *
D = pickle.load(open(V / "is_objs.pkl", "rb")); X = pickle.load(open(V / "v02_out.pkl", "rb"))
df1h, boxes, sig, tr = D["df1h"], D["boxes"], D["sig"], D["trades"]
B, bt = X["B"], X["bt"]
l1, h1, v1 = (df1h[k].to_numpy(float) for k in ("low", "high", "volume"))
tied = [41, 334, 457, 519, 748, 976]
rows = []
for _, t in tr[tr.box_id.isin(tied)].iterrows():
    k = int(t.box_id); s, e = int(B.s[k]), int(B.e[k])
    edges = np.linspace(l1[s:e].min(), h1[s:e].max(), 51)
    hp = levels(hist_uniform(l1[s:e], h1[s:e], v1[s:e], edges), edges, poc_rule="highest")["POC"]
    a, b = int(t.entry_bar) + 1, int(t.exit_bar)
    mfe = h1[a:b].max() if b > a else np.nan
    ep = t.entry_px
    rows.append(dict(entry=t.entry_time, box=k, exit=t.exit_reason, entry_px=round(ep, 1), POC_low=round(t.poc, 1), POC_high=round(hp, 1),
                     strict_mfe=round(mfe, 1) if mfe == mfe else None,
                     above_low=bool(t.poc > ep), reached_low=bool(t.poc > ep and mfe >= t.poc),
                     above_high=bool(hp > ep), reached_high=bool(hp > ep and mfe >= hp)))
R = pd.DataFrame(rows); print(R.to_string(index=False))
print("Figure-1 left-panel relevant (SL, POC above entry, full bars) flips:",
      int(((R.exit == "SL") & ((R.above_low != R.above_high) | (R.reached_low != R.reached_high))).sum()))
# P-7 nuance
stt = sig["state"].to_numpy(); bid = sig["box_id"].to_numpy()
first_bar = {}
for i in np.flatnonzero(np.r_[True, bid[1:] != bid[:-1]]):
    if bid[i] >= 0 and bid[i] not in first_bar:
        first_bar[bid[i]] = i
inv = np.unique(bid[(stt == 5) & (bid >= 0)])
inherited = [k for k in inv if stt[first_bar[k]] == 5]
print(f"boxes with an INVALID bar while current: {len(inv)}; INVALID already on their first bar (memory/hv_block or veto): {len(inherited)};"
      f" turned INVALID later while current: {len(inv) - len(inherited)}")
