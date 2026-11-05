"""v07: split boxes INVALID on their first bar as current into 'inherited before the bar' vs 'breakdown on that very bar'."""
import pickle
from pathlib import Path
import numpy as np
V = Path(__file__).resolve().parent
D = pickle.load(open(V / "is_objs.pkl", "rb")); sig = D["sig"]
stt = sig["state"].to_numpy(); bid = sig["box_id"].to_numpy(); brk = sig["breakdown"].to_numpy()
starts = np.flatnonzero(np.r_[True, bid[1:] != bid[:-1]])
first = {}
for i in starts:
    if bid[i] >= 0 and bid[i] not in first:
        first[bid[i]] = i
inv = set(np.unique(bid[(stt == 5) & (bid >= 0)]).tolist())
on_first = [k for k in inv if stt[first[k]] == 5]
brk_first = [k for k in on_first if brk[first[k]]]
print("INVALID on first bar:", len(on_first), "| of which breakdown on that very bar:", len(brk_first),
      "| inherited at activation (breakdown memory; veto is off in the baseline):", len(on_first) - len(brk_first))
print("so 'turned INVALID while current' (by a breakdown while the box was current) =", len(inv) - len(on_first) + len(brk_first), "of 1149")
