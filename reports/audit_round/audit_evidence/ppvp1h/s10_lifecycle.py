"""s10: box lifecycle table for the boxes used by the 109 IS trades: id, pivot times, confirmed/available time,
first 1H decision that can use it, superseded time and (state-machine) invalidation time while current; plus the
close-only invalidation rule check (intrabar wicks below the band / below the first-test low never invalidate)."""
import sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
d = pickle.load(open(HERE / "is_objects.pkl", "rb"))
df1h, boxes, sig = d["df1h"], d["boxes"], d["sig"]
pv = pd.read_csv(HERE / "profile_versions_per_signal.csv")
tr = pv[pv["status"] == "filled"].reset_index(drop=True)
H = pd.Timedelta(hours=1)
idx = df1h.index
st = sig["state"].to_numpy()
bid = sig["box_id"].to_numpy()
brk = sig["breakdown"].to_numpy()
used = sorted(set(tr.box_at_signal) | set(tr.box_at_seq_start) | set(tr.box_at_t1_low) | set(tr.box_at_t2_start) | set(tr.box_at_chain_start))
rows = []
for k in range(len(boxes)):
    b = boxes.iloc[k]
    bars = np.flatnonzero(bid == k)
    inv = bars[st[bars] == 5]
    first_inv = inv[0] if len(inv) else -1
    rows.append({"box_id": k, "natural_key": f"{b.pivot_a_time:%Y-%m-%d %H:%M}|{b.pivot_b_time:%Y-%m-%d %H:%M}",
                 "pivot_a_time(4H open)": b.pivot_a_time, "pivot_b_time(4H open)": b.pivot_b_time,
                 "confirmed_at=available_time": b.available_time, "first_1h_bar_using_it": idx[bars[0]],
                 "first_decision_close": idx[bars[0]] + H,
                 "superseded_at": boxes.available_time.iloc[k + 1] if k + 1 < len(boxes) else pd.NaT,
                 "invalid_while_current_from_close": (idx[first_inv] + H) if first_inv >= 0 else pd.NaT,
                 "invalid_cause": ("breakdown_bar" if (first_inv >= 0 and brk[first_inv]) else
                                   "switch_into_blocked_zone_or_inherited" if first_inv >= 0 else "")})
lc_all = pd.DataFrame(rows)
lc_all["used_by_109_trades"] = lc_all["box_id"].isin(used)
lc_all.to_csv(HERE / "box_lifecycle_all_IS_boxes.csv", index=False)
print("all IS boxes:", len(lc_all), "| invalidated while current:", int(lc_all.invalid_while_current_from_close.notna().sum()),
      "| superseded within IS:", int(lc_all.superseded_at.notna().sum()))
lc = lc_all[lc_all.used_by_109_trades].reset_index(drop=True)
lc.to_csv(HERE / "box_lifecycle_trade_boxes.csv", index=False)
print("boxes used by trades:", len(lc), "| natural key (pivot_a_time|pivot_b_time) unique over all IS boxes:",
      (boxes.pivot_a_time.astype(str) + boxes.pivot_b_time.astype(str)).is_unique,
      "| available_time unique:", boxes.available_time.is_unique)
print("first 1H bar using a box == its available_time for all:", bool((lc["first_1h_bar_using_it"] == lc["confirmed_at=available_time"]).all()))
print("boxes invalidated (INVALID state) at some point while current:", int(lc.invalid_while_current_from_close.notna().sum()),
      lc.invalid_cause.value_counts().to_dict())
# signal never on/after invalidation of its own box
sigbox = tr.box_at_signal.to_numpy()
inv_t = lc.set_index("box_id")["invalid_while_current_from_close"]
late = [(int(s), int(k)) for s, k in zip(tr.signal_bar, sigbox) if pd.notna(inv_t[k]) and idx[s] + H > inv_t[k]]
print("trades signalled after their signal box had already turned INVALID while current:", len(late), late[:5])
print(lc.head(6).to_string())

# close-only invalidation: intrabar penetration of the band low or of the first-test low never invalidates by itself
zlo = sig["zone_lo"].to_numpy(); c = df1h["close"].to_numpy(); l = df1h["low"].to_numpy()
live = np.isin(st, [1, 2, 3, 4])
prev_live = np.r_[False, live[:-1]]
wick_below = prev_live & (l < zlo) & (c >= zlo)
print("\nbars that wicked below the band low but closed back >= band low while the zone was live (prev state 1-4):",
      int(wick_below.sum()), "-> state INVALID on that bar:", int((wick_below & (st == 5)).sum()))
o = pd.read_csv("/home/user/ppvp_quant/reports/audit_round/signal_opportunities_B_original.csv")
f = o[o.status == "filled"]
print("filled trades whose signal bar low < first-test low (intrabar sweep of t1 low):", int((f.confirmation_low < f.t1_low).sum()),
      "; signal bar low < band low:", int((f.confirmation_low < f.zone_lo).sum()),
      "; signal close < VAL (closed back only inside the lower half-band):", int((f.entry_ref_close < f.VAL).sum()))
