"""s02: compare engine 4H bars, pivots and all IS boxes with the independent loop implementation."""
import sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/ppvp_quant")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from indep import boxes_indep, build_4h_indep, pivots_indep
from ppvp_engine import detect_pivots, pivot_table
from config import BASELINE_STRATEGY as S

d = pickle.load(open(HERE / "is_objects.pkl", "rb"))
df1h, df4h, boxes = d["df1h"], d["df4h"], d["boxes"]

# 4H
d4 = build_4h_indep(df1h)
assert len(d4) == len(df4h), (len(d4), len(df4h))
for c in ("open", "high", "low", "close", "volume"):
    diff = np.abs(d4[c].to_numpy() - df4h[c].to_numpy()).max()
    print(f"4H {c}: max |diff| = {diff}")
print("4H index equal:", (d4.index == df4h.index).all(), " available_time equal:",
      (d4["available_time"].values == df4h["available_time"].values).all())
print("4H open times on UTC 4h grid:", bool((df4h.index.hour % 4 == 0).all() and (df4h.index.minute == 0).all()))
print("4H consecutive (no gaps):", bool((np.diff(df4h.index.asi8) == 4 * 3600 * 10**9).all()) if df4h.index.asi8.dtype else None)

# pivots
ph_e, pl_e = detect_pivots(df4h["high"].to_numpy(), df4h["low"].to_numpy(), S.pivot_left, S.pivot_right)
ph_i, pl_i = pivots_indep(df4h["high"].to_numpy(), df4h["low"].to_numpy(), S.pivot_left, S.pivot_right)
print("pivot highs equal:", (ph_e == ph_i).all(), int(ph_e.sum()), " pivot lows equal:", (pl_e == pl_i).all(), int(pl_e.sum()),
      " HL bars:", int((ph_e & pl_e).sum()), " total pivot bars:", int((ph_e | pl_e).sum()))
pt = pivot_table(df4h, S)
lag = (pd.DatetimeIndex(pt["confirm_time"]) - pd.DatetimeIndex(pt["pivot_bar_open"]))
print("confirm lag (pivot open -> confirm) unique hours:", sorted(set((lag / pd.Timedelta(hours=1)).astype(int))))

# boxes
bi, _, _, _ = boxes_indep(df1h, S.pivot_left, S.pivot_right, S.n_bins, S.va_pct)
print("n boxes engine / indep:", len(boxes), len(bi))
assert len(bi) == len(boxes)
for c in ("available_time", "pivot_a_time", "pivot_b_time", "window_start", "window_end"):
    eq = (pd.DatetimeIndex(bi[c]) == pd.DatetimeIndex(boxes[c])).all()
    print(f"  {c} equal: {eq}")
print("  n_1h equal:", (bi["n_1h"].to_numpy() == boxes["n_1h"].to_numpy()).all(),
      " n_1h == 4*(b-a+1):", (boxes["n_1h"].to_numpy() == 4 * (boxes["pivot_b_idx"] - boxes["pivot_a_idx"] + 1)).all())
print("  types equal:", (bi["type_a"].to_numpy() == boxes["pivot_a_type"].to_numpy()).all(),
      (bi["type_b"].to_numpy() == boxes["pivot_b_type"].to_numpy()).all())
for c in ("VAL", "POC", "VAH", "P_low", "P_high"):
    e, i = boxes[c].to_numpy(), bi[c].to_numpy()
    rel = np.abs(e - i) / np.abs(e)
    print(f"  {c}: exact-equal {int((e == i).sum())}/{len(e)}, max rel diff {rel.max():.3e}")
avail = pd.DatetimeIndex(boxes["available_time"])
print("available_time == pivot_b open + (R+1)*4h for all:",
      bool((avail == pd.DatetimeIndex(boxes["pivot_b_time"]) + pd.Timedelta(hours=4 * (S.pivot_right + 1))).all()))
print("available strictly increasing:", bool((np.diff(avail.asi8) > 0).all()),
      " window_start strictly increasing:", bool((np.diff(pd.DatetimeIndex(boxes['window_start']).asi8) > 0).all()))
print("window_start == previous box's pivot_b (P=2 chaining):",
      bool((pd.DatetimeIndex(boxes["pivot_a_time"])[1:] == pd.DatetimeIndex(boxes["pivot_b_time"])[:-1]).all()))
bi.to_pickle(HERE / "boxes_indep.pkl")
# descriptive
w_h = boxes["n_1h"]
upd = np.diff(avail.asi8) / 3.6e12
wpct = (boxes["VAH"] - boxes["VAL"]) / boxes["VAL"]
print("window hours median / P10 / P90 / min / max:", w_h.median(), w_h.quantile(.1), w_h.quantile(.9), w_h.min(), w_h.max())
print("update interval hours median:", np.median(upd), " min:", upd.min())
print("VA width pct P10/median/P90: %.4f %.4f %.4f" % tuple(wpct.quantile([.1, .5, .9])))
pair = boxes["pivot_a_type"].astype(str) + boxes["pivot_b_type"].astype(str)
print("pivot pairs:", pair.value_counts().to_dict())
same = pair.isin(["HH", "LL"]).sum()
print("same-type (HH or LL) boxes:", int(same), "of", len(boxes), f"= {same/len(boxes):.3%}")
print("P_low<=VAL<=POC<=VAH<=P_high:", bool(((boxes.P_low <= boxes.VAL) & (boxes.VAL <= boxes.POC) & (boxes.POC <= boxes.VAH) & (boxes.VAH <= boxes.P_high)).all()))
z = (df1h["high"] - df1h["low"] <= 0).sum()
print("zero-range 1H bars in IS:", int(z))
