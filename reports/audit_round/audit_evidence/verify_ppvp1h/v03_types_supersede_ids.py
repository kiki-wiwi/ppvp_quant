"""v03: P-3 anchor types, P-5 superseded-at-decision trades, P-7 box ids / lifecycle / prefix stability (IS only)."""
import pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd
V = Path(__file__).resolve().parent; sys.path.insert(0, str(V)); sys.path.insert(0, "/home/user/ppvp_quant")
from vp_indep import ns, H1
D = pickle.load(open(V / "is_objs.pkl", "rb")); X = pickle.load(open(V / "v02_out.pkl", "rb"))
df1h, df4h, boxes, sig, tr = D["df1h"], D["df4h"], D["boxes"], D["sig"], D["trades"]
B, sbox, atr, eb = X["B"], X["sbox"], X["atr"], X["eb"]
st = df1h.index[eb]
# ---------------- P-3
pair = (B.ta + "->" + B.tb)
print("P-3 anchor pairs:", pair.value_counts().to_dict())
same = pair.isin(["H->H", "L->L"])
print(f"same-type boxes {int(same.sum())}/{len(B)} = {same.mean():.4%}")
print("box k pivot_a == box k-1 pivot_b for all:", bool((B.a.to_numpy()[1:] == B.b.to_numpy()[:-1]).all()), "pairs", len(B) - 1)
tp = pair.to_numpy()[sbox]
print("109 trades' signal-box pairs:", pd.Series(tp).value_counts().to_dict())
for t in ("2021-05-20 18:00", "2020-09-27 10:00"):
    i = np.flatnonzero(st == pd.Timestamp(t, tz="UTC"))[0]; k = sbox[i]; r = boxes.loc[k]
    print(f"  signal bar {t}: box {k} {r.pivot_a_type}->{r.pivot_b_type} pivots {r.pivot_a_time} .. {r.pivot_b_time} window_end {r.window_end} n_1h {r.n_1h} VAL {r.VAL:.2f} VAH {r.VAH:.2f}")
# ---------------- P-5
av = ns(boxes["available_time"]); t1 = ns(df1h.index)
sig_close = t1[eb] + H1
nxt = np.where(sbox + 1 < len(boxes), av[np.minimum(sbox + 1, len(boxes) - 1)], -1)
hit = np.flatnonzero(nxt == sig_close)
print("\nP-5 trades whose next box is confirmed exactly at the signal close:", len(hit))
c = df1h["close"].to_numpy()
for i in hit:
    k = sbox[i]; r0, r1 = boxes.loc[k], boxes.loc[k + 1]
    print(f"  signal bar {st[i]} close {c[eb[i]]:.1f} ATR {atr[i]:.1f}: used box {k} VAL {r0.VAL:.1f} VAH {r0.VAH:.1f}; box {k+1} ({r1.pivot_a_type}->{r1.pivot_b_type} {r1.pivot_a_time}..{r1.pivot_b_time})"
          f" VAL {r1.VAL:.1f} ({(r1.VAL-r0.VAL)/atr[i]:+.2f} ATR) VAH {r1.VAH:.1f} ({(r1.VAH-r0.VAH)/atr[i]:+.2f} ATR); new band [{r1.VAL-0.5*atr[i]:.1f},{r1.VAL+0.5*atr[i]:.1f}]")
# mapping rule: bar i uses latest box with available_time <= open_time[i]
m = np.searchsorted(av, t1, side="right") - 1
print("engine box_id == latest box with available_time <= open_time:", bool((m == sig["box_id"].to_numpy()).all()))
op = pd.read_csv("/home/user/ppvp_quant/reports/audit_round/signal_opportunities_B_original.csv")
eq = op["signal_time"] == op["profile_superseded_at"]
print("CSV rows (0-based data index) with signal_time == profile_superseded_at:", op.index[eq].tolist(),
      "-> file line numbers incl. header:", (op.index[eq] + 2).tolist(), "status", op.loc[eq, "status"].tolist())
# ---------------- P-7
print("\nP-7 box_id is positional RangeIndex:", boxes.index.equals(pd.RangeIndex(len(boxes))), boxes.index.name)
print("pivot_a_time == 4H open of pivot_a_idx:", bool((ns(boxes.pivot_a_time) == ns(df4h.index[boxes.pivot_a_idx])).all()),
      "| pivot_b_time == 4H open:", bool((ns(boxes.pivot_b_time) == ns(df4h.index[boxes.pivot_b_idx])).all()))
nk = boxes.pivot_a_time.astype(str) + "|" + boxes.pivot_b_time.astype(str)
print("natural key unique:", nk.is_unique, "| available_time unique:", boxes.available_time.is_unique)
fu = np.searchsorted(t1, av, side="left")
ok = fu < len(t1)
print("first 1H bar using each box opens at available_time:", bool((t1[fu[ok]] == av[ok]).all()), "boxes w/o IS bar:", int((~ok).sum()))
print("BOX_COLUMNS have superseded/invalid fields:", [c for c in boxes.columns if "supersed" in c or "invalid" in c])
stt = sig["state"].to_numpy(); bid = sig["box_id"].to_numpy(); brk = sig["breakdown"].to_numpy()
inv_boxes = np.unique(bid[(stt == 5) & (bid >= 0)])
brk_boxes = np.unique(bid[brk & (bid >= 0)])
print("boxes with an INVALID bar while current:", len(inv_boxes), "| boxes with a breakdown bar while current:", len(brk_boxes))
r = boxes.loc[13]
print(f"box 13: key {r.pivot_a_time}|{r.pivot_b_time} type b {r.pivot_b_type} 4H low at pivot_b {df4h.low.iloc[r.pivot_b_idx]} available {r.available_time}"
      f" superseded {boxes.available_time[14]} INVALID while current: {13 in inv_boxes}")
# prefix stability on IS (sample of cuts) and id shift under a different data start
from ppvp_engine import build_boxes
from config import BASELINE_STRATEGY as S
from ppvp_audit.common import load_is_market
d1, _, _ = load_is_market()
rng = np.random.default_rng(7)
cuts = sorted(rng.choice(np.arange(200, len(d1)), 12, replace=False))
t0 = time.time(); bad = 0
for cpos in cuts:
    bb, _ = build_boxes(d1.iloc[:cpos], S)
    pre = boxes.iloc[:len(bb)]
    bad += not (bb.reset_index(drop=True).equals(pre.reset_index(drop=True)) and bb.index.equals(pre.index))
print(f"prefix cuts {len(cuts)}: tables not equal to the full-table prefix: {bad}  ({time.time()-t0:.0f}s)")
bb2, _ = build_boxes(d1[d1.index >= pd.Timestamp("2020-02-01", tz="UTC")], S)
nk2 = bb2.pivot_a_time.astype(str) + "|" + bb2.pivot_b_time.astype(str)
j = pd.Series(boxes.index, index=nk).reindex(nk2).dropna().astype(int)
print("data start 2020-02-01: common boxes", len(j), "| same box_id for same natural key:", int((j.to_numpy() == np.arange(len(bb2))[nk2.isin(nk).to_numpy()]).sum()),
      "| example offset", int(j.iloc[0]) - int(np.flatnonzero(nk2.isin(nk).to_numpy())[0]))
