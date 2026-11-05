"""v02b: unit checks of the triangular allocation, nonzero-share counts, and the worked examples of P-1/P-2/P-4."""
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
V = Path(__file__).resolve().parent; sys.path.insert(0, str(V))
from vp_indep import *
D = pickle.load(open(V / "is_objs.pkl", "rb")); X = pickle.load(open(V / "v02_out.pkl", "rb"))
df1h, boxes, sig, tr = D["df1h"], D["boxes"], D["sig"], D["trades"]
B, alts, sbox, atr, eb = X["B"], X["alts"], X["sbox"], X["atr"], X["eb"]
# triangular unit checks
e = np.linspace(0, 10, 11)
for m in (0.0, 2.0, 10.0):
    hh = hist_tri(np.array([0.0]), np.array([10.0]), np.array([m]), np.array([1.0]), e)
    print(f"tri mode {m}: sum {hh.sum():.12f} first bins {np.round(hh[:3], 4)} last {np.round(hh[-2:], 4)}")
hh = hist_tri(np.array([0.0]), np.array([10.0]), np.array([2.0]), np.array([1.0]), e)
assert abs(hh[:2].sum() - 0.2) < 1e-12   # mass left of the mode = (m-a)/(b-a)
for k, d in alts.items():
    print(f"{k:>8}: share |dVAL|>0 {np.mean(np.abs(d['VAL'])>0):.3f} ({int((np.abs(d['VAL'])>0).sum())}/109)  share |dVAH|>0 {np.mean(np.abs(d['VAH'])>0):.3f}")
st = df1h.index[eb]
def show(t):
    i = np.flatnonzero(st == pd.Timestamp(t, tz="UTC"))[0]; k = sbox[i]
    print(f"signal bar {t}: box {k} n_1h {int(B.e[k]-B.s[k])} ATR {atr[i]:.2f} VAL {boxes.VAL[k]:.2f} VAH {boxes.VAH[k]:.2f} | " +
          " ".join(f"{a}: dVAL {alts[a]['VAL'][i]:+.2f} dVAH {alts[a]['VAH'][i]:+.2f}" for a in ("typical", "close", "tri_close", "tri_typical", "ext")))
    return i, k
show("2023-12-13 02:00"); show("2023-08-28 22:00")
i, k = show("2020-07-05 07:00")
r = boxes.loc[k]
print(f"  box {k}: pivots {r.pivot_a_type}@{r.pivot_a_time} {r.pivot_b_type}@{r.pivot_b_time} window [{r.window_start}, {r.window_end}) n_1h {r.n_1h} available {r.available_time}")
i, k = show("2020-01-22 19:00")
r = boxes.loc[k]
print(f"  box {k}: window [{r.window_start}, {r.window_end}) n_1h {r.n_1h}; excluded {df1h.index[int(B.e[k])]} .. {df1h.index[int(B.e2[k])-1]} ({int(B.e2[k]-B.e[k])} bars); available {r.available_time}")
# P-4 examples
for t in ("2020-09-29 13:00", "2021-01-05 06:00", "2022-06-18 03:00"):
    i = np.flatnonzero(st == pd.Timestamp(t, tz="UTC"))[0]
    print(f"two-row: signal bar {t} box {sbox[i]} dVAL {alts['tworow']['VAL'][i]:+.2f} ATR")
