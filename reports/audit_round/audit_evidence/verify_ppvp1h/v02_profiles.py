"""v02: independent rebuild of all IS boxes + P-1 (allocation), P-2 (window end), P-4 (VA conventions) diagnostics."""
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
V = Path(__file__).resolve().parent; sys.path.insert(0, str(V))
from vp_indep import *

D = pickle.load(open(V / "is_objs.pkl", "rb"))
df1h, df4h, boxes, sig, tr = D["df1h"], D["df4h"], D["boxes"], D["sig"], D["trades"]
o1, h1, l1, c1, v1 = (df1h[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
t1 = ns(df1h.index); t4 = ns(df4h.index)
assert (h1 > l1).all(), "zero-range 1H bar present"
ph, pl = pivots(df4h["high"].to_numpy(float), df4h["low"].to_numpy(float))
piv = np.flatnonzero(ph | pl)
R = 5
rows = []
for k in range(1, len(piv)):
    a, b = piv[k-1], piv[k]
    ws, we, av = t4[a], t4[b] + H4, t4[b + R] + H4
    s, e, e2 = (np.searchsorted(t1, x) for x in (ws, we, av))
    rows.append(dict(a=a, b=b, ws=ws, we=we, av=av, s=s, e=e, e2=e2,
                     ta="HL" if ph[a] and pl[a] else "H" if ph[a] else "L", tb="HL" if ph[b] and pl[b] else "H" if ph[b] else "L"))
B = pd.DataFrame(rows)
print("pivot bars:", len(piv), "highs", int(ph.sum()), "lows", int(pl.sum()), "both", int((ph & pl).sum()), "| boxes", len(B))
assert len(B) == len(boxes)
for col, ecol in (("av", "available_time"), ("ws", "window_start"), ("we", "window_end")):
    assert (B[col].to_numpy() == ns(boxes[ecol])).all(), col
assert (B["a"].to_numpy() == boxes["pivot_a_idx"].to_numpy()).all() and (B["b"].to_numpy() == boxes["pivot_b_idx"].to_numpy()).all()
print("independent pivots/windows/available_time == engine for all boxes: True")

def prof(k, how="uniform", ext=False, **kw):
    r = B.iloc[k]; s, e = int(r.s), int(r.e2 if ext else r.e)
    lo, hi, v = l1[s:e], h1[s:e], v1[s:e]
    edges = np.linspace(lo.min(), hi.max(), 51)
    if how == "uniform": hst = hist_uniform(lo, hi, v, edges)
    elif how == "typical": hst = hist_point((h1[s:e] + l1[s:e] + c1[s:e]) / 3.0, v, edges)
    elif how == "close": hst = hist_point(c1[s:e], v, edges)
    elif how == "body": hst = hist_body(o1[s:e], c1[s:e], v, edges)
    elif how == "tri_close": hst = hist_tri(lo, hi, c1[s:e], v, edges)
    elif how == "tri_typical": hst = hist_tri(lo, hi, (h1[s:e] + l1[s:e] + c1[s:e]) / 3.0, v, edges)
    return levels(hst, edges, **kw), edges, hst

# ---- 1) engine reproduction on all 1149 boxes
base = [prof(k, count_ties=True)[0] for k in range(len(B))]
bt = pd.DataFrame(base)
for c in ("VAL", "POC", "VAH"):
    d = np.abs(bt[c].to_numpy() - boxes[c].to_numpy())
    print(f"{c}: exact-equal {int((d == 0).sum())}/{len(B)}  max|diff| {d.max():.3e}")
print("VA coverage min/median/max: %.4f %.4f %.4f" % (bt["cov"].min(), bt["cov"].median(), bt["cov"].max()))
print("boxes with POC tie (>=2 bins within 1e-9*max):", int((bt["n_tied"] >= 2).sum()), " VA steps with |up-dn|<=tol:", int(bt["tie_steps"].sum()))

# ---- trades: signal box + ATR at signal bar
eb = tr["entry_bar"].to_numpy(int)
assert (sig["sig_raw"].to_numpy()[eb]).all()
sbox = sig["box_id"].to_numpy(int)[eb]; atr = sig["atr"].to_numpy(float)[eb]
assert (sbox == tr["box_id"].to_numpy(int)).all()
print("trades", len(eb), "unique signal boxes", len(np.unique(sbox)))
width = (B.iloc[sbox][["e"]].to_numpy().ravel() * 0)  # placeholder
bw = np.array([(l1[int(B.s[k]):int(B.e[k])].min(), h1[int(B.s[k]):int(B.e[k])].max()) for k in sbox])
binw_atr = (bw[:, 1] - bw[:, 0]) / 50 / atr
print("bin width / ATR_signal: median %.3f P90 %.3f max %.3f" % (np.median(binw_atr), np.quantile(binw_atr, .9), binw_atr.max()))
va_w = (boxes["VAH"].to_numpy()[sbox] - boxes["VAL"].to_numpy()[sbox]) / atr
print("VA width / ATR_signal: median %.2f" % np.median(va_w))

def summarize(name, alt):
    out = {}
    for c in ("VAL", "POC", "VAH"):
        d = (np.array([alt[k][c] for k in sbox]) - boxes[c].to_numpy()[sbox]) / atr
        a_ = np.abs(d)
        out[c] = d
        print(f"  [{name}] {c}: |d| median {np.median(a_):.3f} P90 {np.quantile(a_, .9):.3f} max {a_.max():.3f} | "
              f">0.5ATR {int((a_ > .5).sum())}/109 >1ATR {int((a_ > 1).sum())}/109 | signed median {np.median(d):+.3f}")
    return out

alts = {}
for how in ("typical", "close", "body", "tri_close", "tri_typical"):
    cache = {k: prof(k, how)[0] for k in np.unique(sbox)}
    alts[how] = summarize(how, cache)
# ---- P-2: window extended to available_time
cache = {k: prof(k, ext=True)[0] for k in np.unique(sbox)}
alts["ext"] = summarize("window->available_time", cache)
wh = np.array([(B.e[k] - B.s[k]) for k in sbox]); xh = np.array([(B.e2[k] - B.e[k]) for k in sbox])
print("excluded bars per box unique:", np.unique(B.e2 - B.e))
print("window hours (trade signal boxes): median %d min %d max %d" % (np.median(wh), wh.min(), wh.max()))
print("excluded share of (window+excluded): median %.3f" % np.median(xh / (wh + xh)))
sig_close = t1[eb] + H1
age = (sig_close - B.we.to_numpy()[sbox]) / H1
print("hours from window_end to signal close: min %d median %d max %d" % (age.min(), np.median(age), age.max()))
# ---- P-4: two-row rule, highest-tie POC
cache = {k: prof(k, rows=2)[0] for k in np.unique(sbox)}
alts["tworow"] = summarize("two-row VA", cache)
tied_tr = [k for k in np.unique(sbox) if bt["n_tied"][k] >= 2]
print("unique signal boxes with POC tie:", tied_tr, " trades:", int(np.isin(sbox, tied_tr).sum()))
for k in tied_tr:
    hb = prof(k, poc_rule="highest")[0]; i = np.flatnonzero(sbox == k)[0]
    print(f"   box {k}: dPOC {(hb['POC']-boxes.POC[k])/atr[i]:+.3f} ATR dVAL {(hb['VAL']-boxes.VAL[k])/atr[i]:+.3f} dVAH {(hb['VAH']-boxes.VAH[k])/atr[i]:+.3f}"
          f"  tied bins {np.flatnonzero(prof(k)[2] >= prof(k)[2].max()*(1-1e-9)).tolist()}  signal {df1h.index[eb[i]]}")
pickle.dump({"B": B, "bt": bt, "alts": alts, "sbox": sbox, "atr": atr, "eb": eb}, open(V / "v02_out.pkl", "wb"))
