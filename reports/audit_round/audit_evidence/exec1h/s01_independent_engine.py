"""s01: independent plain-Python re-implementation of the documented execution/funding/MTM rules
(no import of backtester internals), compared trade-by-trade and bar-by-bar with backtester.py output. IS only."""
import pickle, sys
sys.path.insert(0, "/home/user/ppvp_quant")
from pathlib import Path
import numpy as np, pandas as pd
from config import BASELINE_BACKTEST as B, IS_END

D = Path(__file__).resolve().parent
X = pickle.load(open(D / "baseline.pkl", "rb"))
df, sig, tr, eq_bt, fund = X["df1h"], X["signals"], X["trades"], X["equity"], X["fund"]
assert df.index.max() < IS_END
o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
n = len(df)
H = pd.Timedelta(hours=1)
open_t = df.index
close_t = df.index + H
# funding: settlement T -> rate (all T in (first open, last close])
f = fund[(fund.index > open_t[0]) & (fund.index <= close_t[-1])]
ft = {t: r for t, r in zip(f.index, f["funding_rate"].to_numpy())}
taker, maker, slip = B.perp_taker_fee, B.perp_maker_fee, B.slippage
S = sig["signal"].to_numpy(bool); SL = sig["sl"].to_numpy(float); TP = sig["tp"].to_numpy(float)

wallet = B.initial_capital
pos = None
trades = []
eq = np.empty(n)
for i in range(n):
    if pos is not None:
        # funding at the open of bar i if a settlement is at open_t[i] and entry time < T
        T = open_t[i]
        if T in ft and pos["entry_time"] < T:
            amt = pos["qty"] * o[i] * ft[T]
            wallet -= amt; pos["fund"] += amt; pos["nf"] += 1; pos["fund_events"].append((T, o[i], ft[T], amt))
        if l[i] <= pos["sl"]:
            ref = min(o[i], pos["sl"]); px = ref * (1 - slip); fee = pos["qty"] * px * taker; why = "SL"
        elif h[i] >= pos["tp"]:
            ref = pos["tp"]; px = ref * (1 - slip); fee = pos["qty"] * px * maker; why = "TP"
        else:
            ref = None
        if ref is not None:
            g = pos["qty"] * (px - pos["entry_px"])
            wallet += g - fee
            pos.update(exit_bar=i, exit_ref=ref, exit_px=px, fees=pos["fees"] + fee, gross=g, reason=why,
                       pnl=g - pos["fees"] - fee - pos["fund"])
            trades.append(pos); pos = None
    eq[i] = wallet + (pos["qty"] * (c[i] - pos["entry_px"]) if pos is not None else 0.0)
    if pos is None and S[i]:
        e = eq[i]; efill = c[i] * (1 + slip); notional = e * B.fixed_leverage
        if efill > SL[i] and notional > 0:
            q = notional / efill; fee = notional * taker; wallet -= fee
            pos = dict(entry_bar=i, entry_time=close_t[i], entry_ref=c[i], entry_px=efill, qty=q, sl=SL[i], tp=TP[i],
                       eq_at_entry=e, fees=fee, fund=0.0, nf=0, fund_events=[])
            eq[i] = wallet + q * (c[i] - efill)
assert pos is None, "open trade at IS end"
T2 = pd.DataFrame(trades)
print("independent trades", len(T2), T2["reason"].value_counts().to_dict())
cols = [("entry_bar", "entry_bar"), ("exit_bar", "exit_bar"), ("entry_px", "entry_px"), ("exit_px", "exit_px"),
        ("qty", "qty"), ("fees", "fees"), ("fund", "funding"), ("nf", "n_funding"), ("gross", "gross_pnl"), ("pnl", "pnl"),
        ("sl", "sl"), ("tp", "tp"), ("eq_at_entry", "equity_at_entry")]
for a, b in cols:
    x, y = T2[a].to_numpy(float), tr[b].to_numpy(float)
    print(f"  {b:16s} max|diff| = {np.max(np.abs(x - y)):.3e}")
print("  reason equal:", bool((T2["reason"].to_numpy() == tr["exit_reason"].to_numpy()).all()))
print("hourly equity max|diff| vs backtester:", float(np.max(np.abs(eq - eq_bt.to_numpy()))))
# ---- execution checks ----
eb = tr["entry_bar"].to_numpy(int); xb = tr["exit_bar"].to_numpy(int)
nb = (o[eb + 1] / c[eb] - 1) * 1e4
print("\n[entry] fill = close*(1+slip) exactly:", bool(np.allclose(tr["entry_px"], c[eb] * (1 + slip), rtol=0, atol=1e-9)))
print(f"[entry] next open vs signal close (bp): min {nb.min():+.3f} max {nb.max():+.3f} max|.| {np.abs(nb).max():.3f}; "
      f"n>0 {int((nb>0).sum())} n<0 {int((nb<0).sum())} n==0 {int((nb==0).sum())}")
allsig = np.flatnonzero(sig["sig_raw"].to_numpy(bool))
nb_all = (o[allsig + 1] / c[allsig] - 1) * 1e4
print(f"[entry] all {len(allsig)} raw signals: max|next open - close| = {np.abs(nb_all).max():.3f} bp")
print("[exit] exit_bar > entry_bar for all:", bool((xb > eb).all()), " min hold", int((xb - eb).min()))
skip = 0
for j in range(len(tr)):
    for i in range(eb[j] + 1, xb[j]):
        if l[i] <= tr["sl"].iat[j] or h[i] >= tr["tp"].iat[j]:
            skip += 1
print("[exit] bars strictly between entry and exit that touched SL or TP (should be 0):", skip)
trig_ok = np.where(tr["exit_reason"] == "SL", l[xb] <= tr["sl"], h[xb] >= tr["tp"])
print("[exit] trigger condition holds on every exit bar:", bool(trig_ok.all()))
print("[exit] min_low/max_high equal post-entry extremes:",
      bool(all(np.isclose(tr['min_low'].iat[j], l[eb[j]+1:xb[j]+1].min()) and np.isclose(tr['max_high'].iat[j], h[eb[j]+1:xb[j]+1].max()) for j in range(len(tr)))))
sl_m = (tr["exit_reason"] == "SL").to_numpy(); tp_m = ~sl_m
gap_sl = sl_m & (o[xb] < tr["sl"].to_numpy()); gap_tp = tp_m & (o[xb] > tr["tp"].to_numpy())
print(f"[gap] SL exits filled at the open (open < SL): {int(gap_sl.sum())}; TP exits whose bar opened above TP: {int(gap_tp.sum())}")
if gap_sl.any():
    for j in np.flatnonzero(gap_sl):
        print("   gap SL trade", j, df.index[xb[j]], "open", o[xb[j]], "SL", tr['sl'].iat[j], "prev close", c[xb[j]-1])
amb_tp = sl_m & (h[xb] >= tr["tp"].to_numpy())
poc = tr["poc"].to_numpy(); ep = tr["entry_px"].to_numpy()
amb_poc = sl_m & (poc > ep) & (h[xb] >= poc)
print(f"[ambiguity] SL exit bar also touched TP: {int(amb_tp.sum())}; SL exit bar touched POC (POC>entry): {int(amb_poc.sum())}; "
      f"union {int((amb_tp | amb_poc).sum())} of {len(tr)} = {(amb_tp | amb_poc).mean():.2%}")
for j in np.flatnonzero(amb_tp | amb_poc):
    print(f"   trade {j:3d} entry {tr['entry_time'].iat[j]} exit bar open {df.index[xb[j]]} O={o[xb[j]]:.2f} H={h[xb[j]]:.2f} "
          f"L={l[xb[j]]:.2f} SL={tr['sl'].iat[j]:.2f} POC={poc[j]:.2f} TP={tr['tp'].iat[j]:.2f} hold={xb[j]-eb[j]} "
          f"amb_tp={bool(amb_tp[j])} amb_poc={bool(amb_poc[j])}")
# open price above TP / below SL as order-revealing first tick in ambiguous bars
for j in np.flatnonzero(amb_tp):
    print("   SL&TP bar: open<=SL?", o[xb[j]] <= tr['sl'].iat[j], " open>=TP?", o[xb[j]] >= tr['tp'].iat[j])
# entries on the same bar as the previous exit
same = int((eb[1:] == xb[:-1]).sum())
print("[sequencing] trades entering at the close of the bar in which the previous trade exited:", same)
# ---- funding ----
ev = [(j, *e) for j, t in enumerate(trades) for e in t["fund_events"]]
fe = pd.DataFrame(ev, columns=["trade", "T", "price", "rate", "amt"])
print(f"\n[funding] events {len(fe)}, total {fe['amt'].sum():.4f} vs backtester {tr['funding'].sum():.4f}; n_funding sum {int(tr['n_funding'].sum())}")
et = tr["entry_time"]; xo = tr["exit_bar_open"]
on_entry = sum(1 for t in et if t in ft)
on_exit_open = sum(1 for t in xo if t in ft)
print(f"[funding] trades whose entry time coincides with a settlement (not charged): {on_entry}; "
      f"trades whose exit-bar open is a settlement (charged, then exit intrabar): {on_exit_open}")
# held-at-T check by interval arithmetic: settlements T with entry_time < T <= exit_bar_open
cnt = [sum(1 for T in ft if a < T <= b) for a, b in zip(et, xo)]
print("[funding] n_funding equals #{T: entry_time < T <= exit_bar_open}:", bool(np.array_equal(np.array(cnt), tr["n_funding"].to_numpy())))
print("[funding] price used == open of bar starting at T:", bool(all(np.isclose(p, o[df.index.get_loc(T)]) for T, p in zip(fe['T'], fe['price']))))
print("[funding] rate sign: long pays when rate>0 -> amt sign equals rate sign:", bool((np.sign(fe['amt']) == np.sign(fe['rate'])).all()))
print("[funding] settlements per day in window: ", f.index.floor('D').value_counts().value_counts().to_dict(),
      " hours:", sorted(set(f.index.hour)), " minutes/seconds nonzero:", int(((f.index.minute != 0) | (f.index.second != 0)).sum()))
pickle.dump({"indep_trades": T2, "fund_events": fe, "indep_equity": eq}, open(D / "s01_indep.pkl", "wb"))
