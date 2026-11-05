"""s02: accounting checks on the baseline (IS only): cost decomposition, commission recomputation, R denominator,
daily mark-to-market equity, the two zero-cost counterfactuals (independent recomputation)."""
import pickle, sys
sys.path.insert(0, "/home/user/ppvp_quant")
from pathlib import Path
import numpy as np, pandas as pd
from config import BASELINE_BACKTEST as B, IS_END, IS_START
from metrics import daily_equity, daily_returns, sharpe

D = Path(__file__).resolve().parent
X = pickle.load(open(D / "baseline.pkl", "rb"))
I = pickle.load(open(D / "s01_indep.pkl", "rb"))
df, tr, eq = X["df1h"], X["trades"], X["equity"]
o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
slip, tk, mk = B.slippage, B.perp_taker_fee, B.perp_maker_fee
T2 = I["indep_trades"]
L = pd.read_csv("/home/user/ppvp_quant/reports/audit_round/trades_B_original.csv")
assert len(L) == 109
# --- cost decomposition identities on the published ledger ---
q, ep, xp = L["qty"], L["entry_px"], L["exit_px"]
ref_e, ref_x = T2["entry_ref"].to_numpy(), T2["exit_ref"].to_numpy()        # independent reference prices
print("[ledger] entry_ref == signal close:", bool(np.allclose(L["entry_ref"], ref_e, rtol=0, atol=1e-8)),
      " exit_ref == min(open,SL) or TP:", bool(np.allclose(L["exit_ref"], ref_x, rtol=0, atol=1e-8)))
slip_ind = q * (ep - ref_e) + q * (ref_x - xp)
print("[ledger] slippage_impact == q*(fill_in-ref_in)+q*(ref_out-fill_out):", float(np.max(np.abs(L["slippage_impact"] - slip_ind))))
fee_ind = q * ep * tk + q * xp * np.where(L["exit_reason"] == "SL", tk, mk)
print("[ledger] commission == taker*entry notional + (taker|maker)*exit notional:", float(np.max(np.abs(L["commission"] - fee_ind))))
ident = L["gross_pnl_ref"] - L["slippage_impact"] - L["commission"] - L["funding_cashflow"] - L["net_pnl"]
print("[ledger] gross_ref - slippage - commission - funding - net (max abs):", float(np.abs(ident).max()))
print("[ledger] gross_pnl (at fill) == gross_ref - slippage:", float(np.max(np.abs(L["gross_pnl"] - (L["gross_pnl_ref"] - L["slippage_impact"])))))
tot = {k: float(L[k].sum()) for k in ("gross_pnl_ref", "slippage_impact", "commission", "funding_cashflow", "net_pnl", "gross_pnl")}
print("[ledger] totals:", {k: round(v, 2) for k, v in tot.items()})
print("[ledger] equity change 100000 -> final:", round(float(eq.iloc[-1]) - B.initial_capital, 2), " sum net:", round(tot["net_pnl"], 2))
print("[ledger] double-count check: gross_at_fill - fees - funding - slippage =",
      round(tot["gross_pnl"] - tot["commission"] - tot["funding_cashflow"] - tot["slippage_impact"], 2), "(would be wrong)")
# --- R denominator ---
risk = q * (ep - L["sl"])
print("[R] initial_risk_usd == qty*(entry_fill - initial SL):", float(np.max(np.abs(L["initial_risk_usd"] - risk))),
      " r_multiple == net/risk:", float(np.max(np.abs(L["r_multiple"] - L["net_pnl"] / risk))),
      " net_R == r_multiple:", float(np.max(np.abs(L["net_R"] - L["r_multiple"]))))
sig = X["signals"]
print("[R] ledger SL == signal SL at entry bar:", bool(np.allclose(L["sl"], sig["sl"].to_numpy()[L["entry_bar"]], rtol=0, atol=0)))
print("[R] mean net R incl funding", round(float(L["net_R"].mean()), 4), " ex funding", round(float(L["r_ex_funding"].mean()), 4),
      " gross_R_ref", round(float(L["gross_R_ref"].mean()), 4))
# SL trade realised R: should be about -1 - costs/risk
slm = L["exit_reason"] == "SL"
print("[R] SL trades net R range:", round(float(L.loc[slm, "net_R"].min()), 3), round(float(L.loc[slm, "net_R"].max()), 3),
      " gross_R_ref of SL trades (should be exactly -1 absent gaps):", round(float(L.loc[slm, "gross_R_ref"].min()), 6), round(float(L.loc[slm, "gross_R_ref"].max()), 6))
# --- daily equity is MTM ---
d = daily_equity(type("R", (), {"equity": eq, "trades": tr, "initial_capital": B.initial_capital, "start_time": X["start_time"]})())
pos = X["position"]
mid = eq.index[(eq.index.hour == 0) & (eq.index.minute == 0)]
open_mid = pos.loc[mid] > 0
print(f"[daily] daily equity label D == hourly MTM at D+1 00:00 for all days:",
      bool(np.allclose(d.loc[(mid - pd.Timedelta(days=1)).floor('D')].to_numpy(), eq.loc[mid].to_numpy())))
# realised-balance step curve for comparison
wallet = np.full(len(eq), B.initial_capital)
for _, t in tr.iterrows():
    wallet[int(t["entry_bar"]):] -= t["notional"] * tk          # entry fee hits the cash at entry
    wallet[int(t["exit_bar"]):] += t["pnl"] + t["notional"] * tk
unreal = eq.to_numpy() - wallet
k = np.flatnonzero(open_mid.to_numpy())
print(f"[daily] midnights with an open position: {int(open_mid.sum())} of {len(mid)}; max |MTM - realised cash| at those: {np.abs(unreal[eq.index.get_indexer(mid[k])]).max():.2f} USDT")
ex = int(k[0]); tm = mid[ex]
j = int(np.flatnonzero((tr['entry_time'] < tm) & (tr['exit_time'] > tm))[0])
t = tr.iloc[j]; bi = eq.index.get_loc(tm)
print(f"[daily] example: {tm} position from trade entered {t['entry_time']} qty={t['qty']:.6f} entry_px={t['entry_px']:.2f} "
      f"close={c[bi]:.2f}; equity={eq.iloc[bi]:.2f}, realised cash={wallet[bi]:.2f}, unrealised={unreal[bi]:.2f} "
      f"= qty*(close-entry_px) - funding so far? {t['qty']*(c[bi]-t['entry_px']):.2f}")
# --- zero-cost counterfactual 1: fixed quantities (independent) ---
n = len(df)
mtm = np.zeros(n); real = np.zeros(n)
for j, t in T2.iterrows():
    e, x, qq = int(t["entry_bar"]), int(t["exit_bar"]), float(t["qty"])
    mtm[e:x] += qq * (c[e:x] - t["entry_ref"])
    real[x:] += qq * (t["exit_ref"] - t["entry_ref"])
eq_fq = pd.Series(B.initial_capital + mtm + real, index=eq.index)
lab = (eq_fq.index - pd.Timedelta(1, "ns")).floor("D")
dd = pd.concat([pd.Series([B.initial_capital], index=[pd.Timestamp("2019-12-31", tz="UTC")]), eq_fq.groupby(lab).last()])
print(f"\n[zero-cost fixed qty] final {dd.iloc[-1]:.2f} ({dd.iloc[-1]/1e5-1:+.4%}), sharpe {sharpe(daily_returns(dd)):.4f}")
# --- zero-cost counterfactual 2: independent re-simulation with zero costs and recomputed compounding ---
S = sig["signal"].to_numpy(bool); SL = sig["sl"].to_numpy(float); TP = sig["tp"].to_numpy(float)
w = B.initial_capital; p = None; eq0 = np.empty(n); tr0 = []
for i in range(n):
    if p is not None:
        if l[i] <= p["sl"]:
            px = min(o[i], p["sl"])
        elif h[i] >= p["tp"]:
            px = p["tp"]
        else:
            px = None
        if px is not None:
            w += p["q"] * (px - p["e"]); tr0.append((p["b"], i)); p = None
    eq0[i] = w + (p["q"] * (c[i] - p["e"]) if p is not None else 0.0)
    if p is None and S[i] and c[i] > SL[i]:
        p = dict(b=i, e=c[i], q=eq0[i] / c[i], sl=SL[i], tp=TP[i])
e0 = pd.Series(eq0, index=eq.index)
d0 = pd.concat([pd.Series([B.initial_capital], index=[pd.Timestamp("2019-12-31", tz="UTC")]), e0.groupby(lab).last()])
same = np.array_equal(np.array(tr0), tr[["entry_bar", "exit_bar"]].to_numpy())
print(f"[zero-cost recompounded] final {d0.iloc[-1]:.2f} ({d0.iloc[-1]/1e5-1:+.4%}), sharpe {sharpe(daily_returns(d0)):.4f}, same entry/exit bars: {same}")
# product identity: zero-cost compounded final = prod(1 + gross_ref_return)
gr = (T2["exit_ref"] / T2["entry_ref"]).to_numpy()
print(f"[zero-cost recompounded] = 1e5*prod(exit_ref/entry_ref): {1e5*np.prod(gr):.2f}")
# fixed-qty final = 1e5 + sum(gross_ref)
print(f"[zero-cost fixed qty]  = 1e5 + sum(gross_ref): {1e5 + float((T2['qty']*(T2['exit_ref']-T2['entry_ref'])).sum()):.2f}")
