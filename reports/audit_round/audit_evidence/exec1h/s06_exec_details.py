"""s06: execution-rule details on the 109 baseline IS trades: TP fills by touch vs 1-tick penetration (the C1 entry rule),
TP/SL distance vs fill, leverage/cap, entry at next open instead of close (effect), worst-case trade alternative path,
cost-per-R breakdown claims of the Phase 2 log. IS only, baseline only."""
import pickle, sys
sys.path.insert(0, "/home/user/ppvp_quant")
from pathlib import Path
import numpy as np, pandas as pd
from config import BASELINE_BACKTEST as B
from ppvp_audit.engine_path import tick_size

D = Path(__file__).resolve().parent
X = pickle.load(open(D / "baseline.pkl", "rb"))
df, tr, sig = X["df1h"], X["trades"], X["signals"]
o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
eb = tr["entry_bar"].to_numpy(int); xb = tr["exit_bar"].to_numpy(int)
tick = tick_size(df.index)
tp_m = (tr["exit_reason"] == "TP").to_numpy()
pen = h[xb] - tr["tp"].to_numpy()
touch_only = tp_m & (pen < tick[xb])
print(f"[TP] TP exits {tp_m.sum()}; exit-bar high - TP: min {pen[tp_m].min():.4f}; touch-only (penetration < 1 tick): {int(touch_only.sum())}; "
      f"penetration < 1bp of price: {int((tp_m & (pen / tr['tp'].to_numpy() < 1e-4)).sum())}; < 5bp: {int((tp_m & (pen / tr['tp'].to_numpy() < 5e-4)).sum())}")
for j in np.flatnonzero(touch_only):
    print(f"   trade {j} entry {tr['entry_time'].iat[j]} exit bar {df.index[xb[j]]} high {h[xb[j]]} TP {tr['tp'].iat[j]:.4f} tick {tick[xb[j]]} net {tr['pnl'].iat[j]:.2f}")
d_tp = tr["tp"] / tr["entry_px"] - 1; d_sl = 1 - tr["sl"] / tr["entry_px"]
print(f"[geometry] d_tp vs fill: min {d_tp.min()*1e4:.2f} bp (all > 0: {bool((d_tp>0).all())}); d_sl min {d_sl.min()*1e4:.1f} bp")
print(f"[geometry] TP exits with negative net pnl: {int((tp_m & (tr['pnl'] < 0)).sum())}, their d_tp (bp): {np.round(d_tp[tp_m & (tr['pnl'] < 0)].to_numpy()*1e4, 2).tolist()}")
rt = B.perp_taker_fee + B.perp_maker_fee + 2 * B.slippage
print(f"[geometry] trades with d_tp < 3 x TP round-trip cost ({rt*1e4:.0f} bp): {int((d_tp < 3 * rt).sum())}")
print(f"[sizing] leverage min/max {tr['leverage'].min():.6f}/{tr['leverage'].max():.6f}; capped {int(tr['capped'].sum())}; notional == equity_at_entry: {bool(np.allclose(tr['notional'], tr['equity_at_entry']))}")
# cost per R (Phase 2 log: fees 0.075R, slippage 0.034R, funding 0.010R, total 0.119R)
L = pd.read_csv("/home/user/ppvp_quant/reports/audit_round/trades_B_original.csv")
risk = L["initial_risk_usd"]
print(f"[cost/R] fees {np.mean(L['commission']/risk):.4f}R, slippage {np.mean(L['slippage_impact']/risk):.4f}R, funding {np.mean(L['funding_cashflow']/risk):.4f}R, "
      f"total {np.mean((L['commission']+L['slippage_impact']+L['funding_cashflow'])/risk):.4f}R")
# entry at the next bar's open instead of the signal close: same qty, same exits (exit checks already start at bar e+1)
on = o[eb + 1]
d_entry = tr["qty"].to_numpy() * (on - c[eb]) * (1 + B.slippage)
d_fee = tr["qty"].to_numpy() * (on - c[eb]) * (1 + B.slippage) * B.perp_taker_fee
print(f"[next-open entry] total net pnl change (fixed qty & exits): {-(d_entry + d_fee).sum():+.2f} USDT on {tr['pnl'].sum():.2f}; "
      f"max per-trade |change| {np.abs(d_entry + d_fee).max():.2f} USDT; max |R change| "
      f"{np.max(np.abs((d_entry + d_fee) / (tr['qty'] * (tr['entry_px'] - tr['sl'])))):.4f}R")
# would any trade's SL/TP be crossed between the close and the next open? (open of e+1 is the first price of the exit-check bar)
print("[next-open entry] next open at/below SL or at/above TP:", int(((on <= tr['sl']) | (on >= tr['tp'])).sum()))
# the single SL&TP ambiguous trade: alternative (TP-first) outcome for disclosure only
j = int(np.flatnonzero((tr["exit_reason"] == "SL").to_numpy() & (h[xb] >= tr["tp"].to_numpy()))[0])
q = tr["qty"].iat[j]; ep = tr["entry_px"].iat[j]; tpx = tr["tp"].iat[j] * (1 - B.slippage)
alt = q * (tpx - ep) - q * ep * B.perp_taker_fee - q * tpx * B.perp_maker_fee - tr["funding"].iat[j]
print(f"[ambiguous SL&TP] trade {j} {tr['entry_time'].iat[j]}: worst-case net {tr['pnl'].iat[j]:.2f} USDT ({tr['r_multiple'].iat[j]:+.3f}R); "
      f"TP-first alternative {alt:.2f} USDT ({alt / (q * (ep - tr['sl'].iat[j])):+.3f}R); swing {alt - tr['pnl'].iat[j]:.2f} USDT")
