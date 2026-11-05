"""v01 (skeptic, exec1h): baseline-only facts behind E-1..E-6, recomputed independently from the sanctioned entry points.
BTC IS only, local 1H data only, baseline (B_original) only; no variant is run and no limit price is computed."""
import sys
sys.path.insert(0, "/home/user/ppvp_quant")
import numpy as np, pandas as pd
from ppvp_audit.common import is_run, load_is_market
from config import IS_END, IS_START, BASELINE_BACKTEST as B
from backtester import run_backtest
from ppvp_audit.engine_path import run_path

out = is_run()
df1h, fund, _ = load_is_market()
assert df1h.index.max() < IS_END and out["df1h"].index.max() < IS_END and fund.index.max() <= IS_END
d = out["df1h"]; sig = out["signals"]; res = out["result"]; tr = res.trades
print("== data")
print("1H bars", len(d), d.index[0], d.index[-1], "| same as load_is_market:", d.index.equals(df1h.index))
dt = np.diff(d.index.as_unit("ns").asi8)
print("1H gaps (diff != 1h):", int((dt != 3600 * 10**9).sum()), "| min/max diff h:", dt.min() / 3.6e12, dt.max() / 3.6e12)
ft = fund.index.as_unit("ns").asi8
win = (ft > d.index.as_unit("ns").asi8[0]) & (ft <= d.index.as_unit("ns").asi8[-1] + 3600 * 10**9)
print("funding settlements in window", int(win.sum()), "| all on 8h grid:", bool((ft[win] % (8 * 3600 * 10**9) == 0).all()),
      "| NaN rates in window:", int(np.isnan(fund['funding_rate'].to_numpy()[win]).sum()))
print("trades", len(tr), tr["exit_reason"].value_counts().to_dict())

print("== E-1/E-2: engine_path (market mode) vs backtester on the baseline")
b = run_path(d, sig, B, fund)
print("equity identical:", bool(np.array_equal(res.equity.to_numpy(), b["equity"].to_numpy())))
for col in ("entry_bar", "exit_bar", "entry_px", "exit_px", "qty", "fees", "funding", "pnl", "hold_bars"):
    print(f"  {col}: identical={bool(np.array_equal(tr[col].to_numpy(), b['trades'][col].to_numpy()))}")
print("  total funding backtester %.6f  engine_path %.6f  n_settlements %d" % (tr["funding"].sum(), b["trades"]["funding"].sum(),
                                                                             int(tr["n_funding"].sum())))
# which settlement bars fall right after an entry bar (the only place the index rule could differ from the time rule)
eb = tr["entry_bar"].to_numpy(); xb = tr["exit_bar"].to_numpy()
o_ns = d.index.as_unit("ns").asi8
fbar = np.searchsorted(o_ns, ft[win], side="left")
after_entry = np.isin(fbar, eb + 1)
print("  settlements mapped to entry_bar+1 (not charged by either engine: entry close == T):", int(after_entry.sum()),
      "| of those, bar entry_bar+1 opens exactly at T:", bool((o_ns[fbar[after_entry]] == ft[win][after_entry]).all()))

print("== E-3: gaps between consecutive baseline signals (C1 parents are the tradable baseline signals)")
t = np.flatnonzero(sig["signal"].to_numpy()); r = np.flatnonzero(sig["sig_raw"].to_numpy())
print("tradable", len(t), "min gap", int(np.diff(t).min()), "gaps<=4:", int((np.diff(t) <= 4).sum()),
      "| raw", len(r), "min gap", int(np.diff(r).min()), "gaps<=4:", int((np.diff(r) <= 4).sum()))
print("smallest 5 tradable gaps:", sorted(np.diff(t).tolist())[:5])

print("== E-4: gross R before costs, fill-based vs reference denominator")
s = B.slippage
ep, xp, q, sl = (tr[k].to_numpy() for k in ("entry_px", "exit_px", "qty", "sl"))
cl = d["close"].to_numpy()[eb]
assert np.allclose(ep, cl * (1 + s), rtol=0, atol=1e-9)
g_ref = q * (xp / (1 - s) - ep / (1 + s))
r_fill = g_ref / (q * (ep - sl)); r_ref = g_ref / (q * (cl - sl))
isl = (tr["exit_reason"] == "SL").to_numpy()
o = d["open"].to_numpy()
gapped = isl & (o[xb] < sl)
print("mean gross R (fill denom) %.4f | (ref denom) %.4f" % (r_fill.mean(), r_ref.mean()))
print("SL trades", int(isl.sum()), "gapped through SL at exit-bar open:", int(gapped.sum()))
print("SL fill-denom range [%.4f, %.4f]; ref-denom range [%.6f, %.6f]" % (r_fill[isl].min(), r_fill[isl].max(),
                                                                       r_ref[isl].min(), r_ref[isl].max()))
dsl = 1 - sl / ep
j = int(np.argmin(np.where(isl, dsl, np.inf)))
print("tightest-stop SL trade: idx %d entry %s d_sl %.1fbp gross_R_fill %.4f; hand: -(close-SL)/(fill-SL) = %.4f" %
      (j, tr["entry_time"].iat[j], dsl[j] * 1e4, r_fill[j], -(cl[j] - sl[j]) / (ep[j] - sl[j])))
net_r = tr["pnl"].to_numpy() / (q * (ep - sl))
cost_r = (g_ref - tr["pnl"].to_numpy()) / (q * (ep - sl))
print("additivity with the common fill denominator: gross %.4f - costs %.4f = net %.4f (ledger net mean %.4f)" %
      (r_fill.mean(), cost_r.mean(), r_fill.mean() - cost_r.mean(), net_r.mean()))

print("== E-5: exit-bar ambiguity sets")
h = d["high"].to_numpy(); tp = tr["tp"].to_numpy(); poc = tr["poc"].to_numpy()
mfe = np.array([h[a + 1:x].max() if x > a + 1 else np.nan for a, x in zip(eb, xb)])
poc_above = poc > ep
strict = poc_above & (np.nan_to_num(mfe, nan=-np.inf) >= poc)
left = isl & poc_above & ~np.isnan(mfe)
sl_tp = isl & (h[xb] >= tp)
sl_poc_left_ns = left & ~strict & (h[xb] >= poc)
sl_poc_any = isl & poc_above & (h[xb] >= poc)
print("left panel", int(left.sum()), "strict reached", int((left & strict).sum()))
print("SL&TP idx:", np.flatnonzero(sl_tp).tolist(), "| SL&POC left-panel not-strict idx:", np.flatnonzero(sl_poc_left_ns).tolist())
print("distinct trades behind the '1 + 2' counts:", sorted(set(np.flatnonzero(sl_tp)) | set(np.flatnonzero(sl_poc_left_ns))))
u = np.flatnonzero(sl_tp | sl_poc_any)
print("union SL&TP | SL&POC(any):", u.tolist(), "=", len(u), "/ 109 = %.2f%%" % (100 * len(u) / 109))
for k in u:
    print("  idx %d entry %s exit-bar %s O %.2f H %.2f L %.2f | SL %.2f POC %.2f TP %.2f | strict POC before exit: %s" %
          (k, tr["entry_time"].iat[k], d.index[xb[k]], o[xb[k]], h[xb[k]], d["low"].to_numpy()[xb[k]], sl[k], poc[k], tp[k],
           bool(strict[k])))
print("POC < TP for SL&TP trades:", [(int(k), bool(poc[k] < tp[k]), bool(poc_above[k])) for k in np.flatnonzero(sl_tp)])

print("== E-6: cost block identities")
gross_fill = tr["gross_pnl"].sum(); fees = tr["fees"].sum(); fu = tr["funding"].sum(); net = tr["pnl"].sum()
slip_usd = (q * (ep - ep / (1 + s)) + q * (xp / (1 - s) - xp)).sum()
print("gross_at_fill %.2f fees %.2f funding %.2f slippage %.2f net %.2f" % (gross_fill, fees, fu, slip_usd, net))
print("gross_at_fill - fees - funding = %.2f (net %.2f)" % (gross_fill - fees - fu, net))
print("gross_at_fill - fees - funding - slippage = %.2f (double count)" % (gross_fill - fees - fu - slip_usd))
print("gross_ref = gross_at_fill + slippage = %.2f ; gross_ref - slip - fees - funding = %.2f" %
      (gross_fill + slip_usd, gross_fill + slip_usd - slip_usd - fees - fu))
