# Baseline (B_original) IS run: end-to-end time conventions, funding settlement ordering / basis, box availability,
# early-2020 warm-up dependencies. Only the baseline is run (no variants).
import numpy as np
import pandas as pd
from ppvp_audit.common import is_run, load_is_market
from config import IS_START, IS_END, BASELINE_STRATEGY as S, BASELINE_BACKTEST as B
from metrics import daily_equity
from strategy_state_machine import generate_signals

out = is_run()
df1h, sig, boxes, df4h, res = out["df1h"], out["signals"], out["boxes"], out["df4h"], out["result"]
_, fund, spot = load_is_market()
tr = res.trades
H = pd.Timedelta(hours=1)
print("trades", len(tr), "exit reasons", tr["exit_reason"].value_counts().to_dict(), "IS sharpe", round(out["report"]["IS"]["sharpe"], 4))
print("funding rows passed to engine:", len(fund), "max ts == IS_END:", fund.index.max() == IS_END)

# --- index units / labels ---
print("df1h index unit", df1h.index.dtype, "| equity index unit", res.equity.index.dtype, "| first/last equity ts",
      res.equity.index[0], res.equity.index[-1])
lab = (res.equity.index - pd.Timedelta(1, "ns"))
print("equity index minus 1ns dtype", lab.dtype, "| label of 2020-01-02 00:00 point:",
      lab[res.equity.index == pd.Timestamp("2020-01-02", tz="UTC")].floor("D")[0].date())
de = daily_equity(res)
print("daily equity first rows:", de.iloc[:3].to_dict(), " last label", de.index[-1].date(), "len", len(de))
print("daily equity 2020-01-01 value == equity at 2020-01-02 00:00:",
      de.loc["2020-01-01"] == res.equity.loc[pd.Timestamp("2020-01-02", tz="UTC")])

# --- box availability vs use ---
bid = sig["box_id"].to_numpy()
used = bid >= 0
av = pd.DatetimeIndex(boxes["available_time"]).as_unit("ns").asi8
op = df1h.index.as_unit("ns").asi8
print("\nbars using a box:", int(used.sum()), " violations available_time > open_time:", int((av[bid[used]] > op[used]).sum()),
      " bars where a newer box was already available but not used:",
      int((np.searchsorted(av, op[used], side="right") - 1 != bid[used]).sum()))
we = pd.DatetimeIndex(boxes["window_end"]); ws = pd.DatetimeIndex(boxes["window_start"]); at = pd.DatetimeIndex(boxes["available_time"])
print("boxes", len(boxes), "available_time - window_end values:", pd.Series(at - we).value_counts().to_dict())
print("window_start on 4H boundary:", bool(((ws.hour % 4 == 0) & (ws.minute == 0)).all()),
      " n_1h == hours(window):", bool((boxes["n_1h"].to_numpy() == ((we - ws) / H).astype(int)).all()),
      " min window_start:", ws.min())
print("first box available", at.min(), " first box window", ws[0], "->", we[0])
atr_first = df1h.index[np.flatnonzero(np.isfinite(sig["atr"].to_numpy()))[0]]
vs_first = df1h.index[np.flatnonzero(np.isfinite(sig["vsma"].to_numpy()))[0]]
first_sig = df1h.index[np.flatnonzero(sig["sig_raw"].to_numpy())[0]]
print("first finite ATR at bar open", atr_first, " first vsma", vs_first, " first raw signal bar", first_sig,
      " first trade entry", tr["entry_time"].min())

# --- spot warm-up not used by the baseline signal path ---
s2 = generate_signals(df1h, S, boxes, df4h, None)
same = all(np.array_equal(s2[c].to_numpy(), sig[c].to_numpy(), equal_nan=True) if s2[c].dtype.kind == "f"
           else np.array_equal(s2[c].to_numpy(), sig[c].to_numpy()) for c in sig.columns)
print("\nsignals identical with spot_daily=None (baseline regime_filter=None):", same)
print("spot warm-up series: n", len(spot), "first", spot.index[0].date(), "last", spot.index[-1].date(), "name", spot.name, "dtype", spot.dtype)

# --- funding: independent recomputation of every trade's settlements ---
o = df1h["open"]
fr = fund["funding_rate"]
rows = []
for _, t in tr.iterrows():
    e, xo = t["entry_time"], t["exit_bar_open"]
    Ts = fr.index[(fr.index > e) & (fr.index <= xo)]        # SL/TP: settlements strictly after entry, up to the exit-bar open
    amt = float((t["qty"] * o.reindex(Ts).to_numpy() * fr.reindex(Ts).to_numpy()).sum())
    alt_entry_notional = float((t["qty"] * t["entry_px"] * fr.reindex(Ts).to_numpy()).sum())
    prev_close = df1h["close"].reindex(Ts - H).to_numpy()
    alt_prev_close = float((t["qty"] * prev_close * fr.reindex(Ts).to_numpy()).sum())
    at_entry = e in fr.index
    at_exit_open = xo in fr.index
    gap_fill = bool(t["exit_reason"] == "SL" and df1h.loc[xo, "open"] <= t["sl"])
    rows.append({"entry_time": e, "exit_bar_open": xo, "n_fund_engine": t["n_funding"], "n_fund_indep": len(Ts),
                 "fund_engine": t["funding"], "fund_indep": amt, "alt_entry_notional": alt_entry_notional,
                 "alt_prev_close": alt_prev_close, "entry_at_settlement": at_entry,
                 "rate_at_entry_settlement": float(fr.get(e, np.nan)) if at_entry else np.nan,
                 "notional_entry": t["notional"], "qty": t["qty"], "entry_px": t["entry_px"],
                 "exit_bar_open_is_settlement": at_exit_open, "exit_reason": t["exit_reason"],
                 "sl_gap_fill_at_open": gap_fill, "pnl": t["pnl"], "hold_bars": t["hold_bars"]})
F = pd.DataFrame(rows)
print("\nfunding recomputation: n_funding equal", bool((F["n_fund_engine"] == F["n_fund_indep"]).all()),
      " max |engine - indep| $", float((F["fund_engine"] - F["fund_indep"]).abs().max()),
      " total engine $", round(F["fund_engine"].sum(), 4), " total indep $", round(F["fund_indep"].sum(), 4))
print("trades with >=1 settlement:", int((F["n_fund_indep"] > 0).sum()), " total settlements:", int(F["n_fund_indep"].sum()))
print("notional basis alternatives: entry-notional basis total $ %.4f (diff %.4f), prev-close basis $ %.4f (diff %.4f)" % (
      F["alt_entry_notional"].sum(), F["alt_entry_notional"].sum() - F["fund_engine"].sum(),
      F["alt_prev_close"].sum(), F["alt_prev_close"].sum() - F["fund_engine"].sum()))
# same-timestamp ordering cases
ent = F[F["entry_at_settlement"]]
print("\nentries exactly at a settlement time (fill after settlement => not charged):", len(ent),
      " entry hours:", tr["entry_time"].dt.hour.value_counts().sort_index().to_dict())
alt_entry_charge = float((ent["qty"] * df1h["open"].reindex(ent["entry_time"]).to_numpy() * ent["rate_at_entry_settlement"].to_numpy()).sum())
alt_entry_charge2 = float((ent["notional_entry"] * ent["rate_at_entry_settlement"]).sum())
print("   if those settlements were charged instead: +$%.4f (open-at-T basis) / +$%.4f (entry notional), "
      "vs total funding $%.4f and total |pnl| $%.1f" % (alt_entry_charge, alt_entry_charge2, F["fund_engine"].sum(), F["pnl"].abs().sum()))
print("   entry-time settlement rates: min %.6g max %.6g" % (ent["rate_at_entry_settlement"].min(), ent["rate_at_entry_settlement"].max()))
ex = F[F["exit_bar_open_is_settlement"]]
print("exits whose exit-bar open is a settlement (held through T => charged):", len(ex), " of which SL:",
      int((ex["exit_reason"] == "SL").sum()), " TP:", int((ex["exit_reason"] == "TP").sum()),
      " SL filled at the gap open on such a bar:", int(ex["sl_gap_fill_at_open"].sum()))
exr = fund["funding_rate"].reindex(ex["exit_bar_open"]).to_numpy()
print("   funding charged at those exit-bar-open settlements: $%.4f" % float((ex["qty"] * df1h["open"].reindex(ex["exit_bar_open"]).to_numpy() * exr).sum()))
print("SL fills at the bar open (gap through stop), all trades:", int(F["sl_gap_fill_at_open"].sum()))
same_bar = int((tr["exit_bar"] == tr["entry_bar"] + 1).sum())
print("trades exiting on the first bar after entry:", same_bar)
F.to_csv("/tmp/claude-0/-home-user-ppvp-quant/04a66f10-1e5e-5960-9a9f-ffb5b694e43a/scratchpad/audit_round/data1h/s06_funding_by_trade.csv", index=False)

# entry close vs next open across all trades
nxt = df1h["open"].shift(-1).to_numpy()[tr["entry_bar"].to_numpy()]
cl = df1h["close"].to_numpy()[tr["entry_bar"].to_numpy()]
print("\ntrade entry close vs next open bp: max abs %.3f" % float(np.nanmax(np.abs(nxt / cl - 1) * 1e4)))
