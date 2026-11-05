# D-7: funding convention at settlement timestamps, independent recomputation, and sensitivities (baseline only).
import _setup
import numpy as np
import pandas as pd
from ppvp_audit.common import is_run

out = is_run()
res, df = out["result"], out["df1h"]
tr = res.trades.copy()
_, fund, _ = __import__("ppvp_audit.common", fromlist=["load_is_market"]).load_is_market()
o = df["open"]; c = df["close"]
H = pd.Timedelta(hours=1)
print("IS trades", len(tr), "engine funding total $%.2f" % tr["funding"].sum(), "settlements counted", int(tr["n_funding"].sum()),
      "on", int((tr["n_funding"] > 0).sum()), "trades")

# independent recomputation: settlements T with entry_time < T <= exit_bar_open (SL/TP exits only in baseline)
assert set(tr["exit_reason"]) <= {"SL", "TP"}
rows = []
for k, r in tr.iterrows():
    sel = fund[(fund.index > r["entry_time"]) & (fund.index <= r["exit_bar_open"])]
    for T, fr in sel["funding_rate"].items():
        rows.append({"trade": k, "T": T, "rate": fr, "amt_open": r["qty"] * o[T] * fr,
                     "amt_prevclose": r["qty"] * c[T - H] * fr, "amt_entry_notional": r["notional"] * fr})
ev = pd.DataFrame(rows)
mine = ev.groupby("trade")["amt_open"].sum().reindex(tr.index, fill_value=0.0)
print("independent: %d settlements on %d trades, total $%.2f, max |diff| vs engine %.2e"
      % (len(ev), ev["trade"].nunique(), ev["amt_open"].sum(), (mine - tr["funding"]).abs().max()))
print("basis = previous 1H close instead of open: delta $%.4f" % (ev["amt_prevclose"].sum() - ev["amt_open"].sum()))
print("basis = entry notional instead of qty*open(T): delta $%.4f" % (ev["amt_entry_notional"].sum() - ev["amt_open"].sum()))

# entries exactly at a settlement (entry_time = signal-bar close = T): not charged by convention
on = tr[tr["entry_time"].isin(fund.index)]
alt = sum(r["qty"] * o[r["entry_time"]] * fund.loc[r["entry_time"], "funding_rate"] for _, r in on.iterrows())
print("entries at a settlement:", len(on), "by hour", on["entry_time"].dt.hour.value_counts().sort_index().to_dict(),
      "rates [%.3g, %.3g]" % (fund.loc[on["entry_time"], "funding_rate"].min(), fund.loc[on["entry_time"], "funding_rate"].max()),
      "| charging them would add $%.2f" % alt)
# exits on a bar that opens at a settlement: charged before the SL/TP check
xs = tr[tr["exit_bar_open"].isin(fund.index)]
xamt = sum(r["qty"] * o[r["exit_bar_open"]] * fund.loc[r["exit_bar_open"], "funding_rate"] for _, r in xs.iterrows())
gap_fill = xs[(xs["exit_reason"] == "SL") & (o[xs["exit_bar_open"]].to_numpy() < xs["sl"].to_numpy())]
print("exits on a bar opening at a settlement:", len(xs), xs["exit_reason"].value_counts().to_dict(),
      "charged $%.2f" % xamt, "| SL filled at a gap open among them:", len(gap_fill))
# same-bar entry/exit impossible (checks start next bar)
print("min hold_bars", int(tr["hold_bars"].min()))
