# D-11: does the independent Binance spot 1H series also reach the SL / TP of each IS exit bar?  D-13: first-bar exits.
import _setup
import numpy as np
import pandas as pd
from _setup import read_spot_is
from ppvp_audit.common import is_run

out = is_run()
tr = out["result"].trades
df = out["df1h"]
s = read_spot_is()
xb = tr["exit_bar_open"]
p_low = df["low"].reindex(xb).to_numpy(); p_high = df["high"].reindex(xb).to_numpy()
s_low = s["low"].reindex(xb).to_numpy(); s_high = s["high"].reindex(xb).to_numpy()
sl = tr["sl"].to_numpy(); tp = tr["tp"].to_numpy()
isl = (tr["exit_reason"] == "SL").to_numpy(); itp = (tr["exit_reason"] == "TP").to_numpy()
print("spot rows missing at exit bars:", int(np.isnan(s_low).sum()))
print("SL exits", isl.sum(), "| perp low <= SL:", int((p_low[isl] <= sl[isl]).sum()), "| spot low <= SL:", int((s_low[isl] <= sl[isl]).sum()))
print("TP exits", itp.sum(), "| perp high >= TP:", int((p_high[itp] >= tp[itp]).sum()), "| spot high >= TP:", int((s_high[itp] >= tp[itp]).sum()))
po = isl & (s_low > sl)
t = pd.DataFrame({"exit_bar_open": xb[po].to_numpy(), "SL": sl[po], "perp_low_bp_vs_SL": (p_low[po] / sl[po] - 1) * 1e4,
                  "spot_low_bp_vs_SL": (s_low[po] / sl[po] - 1) * 1e4})
print("perp-only SL breaches:\n", t.round(2).to_string(index=False))
tpo = itp & (s_high < tp)
print("TP touched on perp but not spot:", [(str(a), round(b, 2)) for a, b in zip(xb[tpo], (s_high[tpo] / tp[tpo] - 1) * 1e4)])
# D-13: hold_bars = exit_bar - entry_bar; first-bar exits
hb = tr["hold_bars"]
print("\nD-13 hold_bars == exit_bar - entry_bar for all:", bool((hb == tr["exit_bar"] - tr["entry_bar"]).all()),
      "| exits on the first bar after entry:", int((hb == 1).sum()), tr.loc[hb == 1, "exit_reason"].value_counts().to_dict())
print("exit_time == exit_bar_open + 1h:", bool((tr["exit_time"] == tr["exit_bar_open"] + pd.Timedelta(hours=1)).all()),
      "| entry_time == open(entry_bar)+1h:", bool((tr["entry_time"] == df.index[tr["entry_bar"]] + pd.Timedelta(hours=1)).all()))
led = pd.read_csv("reports/audit_round/trades_B_original.csv")
print("ledger hold_hours == hold_bars:", bool((led["hold_hours"] == led["hold_bars"]).all()), "| ledger first-bar exits:", int((led["hold_hours"] == 1).sum()))
