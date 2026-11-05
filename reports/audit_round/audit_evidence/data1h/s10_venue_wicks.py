# Venue cross-check of the prices that decide the 109 IS fills: perp (traded venue) vs Binance spot (independent).
import numpy as np
import pandas as pd
from ppvp_audit.common import is_run
from config import IS_END
out = is_run()
df1h, tr = out["df1h"], out["result"].trades
sp = pd.read_parquet("/home/user/ppvp_quant/data/processed/BTCUSDT_spot_1h.parquet")
sp = sp[sp.index < IS_END].reindex(df1h.index)
lo_d = (df1h["low"] / sp["low"] - 1) * 1e4
hi_d = (df1h["high"] / sp["high"] - 1) * 1e4
print("IS hours: perp low below spot low by > 50bp:", int((lo_d < -50).sum()), "| perp high above spot high by > 50bp:", int((hi_d > 50).sum()))
print("largest perp-only down-wicks (bp):", lo_d.nsmallest(5).round(1).to_dict())
xb = tr["exit_bar"].to_numpy(); eb = tr["entry_bar"].to_numpy()
sl = tr["exit_reason"].eq("SL").to_numpy(); tp = tr["exit_reason"].eq("TP").to_numpy()
sl_px = tr["sl"].to_numpy(); tp_px = tr["tp"].to_numpy()
spot_low_x = sp["low"].to_numpy()[xb]; spot_high_x = sp["high"].to_numpy()[xb]
print("SL exits:", int(sl.sum()), "| spot low also <= SL:", int((sl & (spot_low_x <= sl_px)).sum()),
      "| spot bar missing:", int((sl & np.isnan(spot_low_x)).sum()),
      "| perp-only stop (spot low > SL):", int((sl & (spot_low_x > sl_px)).sum()))
po = sl & (spot_low_x > sl_px)
if po.any():
    d = pd.DataFrame({"exit_bar_open": tr.loc[po, "exit_bar_open"].astype(str), "SL": sl_px[po],
                      "perp_low": df1h["low"].to_numpy()[xb[po]], "spot_low": spot_low_x[po]})
    d["perp_low_below_SL_bp"] = (d["perp_low"] / d["SL"] - 1) * 1e4
    d["spot_low_above_SL_bp"] = (d["spot_low"] / d["SL"] - 1) * 1e4
    print(d.to_string())
print("TP exits:", int(tp.sum()), "| spot high also >= TP:", int((tp & (spot_high_x >= tp_px)).sum()),
      "| perp-only TP (spot high < TP):", int((tp & (spot_high_x < tp_px)).sum()))
basis_entry = (df1h["close"].to_numpy()[eb] / sp["close"].to_numpy()[eb] - 1) * 1e4
print("entry-bar close perp-spot basis bp: median %.2f min %.2f max %.2f" % (np.nanmedian(basis_entry), np.nanmin(basis_entry), np.nanmax(basis_entry)))
gap = (df1h["open"] / df1h["close"].shift(1) - 1).abs() * 1e4
big = set(np.flatnonzero(gap.to_numpy() > 5))
print("trades whose entry bar / exit bar / bar after entry has |open/prev close| > 5bp:",
      int(sum(e in big for e in eb)), int(sum(x in big for x in xb)), int(sum((e + 1) in big for e in eb)))
