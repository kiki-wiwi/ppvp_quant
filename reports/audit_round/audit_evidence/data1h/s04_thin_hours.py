# Inspect anomalous IS hours (thin volume, open/close discontinuities) and their downstream use.
import numpy as np
import pandas as pd
from ppvp_audit.common import load_is_market
from config import IS_END
df1h, fund, spot = load_is_market()
spot1h = pd.read_parquet("/home/user/ppvp_quant/data/processed/BTCUSDT_spot_1h.parquet")
spot1h = spot1h[spot1h.index < IS_END]
cols = ["open", "high", "low", "close", "volume", "count", "taker_buy_volume"]
for t in ["2021-03-02 01:00", "2020-02-13 14:00", "2020-02-19 12:00"]:
    t = pd.Timestamp(t, tz="UTC")
    w = pd.date_range(t - pd.Timedelta(hours=2), t + pd.Timedelta(hours=3), freq="h", tz="UTC")
    x = df1h[cols].reindex(w)
    x["spot_close"] = spot1h["close"].reindex(w)
    x["spot_volume"] = spot1h["volume"].reindex(w)
    x["open_vs_prev_close_bp"] = (x["open"] / x["close"].shift(1) - 1) * 1e4
    x["range_bp"] = (x["high"] / x["low"] - 1) * 1e4
    print("window around", t)
    print(x.to_string())
    print()
