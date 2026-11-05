# D-10: spot warm-up content, its (non-)use in baseline signals, and its reach into regime features of the ledgers.
import _setup
import numpy as np
import pandas as pd
import pipeline
from config import BASELINE_STRATEGY as S
from ppvp_audit.common import is_run, load_is_market
from strategy_state_machine import daily_closes_with_warmup, generate_signals
from _setup import IS_START

df1h, fund, spot = load_is_market()
print("warm-up series: n", len(spot), "first", spot.index.min().date(), "last", spot.index.max().date(),
      "all < 2020:", bool((spot.index < IS_START).all()), "dtype", spot.dtype, "name", spot.name)
raw = pd.read_parquet("data/processed/BTCUSDT_spot_1h.parquet")
raw = raw[raw.index < IS_START]                      # 2019 only
days = raw.index.floor("D").unique()
has23 = raw.index[raw.index.hour == 23].floor("D").unique()
print("2019 spot days with data", len(days), "with a 23:00 bar", len(has23))
chk = raw["close"].groupby(raw.index.floor("D")).last()
print("warm-up equals the last 1H close of each 2019 day:", bool(np.allclose(chk.reindex(spot.index).to_numpy(), spot.to_numpy())))

out = is_run()
sig_w = out["signals"]
boxes, df4h = pipeline.get_boxes("BTCUSDT", S, _setup.IS_END)
sig_n = generate_signals(df1h, S, boxes, df4h, None)
same = all(np.array_equal(sig_w[c].to_numpy(), sig_n[c].to_numpy(), equal_nan=True) if sig_w[c].dtype.kind == "f"
           else np.array_equal(sig_w[c].to_numpy(), sig_n[c].to_numpy()) for c in sig_w.columns)
print("baseline signal table identical with spot warm-up removed:", same, "| regime_filter =", S.regime_filter)
print("first box: window_start", boxes["window_start"].iloc[0], "available", boxes["available_time"].iloc[0])
fa = sig_w["atr"].first_valid_index(); fv = sig_w["vsma"].first_valid_index()
print("ATR first defined at bar open", fa, "| vol SMA first defined at bar open", fv)

# regime features: which raw signals' windows reach the 2019 spot closes
d = daily_closes_with_warmup(df1h, spot)
is_spot = d.index < IS_START
raw_sig = sig_w.index[sig_w["sig_raw"].to_numpy()]
traded = set(out["result"].trades["entry_time"] - pd.Timedelta(hours=1))
dec = raw_sig + pd.Timedelta(hours=1)                      # decision time = signal-bar close
avail = d.index + pd.Timedelta(days=1)
j = np.searchsorted(avail.asi8 if hasattr(avail, "asi8") else avail.view("i8"), dec.as_unit(avail.unit).asi8, side="right") - 1
def reach(back):   # window of `back` daily closes ending at j includes a spot close?
    lo = j - back + 1
    return np.array([bool(is_spot[max(l, 0):jj + 1].any()) for l, jj in zip(lo, j)])
r200 = reach(200); r220 = reach(220); r365 = reach(365)
trd = np.array([t in traded for t in raw_sig])
print("raw signals", len(raw_sig), "traded", int(trd.sum()))
print("SMA200 window reaches 2019 spot: raw %d, traded %d" % (r200.sum(), (r200 & trd).sum()))
print("sma200_up (SMA200 and its 20-day lag, 220 closes): raw %d, traded %d" % (r220.sum(), (r220 & trd).sum()))
print("365-day windows (dd365): raw %d, traded %d" % (r365.sum(), (r365 & trd).sum()))
k0 = j[0]
print("first signal decision", dec[0], ": SMA200 window = %d spot + %d perp closes" % (is_spot[k0 - 199:k0 + 1].sum(), (~is_spot[k0 - 199:k0 + 1]).sum()))
# splice day return: last 2019 spot close -> first 2020 perp daily close
i0 = int(np.flatnonzero(~is_spot)[0])
print("splice: last spot close %.2f (%s) -> first perp daily close %.2f (%s)" % (d.iloc[i0 - 1], d.index[i0 - 1].date(), d.iloc[i0], d.index[i0].date()))
