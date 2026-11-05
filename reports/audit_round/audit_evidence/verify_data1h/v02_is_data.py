# IS-only data checks: completeness, provenance, thin hours (D-6), the 2023-11-10 12:00 4H bar (D-8), basis
import _setup
import numpy as np
import pandas as pd
from _setup import IS_END, IS_START, read_spot_is
from ppvp_audit.common import load_is_market
from ppvp_engine import build_4h

df, fund, spotd = load_is_market()
H = pd.Timedelta(hours=1)
grid = pd.date_range(IS_START, IS_END - H, freq="h", tz="UTC")
print("1H rows", len(df), "expected", len(grid), "missing", len(grid.difference(df.index)), "extra", len(df.index.difference(grid)))
print("fill_source:", df["fill_source"].value_counts().to_dict())
flat = (df.open == df.high) & (df.high == df.low) & (df.low == df.close)
print("zero-vol rows", int((df.volume == 0).sum()), "flat rows", int(flat.sum()), "NaN OHLCV",
      int(df[["open", "high", "low", "close", "volume"]].isna().sum().sum()), "NaN count", int(df["count"].isna().sum()))

# ---- D-6 thin hours: volume vs median of the PREVIOUS 168 hours (excluding current), and vs including current
v = df["volume"]
med_prev = v.shift(1).rolling(168, min_periods=168).median()
med_incl = v.rolling(168, min_periods=168).median()
r_prev = (v / med_prev).dropna()
r_incl = (v / med_incl).dropna()
print("\nthinnest 4 by vol/median(prev168):"); print(r_prev.nsmallest(4).round(4).to_string())
print("count < 10%:", int((r_prev < 0.10).sum()), " (incl-current median:", int((r_incl < 0.10).sum()), ")")
t = pd.Timestamp("2021-03-02 01:00", tz="UTC")
print("2021-03-02 01:00 vol/med_prev = %.5f, vol/med_incl = %.5f" % (r_prev[t], r_incl[t]))
s = read_spot_is()
w = pd.date_range(t - 2 * H, t + 2 * H, freq="h", tz="UTC")
tab = pd.DataFrame({"p_open": df.open, "p_high": df.high, "p_low": df.low, "p_close": df.close, "p_vol": df.volume,
                    "p_count": df["count"], "s_open": s.open, "s_high": s.high, "s_low": s.low, "s_close": s.close,
                    "s_vol": s.volume, "s_count": s["count"]}).loc[w]
tab["p_range_bp"] = (tab.p_high / tab.p_low - 1) * 1e4
tab["s_range_bp"] = (tab.s_high / tab.s_low - 1) * 1e4
pd.set_option("display.width", 250)
print(tab.round(3).to_string())
# count-vs-median
cmed = df["count"].shift(1).rolling(168, min_periods=168).median()
print("count/median(prev168) at t: %.5f" % (df.loc[t, "count"] / cmed[t]))
# spot thinness at the same hour
sv = s["volume"]; smed = sv.shift(1).rolling(168, min_periods=168).median()
print("spot vol/median(prev168) at t: %.4f" % (sv[t] / smed[t]))

# ---- D-8: the 2023-11-10 12:00 4H bar
d4 = build_4h(df)
b = pd.Timestamp("2023-11-10 12:00", tz="UTC")
print("\nlocal 4H 2023-11-10 12:00:", d4.loc[b, ["open", "high", "low", "close", "volume", "n_bars"]].to_dict())
print("sum of 4 1H volumes: %.3f" % df.loc[b:b + 3 * H, "volume"].sum())
nxt = df.loc[b + 4 * H, "open"]
print("next 1H (16:00) open", nxt, " local close vs next open bp %.2f" % ((d4.loc[b, "close"] / nxt - 1) * 1e4),
      " official close 37118.4 vs next open bp %.2f" % ((37118.4 / nxt - 1) * 1e4))
sc = s.loc[b + 3 * H, "close"]
print("spot 15:00 close", sc, " basis local %.2f bp, official %.2f bp" % ((d4.loc[b, "close"] / sc - 1) * 1e4, (37118.4 / sc - 1) * 1e4))
bas = (df["close"] / s["close"] - 1).dropna() * 1e4
print("adjacent 1H basis bp:", bas.loc[b:b + 7 * H].round(2).to_dict())
print("official vol / local vol = %.5f" % (68307.178 / d4.loc[b, "volume"]))
print("IS perp-spot close basis bp: median %.3f n=%d" % (bas.median(), len(bas)))
