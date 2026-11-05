"""v04 (skeptic, exec1h): scope of E-1 on hand-made bars. Which gap layouts actually mis-charge in engine_path?
(1) one settlement inside a gap in the middle of the holding period; (2) a marketable-at-placement C1 order (market
branch) followed by a gap with a settlement inside; (3) C1 limit fill where the fill bar is the bar right after a gap."""
import sys
sys.path.insert(0, "/home/user/ppvp_quant")
import numpy as np, pandas as pd
from config import BASELINE_BACKTEST as B
from backtester import run_backtest
from ppvp_audit.engine_path import run_path

H = pd.Timedelta(hours=1); T0 = pd.Timestamp("2021-03-01 00:00", tz="UTC")
def frame(rows, hh):
    a = np.asarray(rows, float)
    df = pd.DataFrame({"open": a[:, 0], "high": a[:, 1], "low": a[:, 2], "close": a[:, 3]},
                      index=pd.DatetimeIndex([T0 + h * H for h in hh])); df["volume"] = 100.0; return df
def sigs(df, at, sl, tp):
    s = pd.DataFrame({"signal": False, "sl": np.nan, "tp": np.nan, "poc_tp": np.nan}, index=df.index)
    for x in at: s.iloc[x, :3] = [True, sl, tp]
    return s
F = pd.DataFrame({"funding_rate": [0.001], "funding_interval_hours": 8}, index=pd.DatetimeIndex([T0 + 8 * H]))

# (1) entry at the 01:00-bar close (02:00); bars to 05:00, gap 06:00-09:00 (contains 08:00), resumes 10:00
df = frame([(100, 101, 99, 100), (100, 101, 99, 100.6)] + [(100.6, 100.8, 100.2, 100.4)] * 7, (0, 1, 2, 3, 4, 5, 10, 11, 12))
s = sigs(df, [1], 98.0, 110.0)
p = run_path(df, s, B, F)["trades"].iloc[0]; a = run_backtest(df, s, B, F).trades.iloc[0]
print(f"(1) single settlement in a mid-hold gap: engine_path {p['funding']:.6f} | backtester {a['funding']:.6f} (n={a['n_funding']})"
      f" | expected qty*open(10:00)*r = {p['qty'] * 100.6 * 0.001:.6f}")
# (2) C1 order marketable at placement (L >= close) at the 05:00-bar close, then gap to 12:00 (contains 08:00)
df = frame([(100, 101, 99, 100)] * 5 + [(100, 101, 99, 100.6)] + [(100.6, 100.8, 100.2, 100.4)] * 3, (0, 1, 2, 3, 4, 5, 12, 13, 14))
lim = np.full(len(df), np.nan); lim[5] = 100.7
r = run_path(df, sigs(df, [5], 98.0, 110.0), B, F, limit_price=lim, valid_bars=3)
t = r["trades"].iloc[0]
print(f"(2) marketable-at-placement C1 order then gap: status {r['orders']['status'].iat[0]}, engine_path funding {t['funding']:.6f}"
      f" | documented rule {t['qty'] * 100.6 * 0.001:.6f}")
# (3) limit fill on the first bar after a gap that contains T (fill bar opens 09:00 > T): not charged by either rule
df = frame([(100, 101, 99, 100), (100, 101, 99, 100.6), (100.6, 100.8, 99.9, 100.3)] + [(100.3, 100.5, 100.1, 100.3)] * 3,
           (0, 1, 9, 10, 11, 12))
lim = np.full(len(df), np.nan); lim[1] = 100.0
t = run_path(df, sigs(df, [1], 98.0, 110.0), B, F, limit_price=lim, valid_bars=3)["trades"].iloc[0]
print(f"(3) limit fill on the bar after a gap containing T: fill bar {t['fill_bar']} opens {df.index[int(t['fill_bar'])]}, "
      f"engine_path funding {t['funding']:.6f} (documented rule: 0, the fill happens after T)")
