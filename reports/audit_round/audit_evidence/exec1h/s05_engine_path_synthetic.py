"""s05: synthetic (hand-made bars, no market data) boundary tests of ppvp_audit/engine_path.py: order expiry vs a new signal,
funding at limit/market fills, include_funding flag, gap-through-limit, two settlements in one bar, fill-bar TP."""
import sys
sys.path.insert(0, "/home/user/ppvp_quant")
import numpy as np, pandas as pd
from config import BASELINE_BACKTEST as B
from backtester import run_backtest
from ppvp_audit.engine_path import run_path

H = pd.Timedelta(hours=1)
T0 = pd.Timestamp("2021-03-01 00:00", tz="UTC")
def bars(rows, start=T0):
    a = np.asarray(rows, float)
    idx = pd.date_range(start, periods=len(a), freq="h", tz="UTC")
    return pd.DataFrame({"open": a[:, 0], "high": a[:, 1], "low": a[:, 2], "close": a[:, 3], "volume": 100.0}, index=idx)
def sigs(df, at, sl, tp):
    s = pd.DataFrame({"signal": False, "sl": np.nan, "tp": np.nan, "poc_tp": np.nan}, index=df.index)
    for a in at:
        s.iloc[a, 0] = True; s.iloc[a, 1] = sl; s.iloc[a, 2] = tp
    return s
def fundtab(times, rates):
    return pd.DataFrame({"funding_rate": rates, "funding_interval_hours": 8}, index=pd.DatetimeIndex(times, name="funding_time"))
EMPTY = fundtab([], [])

# A) expiry boundary: parent at bar 1, valid bars 2,3,4 (unfilled); new signal at bar 4 (close of the last valid bar) and bar 5
rows = [(100,101,99,100), (100,101,99,100.6)] + [(100.6,101,100.5,100.6)]*6
df = bars(rows); lim = np.full(len(df), np.nan); lim[[1, 4, 5]] = 100.0
r = run_path(df, sigs(df, [1, 4], 98.0, 110.0), B.with_(include_funding=False), EMPTY, limit_price=lim, valid_bars=3)
print("[A] signal at bar 4 (last valid bar of the bar-1 order):", r["orders"]["status"].to_dict())
r = run_path(df, sigs(df, [1, 5], 98.0, 110.0), B.with_(include_funding=False), EMPTY, limit_price=lim, valid_bars=3)
print("[A] signal at bar 5 (first bar after expiry):           ", r["orders"]["status"].to_dict())

# B) funding with a limit fill: fill in bar 8 (open 08:00 = settlement T1) -> T1 not charged; T2 = 16:00 charged
rows = [(100,101,99,100)]*7 + [(100,101,99,100.6)] + [(100.6,100.8,99.9,100.3)] + [(100.3,100.5,100.1,100.3)]*8 + [(100.3,111,100.2,110)]
df = bars(rows); lim = np.full(len(df), np.nan); lim[7] = 100.0
F = fundtab([T0 + 8*H, T0 + 16*H], [0.001, 0.002])
r = run_path(df, sigs(df, [7], 98.0, 110.0), B, F, limit_price=lim, valid_bars=3)
t = r["trades"].iloc[0]
exp = t["qty"] * df["open"].iloc[16] * 0.002
print(f"[B] limit fill bar {t['fill_bar']} (opens {df.index[int(t['fill_bar'])]}); funding charged {t['funding']:.6f}; "
      f"expected only T=16:00: {exp:.6f}; exit bar {t['exit_bar']} {t['exit_reason']}")
# C) market entry at the close of bar 7 = 08:00 settlement -> not charged; 16:00 charged (path engine and 1H engine)
lim2 = np.full(len(df), np.nan)
r = run_path(df, sigs(df, [7], 98.0, 110.0), B, F)
a = run_backtest(df, sigs(df, [7], 98.0, 110.0), B, F)
print(f"[C] market entry at {a.trades['entry_time'].iat[0]}: path funding {r['trades']['funding'].iat[0]:.6f}, "
      f"1H engine funding {a.trades['funding'].iat[0]:.6f}, n_funding {a.trades['n_funding'].iat[0]}")
# D) include_funding=False with a non-empty funding table
cfg = B.with_(include_funding=False)
r = run_path(df, sigs(df, [7], 98.0, 110.0), cfg, F)
a = run_backtest(df, sigs(df, [7], 98.0, 110.0), cfg, F)
print(f"[D] include_funding=False: path engine funding {r['trades']['funding'].iat[0]:.6f} vs 1H engine {a.trades['funding'].iat[0]:.6f}")
cfg = B.with_(venue="spot")
r = run_path(df, sigs(df, [7], 98.0, 110.0), cfg, F)
a = run_backtest(df, sigs(df, [7], 98.0, 110.0), cfg, F)
print(f"[D] venue='spot': path engine funding {r['trades']['funding'].iat[0]:.6f} vs 1H engine {a.trades['funding'].iat[0]:.6f}; "
      f"qty path {r['trades']['qty'].iat[0]:.6f} vs 1H {a.trades['qty'].iat[0]:.6f}")
cfg = B.with_(compounding=False)
# E) gap down through L and SL on the fill bar: open 97 < SL 98 < L 100
rows = [(100,101,99,100), (100,101,99,100.6), (97.0,99.0,96.0,98.5), (98.5,99,98,98.5)]
df = bars(rows); lim = np.full(len(df), np.nan); lim[1] = 100.0
r = run_path(df, sigs(df, [1], 98.0, 110.0), B.with_(include_funding=False), EMPTY, limit_price=lim, valid_bars=3)
t = r["trades"].iloc[0]
print(f"[E] gap-down fill bar: entry_px {t['entry_px']:.4f} (= L*(1+slip)), exit_px {t['exit_px']:.4f} (= SL*(1-slip)); "
      f"a real limit buy would fill near the open {rows[2][0]} and be stopped near the open -> engine is conservative")
# F) two settlements mapped to the same bar (data gap): 1H engine raises, path engine charges one
rows = [(100,101,99,100)]*3 + [(100,101,99,100.6)] + [(100.6,100.8,100.2,100.4)]*3
idx = list(pd.date_range(T0, periods=4, freq="h", tz="UTC")) + [T0 + 20*H, T0 + 21*H, T0 + 22*H]
df = pd.DataFrame(np.asarray(rows, float), columns=["open","high","low","close"], index=pd.DatetimeIndex(idx)); df["volume"] = 100.0
F2 = fundtab([T0 + 8*H, T0 + 16*H], [0.001, 0.001])
r = run_path(df, sigs(df, [3], 98.0, 110.0), B, F2)
try:
    run_backtest(df, sigs(df, [3], 98.0, 110.0), B, F2); msg = "no error"
except ValueError as e:
    msg = f"ValueError: {e}"
print(f"[F] gap of 16h covering two settlements: path funding {r['trades']['funding'].iat[0]:.6f} "
      f"(one settlement = {r['trades']['qty'].iat[0]*100.6*0.001:.6f}); 1H engine: {msg[:90]}")
# G) fill-bar TP not counted, next bar TP counted
rows = [(100,101,99,100), (100,101,99,100.6), (100.6,111,99.9,100.4), (100.4,111,100.2,110)]
df = bars(rows); lim = np.full(len(df), np.nan); lim[1] = 100.0
r = run_path(df, sigs(df, [1], 98.0, 110.0), B.with_(include_funding=False), EMPTY, limit_price=lim, valid_bars=3)
t = r["trades"].iloc[0]
print(f"[G] fill bar {t['fill_bar']} touches TP but exit bar = {t['exit_bar']} ({t['exit_reason']}) -> fill-bar TP not counted")
