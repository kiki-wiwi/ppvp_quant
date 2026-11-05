"""v02 (skeptic, exec1h): hand-made bars only (no market data, no variant performance). Re-tests E-1, E-2, E-3 against
an independent hand implementation of the documented rules (engine_path.py docstring lines 10, 13-16)."""
import sys
sys.path.insert(0, "/home/user/ppvp_quant")
import numpy as np, pandas as pd
from config import BASELINE_BACKTEST as B
from backtester import run_backtest
from ppvp_audit.engine_path import run_path

H = pd.Timedelta(hours=1)
T0 = pd.Timestamp("2021-03-01 00:00", tz="UTC")


def frame(rows, times):
    a = np.asarray(rows, float)
    df = pd.DataFrame({"open": a[:, 0], "high": a[:, 1], "low": a[:, 2], "close": a[:, 3]}, index=pd.DatetimeIndex(times))
    df["volume"] = 100.0
    return df


def hours(*hh):
    return [T0 + h * H for h in hh]


def sigs(df, at, sl, tp, poc=np.nan):
    s = pd.DataFrame({"signal": False, "sl": np.nan, "tp": np.nan, "poc_tp": np.nan}, index=df.index)
    for a in at:
        s.iloc[a, :4] = [True, sl, tp, poc]
    return s


def ftab(times, rates):
    return pd.DataFrame({"funding_rate": rates, "funding_interval_hours": 8}, index=pd.DatetimeIndex(times, name="funding_time"))


def doc_rule_funding(df, F, qty, entry_time, first_held_bar, last_held_bar):
    """Documented rule: T -> first sub-bar j with open >= T; charge qty*open[j]*rate if entry_time < T and the position is
    open at the start of j (first_held_bar <= j <= last_held_bar). Two settlements on one sub-bar are both charged."""
    o_ns = df.index.as_unit("ns").asi8
    tot, n = 0.0, 0
    for T, r in zip(F.index, F["funding_rate"]):
        if not (df.index[0] < T <= df.index[-1] + H):
            continue
        j = int(np.searchsorted(o_ns, T.value, side="left"))
        if j < len(df) and entry_time < T and first_held_bar <= j <= last_held_bar:
            tot += qty * df["open"].iat[j] * r
            n += 1
    return tot, n


def show(tag, df, F, s, cfg=B, **kw):
    p = run_path(df, s, cfg, F, **kw)
    t = p["trades"].iloc[0]
    ent = df.index[int(t["fill_bar"])] + (H if kw.get("limit_price") is None else 0.5 * H)
    last = int(t["exit_bar"]) if t["exit_bar"] >= 0 else len(df) - 1
    exp, nexp = doc_rule_funding(df, F, t["qty"], ent, int(t["fill_bar"]) + 1, last)
    try:
        a = run_backtest(df, s, cfg, F)
        bt = f"backtester {a.trades['funding'].iat[0]:.6f} (n={a.trades['n_funding'].iat[0]})"
    except ValueError as e:
        bt = f"backtester raises: {str(e)[:70]}"
    print(f"{tag}: engine_path {t['funding']:.6f} | documented rule {exp:.6f} (n={nexp}) | {bt}")


print("== E-1 funding boundary")
# (a) close entry at 06:00 (bar 05:00), next bar opens 12:00, settlement 08:00 inside the gap, held through it
rows = [(100, 101, 99, 100)] * 5 + [(100, 101, 99, 100.6)] + [(100.6, 100.8, 100.2, 100.4)] * 3
df = frame(rows, hours(0, 1, 2, 3, 4, 5, 12, 13, 14))
show("(a) gap after close entry, T inside gap  ", df, ftab(hours(8), [0.001]), sigs(df, [5], 98.0, 110.0))
# (a2) close entry at 04:00 (bar 03:00), next bar opens exactly 08:00 = T
rows = [(100, 101, 99, 100)] * 3 + [(100, 101, 99, 100.6)] + [(100.6, 100.8, 100.2, 100.4)] * 3
df = frame(rows, hours(0, 1, 2, 3, 8, 9, 10))
show("(a2) gap after close entry, next open = T", df, ftab(hours(8), [0.001]), sigs(df, [3], 98.0, 110.0))
# (b) gap in the middle of the holding period covering 08:00 and 16:00 (entry at 02:00)
rows = [(100, 101, 99, 100), (100, 101, 99, 100.6)] + [(100.6, 100.8, 100.2, 100.4)] * 9
df = frame(rows, hours(0, 1, 2, 3, 4, 5, 6, 7, 20, 21, 22))
show("(b) mid-hold gap covering 2 settlements  ", df, ftab(hours(8, 16), [0.001, 0.002]), sigs(df, [1], 98.0, 110.0))
# (c) limit fill in bar 2 (02:00-03:00), then gap to 12:00 covering T=08:00: the index rule should agree with the doc rule
rows = [(100, 101, 99, 100), (100, 101, 99, 100.6), (100.6, 100.8, 99.9, 100.3)] + [(100.3, 100.5, 100.1, 100.3)] * 3
df = frame(rows, hours(0, 1, 2, 12, 13, 14))
lim = np.full(len(df), np.nan); lim[1] = 100.0
show("(c) limit fill then gap, T inside gap     ", df, ftab(hours(8), [0.001]), sigs(df, [1], 98.0, 110.0),
     cfg=B, limit_price=lim, valid_bars=3)
# (c2) gap right before the limit-fill bar: T maps to the fill bar itself -> not charged by either rule
rows = [(100, 101, 99, 100), (100, 101, 99, 100.6), (100.6, 100.8, 99.9, 100.3)] + [(100.3, 100.5, 100.1, 100.3)] * 3
df = frame(rows, hours(0, 1, 9, 10, 11, 12))
show("(c2) gap before limit-fill bar           ", df, ftab(hours(8), [0.001]), sigs(df, [1], 98.0, 110.0),
     cfg=B, limit_price=lim, valid_bars=3)
# (d) contiguous control: close entry exactly at T=08:00, then T=16:00 held
rows = [(100, 101, 99, 100)] * 7 + [(100, 101, 99, 100.6)] + [(100.6, 100.8, 100.2, 100.4)] * 10
df = frame(rows, pd.date_range(T0, periods=len(rows), freq="h", tz="UTC"))
show("(d) contiguous, entry at T, next T held   ", df, ftab(hours(8, 16), [0.001, 0.002]), sigs(df, [7], 98.0, 110.0))

print("== E-2 unsupported config fields (contiguous bars)")
F = ftab(hours(8, 16), [0.001, 0.002])
s1 = sigs(df, [7], 98.0, 110.0)


def both(tag, cfg, s=s1, d=df, f=F):
    p = run_path(d, s, cfg, f)["trades"]
    a = run_backtest(d, s, cfg, f).trades
    cols = ["entry_bar", "exit_bar", "exit_reason", "qty", "notional", "funding", "pnl"]
    print(f"{tag}\n  engine_path: {p[[c for c in cols if c in p]].round(6).to_dict('records')}\n"
          f"  backtester : {a[cols].round(6).to_dict('records')}")
    return p, a


both("(e) include_funding=False", B.with_(include_funding=False))
p, a = both("(f) venue='spot'", B.with_(venue="spot"))
eq0 = B.initial_capital
for name, t in (("engine_path", p.iloc[0]), ("backtester", a.iloc[0])):
    print(f"  {name}: spot cash after buy = {eq0 - t['notional'] - t['notional'] * B.with_(venue='spot').taker_fee:.4f}")
# (g) compounding=False: two trades, the first one loses
rows = [(100, 101, 99, 100), (100, 101, 99, 100.6), (100.6, 100.8, 97.0, 97.5), (97.5, 98, 97, 97.6),
        (97.6, 98, 97.2, 97.8), (97.8, 120, 97.5, 119)]
dg = frame(rows, pd.date_range(T0, periods=len(rows), freq="h", tz="UTC"))
sg = sigs(dg, [1], 98.0, 110.0); sg.iloc[3, :4] = [True, 95.0, 110.0, np.nan]
both("(g) compounding=False (2nd trade sized on initial capital?)", B.with_(include_funding=False, compounding=False), sg, dg, F.iloc[:0])
# (h) sizing='risk'
both("(h) sizing='risk' (1% risk)", B.with_(include_funding=False, sizing="risk"), sg, dg, F.iloc[:0])
# (i) time_stop_bars=2
both("(i) time_stop_bars=2", B.with_(include_funding=False, time_stop_bars=2))
# (j) exit_on_breakdown
sb = s1.copy(); sb["VAL"] = 99.5; sb["k1_atr"] = 0.1; sb["brk_bodyvol"] = False
db = df.copy(); db.iloc[10] = [100.4, 100.5, 99.0, 99.1, 100.0]; sb.iloc[10, sb.columns.get_loc("brk_bodyvol")] = True
both("(j) exit_on_breakdown=True (bar 10 closes 99.1 < VAL-k1ATR 99.4)", B.with_(include_funding=False, exit_on_breakdown=True), sb, db)
# (k) funding table with a missing settlement and with a NaN rate (backtester validates coverage; engine_path does not)
for tag, f in (("missing 16:00", ftab(hours(8), [0.001])), ("NaN at 16:00", ftab(hours(8, 16), [0.001, np.nan]))):
    p = run_path(df, s1, B, f)
    try:
        run_backtest(df, s1, B, f); msg = "no error"
    except ValueError as e:
        msg = f"raises: {str(e)[:60]}"
    print(f"(k) {tag}: engine_path funding {p['trades']['funding'].iat[0]} final equity {p['equity'].iat[-1]} | backtester {msg}")

print("== E-3 pending-order expiry vs a new signal")
rows = [(100, 101, 99, 100), (100, 101, 99, 100.6)] + [(100.6, 101, 100.5, 100.6)] * 7
de = frame(rows, pd.date_range(T0, periods=len(rows), freq="h", tz="UTC"))
for second in (2, 3, 4, 5):
    lim = np.full(len(de), np.nan); lim[[1, second]] = 100.0
    r = run_path(de, sigs(de, [1, second], 98.0, 110.0), B.with_(include_funding=False), F.iloc[:0], limit_price=lim, valid_bars=3)
    st = r["orders"]["status"]
    print(f"  parent bar 1 (valid bars 2,3,4); 2nd signal at bar {second}: {[(int(de.index.get_loc(k)), v) for k, v in st.items()]}")
# analogue for positions: SL inside bar 3, new signal at the close of bar 3 -> accepted by both engines
rows = [(100, 101, 99, 100), (100, 101, 99, 100.6), (100.6, 101, 100.2, 100.5), (100.5, 100.6, 97.5, 98.6),
        (98.6, 99, 98.4, 98.8), (98.8, 120, 98.7, 119)]
dp = frame(rows, pd.date_range(T0, periods=len(rows), freq="h", tz="UTC"))
sp = sigs(dp, [1], 98.0, 110.0); sp.iloc[3, :4] = [True, 97.0, 110.0, np.nan]
p = run_path(dp, sp, B.with_(include_funding=False), F.iloc[:0])
a = run_backtest(dp, sp, B.with_(include_funding=False))
print("  position analogue: trades engine_path", p["trades"][["entry_bar", "exit_bar", "exit_reason"]].to_dict("records"),
      "| backtester", a.trades[["entry_bar", "exit_bar", "exit_reason"]].to_dict("records"))
