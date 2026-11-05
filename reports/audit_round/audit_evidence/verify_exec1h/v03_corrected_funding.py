"""v03 (skeptic, exec1h): what a corrected engine_path funding rule would do (E-1), measured.
The repo file is NOT modified: its source text is patched in memory into a scratch module. The patch replaces the
index proxy (m >= charge_from) by the documented time rule (entry time < T), and charges every settlement mapped to a
sub-bar (no silent collapse). Runs: hand-made bars, and the IS baseline in MARKET mode only (no C1/limit run on IS)."""
import sys, types
sys.path.insert(0, "/home/user/ppvp_quant")
from pathlib import Path
import numpy as np, pandas as pd
from config import BASELINE_BACKTEST as B
from backtester import run_backtest
import ppvp_audit.engine_path as EP
from ppvp_audit.common import is_run, load_is_market

src = Path(EP.__file__).read_text()
rep = [
    ("@njit(cache=True)\ndef _sim(", "@njit(cache=False)\ndef _sim("),
    ("         through):\n", "         through, fund_t, open_ns):\n"),
    ("        if fi < nf and fund_min[fi] == m and qty > 0.0 and m >= charge_from:\n"
     "            f = qty * mo[m] * fund_rate[fi]\n"
     "            wallet -= f\n"
     "            fund += f\n",
     "        while fi < nf and fund_min[fi] == m:\n"
     "            if qty > 0.0 and entry_t < fund_t[fi]:\n"
     "                f = qty * mo[m] * fund_rate[fi]\n"
     "                wallet -= f\n"
     "                fund += f\n"
     "            fi += 1\n"),
    ("    charge_from = 0 ", "    entry_t = np.int64(0)\n    charge_from = 0 "),
    ("                        charge_from = m + 1 ", "                        entry_t = open_ns[m] + 1\n                        charge_from = m + 1 "),
    ("                        charge_from = m + 2 ", "                        entry_t = open_ns[m] + np.int64(3600 * 10**9)\n                        charge_from = m + 2 "),
    ("bool(limit_mode), fill_rule == \"through\")", "bool(limit_mode), fill_rule == \"through\", f_t[keep], ht)"),
]
for a, b in rep:
    assert src.count(a) == 1, a[:60]
    src = src.replace(a, b)
FX = types.ModuleType("engine_path_fixed"); FX.__file__ = "engine_path_fixed.py"
exec(compile(src, "engine_path_fixed.py", "exec"), FX.__dict__)

H = pd.Timedelta(hours=1); T0 = pd.Timestamp("2021-03-01 00:00", tz="UTC")
def frame(rows, hh):
    a = np.asarray(rows, float)
    df = pd.DataFrame({"open": a[:, 0], "high": a[:, 1], "low": a[:, 2], "close": a[:, 3]},
                      index=pd.DatetimeIndex([T0 + h * H for h in hh])); df["volume"] = 100.0; return df
def sigs(df, at, sl, tp):
    s = pd.DataFrame({"signal": False, "sl": np.nan, "tp": np.nan, "poc_tp": np.nan}, index=df.index)
    for x in at: s.iloc[x, :3] = [True, sl, tp]
    return s
def ftab(hh, rates):
    return pd.DataFrame({"funding_rate": rates, "funding_interval_hours": 8}, index=pd.DatetimeIndex([T0 + h * H for h in hh]))

cases = {
    "(a) gap after close entry, T in gap": (frame([(100, 101, 99, 100)] * 5 + [(100, 101, 99, 100.6)] + [(100.6, 100.8, 100.2, 100.4)] * 3,
                                                  (0, 1, 2, 3, 4, 5, 12, 13, 14)), [5], ftab([8], [0.001]), None),
    "(b) mid-hold gap with 2 settlements": (frame([(100, 101, 99, 100), (100, 101, 99, 100.6)] + [(100.6, 100.8, 100.2, 100.4)] * 9,
                                                  (0, 1, 2, 3, 4, 5, 6, 7, 20, 21, 22)), [1], ftab([8, 16], [0.001, 0.002]), None),
    "(d) contiguous control": (frame([(100, 101, 99, 100)] * 7 + [(100, 101, 99, 100.6)] + [(100.6, 100.8, 100.2, 100.4)] * 10,
                                     range(18)), [7], ftab([8, 16], [0.001, 0.002]), None),
}
lim = np.full(6, np.nan); lim[1] = 100.0
cases["(c) limit fill then gap, T in gap"] = (frame([(100, 101, 99, 100), (100, 101, 99, 100.6), (100.6, 100.8, 99.9, 100.3)]
                                                  + [(100.3, 100.5, 100.1, 100.3)] * 3, (0, 1, 2, 12, 13, 14)), [1], ftab([8], [0.001]), lim)
for tag, (df, at, F, lp) in cases.items():
    s = sigs(df, at, 98.0, 110.0)
    kw = {} if lp is None else {"limit_price": lp, "valid_bars": 3}
    cur = EP.run_path(df, s, B, F, **kw)["trades"]["funding"].iat[0]
    fix = FX.run_path(df, s, B, F, **kw)["trades"]["funding"].iat[0]
    print(f"{tag}: current {cur:.6f} | corrected {fix:.6f}")

print("== IS baseline, market mode (B_original only)")
out = is_run(); _, fund, _ = load_is_market()
d, sig, tr = out["df1h"], out["signals"], out["result"].trades
cur, fix = EP.run_path(d, sig, B, fund), FX.run_path(d, sig, B, fund)
print("corrected vs current equity identical:", bool(np.array_equal(cur["equity"].to_numpy(), fix["equity"].to_numpy())))
print("corrected vs backtester per-trade funding identical:", bool(np.array_equal(fix["trades"]["funding"].to_numpy(), tr["funding"].to_numpy())),
      "| total %.6f vs %.6f" % (fix["trades"]["funding"].sum(), tr["funding"].sum()))
for col in ("entry_bar", "exit_bar", "pnl"):
    print(f"  {col}: identical={bool(np.array_equal(fix['trades'][col].to_numpy(), tr[col].to_numpy()))}")
