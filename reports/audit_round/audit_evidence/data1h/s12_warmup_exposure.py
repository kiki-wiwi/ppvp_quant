# Which of the 120 IS raw signals (109 trades) have ledger regime features computed partly on 2019 spot closes.
# Daily label D is available at D+1 00:00; at decision time t (signal close) the newest usable label is D_last.
import pandas as pd
op = pd.read_csv("/home/user/ppvp_quant/reports/audit_round/signal_opportunities_B_original.csv")
t = pd.to_datetime(op["signal_time"], utc=True)
d_last = (t - pd.Timedelta(days=1)).dt.floor("D")            # newest label with D+1 00:00 <= t
first_perp = pd.Timestamp("2020-01-01", tz="UTC")
spot_in = lambda days: (d_last - pd.Timedelta(days=days - 1)) < first_perp
filled = op["status"].eq("filled")
for name, days in (("vol30d (30 closes)", 30), ("SMA200 (200 closes)", 200), ("sma200_up (SMA200 shifted 20)", 220),
                   ("dd365 / vol30d_pct (365 closes)", 365)):
    m = spot_in(days)
    print("%-32s raw signals %3d of %d | trades %3d of %d" % (name, int(m.sum()), len(op), int((m & filled).sum()), int(filled.sum())))
i0 = t.idxmin()
n_spot = int(((first_perp - (d_last[i0] - pd.Timedelta(days=199))) / pd.Timedelta(days=1)))
print("first signal", t[i0], "D_last", d_last[i0].date(), "-> SMA200 window = %d spot + %d perp closes" % (n_spot, 200 - n_spot))
