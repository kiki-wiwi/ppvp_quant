# CP3 descriptive figure guide

All figures use main_17, canonical UM signals and continuous contract sizes.
This is the user-authorized one-shot descriptive evaluation with no automatic
promotion and no parameter selection. USD = USDT at the registered fixed
parity. These figures use completed account outputs, not new simulations.

1. **BTC Post-IS equity and effective 1m drawdown.** USD curves show daily
   closing cash-comparable equity. The shaded drawdown envelope is each day's
   minimum and maximum of the original effective 1m drawdown series; dots mark
   the exact original minute and magnitude of each account's maximum drawdown.
   Bad minute placeholders do not establish a peak or trough. This is not a
   drawdown computed from daily closes. V4 has additional collateral exposure
   (1 + gL), so its USD curve is not an equal-risk carrier ranking. The dashed
   2025-07-01 line divides VAL/OOS without resetting or liquidating the account.
   The shaded chop interval is [2024-03-01, 2024-11-01); its diagnosis inherits
   the February 29 valuation and does not invent a slice-end liquidation.
2. **Signed funding.** Curves are the exact daily accumulation of account
   funding ledger flows; positive means received. CM funding uses the existing
   event-time Spot conversion. IS-Common and Post-IS have independent initial
   accounts, so they are displayed separately with no connecting wealth path.
   Post funding remains cumulative through the VAL/OOS boundary. Three bars
   per carrier show the supplied IS, VAL and OOS funding totals, not differences
   between independently rescaled cumulative curves.
3. **V4 versus HODL coin accumulation.** C / C0 uses each independent window's
   actual initial conversion; C is total coin equity, including unrealized PnL.
   A ratio of 1 is zero coin growth. Diamonds locate the original maximum USD
   drawdown minute at its actual coin balance; crosses locate modeled pressure
   liquidation at its residual coin balance. The fixed 0.5% MMR and contract
   OHLC extreme are a simplified marked-price proxy, not the historical exchange
   liquidation engine. V4's additional collateral exposure is shown against
   HODL, not promoted as a same-risk substitute for V1/V2/V3.
4. **ETH same-target-exposure comparison.** V1/V2/V3 and their corresponding
   P controls are aligned within each independently initialized IS-Common or
   Post-IS panel. Solid lines are trend accounts and dashed lines are passive
   controls. Actual daily mean exposure may drift between rebalances. The
   Sharpe/MaxDD overview uses supplied registered daily Sharpe and effective
   1m MaxDD; undefined Sharpe remains NA, not zero. No 69-month continuous
   account is implied. Any separately reported (1+R_IS)*(1+R_Post)-1 is an
   independent-account compound reference, not continuous positions.

Partial-period valuations use the existing continuous account and preceding
close; no hypothetical fees or FX are added at VAL/OOS or chop boundaries.
Only the actual full-account endpoint includes terminal closing/conversion.
Coin endpoints retain final coin before final FX. Statistical references, if
reported elsewhere, are descriptive within one venue/window and do not create
a global multi-family conclusion or an automatic promotion decision.
