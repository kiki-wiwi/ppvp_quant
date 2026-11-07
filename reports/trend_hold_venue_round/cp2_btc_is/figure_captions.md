# CP2 BTC IS figure guide

1. **Equity and coin balances.** Main-cost cash-comparable daily equity for each
   strategy and its within-venue passive control. The Full and Common windows
   use separate initial accounts. V4 carries additional collateral exposure
   (`1 + gL`) and is compared separately with HODL; its coin index is not a USD
   risk-adjusted ranking. Terminal cash-equity points include the actual closing
   trade and Spot conversion, while the coin index retains final coin balances.
2. **Cost steps.** Returns use each account's actual endpoint. In gross_0,
   funding is disabled; gross-to-main differences therefore combine funding and
   transaction friction, not friction alone. V4/HODL coin growth is kept in a
   separate panel from equal-target-exposure cash returns.
3. **Funding, friction and gates.** Funding is signed and translated at each
   event's Spot reference for CM. Friction is positive fees minus signed
   slippage and conversion PnL. MaxDD is the positive loss fraction from valid
   minute equity. The V1–V6 matrix is an IS diagnostic; Holm controls only two
   hypotheses within one venue/window, not a global cross-venue family.
4. **Signal and rounding diagnostics.** The 38-month panel compares canonical
   UM and native CM decision schedules, not independently initialized account
   PnL. Whole-contract differences are whole minus continuous face, from the
   same Common-window start and main cost. These are diagnostics, not parameter
   selection or additional promotion gates.
