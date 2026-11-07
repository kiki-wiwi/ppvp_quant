# CP2 BTC IS implementation registration

Base: deadf01a4570386e14bbbec41d6e8eadd3345cac. Additions only; 32 CP1 files and 142 historical protection entries unchanged. Both physical locks remain False in this commit.

## Approved input provenance
- Reuse R5 UM IS inputs and its frozen load-entry time predicates. The existing UM Parquet byte SHA differs from the historical writer output, but user-approved bounded content-only validation exactly matched the frozen content SHA256. No cache or manifest was rewritten.
- 48 Spot 1m and 41 CM 1m official monthly ZIPs: raw bytes checked against official CHECKSUM; no real Kline parsing before the BTC unlock.
- 18 CM Vision funding archives contain 1593 records, each Decimal-exact against saved official dapi responses. Only the 54 specifically approved month-end UTC 00/08/16 settlements use dapi supplements. Continuous overlap: 1647; Common window: 3651. August launch metadata and all 41 raw API pages/parameters/timestamps/hashes are retained. Rate tolerance remains <=1e-8; no new gaps permitted.
- Prior stops, read-only evidence and the passing pre-patch synthetic run are preserved byte-for-byte under validation_history.

## Fixed execution and accounting
- IS-Full: four versions x three costs = 12 accounts. Six UM outputs must match archived R5 daily equity, daily returns and terminal equity exactly.
- IS-Common: eight versions x three costs = 24 accounts; plus three main-cost whole-contract diagnostics.
- IS-CM-SelfSignal: three main-cost accounts plus daily switch concordance and budget comparison. Total fixed accounts: 42; no extra reference simulation.
- All frozen CP1 formulas, seeds, diagnostics and V1-V6 criteria remain unchanged. Zero-floor and Spot cash-roundoff adjustments are explicit bridge components.
- Export account ledgers, dynamic reconciliation, five within-venue paired statistics families, cost/coin/basis diagnostics and four descriptive figures. CP3/post-IS/ETH remain unauthorized.

## Validation and release sequence
175 synthetic tests passed under the fixed pandas 3.0.5 environment and market/network guard. Only the explicitly authorized historical R5 lock node was deselected; details are in synthetic_validation.json.
After this implementation commit is pushed, create a separate commit changing only BTC_IS_UNLOCKED=False to True. POST_IS_AND_ETH_UNLOCKED remains False. Formal run requires a clean worktree, exact registered hashes and an absent output directory. No retry or repair after a formal error. The third commit only archives CP2 outputs and the requested phase log.
