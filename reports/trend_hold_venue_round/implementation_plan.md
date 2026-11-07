# Cross-venue CP1 implementation plan

Goal: implement approved venue accounting and synthetic validation without opening market data.
Base: 3846971ad9014043cc0371a96c693924b7ca1e61, verified published.
Spec: approved_spec.md plus boundary_addendum.md. User-approved zero-floor adjustment included.
Environment: inherited fixed R5 Python/pandas/numpy/pyarrow versions; no dependency changes.

## Constraints
Only new files. No old source, manifest or result edits. Two independent data locks False.
No real reads, downloads or performance calculations. Synthetic sources explicitly tagged.
User first-failure stop overrides a deliberate failing-test/fix cycle: write tests first, run complete synthetic verification with -x only after source integration. Preserve any first failure, do not repair and retry.
No additional optimization variants. Future real CP2/CP3 separately approved.

## Interfaces
- config.Window(name,start,end,signal_origin); COSTS reuses frozen R5 immutable Cost values.
- scope.Context(index,O,H,L,C,V,bad,archive_gap,funding_rate,source='synthetic',asset='BTC',venue='UM'), immutable UTC ns minute grid.
- features.build_features(ctx, signal_origin, end=None) produces R5-compatible columns; account slices retain canonical budget.
- features.references(ctx, days) supplies each venue's own prior valid close.
- linear.run_linear(ctx, features, version, cost, window) -> R5-compatible Result; retain operation order, accept V1_USDM/P_USDM.
- portfolios.run_spot(ctx,features,version,cost,window); run_coin(cm,spot,features,version,cost,window,whole_contracts=False).
- Result exposes minute,daily_equity,daily_returns,orders,decisions,funding,metadata plus attribution and pressure_events for new venues.
- accounting pure signed-N inverse PnL, initial conversion/estimate, trade adjustment, funding, cash clipping, pressure line/trigger and dynamic attribution.
- statistics.analyze(t,p,harsh_t,harsh_p,features,t_order_count) uses full input calendar, shared frozen draws and two-member family. IS diagnostics split at frozen 2022-01-01. Post-IS statistical references descriptive, no promotion.
- reporting.metrics, period slicing preserve continuous equity; no real plotting/run entry opened in CP1.
- registration verifies baseline sources and approved docs. tools/check_trend_hold_venue_cp1.py is source-only; no real execution CLI is included in CP1.

## Tasks
- [x] Preserve authoritative spec and addendum; register baseline hashes.
- [x] Implement independent scopes and causal canonical features; synthetic prefix and scope tests.
- [x] Implement inverse accounting primitives and hand-computable A-D tests.
- [x] Generalize independent linear wrapper without modifying frozen modules; exact minute R5 regression.
- [x] Implement Spot and CM portfolios, separate event ledgers, zero-floor credit, terminal cash/coin valuations; synthetic path tests.
- [x] Implement calendar-aware descriptive statistics/metrics with inherited seeds; synthetic pairing and zero-equity tests.
- [x] Integrate under explicit market/network I/O guard; one focused source review, then required synthetic tests with first-failure stop.
- [x] Register new-file hashes, validation log and phase log; commit/push independent branch; stop before CP2.

## Review focus
First-day estimated versus actual conversion; pre-funding target freeze; pressure open gap before extrema; realized wallet versus inverse unrealized PnL; residual coin mark after terminal pressure; no double liquidation/fees; exact dynamic attribution including floor credit; spot nonnegative cash; unmodified frozen dependencies.

## Implementation choices
The main canonical ETH budget starts on 2020-08-31 to supply the specified first inherited Monday budget. Self-signal primary comparator uses a separately initialized account on its 38-month diagnostic window, while retaining primary canonical budget history.
Both synthetic and real account operations require explicit venue, asset and provenance. Synthetic future/ETH fixtures are allowed; real ones require the separate stage lock.

## CP1 validation closure
129 passed; 1 historical R5 stage-lock node deselected by explicit user approval.
78 new tests (including both independent locks and A–E) and 51 applicable R5 regressions passed.
The sole exclusion and reason are registered in synthetic_validation.json and synthetic_pytest.txt.
The first stopped attempt is retained byte-for-byte under validation_history/attempt_1_stopped/.
No strategy/test source changed during the approved resume. Both new physical locks remain False.
The implementation and evidence are prepared as a single CP1 commit; remote publication is confirmed in the delivery response.

## Registered implementation details and remaining stage boundary
- Exact float64 trade fills and friction expressions are retained; extended-precision accumulation supports the prescribed attribution tolerances, without a hidden rounding-profit item.
- At a funding/decision boundary: freeze target, mark old position, settle old funding, check old opening pressure, execute a surviving net order, then check new opening/intrabar pressure. Normal minutes still check opening before adverse extreme.
- Spot price at an intraminute CM pressure event uses the latest available Spot opening reference (last valid reference for bad Spot minutes); subsequent close marks normally. This does not infer an unavailable intraminute Spot price.
- Whole-contract diagnostic targets round signed face toward zero to the specified BTC/ETH contract size; primary targets retain continuous face value.
- Ratios with a zero actual initial denominator in a post-liquidation slice are NA; the stipulated zero-to-zero daily returns remain zero.
- The CP1 delivery is the blind engine, statistics/metrics, source verification and synthetic suite. Real loaders, native venue input/funding/cache registration, batch export and chart entrypoints remain CP2 preparation work; CP1 does not provide a runnable real batch.
- No real BTC IS, Post-IS or ETH data were read. Synthetic future/ETH calendars test isolation and continuous slicing only. CP2 and CP3 each still require explicit stage approval and their independent unlock commits.
