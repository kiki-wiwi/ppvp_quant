# CP3 implementation plan and preregistration

> Execute the approved specification in the existing isolated worktree using executing-plans; independent implementation tasks may be delegated. User first-error-stop and exact three-commit rules override generic RED/GREEN and frequent-commit skill defaults.

Goal: one user-authorized descriptive BTC Post-IS and ETH cross-venue evaluation, with no parameter search or automatic promotion.
Architecture: new CP3 adapters and orchestration wrap the frozen CP1 features, engines, accounting, metrics and statistics. Existing CP2 code/results and historical protected files remain byte-identical. Raw input preparation is separate from Context construction and strategy imports.
Tech stack: fixed Python 3.14.4 / pandas 3.0.5 / NumPy 2.5.4 / PyArrow 26.0.1 environment.
Spec: ../approved_spec.md and ../boundary_addendum.md, plus the user's CP3 authorization, 87-account enumeration, data-format amendments, three exact test exclusions, and dedicated-volume authorization.
Baseline: f0c21acdd9320d87272d3c8cd46fe95b8d72ec5d.

## Global constraints
- BTC_IS_UNLOCKED remains True. POST_IS_AND_ETH_UNLOCKED remains False until the separate one-line unlock commit.
- Preserve CP1 core bytes (normalizing only approved physical lock lines when hashing), 142 historical protections, CP2 effective registrations, and all 407 CP2 delivery files plus their manifest.
- Frozen SMA60, rolling volatility20, target20%, cap1x, Monday updates, costs gross_0/main_17/harsh_34, cash/pressure/floor accounting, PCG64 Webb99999 seed2026111001 and month10000 seed2026111002.
- Before unlock: raw ZIP and dapi download, checksums, funding reconciliation, structural OHLCV parsing/grid validation and hashes only. No new-real features, signals, budgets, exposure summaries, accounts or returns.
- Test files are written before implementation; no deliberate red execution. A genuine first validation/run failure is recorded and stops the attempt. No retries, repair-and-resume, partial reruns, seeds/config reductions, or existing-file edits.
- Exactly 3 approved historical test node exclusions; new tests cover actual BTC=True/POST=False baseline, blocked Post/ETH, and monkeypatched authorized transition.
- Data/reports/LFS/scratch/temp write to UUID d60bc8c8-11ef-4540-8d13-e2327feac6a1 mounted at /mnt/ppvp-backtests. Check actual device and >=40 GiB available before acquisition and before formal run. The original absolute paths are persistent bind mounts, not symlink substitutions.
- Formal run starts on the clean unlock commit and creates cp3_post_is_and_eth with exist_ok=False. Result commit adds only new CP3 outputs and the final phase report. Never put its own commit SHA into the report.

## Fixed matrix and directory order (87)
Each main matrix loops versions in frozen config.VERSIONS order, then gross_0/main_17/harsh_34. Each diagnostic loops V3_CM_DeltaMatched, P_CM_DeltaMatched, V4_CM_CoinAccum at main_17.
1. BTC Post-IS main24; whole-contract3 (100USD); CM-native self-signal3.
2. ETH IS-Common main24.
3. ETH Post-IS main24.
4. ETH IS-CM-SelfSignal3; ETH Post-IS CM-self3; ETH Post-IS whole-contract3 (10USD).
Job IDs include asset, window, signal_source, version, cost and whole-contract flag. The obsolete heading 51 is superseded by the user's final explicit total87 and its itemized components.
Post-IS accounts run continuously [2024-01-01,2026-06-01); no reset, liquidation or FX conversion at 2025-07-01. VAL/OOS are exact output slices. BTC warmup is [2023-11-01,2024-01-01); ETH UM/Spot native coverage starts2020-07-01, CM starts its first valid native minute no earlier than2020-08-18.

## Tasks and interfaces
### Input preparation: cp3_archives.py, cp3_funding.py, cp3_data.py; new synthetic tests
- [ ] Fixed official archive catalog: BTC2024-01..2026-05 UM/Spot/CM; ETH2020-07..2026-05 UM/Spot and CM2020-08..2026-05. Funding UM native archives; CM SchemeA Vision available months plus dapi raw pages.
- [ ] Pure ms/us, header/no-header parsers; every authorized abnormal close_time recorded; OHLCV unchanged, decision availability open+1min; deterministic CP2 bad/archive_gap grid. Reject invalid/order/out-of-range data and any full UTC day without effective minutes. Explicit official daily supplements only under the approved missing/partial-month rule, with complete source registration.
- [ ] CM Decimal overlap <=1e-8, source-tag early and approved month-end fills, preserve pre-native funding in canonical source and exclude only from Context.
- [ ] prepare_inputs(root) runs with BTC=True/POST=False and persists source hashes, preflight evidence and new caches in reports/.../cp3_registration and data/processed_1m/venue_cp3_*; no overwrite of existing source/cache.
- [ ] load_inputs(root) checks require_cp3() before I/O and returns {asset:{UM:Context,SPOT:Context,CM:Context}}. BTC contexts start2023-11-01; ETH starts native history. Cache registration binds both bytes and canonical content hashes. Existing BTC warmup uses registered CP2 inputs with bounded reads.

### Runtime and analysis: cp3_runtime.py, cp3_analysis.py; new synthetic tests
- [ ] jobs() returns exact87; execute(contexts,out) validates real/synthetic scope and fixed calendars, computes frozen features only after authorization, invokes frozen engines once/job, and exports unchanged account formats.
- [ ] Reuse frozen export_account and paired analyze primitives; add independent orchestration. Each account uses fresh immutable Contexts and copied version schedules.
- [ ] Generate full/VAL/OOS segment metrics with actual preceding equity/coin and effective-minute drawdowns. Zero initial denominator stays NA; frozen inference results do not replace actual segment wealth ratios.
- [ ] 21 descriptive main_17 paired families: BTC Post/VAL/OOS x3 carriers and ETH IS-Common/Post/VAL/OOS x3. Use corresponding harsh returns, fixed draws, two-hypothesis within-family Holm; no global family claim. Frozen Post gates stay NA; ETH IS gates use frozen2022 split.
- [ ] IS/Post independent compound reference is (1+R_IS_Common)*(1+R_Post)-1 for matching asset/version/cost, explicitly not69-month continuous positions. BTC IS comes from frozen CP2 only.
- [ ] Export reconciliation, funding/friction/basis differences, clipping, pressure events, diagnostic concordance and whole-contract differences. Complete-window coin growth and segment coin growth remain distinct.

### Charts and final analysis exports: cp3_plotting.py; new synthetic tests
- [ ] Four mandated figures: BTC Post equity/minuteDD with split/chop annotation; BTC+ETH funding curves and IS/VAL/OOS bars; V4/HODL coin curves with pressure/extremeDD marks; ETH IS/Common+Post exposures and Sharpe/DD comparison.
- [ ] Provide chart captions that distinguish independent accounts, descriptive references, extra V4 collateral exposure and marked-price proxies.
- [ ] 2024-03-01..2024-11-01 chop diagnosis inherits Feb29 valuation, does not invent slice-end liquidation. Register exact churn aggregation and funding-offset denominator before unlock; use existing account attribution rather than counterfactual new simulations.

### Protection and execution entry: cp3_control.py, cp3_storage.py, tools/run_trend_hold_venue_cp3.py; new synthetic tests
- [ ] Bind current baseline, every effective CP1/CP2/historical file and CP2 delivered output; normalize only the exact two authorized scope lines for CP1 digest verification. Input registrations and new implementation hashes cannot self-reference.
- [ ] Enforce actual mounted UUID/device/free-space and preflight feature-import barrier; archive earlier stop/storage evidence with hashes.
- [ ] Validate exact POST-only unlock diff and clean worktree, fixed environment, fixed inputs and fresh output directory. Log first failure and retain partial outputs.
- [ ] Fixed validation command: pytest -p trend_hold_venue_study.testing_guard -x with only the 3 exclusions below; explicitly select applicable existing/new synthetic suites so no historical real-data tests are collected.
- [ ] Commit/push implementation+input registration; commit/push single-line POST unlock; run once; verify delivery bytes and LFS objects; commit/push results and final report; stop.

## Exact approved exclusions
- tests/test_trend_hold_engine.py::test_rejects_missing_or_invalid_daily_inputs_and_real_context_before_access — old R5 lock-False assertion.
- tests/test_trend_venue_scope_features.py::test_two_real_locks_and_provenance — old CP1 double-False assertion, incompatible with current BTC=True.
- tests/test_trend_venue_cp2_runtime.py::test_runtime_rejects_post_is_eth_and_real_locked_before_feature_work — old CP2 pre-unlock entry expectation, incompatible with current BTC=True.

## Review focus
Timestamp units/close-time exception inventory; true first-native CM minute and pre-launch funding; no unregistered funding fills; exact continuous slice denominators; ETH contract size; finite/zero-equity slices; no signal mutation/shared state; matrix IDs/order/count; actual storage routing; strict one-shot run and artifact-only publication.

## Registered reporting definitions before any new-real input preparation
- Chop episodes are maximal intervals of the shared canonical UM trend state g, clipped to [2024-03-01,2024-11-01). The same state grouping is applied to passive/HODL controls; its switch count is not a claim that those accounts traded. Actual order count is reported separately. Boundary-clipped episodes are identified and never assigned a hypothetical exit.
- Negative price-attribution episode loss is sum(max(0,-episode observed price PnL)). CM price PnL is holding_spot_usd + contract_mtm_spot_usd; UM reconstructs observed price PnL from exact equity changes and recorded funding/fees/slippage; Spot uses its existing price attribution. ON-only and CM contract-only negative losses are disclosed separately.
- Actual trading/FX friction is positive fee expense minus signed slippage and conversion PnL. Funding-offset ratio is signed observed funding receipts divided by (negative episode price loss + actual friction), only for a positive denominator; otherwise NA. This descriptive ledger decomposition is neither a realized-stop-loss estimate nor a no-funding counterfactual account. Switch-day friction is a separate observed subset.
- New-real runs must render all four final figures. Synthetic-only execution has no real CP2 archive and explicitly skips these archive-dependent final figures; independent plot tests supply fully synthetic curves.
- Formal runtime receives the full frozen CP2 delivery hash map. Necessary existing BTC IS comparison/daily/minute/attribution files are hash-checked and read for archived plot series; no BTC IS account, feature or inference run is repeated.

## Approved revision and corrected launch boundary (2026-11-08)
- Post-IS ends2026-06-01 exclusive,882days29months;VAL18months/OOS11months335days;May31 23:59 terminal liquidation;Jul1 2025 carries Jun30equity. CP1config andscope remain unchanged, apart from laterPOST-only unlock. Use newcp3_windows.window adapter.
- ETH CM native firstminute2020-08-18 07:01; firstfunding08:00. Midnight was a prelaunch expected slot, not an actualsource record. Preserve all actualrecords; Context excludes0 actualpre-native records.
- ETHCM Aug28–31 2023 uses4 verifiedofficial dailyarchives. ETHSPOT exact1424missing minutes across10registeredranges use inheritedbad/archive_gap andcausalprior-onlyvaluation; preserve21extraDec21monthlybars.
- All rawsourcesandAPIrequests end<=May2026. OctoberREST supplementation andJuly–Octoberfunding replacement disabled. Reuse previouslyauthenticated bytes viahardlinks orreadonlycopies, withsourceprovenancebound inreuse_sources.json; never overwrite historicalrawrecords.
- Historicalstopanddiagnostic summaries archived under validation_history; full rawdiagnostic evidence remains ondedicatedvolume withitsoriginalhashmanifest.
