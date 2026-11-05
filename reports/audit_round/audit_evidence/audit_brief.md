# Brief: implementation audit of the PPVP baseline (BTC IS only, 1H data only)

Repo: /home/user/ppvp_quant (branch claude/sharp-darwin-k34sz1, HEAD 4949052). Python research code for a long-only
BTCUSDT (Binance USD-M perp) strategy: 4H pivot-point volume profile (PPVP) boxes; on 1H, two pullbacks ("tests")
into a support band around the box's VAL, then a bullish confirmation candle -> long. Initial stop below the lower of
the two pullback lows; initial target = VAH. Costs: taker 5bp, maker 2bp, 2bp slippage per fill, real funding.
1x notional of equity per trade, compounding, one position at a time.

## HARD research policy (violating it invalidates your work)
- Only BTC IS data: [2020-01-01 00:00 UTC, 2024-01-01 00:00 UTC). Never run, view or compute anything on data at or
  after 2024-01-01 (VAL/OOS), and never touch ETHUSDT data or results.
- Only the locally cached 1H data (and the 4H aggregated from it, funding, pre-2020 spot daily closes). NO data finer
  than 1H exists or may be used, and you must NOT download anything from the network (the user explicitly forbids
  bulk-downloading 1m archives). Intrabar ordering is handled by the pre-declared worst-case rule (stop-loss first;
  a POC/TP touch inside the same bar as the stop does not count) - count and report ambiguous cases instead of
  resolving them.
- Use only these entry points for data and backtests:
  `from ppvp_audit.common import is_run, load_is_market` (they truncate at IS_END before the engine and assert it).
  `pipeline.run(...)` without `end=IS_END` is FORBIDDEN. The parquet files under data/processed contain rows after
  2024; if you read one directly, truncate to < 2024-01-01 immediately and never print/compute on later rows.
- Do NOT run tools/run_phase2.py, test_lookahead.py, test_attribution.py or tools/mutation_test.py (they compute
  VAL/OOS/ETH). Do not modify any file in the repo. Write scripts and outputs only under your scratch folder
  /tmp/claude-0/-home-user-ppvp-quant/04a66f10-1e5e-5960-9a9f-ffb5b694e43a/scratchpad/audit_round/<your-label>/
  and set NUMBA_CACHE_DIR to a folder inside it. Run python with cwd=/home/user/ppvp_quant so imports work
  (e.g. `cd /home/user/ppvp_quant && NUMBA_CACHE_DIR=<dir>/nb python3 <dir>/script.py`).
- Do not run or report the performance of any strategy variant other than the baseline (the experiment versions are
  pre-registered separately and must not be previewed).

## Where things are
- data_loader.py (archive download/cleaning, resample_ohlcv, funding parsing), pipeline.py (entry, caches),
  ppvp_engine.py (4H aggregation, pivots, volume profile, boxes, box->1H mapping), strategy_state_machine.py
  (explicit 1H state machine; long docstring lists every rule and its interpretation), backtester.py (fills, costs,
  funding, MTM equity), metrics.py (daily equity, Sharpe), attribution.py (Phase 2 analysis: trade_frame, strict
  MFE/MAE, random-entry nulls random_entry_null/_bracket, thinning, bucket stats), config.py (all parameters; the
  baseline is the dataclass defaults).
- Phase logs: docs/phase_logs/PHASE0_data_engineering.md, PHASE1_engine.md, PHASE2_baseline.md.
- This round so far (ppvp_audit/ and reports/audit_round/): figure_checks.json (five-figure reproduction),
  baseline_manifest.json, signal_opportunities_B_original.csv (120 raw IS signals with status, profile ids at each
  test, departure diagnostics), trades_B_original.csv (109 trades, cost decomposition, ambiguity flags),
  trade_reconciliation_B_original.csv, random_baseline_review.json; ppvp_audit/engine_path.py (a sub-bar engine run on
  the 1H bars themselves that reproduces backtester.py bit for bit).
- Facts already verified this round: IS 1H has 35,064 bars (complete, all Binance, no OKX fill, no zero-volume bar);
  all 8,766 IS 4H bars have n_bars = 4; baseline IS Sharpe -1.174, 109 trades (65 SL, 44 TP); entry close vs next open
  differs by at most 0.9 bp; 4 of 109 exit bars are ambiguous under 1H data (worst case applied).

## Classify every finding as exactly one of
- implementation_error: the code does not do what its own documented rule says, or produces an outcome that is
  impossible/unexecutable, provable with independent evidence (a script or a hand computation on 1H data).
- definition_question: the rule is ambiguous or under-specified (the user wants these registered, NOT fixed by
  picking whichever variant performs better).
- design_choice: a deliberate rule that differs from the user's stated core rules or from common practice
  (report, do not call it a bug).
- labeling_or_documentation: a number/label/doc statement that misdescribes what the code computes.
- data_issue: a data problem (missing/partial/imputed bars, source mixing, timestamp errors).
Changing box definitions, entry filters, stop, target or sizing is a strategy experiment, never a bug fix.
For each implementation_error, state what B_audited should do and estimate its effect on the 109 IS trades if you can
compute it (old vs new), but do not tune anything.
