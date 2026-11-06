"""Synthetic-only tests for the pre-unlock section-11 implementation.

No data loader is called. Expected-invalid-input tests are intentionally absent
because the authorized workflow stops immediately on unexpected runtime errors.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from impulse_study import cp3_sensitivity as Q
from impulse_study import engine as E
from impulse_study import features as F
from impulse_study import synthetic as SY
from impulse_study import versions as VS

START = pd.Timestamp("2020-01-01", tz="UTC")
MAIN = VS.COSTS[VS.MAIN_COST]
GROSS = VS.COSTS["gross_0bp"]
RISK = VS.SIZINGS[VS.MAIN_SIZING]
B1 = VS.VERSIONS["B1_pure_long"]
B4 = VS.VERSIONS["B4_pure_1bar"]


def fixture_context(n=2000, *, arrays=None, bad=None, funding=None, gaps=(), start=400, box=None):
    arrays = SY.flat_bars(n, spread=0.0005) if arrays is None else arrays
    grid = SY.grid_from_arrays(START, *arrays, bad=bad)
    ctx = SY.synthetic_context(grid, grid.index[start], grid.index[n - 40], box_ok=box,
                               funding=funding, archive_gap_minutes=gaps)
    return ctx


def test_causal_median_exact_day_cut_lower_bound_and_contamination_inclusion():
    index = pd.date_range(START, periods=3 * 1440, freq="min").as_unit("ns")
    sigma = np.full(len(index), np.nan)
    sigma[0:15] = 1000.0                      # strictly outside the defined minute set
    sigma[15:20] = [1, 3, 8, np.nan, 4]      # prior-day nanmedian = 3.5, even count
    sigma[1440:1444] = [100, 200, np.nan, 2]
    frame = pd.DataFrame({"sigma7d": sigma, "contaminated": True}, index=index)
    ctx = SimpleNamespace(index=index, frame=frame)
    reference = Q.causal_sigma_reference(ctx)
    assert reference.iloc[:1440].isna().all()
    assert (reference.iloc[1440:2880] == 3.5).all()
    expected = np.nanmedian(sigma[15:2880])
    assert (reference.iloc[2880:] == expected).all()
    frame_later = frame.copy()
    frame_later.loc[index[2880]:, "sigma7d"] = 9e9
    later = Q.causal_sigma_reference(SimpleNamespace(index=index, frame=frame_later))
    pd.testing.assert_series_equal(reference, later)


@pytest.mark.parametrize("unit,tz", [("us", "UTC"), ("ns", "UTC"), ("us", "Pacific/Auckland")])
def test_causal_median_normalizes_units_and_utc_day_boundaries(unit, tz):
    index = pd.date_range(START, periods=3 * 1440, freq="min").as_unit(unit).tz_convert(tz)
    sigma = np.full(len(index), np.nan)
    sigma[15:19] = [1.0, 3.0, 4.0, 8.0]
    sigma[1440] = 100.0
    frame = pd.DataFrame({"sigma7d": sigma}, index=index)
    reference = Q.causal_sigma_reference(SimpleNamespace(index=index, frame=frame))
    assert reference.index.unit == "ns" and str(reference.index.tz) == "UTC"
    assert reference.iloc[:1440].isna().all()
    assert (reference.iloc[1440:2880] == 3.5).all()
    assert (reference.iloc[2880:] == 4.0).all()


def test_new_contexts_and_runs_leave_every_original_field_bitwise_unchanged():
    grid = SY.random_grid(12 * 1440, start="2020-01-01", seed=31, impulse_rate=0.002)
    ctx = SY.synthetic_context(grid, grid.index[9 * 1440], grid.index[-40], box_ok=np.ones(len(grid), bool))
    original = Q.context_fingerprint(ctx)
    causal = Q.causal_z_context(ctx)
    full = Q.full_period_context(ctx)
    box = Q.fresh_context(ctx, box_ok=np.zeros(len(grid), bool))
    assert all(other is not ctx for other in (causal, full, box))
    assert causal.cond is not ctx.cond and causal.C is ctx.C
    assert full.C is not ctx.C and box.C is not ctx.C
    assert full.window[0] and not ctx.window[0]
    assert not box.box_ok.any() and ctx.box_ok.all()
    for other, key in ((causal, Q.Z_VERSIONS[1]), (full, "B1_pure_long"), (box, "A1_box_opt")):
        vin = F.build_version_inputs(other, VS.VERSIONS[key])
        assert vin.context is other
        E.run_portfolio(other, vin, MAIN, RISK)
    assert Q.context_fingerprint(ctx) == original
    expected = Q.causal_sigma_reference(ctx)
    for day in ctx.index.floor("D").unique()[7:]:
        selected = ctx.frame.loc[(ctx.index >= START + pd.Timedelta(minutes=15)) & (ctx.index < day), "sigma7d"]
        assert expected.loc[day] == np.nanmedian(selected.to_numpy())


@pytest.mark.parametrize("cost", [GROSS, MAIN])
def test_reference_kernel_matches_frozen_engine_and_independent_equity(cost):
    grid = SY.random_grid(3000, seed=32, impulse_rate=0.001)
    funding = pd.Series([0.0001, -0.0002], index=[grid.index[960], grid.index[1440]])
    ctx = SY.synthetic_context(grid, grid.index[400], grid.index[-40], funding=funding)
    signals = {i: (1 if (i // 73) % 2 else -1,
                   ctx.C[i] - (1 if (i // 73) % 2 else -1) * 150.0) for i in range(500, 2800, 73)}
    vin = SY.manual_inputs(ctx, B1, signals)
    frozen = E.run_portfolio(ctx, vin, cost, RISK)
    alternate = Q.run_path_variant(ctx, vin, cost, RISK, variant="reference")
    pd.testing.assert_frame_equal(frozen.trades, alternate.trades)
    pd.testing.assert_series_equal(frozen.equity, alternate.equity)
    equity, daily = Q.reconstruct_equity(ctx, frozen.trades, include_funding=cost.include_funding)
    np.testing.assert_allclose(equity, frozen.equity, rtol=1e-12, atol=1e-7, equal_nan=True)
    np.testing.assert_allclose(daily, frozen.daily_equity, rtol=1e-12, atol=1e-7)


@pytest.mark.parametrize("continues,fixed,conflict", [(False, False, True), (True, False, False), (True, True, True)])
def test_first_boundary_conflict_recomputes_missing_continuation(continues, fixed, conflict):
    arrays = SY.flat_bars(700, spread=0.0005)
    O, H, L, C, V = arrays
    i, x = 400, 415
    L[x] = 9900.0
    if continues:
        V[i + 1:x + 1] = 30.0
        C[x] = 10004.0
    ctx = fixture_context(700, arrays=arrays)
    vin = SY.manual_inputs(ctx, B4 if fixed else B1, {i: (1, 9950.0)})
    main = E.run_portfolio(ctx, vin, MAIN, RISK)
    assert main.trades.iloc[0]["reason"] == E.STOP
    assert main.trades.iloc[0]["cont"] == -1
    classification = Q.classify_path_ambiguities(ctx, vin, main.trades)
    diag = classification["trades"].iloc[0]
    assert bool(diag["stop_close_conflict"]) is conflict
    assert np.isfinite(diag["diagnostic_v1"]) and np.isfinite(diag["diagnostic_k1"])
    alternate = Q.run_path_variant(ctx, vin, MAIN, RISK, variant="optimistic")
    assert alternate.trades.iloc[0]["reason"] == (E.TIME1 if conflict else E.STOP)
    assert alternate.trades.iloc[0]["exit_ref"] == (C[x] if conflict else 9950.0)


def test_second_boundary_conflict_unconditional_and_archive_gap_listing():
    arrays = SY.flat_bars(700, spread=0.0005)
    O, H, L, C, V = arrays
    V[401:416] = 30.0
    C[415] = 10004.0
    L[430] = 9900.0
    ctx = fixture_context(700, arrays=arrays, gaps=[START + pd.Timedelta(minutes=430)])
    vin = SY.manual_inputs(ctx, B1, {400: (1, 9950.0)})
    main = E.run_portfolio(ctx, vin, MAIN, RISK)
    tr = main.trades.iloc[0]
    assert tr["reason"] == E.STOP and tr["exit_bar"] == 430 and tr["cont"] == 1
    classified = Q.classify_path_ambiguities(ctx, vin, main.trades)
    assert classified["counts"]["stop_close_conflict"]["n"] == 1
    assert len(classified["archive_gap_trades"]) == 1
    optimistic = Q.run_path_variant(ctx, vin, MAIN, RISK, variant="optimistic")
    assert optimistic.trades.iloc[0]["reason"] == E.TIME2


@pytest.mark.parametrize("kind,reason,flag", [
    ("entry_gap", E.GAP_ENTRY, "entry_open_gap"),
    ("holding_gap", E.GAP_STOP, "holding_open_gap"),
    ("entry_intrabar", E.STOP, "entry_minute_intrabar_stop"),
])
def test_other_path_categories_unchanged_by_optimism(kind, reason, flag):
    arrays = SY.flat_bars(700, spread=0.0005)
    O, H, L, C, V = arrays
    if kind == "entry_gap":
        O[401], L[401] = 9900, 9900
    elif kind == "holding_gap":
        O[405], L[405] = 9900, 9900
    else:
        L[401] = 9900
    ctx = fixture_context(700, arrays=arrays)
    vin = SY.manual_inputs(ctx, B1, {400: (1, 9950)})
    main = E.run_portfolio(ctx, vin, MAIN, RISK)
    optimistic = Q.run_path_variant(ctx, vin, MAIN, RISK, variant="optimistic")
    pd.testing.assert_frame_equal(main.trades, optimistic.trades)
    assert main.trades.iloc[0]["reason"] == reason
    classified = Q.classify_path_ambiguities(ctx, vin, main.trades)
    assert classified["counts"][flag] == {"n": 1, "share": 1.0}
    assert classified["counts"]["stop_close_conflict"]["n"] == 0


def test_natural_bar_includes_preentry_ohlcv_but_not_preentry_pnl():
    arrays = SY.flat_bars(700, spread=0.0005)
    O, H, L, C, V = arrays
    H[390] = 10100.0
    V[390:401] = 100.0
    C[404] = 10004.0
    ctx = fixture_context(700, arrays=arrays)
    vin = SY.manual_inputs(ctx, B1, {400: (1, 9900.0)})
    natural = Q.run_path_variant(ctx, vin, MAIN, RISK, variant="natural")
    tr = natural.trades.iloc[0]
    assert tr["exit_bar"] == 404 and tr["reason"] == E.TIME1
    assert tr["minutes_held"] == 4
    assert tr["v1"] == (11 * 100 + 4 * 10) / 150
    assert tr["k1"] == (2 * 10004 - 10100 - 9995) / (10100 - 9995)
    assert tr["mfe_px"] == 5 and tr["mae_px"] == -5


@pytest.mark.parametrize("entry,duration", [(390, 30), (404, 16)])
def test_natural_two_bar_holding_extremes(entry, duration):
    arrays = SY.flat_bars(700, spread=0.0005)
    O, H, L, C, V = arrays
    V[390:405] = 30.0
    C[404] = 10004.0
    ctx = fixture_context(700, arrays=arrays, start=350)
    vin = SY.manual_inputs(ctx, B1, {entry - 1: (1, 9900.0)})
    natural = Q.run_path_variant(ctx, vin, MAIN, RISK, variant="natural")
    tr = natural.trades.iloc[0]
    assert tr["reason"] == E.TIME2 and tr["exit_bar"] == 419
    assert tr["minutes_held"] == duration and tr["v1"] == 3.0


@pytest.mark.parametrize("trigger,halt", [(1424, False), (1430, False), (1430, True)])
def test_independent_daily_equity_midnight_and_halt_funding(trigger, halt):
    arrays = SY.flat_bars(2000, spread=0.0005)
    bad = np.zeros(2000, bool)
    if halt:
        bad[1435:1442] = True
    funding = pd.Series([0.0003], index=[START + pd.Timedelta(days=1)])
    ctx = fixture_context(arrays=arrays, bad=bad, funding=funding)
    vin = SY.manual_inputs(ctx, B4, {trigger: (1, 9900.0)})
    result = E.run_portfolio(ctx, vin, MAIN, RISK)
    equity, daily = Q.reconstruct_equity(ctx, result.trades, include_funding=True)
    np.testing.assert_allclose(daily, result.daily_equity, rtol=1e-12)
    np.testing.assert_allclose(equity, result.equity, rtol=1e-12, equal_nan=True)
    tr = result.trades.iloc[0]
    assert tr["n_funding"] == 1 and tr["funding"] < 0
    if trigger == 1424:
        assert tr["exit_time_idx"] == 1440 and equity.iloc[1439] == pytest.approx(result.meta["final_equity"])
    if halt:
        assert tr["reason"] == E.HALT and tr["exit_bar"] == 1442


def test_defect_union_filter_preserves_retained_trade_quantities_and_rebuilds():
    funding = pd.Series([0.0001], index=[START + pd.Timedelta(minutes=505)])
    bad = np.zeros(2000, bool)
    bad[705:708] = True
    gaps = [START + pd.Timedelta(minutes=505), START + pd.Timedelta(minutes=590)]
    ctx = fixture_context(bad=bad, funding=funding, gaps=gaps)
    vin = SY.manual_inputs(ctx, B4, {500: (1, 9900), 600: (1, 9900), 700: (1, 9900), 800: (1, 9900)})
    result = E.run_portfolio(ctx, vin, MAIN, RISK)
    filtered = Q.defect_filter(ctx, result, MAIN)
    assert len(filtered["excluded"]) == 3
    assert filtered["retained_trigger_indices"].tolist() == [800]
    retained = filtered["result"]
    assert retained.trades.iloc[0]["qty"] == result.trades.iloc[-1]["qty"]
    assert retained.equity.iloc[-1] == pytest.approx(VS.INITIAL_CAPITAL + result.trades.iloc[-1]["net_pnl"])


def test_all_reconciliation_identities_with_built_version_inputs():
    grid = SY.random_grid(6000, seed=44, impulse_rate=0.01)
    ctx = SY.synthetic_context(grid, grid.index[400], grid.index[-40])
    vin = F.build_version_inputs(ctx, B1)
    result = E.run_portfolio(ctx, vin, MAIN, RISK)
    assert len(result.trades) > 0
    report = Q.reconcile_portfolio(ctx, vin, result, MAIN)
    assert report["all_identities_passed"]
    assert report["trigger_minutes"] == report["signals"] + report["exclusive_exclusions"]
    assert report["signals"] == report["trades"] + report["holding_or_cooldown_skips"]


def test_box_hook_uses_ten_variants_four_versions_and_direction_specific_masks(monkeypatch):
    from impulse_study import daily as DY
    from tools import run_impulse_cp1 as CP1
    ctx = fixture_context(700)
    before = Q.context_fingerprint(ctx)
    calls = []

    def states(*args, **kwargs):
        assert len(kwargs["variants"]) == 8
        return {"sentinel": True}

    def mask(variant, cond, direction, ns, close, c15, c5, states):
        calls.append((variant, direction))
        assert states["sentinel"]
        assert np.isnan(c15[:15]).all() and np.isnan(c5[:5]).all()
        return np.ones(len(close), bool)

    monkeypatch.setattr(DY, "build_box_states", states)
    monkeypatch.setattr(CP1, "box_mask", mask)
    keys = []
    for (variant, name), alternate in Q.box_contexts(ctx, object(), object(), object()):
        keys.append((variant, name))
        assert alternate is not ctx and alternate.C is not ctx.C
        assert alternate.box_ok.all()
        vin = F.build_version_inputs(alternate, VS.VERSIONS[name])
        E.run_portfolio(alternate, vin, MAIN, RISK)
        assert calls[-1][1] == ("dn" if name == "A4_box_flush" else "up")
    assert len(keys) == 40 and len(set(v for v, _ in keys)) == 10
    assert all(v != VS.BOX_STATE_VARIANT for v, _ in keys)
    assert Q.context_fingerprint(ctx) == before


def test_cp3_entrypoint_runs_exactly_causal_z_and_full_b_main_configuration(monkeypatch):
    ctx = fixture_context(700)
    before = Q.context_fingerprint(ctx)
    calls = []

    def portfolio(other, vin, cost, sizing):
        calls.append((other, vin.spec.name, cost.name, sizing.name))
        assert vin.context is other
        return object()

    monkeypatch.setattr(E, "run_portfolio", portfolio)
    result = Q.run_cp3_sensitivities(ctx)
    assert [row[1] for row in calls] == list(Q.Z_VERSIONS) + list(Q.B_VERSIONS)
    assert all(row[2:] == (VS.MAIN_COST, VS.MAIN_SIZING) for row in calls)
    assert all(row[0] is result["causal_context"] for row in calls[:2])
    assert all(row[0] is result["full_period_context"] for row in calls[2:])
    assert result["appendix_enters_gates"] is False
    assert Q.context_fingerprint(ctx) == before


def test_optimistic_midnight_close_adds_boundary_funding_and_reconciles():
    arrays = SY.flat_bars(2000, spread=0.0005)
    arrays[2][1439] = 9900.0
    funding = pd.Series([0.0003], index=[START + pd.Timedelta(days=1)])
    ctx = fixture_context(arrays=arrays, funding=funding)
    vin = SY.manual_inputs(ctx, B4, {1424: (1, 9950.0)})
    main = E.run_portfolio(ctx, vin, MAIN, RISK)
    alternate = Q.run_path_variant(ctx, vin, MAIN, RISK, variant="optimistic")
    assert main.trades.iloc[0]["n_funding"] == 0
    assert alternate.trades.iloc[0]["n_funding"] == 1
    assert alternate.trades.iloc[0]["reason"] == E.TIME1
    _, daily = Q.reconstruct_equity(ctx, alternate.trades, include_funding=True)
    np.testing.assert_allclose(daily, alternate.daily_equity, rtol=1e-12)
    classified = Q.classify_path_ambiguities(ctx, vin, main.trades)
    assert classified["counts"]["economic_result_changes"]["n"] == 1


@pytest.mark.parametrize("variant", ["optimistic", "natural"])
def test_variants_keep_halt_priority_at_first_boundary(variant):
    bad = np.zeros(2000, bool)
    bad[1435:1442] = True
    funding = pd.Series([0.0003], index=[START + pd.Timedelta(days=1)])
    ctx = fixture_context(bad=bad, funding=funding)
    vin = SY.manual_inputs(ctx, B4, {1430: (1, 9900.0)})
    result = Q.run_path_variant(ctx, vin, MAIN, RISK, variant=variant)
    tr = result.trades.iloc[0]
    assert tr["reason"] == E.HALT and tr["exit_bar"] == 1442 and tr["cont"] == -1
    assert tr["n_funding"] == 1
    _, daily = Q.reconstruct_equity(ctx, result.trades, include_funding=True)
    np.testing.assert_allclose(daily, result.daily_equity, rtol=1e-12)


def test_cp4_orchestration_covers_sections_and_frozen_seed_offsets(monkeypatch):
    ctx = fixture_context(700)
    vin = F.build_version_inputs(ctx, B1)
    empty = E.run_portfolio(ctx, vin, MAIN, RISK)
    all_results = {(name, cost, size): empty for name in VS.VERSIONS for cost in VS.COSTS for size in VS.SIZINGS}
    cp3 = {"causal_context": ctx, "causal_results": {key: empty for key in Q.Z_VERSIONS},
           "full_period_context": ctx, "full_period_results": {key: empty for key in Q.B_VERSIONS}}
    monkeypatch.setattr(Q, "_run_versions", lambda other, names, **kwargs: {name: empty for name in names})
    monkeypatch.setattr(Q, "box_contexts", lambda *args: iter([(("main", "A1_box_opt"), ctx)]))
    payloads = []

    def hook(payload):
        payloads.append(payload.copy())
        return {"called": True}

    result = Q.run_cp4_sensitivities(ctx, all_results, cp3_results=cp3,
                                   box_data=(object(), object(), object()), statistics_hook=hook)
    assert {payload["section"] for payload in payloads} == set(range(1, 10))
    for payload in payloads:
        k = payload["section"]
        assert payload["seed_shift"] == (10000 * k if 2 <= k <= 7 else 0)
        assert payload["enters_primary_gates"] is False
    assert len(result["costs_and_sizing"]["results"]) == 80
    assert result["b_full_period_appendix"]["matched_controls"] is False
    assert result["daily_clusters_and_stationary_bootstrap"]["cluster"] == "day"
    assert "harsh_results" in result["exclude_data_defects"]
