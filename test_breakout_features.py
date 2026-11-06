"""Synthetic-only checks of the preregistered channel/VAH causal features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def context(frame):
    """Use the public immutable input boundary, with explicit synthetic provenance."""
    from breakout_study.scope import make_context
    frame.attrs["provenance"] = "synthetic"
    return make_context(frame)


def flat(minutes, start="2020-03-01", price=100.0):
    idx = pd.date_range(start, periods=minutes, freq="min", tz="UTC").as_unit("ns")
    return pd.DataFrame({"open": price, "high": price + 1, "low": price - 1,
                         "close": price, "volume": 1.0, "bad": False,
                         "archive_gap": False}, index=idx)


def warm_frame(close=111.0):
    """36 days of sparse valid 4H history, followed by 21 full hours."""
    f = flat(36 * 1440 + 60, "2020-02-01")
    f.loc[:, "bad"] = True
    f.loc[:, ["high", "low"]] = 100.0
    for k in range(0, len(f), 240):
        f.iloc[k, f.columns.get_indexer(["high", "low", "volume", "bad"])] = [110.0, 90.0, 0.0, False]
        if k + 1 < len(f):
            f.iloc[k + 1, f.columns.get_loc("bad")] = False
    h = f.index[-60]
    f.loc[h - pd.Timedelta(hours=20):, "bad"] = False
    f.loc[h:, "volume"] = 2.0
    f.loc[f.index[-1], ["high", "close"]] = [max(close, 100.0), close]
    return f, h


def features(f):
    from breakout_study.features import build_features
    return build_features(context(f))


def signal(f, version="P_channel", built=None):
    from breakout_study.features import build_signals
    return build_signals(context(f), version, built)


def test_closed_natural_buckets_exclude_bad_and_unfinished_suffix():
    f = flat(4 * 60 + 37, "2020-03-01 00:17")
    f.iloc[0, f.columns.get_indexer(["high", "low", "close", "volume", "bad"])] = [999, 1, 999, 999, True]
    out = features(f)
    assert list(out.hours.index.hour) == [0, 1, 2, 3]
    first = out.hours.iloc[0]
    assert first.n_valid == 42
    assert first.high == 101 and first.low == 99 and first.volume == 42
    assert list(out.references.index) == [pd.Timestamp("2020-03-01 04:00", tz="UTC")]
    assert out.hours.index.unit == "ns" and str(out.hours.index.tz) == "UTC"


def test_atr_initializes_on_14_valid_hours_and_keeps_state_through_empty_hour():
    f = flat(17 * 60)
    f.loc[f.index[5 * 60:6 * 60], "bad"] = True
    # Partial valid hours count as one ATR observation; the bad price is excluded.
    f.loc[f.index[2 * 60:3 * 60 - 1], "bad"] = True
    f.loc[f.index[15 * 60:16 * 60], ["open", "high", "low", "close"]] = [109, 110, 108, 109]
    f.loc[f.index[16 * 60:], "bad"] = True
    out = features(f).hours
    assert out.atr.iloc[:14].isna().all()
    assert out.atr.iloc[14] == 2.0
    assert out.atr.iloc[15] == pytest.approx(36 / 14)  # TR=10 after previous valid close=100.
    assert out.atr.iloc[16] == out.atr.iloc[15]


def test_reference_levels_use_all_valid_minutes_and_readonly_profile_definition():
    f, h = warm_frame()
    out = features(f)
    r = out.references.loc[h]
    assert r.window_valid and r.U == 110 and r.L == 90 and r.C == 100
    assert r.width == 0.2 and r.compression_q == 0.2 and r.compressed
    # All positive volume is a one-price candle at 100. The 50 bins are 0.4 wide.
    assert r.POC == pytest.approx(100.2)
    assert r.VAL == pytest.approx(100.0)
    assert r.VAH == pytest.approx(100.4)


def test_quantile_needs_180_contiguous_past_widths_and_excludes_current_width():
    f = flat(210 * 240, "2020-02-01")
    f.loc[:, "bad"] = True
    for k in range(210):
        f.iloc[k * 240, f.columns.get_indexer(["open", "high", "low", "close", "volume", "bad"])] = [100, 101 + k, 100, 100, 1, False]
    out = features(f).references
    assert out.compression_q.iloc[-2] != out.compression_q.iloc[-2]  # Only 179 prior widths.
    assert out.width.iloc[-1] == 2.10
    assert out.compression_q.iloc[-1] == pytest.approx(0.7475)  # .30 ... 2.09, linear 25%.
    assert not out.compressed.iloc[-1]


def test_current_reference_does_not_rewrite_signal_at_four_hour_close():
    f, h = warm_frame()
    # h+3 is the last hour before a new reference at h+4. Its spike must not enter B.
    extra = flat(3 * 60, str((h + pd.Timedelta(hours=1)).tz_localize(None)))
    extra.loc[:, ["high", "low"]] = 100.0
    f = pd.concat([f, extra])
    f.loc[h + pd.Timedelta(hours=3):, "volume"] = 2.0
    f.iloc[-1, f.columns.get_indexer(["high", "close"])] = [112, 112]
    row = signal(f).loc[h + pd.Timedelta(hours=4)]
    assert row.reference_time == h
    assert row.boundary == 110 and row.eref == 112 and row.crossover


def test_same_current_boundary_is_used_for_both_sides_of_the_cross():
    f, h = warm_frame(close=111)
    built = features(f)
    # A controlled precomputed feature snapshot makes the moving-boundary error observable.
    # At h, previous close=109 exceeded the OLD boundary 108, but is below CURRENT 110.
    built.hours.loc[h - pd.Timedelta(hours=1), "close"] = 109
    built.references.loc[h - pd.Timedelta(hours=4), "U"] = 108
    row = signal(f, built=built).loc[h + pd.Timedelta(hours=1)]
    assert row.boundary == 110 and row.previous_close == 109 and row.eligible
    # Conversely, old boundary 120 cannot manufacture a cross above CURRENT 110.
    built.hours.loc[h - pd.Timedelta(hours=1), "close"] = 111
    built.references.loc[h - pd.Timedelta(hours=4), "U"] = 120
    row = signal(f, built=built).loc[h + pd.Timedelta(hours=1)]
    assert not row.crossover and not row.eligible


def test_variants_share_all_eligibility_inputs_except_the_boundary_cross():
    f, h = warm_frame(close=101)
    built = features(f)
    p = signal(f, "P_channel", built)
    v = signal(f, "V_vah", built)
    shared = [c for c in p if c not in {"boundary", "crossover", "eligible", "skip_reason", "version"}]
    pd.testing.assert_frame_equal(p[shared], v[shared])
    entry = h + pd.Timedelta(hours=1)
    assert not p.loc[entry, "eligible"] and v.loc[entry, "eligible"]
    assert v.loc[entry, "baseline_volume"] == 60
    assert v.loc[entry, "volume_ok"]
    assert v.loc[entry, "sl"] == v.loc[entry, "eref"] - 2 * v.loc[entry, "atr"]


def test_decision_survives_absent_or_bad_next_minute_and_ignores_its_price():
    f, h = warm_frame()
    before = signal(f)
    entry = h + pd.Timedelta(hours=1)
    assert before.loc[entry, "eligible"]  # No entry minute is present yet.
    next_minute = flat(1, str(entry.tz_localize(None)), price=1_000_000.0)
    next_minute.loc[:, "bad"] = True
    after = signal(pd.concat([f, next_minute]))
    pd.testing.assert_frame_equal(before, after)


@pytest.mark.parametrize("where", ["current", "previous"])
def test_signal_and_all_twenty_volume_hours_need_sixty_valid_minutes(where):
    f, h = warm_frame()
    t = h if where == "current" else h - pd.Timedelta(hours=20)
    f.loc[t, "bad"] = True
    row = signal(f).loc[h + pd.Timedelta(hours=1)]
    assert not row.eligible and not row.volume_ok
    assert row.skip_reason in {"incomplete_signal_hour", "incomplete_volume_history"}


def test_empty_4h_bucket_invalidates_reference_and_later_fixed_width_history():
    f, h = warm_frame()
    gap = f.index[60 * 240]
    f.loc[gap:gap + pd.Timedelta(hours=4) - pd.Timedelta(minutes=1), "bad"] = True
    out = features(f).references
    assert not out.loc[gap + pd.Timedelta(hours=4), "window_valid"]
    assert out.loc[h, "window_valid"]
    assert np.isnan(out.loc[h, "compression_q"])  # Cannot replace invalid widths with older observations.
    assert not signal(f, built=features(f)).loc[h + pd.Timedelta(hours=1), "eligible"]


def test_zero_volume_and_zero_range_windows_are_invalid():
    f = flat(31 * 240)
    f.loc[:, "volume"] = 0.0
    assert not features(f).references.window_valid.any()
    f.loc[:, ["high", "low", "open", "close"]] = 100.0
    f.loc[:, "volume"] = 1.0
    assert not features(f).references.window_valid.any()


def test_archive_gap_is_audit_flag_not_an_exclusion():
    f, h = warm_frame()
    plain = signal(f).loc[h + pd.Timedelta(hours=1)]
    f.loc[h - pd.Timedelta(hours=2), "archive_gap"] = True
    flagged = signal(f).loc[h + pd.Timedelta(hours=1)]
    assert plain.eligible and flagged.eligible
    assert flagged.signal_lookback_has_archive_gap and not plain.signal_lookback_has_archive_gap
    assert plain.eref == flagged.eref and plain.atr == flagged.atr


def test_prefix_causality_including_partial_final_hour_and_timezone_resolution():
    f, _ = warm_frame()
    full = features(f)
    cut = len(f) - 23
    prefix = f.iloc[:cut].copy()
    prefix.index = prefix.index.tz_convert("Asia/Shanghai").as_unit("us")
    truncated = features(prefix)
    pd.testing.assert_frame_equal(truncated.hours, full.hours.loc[truncated.hours.index])
    pd.testing.assert_frame_equal(truncated.references, full.references.loc[truncated.references.index])
    s = signal(prefix)
    pd.testing.assert_frame_equal(s, signal(f).loc[s.index])


def test_start_and_last_entry_limits_are_decision_time_constraints():
    f, h = warm_frame()
    for new_h, permitted in [(pd.Timestamp("2020-02-29 23:00", tz="UTC"), False),
                              (pd.Timestamp("2020-03-01 00:00", tz="UTC"), True),
                              (pd.Timestamp("2023-12-28 22:00", tz="UTC"), True),
                              (pd.Timestamp("2023-12-28 23:00", tz="UTC"), False)]:
        delta = new_h - h
        shifted = f.copy()
        shifted.index += delta
        row = signal(shifted).loc[new_h + pd.Timedelta(hours=1)]
        assert bool(row.eligible) is permitted


def test_short_prefix_has_no_closed_hours_and_invalid_version_is_explicit():
    f = flat(1)
    assert features(f).hours.empty and features(f).references.empty and signal(f).empty
    with pytest.raises(ValueError, match="version"):
        signal(f, "unregistered")


def test_volume_baseline_excludes_current_hour_when_prior_volumes_vary():
    f, h = warm_frame()
    for j in range(20):
        start = h - pd.Timedelta(hours=20 - j)
        f.loc[start:start + pd.Timedelta(minutes=59), "volume"] = float(j + 1)
    f.loc[h:, "volume"] = 100.0
    row = signal(f).loc[h + pd.Timedelta(hours=1)]
    # Prior volumes are 60, 120, ... 1200; central values 600 and 660 average to 630.
    assert row.baseline_volume == 630.0
    assert row.volume == 6000.0 and row.volume_ok and row.eligible
