from trend_hold_venue_study import config, cp3_windows
import pandas as pd

def test_approved_post_window_is_separate_and_keeps_frozen_config():
    for asset in ('BTC','ETH'):
        actual=cp3_windows.window('Post-IS',asset)
        assert actual.start==pd.Timestamp('2024-01-01',tz='UTC')
        assert actual.end==pd.Timestamp('2026-06-01',tz='UTC')
        assert actual.signal_origin==actual.start
        assert (actual.end-actual.start).days==882
        assert config.window('Post-IS',asset).end==pd.Timestamp('2026-11-01',tz='UTC')
        assert cp3_windows.window('IS-Common',asset)==config.window('IS-Common',asset)

def test_post_slices_keep_complete_calendar_months_and_continuity():
    post=cp3_windows.window('Post-IS')
    split=pd.Timestamp('2025-07-01',tz='UTC')
    assert len(pd.date_range(post.start,split,freq='MS',inclusive='left'))==18
    assert len(pd.date_range(split,post.end,freq='MS',inclusive='left'))==11
    assert (post.end-split).days==335
    assert post.end-pd.Timedelta(minutes=1)==pd.Timestamp('2026-05-31T23:59Z')
