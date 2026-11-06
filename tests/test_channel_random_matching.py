"""Synthetic exact conditional randomization; no realized performance is supplied."""
import hashlib
import importlib
import struct

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


def opportunities():
    rows = []
    for hour, bucket, days, parents in [(1,0,2,{1,2}), (2,1,1,set()),
                                        (3,2,5,{1,5}), (4,5,3,{2})]:
        for day in range(1,days+1):
            e = pd.Timestamp(f'2020-03-{day:02d} {hour:02d}:00', tz='UTC')
            rows.append(dict(signal_time=e-pd.Timedelta(hours=1), entry_time=e,
                             month='2020-03', utc_hour=hour, vol_bin=bucket,
                             eligible=True, is_parent=day in parents, eref=100., atr=1., sl=98.))
    e = pd.Timestamp('2020-03-01 05:00', tz='UTC')
    rows.append(dict(signal_time=e-pd.Timedelta(hours=1), entry_time=e,
                     month='2020-03', utc_hour=5, vol_bin=0, eligible=False,
                     is_parent=False, eref=100., atr=1., sl=98.))
    result = pd.DataFrame(rows).set_index('entry_time', drop=False).sort_index()
    result.index = result.index.as_unit('ns')
    return result


def times(*items):
    return pd.DatetimeIndex([pd.Timestamp('2020-03-'+item, tz='UTC') for item in items]).as_unit('ns')


def test_exact_spawn_choices_skip_rng_for_empty_and_saturated_strata():
    m = importlib.import_module('channel_random_study.matching')
    result = m.build_schedules(opportunities())
    assert len(result.schedules) == 200
    assert result.schedules[0].index.equals(times('01 01:00','01 03:00','02 01:00','03 03:00','03 04:00'))
    assert result.schedules[1].index.equals(times('01 01:00','01 04:00','02 01:00','02 03:00','04 03:00'))
    assert result.schedules[199].index.equals(times('01 01:00','02 01:00','02 03:00','02 04:00','04 03:00'))
    assert list(result.strata[['utc_hour','vol_bin','N','n']].itertuples(index=False, name=None)) == [
        (1,0,2,2),(2,1,1,0),(3,2,5,2),(4,5,3,1)]
    assert result.counters['parent_count'] == 5
    assert result.counters['missing_vol_parent_count'] == 1
    assert result.counters['saturated_parent_count'] == 2
    assert result.counters['nonsaturated_parent_fraction'] == .6


def test_every_schedule_matches_counts_without_replacement_and_preserves_candidate_snapshots():
    m = importlib.import_module('channel_random_study.matching')
    source = opportunities()
    result = m.build_schedules(source)
    assert len(result.candidates) == 11 and len(result.parents) == 5
    assert result.parents.index.isin(result.candidates.index).all()
    for schedule in result.schedules:
        assert schedule.index.is_unique and schedule.index.is_monotonic_increasing
        assert schedule.index.isin(result.candidates.index).all()
        assert_frame_equal(schedule, source.loc[schedule.index])
        assert schedule.groupby(['month','utc_hour','vol_bin']).size().to_dict() == {
            ('2020-03',1,0):2, ('2020-03',3,2):2, ('2020-03',4,5):1}
    # Real parent opportunities are eligible for the random schedules.
    assert result.parents.index.intersection(result.schedules[0].index).size > 0


def test_determinism_sorted_input_and_portable_time_hashes():
    m = importlib.import_module('channel_random_study.matching')
    data = opportunities()
    a = m.build_schedules(data)
    b = m.build_schedules(data.iloc[::-1])
    assert a.metadata == b.metadata
    for left,right in zip(a.schedules,b.schedules):
        assert_frame_equal(left,right)
    assert a.metadata['seed'] == 2026110901
    assert a.metadata['bit_generator'] == 'PCG64'
    assert a.metadata['replicates'][199]['spawn_key'] == [199]
    expected = hashlib.sha256(b''.join(struct.pack('<q',int(t.value)) for t in a.candidates.index)).hexdigest()
    assert a.metadata['candidate_time_sha256'] == expected
    assert a.metadata['replicates'][0]['time_sha256'] == hashlib.sha256(
        b''.join(struct.pack('<q',int(t.value)) for t in a.schedules[0].index)).hexdigest()


def test_no_parents_keeps_two_hundred_empty_schedules_and_zero_count_strata():
    m = importlib.import_module('channel_random_study.matching')
    data = opportunities()
    data['is_parent'] = False
    result = m.build_schedules(data)
    assert len(result.schedules) == 200 and all(s.empty for s in result.schedules)
    assert len(result.strata) == 4 and result.strata.n.sum() == 0
    assert result.counters['nonsaturated_parent_fraction'] == 0.


@pytest.mark.parametrize('damage', ['duplicate','parent_outside_candidates','month','hour','bin','boundary'])
def test_malformed_candidate_grid_stops_instead_of_relaxing_matching(damage):
    m = importlib.import_module('channel_random_study.matching')
    data = opportunities()
    if damage == 'duplicate':
        data = pd.concat([data,data.iloc[:1]])
    elif damage == 'parent_outside_candidates':
        data.loc[~data.eligible, 'is_parent'] = True
    elif damage == 'month':
        data.iloc[0,data.columns.get_loc('month')] = '2020-04'
    elif damage == 'hour':
        data.iloc[0,data.columns.get_loc('utc_hour')] = 19
    elif damage == 'bin':
        data.iloc[0,data.columns.get_loc('vol_bin')] = 6
    else:
        data.index = data.index + pd.DateOffset(years=4)
        data['entry_time'] = data.index
        data['signal_time'] = data.index - pd.Timedelta(hours=1)
    with pytest.raises(ValueError):
        m.build_schedules(data)
