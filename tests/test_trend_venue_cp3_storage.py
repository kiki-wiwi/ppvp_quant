"""Capacity checks bind actual data/result/LFS destinations, not the cwd alone."""
import pytest
from trend_hold_venue_study import cp3_storage as storage


def sample():
    return {k:{'device_id':17,'free_bytes':40*2**30,'path':'/synthetic/'+k} for k in storage.REQUIRED_LOCATIONS}


def test_actual_destinations_must_share_registered_volume_and_capacity():
    rows=sample()
    storage.validate_measurements(rows,expected_device=17,uuid=storage.VOLUME_UUID,mounted=True)
    for field,value in [('device_id',18),('free_bytes',40*2**30-1)]:
        bad={k:dict(v) for k,v in rows.items()};bad['lfs_temp'][field]=value
        with pytest.raises(ValueError):storage.validate_measurements(bad,expected_device=17,uuid=storage.VOLUME_UUID,mounted=True)
    with pytest.raises(ValueError):storage.validate_measurements(rows,expected_device=17,uuid='wrong',mounted=True)
    with pytest.raises(ValueError):storage.validate_measurements(rows,expected_device=17,uuid=storage.VOLUME_UUID,mounted=False)
    del rows['reports']
    with pytest.raises(ValueError):storage.validate_measurements(rows,expected_device=17,uuid=storage.VOLUME_UUID,mounted=True)
