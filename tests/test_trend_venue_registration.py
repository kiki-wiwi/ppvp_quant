"""Synthetic registry checks; no market reads or network."""
from hashlib import sha256
import json
import pytest
from trend_hold_venue_study.registration import verify, digest
from trend_hold_venue_study.config import BASELINE
from trend_hold_venue_study.testing_guard import _market_path

def test_registration_detects_byte_changes_and_normalizes_only_new_locks(tmp_path):
    (tmp_path/'old.py').write_text('protected\n')
    (tmp_path/'trend_hold_venue_study').mkdir()
    p=tmp_path/'trend_hold_venue_study/scope.py'
    p.write_text('BTC_IS_UNLOCKED = False\nPOST_IS_AND_ETH_UNLOCKED = False\n')
    reg=dict(baseline_commit=BASELINE,protected_sources_sha256={'old.py':digest(tmp_path/'old.py')})
    own={'sha256':{'trend_hold_venue_study/scope.py':digest(p)}}
    assert verify(tmp_path,reg,own)['status']=='verified'
    p.write_text('BTC_IS_UNLOCKED = True\nPOST_IS_AND_ETH_UNLOCKED = False\n')
    assert verify(tmp_path,reg,own)['status']=='verified'
    (tmp_path/'old.py').write_text('changed\n')
    with pytest.raises(ValueError,match='protected'):verify(tmp_path,reg,own)

def test_guard_rejects_market_locations_and_allows_sources():
    assert _market_path('data/raw/spot/BTCUSDT.zip')
    assert _market_path('/x/data/processed_1m/BTCUSDT.parquet')
    assert not _market_path('reports/trend_hold_venue_round/approved_spec.md')
    with pytest.raises(PermissionError,match='CP1 guard'):
        open('data/raw/never_accessed_synthetic_probe','rb')
