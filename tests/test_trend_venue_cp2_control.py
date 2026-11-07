"""Independent CP2 authorization and one-line transition validation."""
import pytest
from trend_hold_venue_study import scope,cp2_control as control

def test_formal_guard_never_accepts_post_is_unlock(monkeypatch):
    monkeypatch.setattr(scope,'BTC_IS_UNLOCKED',False)
    monkeypatch.setattr(scope,'POST_IS_AND_ETH_UNLOCKED',False)
    with pytest.raises(PermissionError):control.require_btc_is()
    monkeypatch.setattr(scope,'BTC_IS_UNLOCKED',True)
    control.require_btc_is()
    monkeypatch.setattr(scope,'POST_IS_AND_ETH_UNLOCKED',True)
    with pytest.raises(PermissionError):control.require_btc_is()

def test_unlock_transition_is_exactly_one_line_and_preserves_other_lock():
    before=b'BTC_IS_UNLOCKED = False\nPOST_IS_AND_ETH_UNLOCKED = False\n'
    after=before.replace(control.FALSE,control.TRUE)
    control.validate_transition(before,after)
    for changed in (after+b'# extra\n',after.replace(b'POST_IS_AND_ETH_UNLOCKED = False',b'POST_IS_AND_ETH_UNLOCKED = True'),before):
        with pytest.raises(ValueError):control.validate_transition(before,changed)

def test_cp2_controls_are_source_only_until_real_authorization(tmp_path,monkeypatch):
    monkeypatch.setattr(scope,'BTC_IS_UNLOCKED',False)
    with pytest.raises(PermissionError):control.verify_unlock(tmp_path)
    assert not list(tmp_path.iterdir())
