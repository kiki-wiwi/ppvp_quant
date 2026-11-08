"""CP3 authorization transitions and immutable baseline protection."""
import pytest
from trend_hold_venue_study import scope
from trend_hold_venue_study import cp3_control as control


def test_current_cp3_preunlock_baseline_and_bounded_authorization(monkeypatch):
    assert scope.BTC_IS_UNLOCKED is True
    assert scope.POST_IS_AND_ETH_UNLOCKED is False
    scope.require_real_authorization('BTC','UM','2020-01-01T00:00Z','2024-01-01T00:00Z')
    for asset,start in [('BTC','2024-01-01T00:00Z'),('ETH','2020-07-01T00:00Z')]:
        with pytest.raises(scope.RealDataLockedError):
            scope.require_real_authorization(asset,'UM',start,'2026-11-01T00:00Z')
    control.require_preparation()
    with pytest.raises(scope.RealDataLockedError):control.require_cp3()
    monkeypatch.setattr(scope,'POST_IS_AND_ETH_UNLOCKED',True)
    control.require_cp3()
    with pytest.raises(scope.RealDataLockedError):control.require_preparation()
    for asset,start in [('BTC','2024-01-01T00:00Z'),('ETH','2020-07-01T00:00Z')]:
        scope.require_real_authorization(asset,'UM',start,'2026-11-01T00:00Z')
    with pytest.raises(scope.ScopeError):
        scope.require_real_authorization('ETH','UM','2020-07-01T00:00Z','2026-11-02T00:00Z')
    monkeypatch.setattr(scope,'BTC_IS_UNLOCKED',False)
    with pytest.raises(scope.RealDataLockedError):control.require_cp3()


def test_post_unlock_is_exactly_one_line_and_cannot_reclose_btc():
    before=b'BTC_IS_UNLOCKED = True\nPOST_IS_AND_ETH_UNLOCKED = False\n'
    after=before.replace(control.POST_FALSE,control.POST_TRUE)
    control.validate_transition(before,after)
    assert control.normalize_scope(before,post_unlocked=False)==b'BTC_IS_UNLOCKED = False\nPOST_IS_AND_ETH_UNLOCKED = False\n'
    assert control.normalize_scope(after,post_unlocked=True)==control.normalize_scope(before,post_unlocked=False)
    for bad in (before,after+b'# extra\n',after.replace(control.BTC_TRUE,control.BTC_FALSE),after.replace(b' = True',b'=True')):
        with pytest.raises(ValueError):control.validate_transition(before,bad)
    with pytest.raises(ValueError):control.normalize_scope(after+control.POST_TRUE,post_unlocked=True)


def test_unauthorized_formal_entry_precedes_all_path_access(tmp_path,monkeypatch):
    monkeypatch.setattr(scope,'POST_IS_AND_ETH_UNLOCKED',False)
    with pytest.raises(scope.RealDataLockedError):control.verify_unlock(tmp_path)
    assert list(tmp_path.iterdir())==[]


def test_new_file_scope_is_narrow_and_old_files_cannot_be_modified():
    assert control.allowed_addition('trend_hold_venue_study/cp3_control.py')
    assert control.allowed_addition('tests/test_trend_venue_cp3_control.py')
    assert control.allowed_addition(control.REG+'/preflight/manifest.json')
    assert not control.allowed_addition('trend_hold_venue_study/features.py')
    assert not control.allowed_addition(control.OUT+'/premature.csv')
    assert not control.allowed_addition(control.REG+'/../approved_spec.md')
