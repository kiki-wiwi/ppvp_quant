"""The executable entry never reaches filesystem loading before authorization."""
import runpy
from pathlib import Path
import pytest
from trend_hold_venue_study import scope


def test_real_cli_rejects_locked_post_before_accessing_output_or_loader(monkeypatch):
    module=runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/run_trend_hold_venue_cp3.py'))
    monkeypatch.setattr(scope,'BTC_IS_UNLOCKED',True)
    monkeypatch.setattr(scope,'POST_IS_AND_ETH_UNLOCKED',False)
    with pytest.raises(scope.RealDataLockedError):module['main'](['--real'])
