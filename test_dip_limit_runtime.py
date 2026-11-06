"""Only synthetic control-flow and entry-boundary checks."""
import pandas as pd
import pytest
from test_dip_limit_scope import frame


def test_cli_real_is_locked_before_output_or_input(tmp_path):
    from tools.run_dip_limit_study import main
    from dip_limit_study.scope import RealDataLockedError
    with pytest.raises(RealDataLockedError): main(['--real'])
    assert not (tmp_path/'results').exists()


def test_incomplete_official_calendar_rejected_before_output(tmp_path):
    from dip_limit_study.runtime import execute
    from dip_limit_study.scope import make_context
    with pytest.raises(ValueError,match='complete IS'):
        execute(make_context(frame()),tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_failure_aggregation_includes_limit_stress():
    from dip_limit_study.runtime import execution_failures
    from types import SimpleNamespace
    from dip_limit_study.config import VERSIONS,COSTS
    results={(v,c,False):SimpleNamespace(metadata={'failed':False}) for v in VERSIONS for c in COSTS}
    results[('L_limit','main',True)]=SimpleNamespace(metadata={'failed':True})
    results[('L_limit','harsh',True)]=SimpleNamespace(metadata={'failed':False})
    assert execution_failures(results)=={'M_market':False,'L_limit':True}
