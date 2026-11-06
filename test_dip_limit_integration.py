"""Opt-in full-calendar invented-price validation; never a real-data run."""
import os
from pathlib import Path
import pytest

@pytest.mark.skipif(not os.environ.get('DIP_LIMIT_INTEGRATION_OUTPUT'),reason='one explicit full synthetic integration, outside repository')
def test_complete_synthetic_registered_batch():
    from dip_limit_study.runtime import execute
    from dip_limit_study.synthetic import study_context
    out=Path(os.environ['DIP_LIMIT_INTEGRATION_OUTPUT'])
    got=execute(study_context(),out,{'source':'synthetic','market_reads':0})
    assert got['status']=='completed' and got['source']=='synthetic'
    assert got['combination_count']==8 and len(got['trade_counts'])==8
    assert got['eligible_father_count']>0 and got['execution_attribution']['father_count']>0
    assert len(list(out.glob('*_minute_equity.parquet')))==8
    assert len(list(out.glob('figure_*.png')))==3
    assert not (out/'STOPPED.json').exists()
