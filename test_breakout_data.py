"""Only generated frames are returned by mocked disk readers."""
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from breakout_study import data
from breakout_study.config import IS_START, IS_END


def test_funding_reader_filters_before_loading_and_checks_grid():
    idx=pd.date_range(IS_START,IS_END,freq='8h',inclusive='left')
    f=pd.DataFrame({'funding_rate':.0001}, index=idx+pd.Timedelta(milliseconds=2))
    with patch.object(data,'require_real_authorization'), patch.object(pd,'read_parquet',return_value=f) as reader:
        got=data.load_funding_is()
    assert len(got)==4383 and got.index.equals(idx.as_unit('ns'))
    assert reader.call_args.kwargs['filters']==[('funding_time','>=',IS_START),('funding_time','<',IS_END)]
    assert reader.call_args.args[0].name=='BTCUSDT_funding.parquet'
    for invalid in (f.iloc[1:],pd.concat([f,f.iloc[:1]])):
        with patch.object(data,'require_real_authorization'),patch.object(pd,'read_parquet',return_value=invalid),pytest.raises(ValueError):
            data.load_funding_is()


def test_perp_reader_filters_and_exact_content_hash(tmp_path):
    import json
    from impulse_study.data_1m import content_sha256
    f=pd.DataFrame(dict(open=[100.],high=[101.],low=[99.],close=[100.],volume=[2.]),index=pd.DatetimeIndex([IS_START]).as_unit('ns'))
    folder=tmp_path/'data'; folder.mkdir()
    manifest={'caches':{'perp_1m':{'content_sha256':content_sha256(f)}}}
    (folder/'manifest_impulse_1m.json').write_text(json.dumps(manifest))
    with patch.object(data,'require_real_authorization'),patch.object(pd,'read_parquet',return_value=f.copy()) as reader:
        pd.testing.assert_frame_equal(data.load_perp_is(tmp_path),f)
    assert reader.call_args.kwargs['filters']==[('open_time','>=',IS_START),('open_time','<',IS_END)]
    assert reader.call_args.args[0].name=='BTCUSDT_perp_1m_IS.parquet'
    changed=f.copy(); changed['volume']=3.
    with patch.object(data,'require_real_authorization'),patch.object(pd,'read_parquet',return_value=changed),pytest.raises(ValueError,match='content hash'):
        data.load_perp_is(tmp_path)


def test_each_public_loader_is_locked_before_any_io():
    from pathlib import Path
    with patch.object(pd,'read_parquet',side_effect=AssertionError('disk')),patch.object(Path,'read_text',side_effect=AssertionError('disk')):
        for name in ('load_funding_is','load_perp_is','load_real_context'):
            with pytest.raises(data.RealDataLockedError): getattr(data,name)()


def test_archive_gap_metadata_interprets_naive_frozen_literals_as_utc():
    gaps=data.archive_gap_index()
    assert len(gaps)==10 and str(gaps.dtype)=='datetime64[ns, UTC]'
    assert gaps[0]==pd.Timestamp('2021-01-12 06:10',tz='UTC')
    assert gaps[-1]==pd.Timestamp('2022-08-21 08:10',tz='UTC')
