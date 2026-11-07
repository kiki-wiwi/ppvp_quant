"""Pure synthetic coverage for the user-approved nine exact input exceptions."""
import numpy as np
import pandas as pd
import pytest
from trend_hold_venue_study import data, scope

# Independent copy of the approved diagnostic timestamps, not derived from code constants.
CASES=(
    ('2020-02',1582112100000,1582112132286,False),
    ('2020-03',1583313660000,1583313706694,False),
    ('2020-12',1608559740000,1608558440521,True),
    ('2021-02',1613014800000,1613014854773,True),
    ('2021-04',1619323200000,1619323258146,False),
    ('2021-08',1628819940000,1628819999000,False),
    ('2021-12',1640321940000,1640321994362,False),
    ('2023-03',1679661540000,1679661581646,True),
)


def csv_row(open_ms,close_ms,prices):
    return ','.join(map(str,(open_ms,*prices,close_ms,100.,1,.5,50.,0)))+'\n'


def exception_fixture(case):
    month,open_ms,close_ms,placeholder=case
    previous=(90.,91.,89.,90.,2.)
    prices=(200.,200.,200.,200.,0.) if placeholder else (100.,105.,95.,102.,1.25)
    raw=(csv_row(open_ms-60_000,open_ms-1,previous)+csv_row(open_ms,close_ms,prices)).encode()
    return raw,np.asarray([previous,prices],dtype=np.float64)


@pytest.mark.parametrize('case',CASES)
def test_each_exact_spot_clock_exception_keeps_ohlcv_and_existing_bad_rule(case):
    month,open_ms,close_ms,placeholder=case
    raw,expected_prices=exception_fixture(case)
    parsed=data.parse_klines(raw,month,'SPOT')
    np.testing.assert_array_equal(parsed.loc[:,list(data.OHLCV)].to_numpy(),expected_prices)
    entries=parsed.attrs['raw_close_time_exceptions']
    assert len(entries)==1
    entry=entries[0]
    assert entry['venue']=='SPOT' and entry['month']==month
    assert entry['open_time_ns']==open_ms*1_000_000
    assert entry['raw_close_time_ns']==close_ms*1_000_000
    assert entry['decision_clock_close_ns']==(open_ms+60_000)*1_000_000
    assert entry['existing_R5_placeholder_rule_marks_bad'] is placeholder
    assert entry['raw_ohlcv_unchanged'] is True
    start=pd.Timestamp((open_ms-60_000)*1_000_000,tz='UTC')
    end=pd.Timestamp((open_ms+60_000)*1_000_000,tz='UTC')
    grid=data.build_grid(parsed,start,end)
    assert grid['bad'].tolist()==[False,placeholder]
    assert not grid['archive_gap'].any()
    if placeholder:
        assert grid.iloc[-1][['open','high','low','close']].tolist()==[90.]*4
    else:
        np.testing.assert_array_equal(grid.loc[:,list(data.OHLCV)].to_numpy(),expected_prices)
    # Building the trading grid must not mutate the raw parsed OHLCV values.
    np.testing.assert_array_equal(parsed.loc[:,list(data.OHLCV)].to_numpy(),expected_prices)


@pytest.mark.parametrize('case',CASES)
def test_changed_exception_endpoint_rejected_even_if_changed_to_normal_close(case):
    month,open_ms,close_ms,_=case
    for changed_close in (close_ms+1,open_ms+59_999):
        raw=csv_row(open_ms,changed_close,(100.,101.,99.,100.,1.)).encode()
        with pytest.raises(ValueError,match='unregistered native close time'):
            data.parse_klines(raw,month,'SPOT')


def test_unknown_exception_changed_open_wrong_month_and_wrong_venue_rejected():
    month,open_ms,close_ms,_=CASES[2]  # Valid CM month, but whitelist belongs only to SPOT.
    raw=csv_row(open_ms,close_ms,(100.,101.,99.,100.,1.)).encode()
    with pytest.raises(ValueError,match='unregistered native close time'):
        data.parse_klines(raw,month,'CM')
    with pytest.raises(ValueError,match='month'):
        data.parse_klines(raw,'2020-11','SPOT')
    changed=csv_row(open_ms+60_000,close_ms,(100.,101.,99.,100.,1.)).encode()
    with pytest.raises(ValueError,match='unregistered native close time'):
        data.parse_klines(changed,month,'SPOT')
    unknown=csv_row(open_ms+120_000,open_ms+150_000,(100.,101.,99.,100.,1.)).encode()
    with pytest.raises(ValueError,match='unregistered native close time'):
        data.parse_klines(unknown,month,'SPOT')


def test_inventory_reports_exact_five_valid_and_three_existing_placeholders():
    entries=[]
    for case in CASES:
        raw,_=exception_fixture(case)
        entries.extend(data.parse_klines(raw,case[0],'SPOT').attrs['raw_close_time_exceptions'])
    summary=data.validate_clock_exception_inventory(entries)
    assert summary=={'count':8,'valid_rows':5,'existing_bad_placeholders':3,
                     'additional_bad_flags':0,'ohlcv_changes':0,'clock':'open_time + 1min'}
    with pytest.raises(ValueError,match='exactly 8'):
        data.validate_clock_exception_inventory(entries[:-1])
    with pytest.raises(ValueError,match='exactly 8'):
        data.validate_clock_exception_inventory(entries+[dict(entries[0])])


def canonical_fixture():
    index=pd.date_range('2020-08-11','2024-01-01',tz='UTC',freq='8h',inclusive='left').as_unit('ns')
    result=pd.Series(np.resize([.0001,-.0002,.0003],len(index)),index=index,name='funding_rate')
    result.attrs['canonical_sha256']='synthetic-registration-unchanged'
    return result


def test_single_approved_funding_record_excluded_only_from_context_and_logged():
    canonical=canonical_fixture();original=canonical.copy(deep=True)
    result=data.funding_for_context(canonical,pd.Timestamp('2020-08-11T07:02Z'))
    assert len(canonical)==3714 and len(result)==3713
    pd.testing.assert_series_equal(canonical,original,check_exact=True)
    np.testing.assert_array_equal(result.to_numpy(),canonical.iloc[1:].to_numpy())
    assert result.index.equals(canonical.index[1:])
    assert (result.index>=pd.Timestamp('2020-09-01T00:00Z')).sum()==3651
    assert result.attrs['canonical_record_count']==3714
    assert result.attrs['context_record_count']==3713
    assert result.attrs['canonical_sha256']=='synthetic-registration-unchanged'
    assert canonical.attrs==original.attrs
    exclusions=result.attrs['context_funding_exclusions']
    assert len(exclusions)==1
    assert exclusions[0]['funding_time_utc']=='2020-08-11T00:00:00+00:00'
    assert exclusions[0]['funding_rate']=='0.0001'
    assert exclusions[0]['native_context_start_utc']=='2020-08-11T07:02:00+00:00'
    assert exclusions[0]['preserved_in_canonical_registration'] is True


def test_pre_context_funding_with_changed_rate_start_or_record_count_is_rejected():
    canonical=canonical_fixture();native=pd.Timestamp('2020-08-11T07:02Z')
    changed=canonical.copy();changed.iloc[0]=.0002
    with pytest.raises(ValueError,match='unapproved pre-context'):
        data.funding_for_context(changed,native)
    with pytest.raises(ValueError,match='unapproved pre-context'):
        data.funding_for_context(canonical,native+pd.Timedelta(minutes=1))
    with pytest.raises(ValueError,match='unapproved pre-context'):
        data.funding_for_context(canonical,pd.Timestamp('2020-08-11T15:00Z'))
    missing_august=canonical.drop(pd.Timestamp('2020-08-12T00:00Z'))
    with pytest.raises(ValueError,match='unapproved pre-context'):
        data.funding_for_context(missing_august,native)
    extra=pd.concat([pd.Series([.0001],index=pd.DatetimeIndex(['2020-08-10T16:00Z']).as_unit('ns')),canonical])
    with pytest.raises(ValueError,match='native IS'):
        data.funding_for_context(extra,native)
    duplicate=pd.concat([canonical.iloc[:1],canonical])
    with pytest.raises(ValueError,match='unique'):
        data.funding_for_context(duplicate,native)


def test_ordinary_no_pre_context_funding_path_retains_all_values():
    source=canonical_fixture().iloc[2:].copy()  # First settlement 16:00, no event before 15:00.
    result=data.funding_for_context(source,pd.Timestamp('2020-08-11T15:00Z'))
    pd.testing.assert_series_equal(result,source,check_exact=True)
    assert result.attrs['context_funding_exclusions']==[]
    assert result.attrs['context_record_count']==len(source)


def test_load_cm_funding_calls_registered_verifier_and_applies_only_approved_filter(monkeypatch):
    from trend_hold_venue_study import funding_sources
    monkeypatch.setattr(scope,'BTC_IS_UNLOCKED',True)
    monkeypatch.setattr(scope,'POST_IS_AND_ETH_UNLOCKED',False)
    canonical=canonical_fixture();seen=[]
    def verifier(root):
        seen.append(root)
        return canonical
    monkeypatch.setattr(funding_sources,'verify_registered',verifier)
    result=data.load_cm_funding('/synthetic/root',pd.Timestamp('2020-08-11T07:02Z'))
    assert seen==[data.Path('/synthetic/root')]
    assert len(result)==3713 and len(canonical)==3714
    assert result.attrs['context_funding_exclusions'][0]['funding_rate']=='0.0001'
