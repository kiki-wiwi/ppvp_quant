"""Official archive preparation helpers tested only with synthetic byte strings."""
import hashlib
import pytest
from trend_hold_venue_study import archives,scope

def test_fixed_107_monthly_archives_have_no_other_market_or_future_period():
    rows=archives.archive_specs()
    assert len(rows)==107
    assert len({r['relative_path'] for r in rows})==107
    assert sum(r['venue']=='SPOT' for r in rows)==48
    assert sum(r['venue']=='CM' and r['kind']=='klines' for r in rows)==41
    assert sum(r['kind']=='fundingRate' for r in rows)==18
    for row in rows:
        assert row['month']<'2024-01'
        assert 'ETH' not in row['relative_path']
        assert archives.validate_archive(row)==row['relative_path']

def test_archive_request_cannot_escape_fixed_registration():
    row=archives.archive_specs()[0]
    for key,value in [('month','2024-01'),('venue','ETH'),('relative_path','../BTC.zip'),('url','https://example.org/file.zip')]:
        changed={**row,key:value}
        with pytest.raises(ValueError):archives.validate_archive(changed)

def test_checksum_exact_bytes_and_filename_are_required():
    payload=b'only synthetic ZIP bytes'
    digest=hashlib.sha256(payload).hexdigest()
    assert archives.checksum_hash((digest+'  BTCUSDT-1m-2020-01.zip\n').encode(),'BTCUSDT-1m-2020-01.zip')==digest
    with pytest.raises(ValueError):archives.checksum_hash((digest+'  other.zip').encode(),'BTCUSDT-1m-2020-01.zip')
    with pytest.raises(ValueError):archives.checksum_hash(b'bad checksum','BTCUSDT-1m-2020-01.zip')

def test_archive_preparation_requires_both_locks_closed_before_filesystem(monkeypatch,tmp_path):
    monkeypatch.setattr(scope,'BTC_IS_UNLOCKED',True)
    with pytest.raises(PermissionError):archives.prepare(tmp_path)
    monkeypatch.setattr(scope,'BTC_IS_UNLOCKED',False)
    monkeypatch.setattr(scope,'POST_IS_AND_ETH_UNLOCKED',True)
    with pytest.raises(PermissionError):archives.prepare(tmp_path)
