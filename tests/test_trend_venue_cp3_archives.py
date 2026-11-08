"""Supplied synthetic archive bytes only; no market-directory fixtures."""
from hashlib import sha256
import io
import zipfile
import pytest
from trend_hold_venue_study import cp3_archives as archives


def zipped(name, payload):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as stream:
        stream.writestr(name.removesuffix('.zip') + '.csv', payload)
    return buffer.getvalue()


def test_catalog_has_fixed_assets_venues_months_and_official_urls():
    rows = archives.archive_specs()
    identities = [(r['asset'], r['venue'], r['kind'], r['month']) for r in rows]
    assert len(identities) == len(set(identities))
    for asset, venue, first in [('BTC', 'UM', '2024-01'), ('BTC', 'SPOT', '2024-01'),
                                ('BTC', 'CM', '2024-01'), ('ETH', 'UM', '2020-07'),
                                ('ETH', 'SPOT', '2020-07'), ('ETH', 'CM', '2020-08')]:
        selected = [r for r in rows if (r['asset'], r['venue'], r['kind']) == (asset, venue, 'klines')]
        assert selected[0]['month'] == first
        assert selected[-1]['month'] == '2026-05'
    assert ('ETH', 'UM', 'fundingRate', '2020-07') in identities
    assert ('ETH', 'CM', 'fundingRate', '2022-07') in identities
    assert ('ETH', 'CM', 'fundingRate', '2022-06') not in identities
    for row in rows:
        archives.validate_spec(row)
        assert row['url'] == archives.BASE + row['relative_path']
        assert row['path'].split('/')[-1].startswith('venue_cp3_')


def test_supplied_zip_and_official_checksum_authenticate_exact_filename():
    row = archives.archive_specs()[0]
    name = row['relative_path'].split('/')[-1]
    raw = zipped(name, b'synthetic,only\n')
    checksum = (sha256(raw).hexdigest() + '  ' + name + '\n').encode()
    entry = archives.verify_payload(row, raw, checksum)
    assert entry['sha256'] == sha256(raw).hexdigest()
    assert archives.csv_payload(entry, raw, checksum) == b'synthetic,only\n'
    with pytest.raises(ValueError, match='SHA256'):
        archives.verify_payload(row, raw + b'changed', checksum)
    with pytest.raises(ValueError, match='checksum'):
        archives.verify_payload(row, raw, checksum.replace(name.encode(), b'other.zip'))


def test_archive_identity_does_not_allow_path_or_url_injection():
    row = archives.archive_specs()[0]
    for key, value in [('url', 'https://example.com/x'), ('path', '../outside.zip'),
                       ('month', '2026-11')]:
        with pytest.raises(ValueError):
            archives.validate_spec({**row, key: value})


def test_daily_spec_requires_explicit_structural_reason():
    row = archives.archive_specs()[0]
    daily = archives.daily_spec(row, '2024-01-31', 'missing_month')
    assert '/daily/klines/' in daily['url']
    assert daily['daily_reason'] == 'missing_month'
    with pytest.raises(ValueError):
        archives.daily_spec(row, '2024-02-01', 'missing_month')
    with pytest.raises(ValueError):
        archives.daily_spec(row, '2024-01-31', 'arbitrary_repair')


def inventory_fixture(tmp_path, spec, *, corrupt=False):
    import json
    source = tmp_path / 'saved' / spec['relative_path']
    source.parent.mkdir(parents=True)
    name = source.name
    raw = zipped(name, b'synthetic,only\n')
    source.write_bytes(raw)
    checksum = (sha256(raw).hexdigest() + '  ' + name + '\n').encode()
    companion = source.with_name(name + '.CHECKSUM')
    companion.write_bytes(checksum)
    entry = dict(path=str(source), sha256=sha256(raw).hexdigest(),
                 checksum_path=str(companion), checksum_sha256=sha256(checksum).hexdigest(),
                 relative_path=spec['relative_path'], url=spec['url'], provenance_metadata={'source': 'synthetic'})
    inventory = {'archives': {spec['relative_path']: entry}, 'listings': {}, 'api_pages': {}}
    path = tmp_path / archives.REG / 'reuse_sources.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(inventory))
    if corrupt:
        source.write_bytes(raw + b'changed')
    return source, companion


def test_authorized_archive_reuse_authenticates_and_hardlinks_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(archives, 'RAW', 'synthetic_raw/venue_cp3_revision_1')
    spec = archives.archive_specs()[0]
    source, checksum = inventory_fixture(tmp_path, spec)
    def forbidden(*args, **kwargs):
        raise AssertionError('verified saved source caused a network request')
    monkeypatch.setattr(archives, '_get', forbidden)
    record = {'archives': [], 'availability_listings': [], 'source_registrations': []}
    acquisition = archives.Acquisition(tmp_path, record, lambda: None)
    assert acquisition.available(spec)
    entry, raw = acquisition.obtain(spec)
    assert raw == b'synthetic,only\n'
    assert entry['reused_from']['path'] == str(source)
    assert (tmp_path / spec['path']).stat().st_ino == source.stat().st_ino
    assert (tmp_path / entry['checksum_path']).stat().st_ino == checksum.stat().st_ino
    assert spec['path'].startswith('synthetic_raw/venue_cp3_revision_1/')
    assert record['source_registrations'][0]['path'].endswith('/reuse_sources.json')
    with pytest.raises(FileExistsError):
        acquisition.obtain(spec)


def test_corrupt_reuse_stops_without_network_or_destination_write(tmp_path, monkeypatch):
    monkeypatch.setattr(archives, 'RAW', 'synthetic_raw/venue_cp3_revision_1')
    spec = archives.archive_specs()[0]
    inventory_fixture(tmp_path, spec, corrupt=True)
    monkeypatch.setattr(archives, '_get', lambda *args: pytest.fail('network fallback for corrupt registered source'))
    record = {'archives': [], 'availability_listings': [], 'source_registrations': []}
    acquisition = archives.Acquisition(tmp_path, record, lambda: None)
    with pytest.raises(ValueError, match='hash'):
        acquisition.obtain(spec)
    assert not (tmp_path / spec['path']).exists()


def test_reused_listing_is_exactly_bounded_and_retains_raw_bytes(tmp_path, monkeypatch):
    import json
    import urllib.parse
    spec = archives.archive_specs()[0]
    prefix = 'data/' + spec['relative_path'].rsplit('/', 1)[0] + '/'
    url = archives.BUCKET + '?' + urllib.parse.urlencode({'list-type': '2', 'prefix': prefix, 'max-keys': '1000'})
    raw = ('<ListBucketResult><IsTruncated>false</IsTruncated></ListBucketResult>').encode()
    source = tmp_path / 'saved-listing.xml'
    source.write_bytes(raw)
    entry = dict(path=str(source), sha256=sha256(raw).hexdigest(), url=url, prefix=prefix)
    inventory = {'archives': {}, 'listings': {prefix: [entry]}, 'api_pages': {}}
    path = tmp_path / archives.REG / 'reuse_sources.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(inventory))
    monkeypatch.setattr(archives, '_get', lambda *args: pytest.fail('saved listing requested again'))
    record = {'archives': [], 'availability_listings': [], 'source_registrations': []}
    acquisition = archives.Acquisition(tmp_path, record, lambda: None)
    assert acquisition.available(spec) is False
    saved = record['availability_listings'][0]
    assert saved['reused_from']['path'] == str(source)
    assert (tmp_path / saved['path']).read_bytes() == raw


def test_missing_inventory_source_checks_actual_storage_before_network(tmp_path, monkeypatch):
    from trend_hold_venue_study import cp3_reuse
    monkeypatch.setattr(cp3_reuse.shutil, 'disk_usage', lambda path: type('Usage', (), {'free': 39 * 2**30})())
    with pytest.raises(ValueError, match='40 GiB'):
        cp3_reuse.require_download_space(tmp_path)
