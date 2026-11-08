"""Fixed CP3 official archive identities, byte authentication and acquisition.

No market parsing, feature imports, automatic retry or existing-file replacement.
Availability comes from saved, bounded official bucket listings before GET.
"""
from __future__ import annotations
from datetime import date, datetime, timezone
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from .cp3_windows import LAST_MONTH
from .cp3_reuse import ReuseInventory, hardlink_new, require_download_space

BASE = 'https://data.binance.vision/data/'
BUCKET = 'https://s3-ap-northeast-1.amazonaws.com/data.binance.vision'
REG = 'reports/trend_hold_venue_round/cp3_registration'
RAW = 'data/raw/venue_cp3_revision_1'
KINDS = ('klines', 'fundingRate')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def digest(raw):
    return sha256(raw).hexdigest()


def file_hash(path):
    result = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_new(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw)


def months(first, last=LAST_MONTH):
    year, month = map(int, first.split('-'))
    while f'{year:04d}-{month:02d}' <= last:
        yield f'{year:04d}-{month:02d}'
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)


def month_bounds(month):
    if not re.fullmatch(r'20\d{2}-(0[1-9]|1[0-2])', str(month)):
        raise ValueError('valid registered month required')
    year, number = map(int, month.split('-'))
    start = date(year, number, 1)
    end = date(year + 1, 1, 1) if number == 12 else date(year, number + 1, 1)
    return start, end


def _spec(asset, venue, kind, month, *, day=None, reason=None):
    symbol = asset + ('USD_PERP' if venue == 'CM' else 'USDT')
    market = {'UM': 'futures/um', 'SPOT': 'spot', 'CM': 'futures/cm'}[venue]
    frequency = 'daily' if day else 'monthly'
    suffix = day or month
    relative = (f'{market}/{frequency}/klines/{symbol}/1m/{symbol}-1m-{suffix}.zip'
                if kind == 'klines' else
                f'{market}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{suffix}.zip')
    path = f'{RAW}/{asset}/{venue}/{kind}/venue_cp3_{PurePosixPath(relative).name}'
    row = dict(asset=asset, venue=venue, kind=kind, month=month, symbol=symbol,
               relative_path=relative, path=path, url=BASE + relative)
    if day:
        row.update(day=day, daily_reason=reason)
    return row


def archive_specs():
    result = []
    for asset in ('BTC', 'ETH'):
        for venue in ('UM', 'SPOT', 'CM'):
            first = '2024-01' if asset == 'BTC' else ('2020-08' if venue == 'CM' else '2020-07')
            for month in months(first):
                result.append(_spec(asset, venue, 'klines', month))
            if venue != 'SPOT':
                funding_first = max(first, '2022-07') if venue == 'CM' else first
                for month in months(funding_first):
                    result.append(_spec(asset, venue, 'fundingRate', month))
    return result


def validate_spec(row):
    identity = tuple(row.get(key) for key in ('asset', 'venue', 'kind', 'month'))
    matches = [item for item in archive_specs()
               if tuple(item[key] for key in ('asset', 'venue', 'kind', 'month')) == identity]
    if len(matches) != 1:
        raise ValueError('archive outside fixed CP3 catalog')
    expected = matches[0]
    if 'day' in row:
        expected = daily_spec(expected, row['day'], row.get('daily_reason'))
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError('archive identity/path/URL mismatch')
    return expected


def daily_spec(monthly, day, reason):
    if monthly.get('kind') != 'klines' or reason not in ('missing_month', 'absent_whole_utc_date'):
        raise ValueError('daily supplement requires approved structural reason')
    date.fromisoformat(day)
    if day[:7] != monthly['month']:
        raise ValueError('daily supplement outside registered month')
    return _spec(monthly['asset'], monthly['venue'], 'klines', monthly['month'], day=day, reason=reason)


def checksum_hash(raw, filename):
    text = raw.decode('ascii').strip()
    found = re.fullmatch(r'([0-9a-fA-F]{64})[ \t]+\*?([^\r\n]+)', text)
    if found is None or found.group(2) != filename:
        raise ValueError('invalid official checksum record or filename')
    return found.group(1).lower()


def verify_payload(spec, blob, checksum):
    validate_spec(spec)
    expected = checksum_hash(checksum, PurePosixPath(spec['relative_path']).name)
    if digest(blob) != expected:
        raise ValueError('official archive SHA256 mismatch: ' + spec['relative_path'])
    return {**spec, 'sha256': expected, 'bytes': len(blob),
            'checksum_path': spec['path'] + '.CHECKSUM', 'checksum_sha256': digest(checksum),
            'verified_against': 'official CHECKSUM'}


def csv_payload(entry, blob, checksum):
    from .data import unzip_csv
    verified = verify_payload(entry, blob, checksum)
    for key in ('sha256', 'checksum_sha256', 'checksum_path'):
        if entry.get(key) != verified[key]:
            raise ValueError('registered archive hash/path mismatch: ' + key)
    return unzip_csv(blob, PurePosixPath(entry['relative_path']).name)


def _get(url):
    if not (url.startswith(BASE) or url.startswith(BUCKET + '?')):
        raise ValueError('only official Binance archive endpoints allowed')
    request = urllib.request.Request(url, headers={'User-Agent': 'ppvp-quant-preregistered-research/CP3'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


class Acquisition:
    """One-shot acquisition, with a registration callback after each raw write."""
    def __init__(self, root, record, persist, reuse=None):
        self.root, self.record, self.persist = Path(root), record, persist
        self.reuse = reuse if reuse is not None else ReuseInventory(root, record, persist)
        self.listings = {}

    def available(self, spec):
        validate_spec(spec)
        relative = spec['relative_path']
        if self.reuse.has_archive(spec):
            return True
        prefix = 'data/' + str(PurePosixPath(relative).parent) + '/'
        if 'day' in spec:
            prefix += spec['symbol'] + '-1m-' + spec['month']
        if prefix not in self.listings:
            self.listings[prefix] = self._list(prefix)
        keys = self.listings[prefix]
        archive, checksum = 'data/' + relative, 'data/' + relative + '.CHECKSUM'
        if (archive in keys) != (checksum in keys):
            raise ValueError('archive/CHECKSUM availability differs: ' + relative)
        return archive in keys

    def _list(self, prefix):
        keys = set()
        token = None
        saved = self.reuse.listing_sources(prefix)
        for page in range(1, 5):
            parameters = {'list-type': '2', 'prefix': prefix, 'max-keys': '1000'}
            if token:
                parameters['continuation-token'] = token
            url = BUCKET + '?' + urllib.parse.urlencode(parameters)
            source = None
            if saved is not None:
                if page > len(saved):
                    raise ValueError('saved official listing pagination is incomplete')
                source = saved[page - 1]
                raw = self.reuse.listing_page(source, url)
            else:
                require_download_space(self.root)
                raw = _get(url)
            path = f'{REG}/venue_cp3_revision_1_listing_{digest(prefix.encode())[:16]}_{page:02d}.xml'
            if source is None:
                write_new(self.root / path, raw)
            else:
                hardlink_new(source['path'], self.root / path)
            entry = {'path': path, 'url': url, 'sha256': digest(raw), 'bytes': len(raw), 'prefix': prefix}
            if source is not None:
                entry.update(reused_from=source, provenance='authenticated saved bounded official listing; no request')
            self.record['availability_listings'].append(entry)
            self.persist()
            tree = ET.fromstring(raw)
            def fields(name):
                return [node.text for node in tree.iter() if node.tag.rsplit('}', 1)[-1] == name]
            found = fields('Key')
            if any(not key or not key.startswith(prefix) for key in found):
                raise ValueError('bucket listing escaped bounded prefix')
            keys.update(found)
            truncated = fields('IsTruncated')
            if truncated == ['false']:
                if saved is not None and page != len(saved):
                    raise ValueError('saved official listing has extra pages after terminal page')
                return keys
            tokens = fields('NextContinuationToken')
            if truncated != ['true'] or len(tokens) != 1 or tokens[0] == token:
                raise ValueError('invalid official listing pagination')
            token = tokens[0]
        raise ValueError('bounded official listing exceeded four pages')

    def obtain(self, spec):
        validate_spec(spec)
        path = self.root / spec['path']
        if path.exists() or Path(str(path) + '.CHECKSUM').exists():
            raise FileExistsError('CP3 revision source already exists; never overwrite or retry')
        saved = self.reuse.archive_source(spec)
        if saved is not None:
            source, blob, checksum = saved
            entry = verify_payload(spec, blob, checksum)
            hardlink_new(source['path'], path)
            hardlink_new(source['checksum_path'], self.root / entry['checksum_path'])
            entry.update(reused_from=source, provenance='authenticated saved archive and official CHECKSUM; no request')
        else:
            if not self.available(spec):
                raise FileNotFoundError('official archive absent: ' + spec['relative_path'])
            require_download_space(self.root)
            checksum = _get(spec['url'] + '.CHECKSUM')
            require_download_space(self.root)
            blob = _get(spec['url'])
            entry = verify_payload(spec, blob, checksum)
            write_new(path, blob)
            write_new(self.root / entry['checksum_path'], checksum)
            entry['provenance'] = 'new official archive and CHECKSUM download; inventory contained no source'
        self.record['archives'].append(entry)
        self.persist()
        return entry, csv_payload(entry, blob, checksum)
