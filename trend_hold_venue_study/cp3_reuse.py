"""Explicitly authorized saved-source reuse, with immutable byte provenance.

Inventory entries are bound to the new input registration. A registered source
that fails authentication is terminal; it never falls back to a network request.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import shutil
from hashlib import sha256

INVENTORY = 'reports/trend_hold_venue_round/cp3_registration/reuse_sources.json'
MIN_FREE_BYTES = 40 * 2**30


def require_download_space(root):
    """Check actual destinations immediately before each new network download."""
    root = Path(root)
    for name in ('data', 'reports'):
        path = root / name
        while not path.exists():
            path = path.parent
        if shutil.disk_usage(path).free < MIN_FREE_BYTES:
            raise ValueError('actual download destination has less than 40 GiB: ' + str(path))


PRIMARY_MOUNT = Path('/mnt/ppvp-backtests')


def _mount_records():
    """Use kernel mount roots, since equal st_dev alone does not permit link()."""
    def decode(value):
        return re.sub(r'\\([0-7]{3})', lambda m: chr(int(m.group(1), 8)), value)
    rows = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        fields = line.split()
        rows.append(dict(id=fields[0], device=fields[2],
                         root=Path(decode(fields[3])), point=Path(decode(fields[4]))))
    return rows


def _mount_for(path, rows):
    matches = [row for row in rows if path.is_relative_to(row['point'])]
    if not matches:
        raise ValueError('path has no verified mount entry: ' + str(path))
    return max(matches, key=lambda row: len(str(row['point'])))


def _canonical_existing(path, rows):
    path = Path(os.path.abspath(path))
    current = _mount_for(path, rows)
    primary = next((row for row in rows if row['point'] == PRIMARY_MOUNT), None)
    if primary is None or current['device'] != primary['device']:
        # Standalone synthetic directories can be on another filesystem. The
        # two normalized endpoints must still share one mount before linking.
        return path.resolve(strict=True)
    if not current['root'].is_relative_to(primary['root']):
        raise ValueError('bind root is outside the primary volume')
    candidate = (PRIMARY_MOUNT / current['root'].relative_to(primary['root'])
                 / path.relative_to(current['point'])).resolve(strict=True)
    if not candidate.is_relative_to(PRIMARY_MOUNT.resolve(strict=True)):
        raise ValueError('mapped path escaped the primary volume')
    if not path.samefile(candidate):
        raise ValueError('mapped path does not identify the same file or directory')
    return candidate


def _sha256_file(path):
    digest = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def hardlink_new(source, destination):
    """Authenticate aliases, then link through one mount; no copy or overwrite."""
    source, destination = Path(os.path.abspath(source)), Path(os.path.abspath(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = _mount_records()
    canonical_source = _canonical_existing(source, rows)
    canonical_parent = _canonical_existing(destination.parent, rows)
    canonical_destination = canonical_parent / destination.name
    if _mount_for(canonical_source, rows)['id'] != _mount_for(canonical_parent, rows)['id']:
        raise ValueError('hardlink endpoints do not share one normalized mount')
    expected = _sha256_file(source)
    if _sha256_file(canonical_source) != expected:
        raise ValueError('normalized source SHA256 differs from authenticated alias')
    os.link(canonical_source, canonical_destination)
    if not destination.samefile(source) or _sha256_file(destination) != expected:
        raise ValueError('hardlink result differs from authenticated source')


class ReuseInventory:
    def __init__(self, root, record, persist, *, required=False):
        from .cp3_archives import digest
        self.root = Path(root)
        self.document = {'archives': {}, 'listings': {}, 'api_pages': {}}
        path = self.root / INVENTORY
        if not path.exists():
            if required:
                raise FileNotFoundError('authorized CP3 reuse inventory is required')
            return
        raw = path.read_bytes()
        document = json.loads(raw)
        if not isinstance(document, dict) or any(not isinstance(document.get(key), dict)
                                                for key in self.document):
            raise ValueError('invalid registered reuse inventory schema')
        self.document = document
        binding = dict(path=INVENTORY, sha256=digest(raw), bytes=len(raw),
                       role='authorized immutable saved-source inventory')
        prior = [row for row in record.setdefault('source_registrations', []) if row['path'] == INVENTORY]
        if prior and prior != [binding]:
            raise ValueError('reuse inventory changed within input attempt')
        if not prior:
            record['source_registrations'].append(binding)
            persist()

    @staticmethod
    def _bytes(row, path_key='path', hash_key='sha256'):
        from .cp3_archives import digest
        if not isinstance(row, dict):
            raise ValueError('registered reuse entry must be an object')
        path = Path(row[path_key])
        expected = row[hash_key]
        if not path.is_absolute() or not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
            raise ValueError('saved source requires absolute path and SHA256')
        raw = path.read_bytes()
        if digest(raw) != expected:
            raise ValueError('registered saved-source hash mismatch: ' + str(path))
        return raw

    def has_archive(self, spec):
        return spec['relative_path'] in self.document['archives']

    def archive_source(self, spec):
        from .cp3_archives import verify_payload
        row = self.document['archives'].get(spec['relative_path'])
        if row is None:
            return None
        metadata = row.get('provenance_metadata', {})
        if not isinstance(metadata, dict):
            raise ValueError('saved archive provenance metadata must be an object')
        for key in ('relative_path', 'url'):
            if row.get(key, metadata.get(key)) != spec[key]:
                raise ValueError('saved archive source identity mismatch: ' + key)
        raw = self._bytes(row)
        checksum = self._bytes(row, 'checksum_path', 'checksum_sha256')
        verified = verify_payload(spec, raw, checksum)
        if verified['sha256'] != row['sha256'] or verified['checksum_sha256'] != row['checksum_sha256']:
            raise ValueError('saved archive differs from official CHECKSUM')
        return row, raw, checksum

    def listing_sources(self, prefix):
        rows = self.document['listings'].get(prefix)
        if rows is None:
            return None
        if not isinstance(rows, list) or not 1 <= len(rows) <= 4:
            raise ValueError('saved official listing requires one to four bounded pages')
        if any(not isinstance(row, dict) or row.get('prefix') != prefix for row in rows):
            raise ValueError('saved official listing prefix identity mismatch')
        return rows

    def listing_page(self, row, url):
        if row.get('url') != url:
            raise ValueError('saved official listing URL or pagination mismatch')
        return self._bytes(row)

    def api_source(self, url, parameters):
        row = self.document['api_pages'].get(url)
        if row is None:
            return None
        if row.get('url') != url or ('parameters' in row and row['parameters'] != parameters):
            raise ValueError('saved API source identity mismatch')
        return row, self._bytes(row)
