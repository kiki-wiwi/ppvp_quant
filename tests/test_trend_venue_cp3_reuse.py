"""Synthetic aliases emulate separate bind-mount entries; no market data."""
from pathlib import Path
import hashlib
import pytest
from trend_hold_venue_study import cp3_reuse as reuse

def mounts(primary, alias):
    return [dict(id='1', device='8:17', root=Path('/'), point=primary),
            dict(id='2', device='8:17', root=Path('/data/worktree'), point=alias)]

def test_cross_mount_hardlink_uses_primary_entry_and_preserves_bytes(tmp_path, monkeypatch):
    primary=tmp_path/'volume'; real=primary/'data/worktree';real.mkdir(parents=True)
    alias=tmp_path/'repo_data';alias.symlink_to(real,target_is_directory=True)
    source=primary/'evidence.zip';source.write_bytes(b'synthetic authenticated zip bytes')
    target=alias/'raw/new.zip'
    monkeypatch.setattr(reuse,'PRIMARY_MOUNT',primary)
    monkeypatch.setattr(reuse,'_mount_records',lambda:mounts(primary,alias))
    original_link=reuse.os.link;calls=[]
    def only_primary(src,dst):
        assert Path(src).is_relative_to(primary)
        assert Path(dst).is_relative_to(primary)
        calls.append((str(src),str(dst)));original_link(src,dst)
    monkeypatch.setattr(reuse.os,'link',only_primary)
    reuse.hardlink_new(source,target)
    assert len(calls)==1
    assert target.samefile(source)
    assert target.read_bytes()==b'synthetic authenticated zip bytes'
    assert hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256(source.read_bytes()).digest()
    with pytest.raises(FileExistsError):reuse.hardlink_new(source,target)

def test_alias_to_alias_hardlink_maps_both_sides(tmp_path, monkeypatch):
    primary=tmp_path/'volume';real=primary/'data/worktree';real.mkdir(parents=True)
    alias=tmp_path/'repo_data';alias.symlink_to(real,target_is_directory=True)
    (real/'input').write_bytes(b'unchanged')
    monkeypatch.setattr(reuse,'PRIMARY_MOUNT',primary)
    monkeypatch.setattr(reuse,'_mount_records',lambda:mounts(primary,alias))
    reuse.hardlink_new(alias/'input',alias/'new/output')
    assert (alias/'new/output').samefile(real/'input')

def test_wrong_mapping_is_rejected_before_link(tmp_path, monkeypatch):
    primary=tmp_path/'volume';real=primary/'data/worktree';real.mkdir(parents=True)
    alias=tmp_path/'not_a_bind';alias.mkdir()
    (alias/'source').write_bytes(b'same bytes do not mean same inode')
    (real/'source').write_bytes(b'same bytes do not mean same inode')
    monkeypatch.setattr(reuse,'PRIMARY_MOUNT',primary)
    monkeypatch.setattr(reuse,'_mount_records',lambda:mounts(primary,alias))
    with pytest.raises(ValueError,match='same file'):
        reuse.hardlink_new(alias/'source',primary/'output')
    assert not (primary/'output').exists()

def test_mapping_cannot_escape_primary_volume(tmp_path, monkeypatch):
    primary=tmp_path/'volume';primary.mkdir()
    outside=tmp_path/'outside';outside.write_bytes(b'synthetic')
    (primary/'escape').symlink_to(outside)
    monkeypatch.setattr(reuse,'PRIMARY_MOUNT',primary)
    monkeypatch.setattr(reuse,'_mount_records',lambda:[dict(id='1',device='8:17',root=Path('/'),point=primary)])
    with pytest.raises(ValueError,match='escaped'):
        reuse.hardlink_new(primary/'escape',primary/'output')
    assert not (primary/'output').exists()
