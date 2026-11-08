#!/usr/bin/env python3
"""Read-only audit byte verification. No project imports, Git, network or extraction."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile


AUDIT = "reports/external_audit_20261108"
POINTER = b"version https://git-lfs.github.com/spec/v1"


def safe_name(name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or "\0" in name or any(p in ("", ".", "..") for p in name.split("/"))):
        raise ValueError("unsafe path: " + repr(name))
    return name


def local_file(root, name):
    candidate = root / safe_name(name)
    if not candidate.resolve().is_relative_to(root) or candidate.is_symlink():
        raise ValueError("unsafe path (symlink or escape): " + name)
    return candidate


def specification(item):
    if not isinstance(item, dict):
        raise ValueError("invalid file/object specification")
    if not isinstance(item.get("sha256"), str) or not re.fullmatch("[0-9a-f]{64}", item["sha256"]):
        raise ValueError("invalid sha256")
    if type(item.get("size")) is not int or item["size"] < 0:
        raise ValueError("invalid size")
    if "lfs_oid" in item and item["lfs_oid"] not in (item["sha256"], "sha256:" + item["sha256"]):
        raise ValueError("lfs_oid differs from expected sha256")


def check_stream(stream, item):
    digest, size, prefix = hashlib.sha256(), 0, b""
    while chunk := stream.read(1024 * 1024):
        if not prefix:
            prefix = chunk[:256]
        digest.update(chunk)
        size += len(chunk)
    if prefix.startswith(POINTER):
        raise ValueError("unmaterialized LFS pointer; download LFS content first")
    if size != item["size"] or digest.hexdigest() != item["sha256"]:
        raise ValueError("SHA256/size mismatch (observed " + digest.hexdigest() + ", " + str(size) + ")")


def check_file(root, item):
    path = local_file(root, item["path"])
    if not path.is_file():
        raise ValueError("missing or non-regular file")
    with path.open("rb") as stream:
        check_stream(stream, item)


def check_pack(root, pack, expected):
    seen = set()
    with tarfile.open(local_file(root, pack), mode="r|*") as archive:
        for member in archive:
            name = safe_name(member.name)
            if not member.isfile() or member.issparse():
                raise ValueError(name + ": non-regular tar member")
            if name in seen:
                raise ValueError(name + ": duplicate member")
            if name not in expected:
                raise ValueError(name + ": unexpected member")
            seen.add(name)
            try:
                with archive.extractfile(member) as stream:
                    check_stream(stream, expected[name])
            except (OSError, ValueError) as exc:
                raise ValueError(name + ": " + str(exc)) from exc
    missing = expected.keys() - seen
    if missing:
        raise ValueError("missing member: " + ", ".join(sorted(missing)))
    return len(seen)


def verify(root, metadata_only=False):
    root = root.resolve()
    counts = dict(snapshot_files=0, delivery_files=0, packs=0, objects=0, skipped_files=0)
    report = {"status": "FAIL", "counts": counts, "errors": []}

    def attempt(label, function):
        try:
            return True, function()
        except (OSError, ValueError, TypeError, KeyError, EOFError, tarfile.TarError) as exc:
            report["errors"].append({"path": label, "reason": str(exc)})
            return False, None

    def read_manifest(name):
        with local_file(root, AUDIT + "/" + name).open(encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ValueError("manifest must be a JSON object")
        return value

    manifests = {}
    for name in ("snapshot.json", "inputs/index.json", "delivery.json"):
        ok, value = attempt(AUDIT + "/" + name, lambda n=name: read_manifest(n))
        if ok:
            manifests[name] = value
    if len(manifests) != 3:
        return report

    def validate_metadata():
        snapshot, index, delivery = (manifests[n] for n in ("snapshot.json", "inputs/index.json", "delivery.json"))
        if not isinstance(snapshot.get("baseline_commit"), str) or not snapshot["baseline_commit"]:
            raise ValueError("snapshot.json: missing baseline_commit")
        report["baseline_commit"] = snapshot["baseline_commit"]
        for manifest, key in ((snapshot, "files"), (delivery, "files"), (index, "packs"), (index, "objects")):
            if not isinstance(manifest.get(key), list):
                raise ValueError("manifest requires list: " + key)
        packs, expected = {}, {}
        for item in index["packs"]:
            name = safe_name(item["path"])
            specification(item)
            if name in packs:
                raise ValueError(name + ": duplicate pack")
            packs[name], expected[name] = item, {}
        for item in index["objects"]:
            pack, member = safe_name(item["pack"]), safe_name(item["member"])
            specification(item)
            if pack not in packs:
                raise ValueError(pack + ": unknown pack")
            if member in expected[pack]:
                raise ValueError(pack + ": duplicate object member " + member)
            if not isinstance(item.get("aliases"), list) or any(not isinstance(a, str) for a in item["aliases"]):
                raise ValueError(member + ": invalid aliases")
            expected[pack][member] = item
        for label, manifest in (("snapshot", snapshot), ("delivery", delivery)):
            seen = set()
            for item in manifest["files"]:
                name = safe_name(item["path"])
                specification(item)
                if name in seen:
                    raise ValueError(name + ": duplicate " + label + " file")
                seen.add(name)
                if name in packs and any(item[k] != packs[name][k] for k in ("sha256", "size")):
                    raise ValueError(name + ": conflicting pack specification")
        return snapshot, index, delivery, packs, expected

    ok, validated = attempt("manifest metadata", validate_metadata)
    if not ok:
        return report
    snapshot, index, delivery, packs, expected = validated
    for label, entries in (("snapshot_files", snapshot["files"]), ("delivery_files", delivery["files"]), ("packs", index["packs"])):
        for item in entries:
            name = item["path"]
            if metadata_only and ("lfs_oid" in item or name in packs):
                counts["skipped_files"] += 1
                continue
            ok, _ = attempt(name, lambda i=item: check_file(root, i))
            counts[label] += int(ok)
    if not metadata_only:
        for pack in packs:
            if any(error["path"] == pack for error in report["errors"]):
                continue
            ok, total = attempt(pack, lambda p=pack: check_pack(root, p, expected[p]))
            if ok:
                counts["objects"] += total
    if not report["errors"]:
        report["status"] = "PARTIAL" if metadata_only else "PASS"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only", action="store_true", help="skip LFS payloads and input packs; reports PARTIAL, never PASS")
    args = parser.parse_args()
    report = verify(Path(__file__).resolve().parents[2], args.metadata_only)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
