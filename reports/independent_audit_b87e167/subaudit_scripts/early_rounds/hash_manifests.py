"""Verify archived output-hash manifests byte-for-byte.
Usage: python -I hash_manifests.py <repo_root>"""
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check(manifest, base, entries):
    ok = bad = miss = 0
    bads = []
    for rel, want in entries:
        p = base / rel
        if not p.exists():
            miss += 1; bads.append(("MISSING", rel)); continue
        if sha(p) == want:
            ok += 1
        else:
            bad += 1; bads.append(("MISMATCH", rel))
    print(f"{manifest}: ok={ok} mismatch={bad} missing={miss}")
    for b in bads[:15]:
        print("   ", b)


m1 = root / "reports/channel_vs_vah_round/results/provenance/original_output_hashes.json"
j = json.loads(m1.read_text())
check(m1.relative_to(root), root / "reports/channel_vs_vah_round/results", [(k, v["sha256"]) for k, v in j.items()])
m2 = root / "reports/trend_dip_limit_round/results/provenance/original_output_hashes.json"
j = json.loads(m2.read_text())
check(m2.relative_to(root), root / "reports/trend_dip_limit_round/results", [(k, v["sha256"]) for k, v in j.items()])
m3 = root / "reports/trend_dip_limit_round/results/delivery_hashes.json"
j = json.loads(m3.read_text())
check(m3.relative_to(root), root, list(j["sha256"].items()))
m4 = root / "reports/channel_random_round/results/output_hashes.json"
j = json.loads(m4.read_text())
print("channel_random output_hashes keys:", [k for k in j if k != "sha256"])
check(m4.relative_to(root), root / "reports/channel_random_round/results", list(j["sha256"].items()))
print("sha of channel_random output_hashes.json", sha(m4))
# Impulse CP3: summary provenance
s = json.loads((root / "reports/impulse_vs_box_round/cp3/summary.json").read_text())
prov = s.get("provenance", {})
print("impulse provenance keys", list(prov.keys())[:30])
