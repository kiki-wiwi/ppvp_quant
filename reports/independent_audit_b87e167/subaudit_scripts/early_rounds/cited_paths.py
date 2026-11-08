"""Extract repo-relative paths cited in in-scope documents and check existence.
Usage: python -I cited_paths.py <repo_root>"""
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
docs = [
    "docs/phase_logs/PHASE0_data_engineering.md", "docs/phase_logs/PHASE1_engine.md", "docs/phase_logs/PHASE2_baseline.md",
    "reports/audit_round/audit_report.md", "reports/audit_round/decision_report.md", "reports/audit_round/experiment_plan.md",
    "reports/audit_round/codex_gpt61sol_high_diagnosis.md", "reports/audit_round/audit_evidence/README.md",
    "docs/phase_logs/IMPULSE_CP1_data.md", "docs/phase_logs/IMPULSE_CP2_engine_and_plan.md", "docs/phase_logs/IMPULSE_CP3_results.md",
    "reports/impulse_vs_box_round/experiment_plan_impulse.md",
    "docs/phase_logs/BREAKOUT_IS_results.md", "docs/phase_logs/BREAKOUT_implementation_and_synthetic.md",
    "reports/channel_vs_vah_round/approved_plan.md", "reports/channel_vs_vah_round/README.md", "reports/channel_vs_vah_round/results/README.md",
    "docs/phase_logs/DIP_LIMIT_IS_results.md", "docs/phase_logs/DIP_LIMIT_implementation_and_synthetic.md",
    "reports/trend_dip_limit_round/approved_plan.md", "reports/trend_dip_limit_round/implementation_plan.md",
    "reports/trend_dip_limit_round/README.md", "reports/trend_dip_limit_round/results/README.md",
    "docs/phase_logs/CHANNEL_RANDOM_results.md", "docs/phase_logs/CHANNEL_RANDOM_implementation_and_synthetic.md",
    "reports/channel_random_round/approved_plan.md", "reports/channel_random_round/README.md",
    "reports/external_audit_20261108/RESEARCH_MAP.md",
]
pat_md = re.compile(r"\]\(([^)#\s]+)")
pat_code = re.compile(r"`([A-Za-z0-9_./\-]+\.(?:py|json|csv|md|txt|png|parquet|gz|npy|npz|log|tar\.xz))`")
pat_bare = re.compile(r"(?<![\w/])((?:reports|docs|tools|data|ppvp_audit|impulse_study|breakout_study|dip_limit_study|channel_random_study)/[A-Za-z0-9_./\-]+)")
missing = {}
checked = 0
for d in docs:
    p = root / d
    if not p.exists():
        print("DOC MISSING", d); continue
    text = p.read_text(encoding="utf-8")
    cands = set()
    for m in pat_md.finditer(text):
        t = m.group(1)
        if t.startswith("http"):
            continue
        cands.add(("rel", t))
    for pat in (pat_code, pat_bare):
        for m in pat.finditer(text):
            cands.add(("abs", m.group(1).rstrip(".,;:)")))
    for kind, t in sorted(cands):
        checked += 1
        options = []
        if kind == "rel":
            options.append((p.parent / t).resolve())
        options.append((root / t).resolve())
        # also relative to the doc dir
        options.append((p.parent / t).resolve())
        # also relative to results dir/cp3 etc for bare names
        ok = any(o.exists() for o in options)
        if not ok:
            # try glob for bare file names (no slash)
            if "/" not in t:
                hits = list(root.glob(f"**/{t}"))
                if hits:
                    continue
            missing.setdefault(d, []).append(t)
print("checked", checked)
for d, ts in missing.items():
    print("==", d)
    for t in ts:
        print("   MISSING:", t)
