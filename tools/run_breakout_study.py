#!/usr/bin/env python3
"""Default: verify source registration. Real execution remains locked."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--check',action='store_true')
    modes.add_argument('--synthetic',action='store_true')
    modes.add_argument('--real',action='store_true')
    parser.add_argument('--output',type=Path,help='Synthetic output directory only; must not exist')
    args=parser.parse_args(argv)
    from breakout_study import scope
    from breakout_study.data import verify_registered_assets
    if args.real:
        scope.require_real_authorization()  # Before any file or market read/output.
    if args.output and not args.synthetic:
        parser.error('--output is supported only with --synthetic')
    if not args.real and not args.synthetic:
        reg=verify_registered_assets()
        print(json.dumps({'real_data_unlocked':scope.REAL_DATA_UNLOCKED,
            'legacy_frozen_files_verified':len(reg['legacy_frozen_files']),
            'new_files_verified':reg.get('new_files_verified',0),
            'approved_plan_sha256':reg['approved_plan_sha256'],'status':'awaiting_real_run_approval'}))
        return 0
    from breakout_study.runtime import execute,write_json
    if args.synthetic:
        verify_registered_assets()
        if args.output is None: parser.error('--synthetic requires --output outside the repository')
        if args.output.resolve().is_relative_to(ROOT): parser.error('synthetic output must be outside repository')
        from breakout_study.synthetic import study_context
        execute(study_context(),args.output,{'source':'synthetic','market_reads':0})
    else:
        out=ROOT/'reports/channel_vs_vah_round/results'
        if out.exists(): raise FileExistsError(out)
        commit=None
        try:
            reg=verify_registered_assets()
            status=subprocess.run(['git','status','--porcelain'],cwd=ROOT,check=True,capture_output=True,text=True).stdout
            if status: raise ValueError('formal execution requires a clean working tree')
            commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip()
            from breakout_study.data import load_real_context
            ctx,metadata=load_real_context()
        except Exception as exc:
            out.mkdir(parents=True,exist_ok=False)
            write_json(out/'STOPPED.json',{'stage':'input_verification','commit':commit,'error':str(exc),'automatic_retry':False})
            raise
        execute(ctx,out,{**metadata,'commit':commit,'starting_worktree_status':status,'registration':reg})
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr)
        raise SystemExit(1)
