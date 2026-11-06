#!/usr/bin/env python3
"""Default is source-only registration check. Formal run needs a separate unlock."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    for mode in ('check','synthetic','real'):modes.add_argument('--'+mode,action='store_true')
    parser.add_argument('--output',type=Path,help='Synthetic only: new directory outside the repository')
    args=parser.parse_args(argv)
    from channel_random_study import scope
    if args.real:scope.require_real_authorization()
    from channel_random_study.data import verify_registered_assets,verify_environment,verify_unlock
    if args.output and not args.synthetic:parser.error('--output requires --synthetic')
    if not args.real and not args.synthetic:
        reg=verify_registered_assets();environment=verify_environment()
        print(json.dumps(dict(status='awaiting_formal_run_approval',real_data_unlocked=scope.REAL_DATA_UNLOCKED,
            protected_sources_verified=reg['protected_sources_verified'],new_files_verified=reg.get('new_files_verified',0),
            environment=environment,approved_plan_sha256=reg['approved_plan_sha256'])))
        return 0
    from channel_random_study.runtime import execute,write_json
    if args.synthetic:
        verify_registered_assets();verify_environment()
        if args.output is None or args.output.resolve().is_relative_to(ROOT):
            parser.error('synthetic output must be a new directory outside repository')
        from channel_random_study.synthetic import study_context
        execute(study_context(),args.output,{'source':'synthetic','market_reads':0})
    else:
        out=ROOT/'reports/channel_random_round/results'
        if out.exists():raise FileExistsError(out)
        metadata={}
        try:
            metadata=verify_unlock();verify_registered_assets();verify_environment()
            from channel_random_study.data import load_real_context
            ctx,inputs=load_real_context()
        except Exception as exc:
            out.mkdir(parents=True,exist_ok=False)
            write_json(out/'STOPPED.json',dict(stage='pre_match_input_verification',error=str(exc),
                starting_state=metadata,automatic_retry=False));raise
        execute(ctx,out,{**inputs,**metadata})
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr);raise SystemExit(1)
