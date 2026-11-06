from pathlib import Path
from datetime import datetime,timezone
import sys,json,hashlib,subprocess,traceback,platform,importlib.metadata
ROOT=Path('/home/ubuntu/.codex/worktrees/channel-vah-prereg/ppvp_quant')
AUDIT=Path(__file__).parent
sys.path.insert(0,str(ROOT))
state={'stage':'input_preflight','started_at':datetime.now(timezone.utc).isoformat(),'matching_runs':0}
try:
    base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
    assert base=='e9dcc33d93225af314028668ff4200ad2bf81f36' and not status
    assert not (ROOT/'reports/trend_dip_limit_round/results').exists()
    from dip_limit_study import scope
    assert scope.REAL_DATA_UNLOCKED is False
    from dip_limit_study.data import verify_registered_assets
    from breakout_study import data as B
    reg=verify_registered_assets();old=B.verify_registered_assets()
    pinned=json.loads((ROOT/'reports/trend_dip_limit_round/implementation_hashes.json').read_text())
    environment={k:platform.python_version() if k=='python' else importlib.metadata.version(k) for k in pinned['environment']}
    assert environment==pinned['environment'],(environment,pinned['environment'])
    state.update(baseline=base,starting_worktree_status=status,environment=environment,protected_sources=reg['protected_sources_verified'],new_files=reg['new_files_verified'],legacy_frozen_files=len(old['legacy_frozen_files']),direction_one_files=old['new_files_verified'],legacy_frozen_key=reg['legacy_frozen_key'])
    # User authorized input verification. Reuse the existing unlocked, time-filtered
    # data validator only; no strategy, feature or performance engine is invoked.
    ctx,metadata=B.load_real_context(ROOT)
    import numpy as np
    manifest=json.loads((ROOT/'data/manifest_impulse_1m.json').read_text())
    state.update(status='passed',input=metadata,first_minute=str(ctx.index[0]),last_minute=str(ctx.index[-1]),bad_minutes=int(ctx.bad.sum()),perp_content_sha256=manifest['caches']['perp_1m']['content_sha256'],input_method='existing cache, no zip rebuild; both input loaders apply lower and upper IS predicates',funding_reference_sha256=hashlib.sha256(ctx.funding_rate.tobytes()).hexdigest(),source_paths={n:str((ROOT/n).resolve()) for n in ['data/processed_1m/BTCUSDT_perp_1m_IS.parquet','data/processed/BTCUSDT_funding.parquet']},forbidden_inputs_read=False)
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
except Exception as exc:
    state.update(status='STOPPED',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),automatic_retry=False)
    raise
finally:
    state['ended_at']=datetime.now(timezone.utc).isoformat()
    (AUDIT/'preflight.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(state,indent=2),flush=True)
