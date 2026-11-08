"""Dedicated-volume routing and start-time capacity checks; no market access."""
from pathlib import Path
import json
import os
import shutil
import subprocess

VOLUME=Path('/mnt/ppvp-backtests')
VOLUME_UUID='d60bc8c8-11ef-4540-8d13-e2327feac6a1'
MIN_FREE_BYTES=40*2**30
REQUIRED_LOCATIONS=('data','reports','lfs_objects','lfs_temp','scratch','temp','cache')

def validate_measurements(rows,*,expected_device,uuid,mounted):
    if not mounted or uuid!=VOLUME_UUID:raise ValueError('registered dedicated volume is not mounted')
    if set(rows)!=set(REQUIRED_LOCATIONS):raise ValueError('incomplete actual storage destination inventory')
    for label,row in rows.items():
        if row['device_id']!=expected_device:raise ValueError('storage path is on wrong filesystem: '+label)
        if row['free_bytes']<MIN_FREE_BYTES:raise ValueError('actual destination has less than 40 GiB: '+label)
    return {'status':'passed','uuid':uuid,'required_free_bytes':MIN_FREE_BYTES,'locations':rows}

def verify_storage(root):
    root=Path(root)
    if not os.path.ismount(VOLUME):raise ValueError('dedicated backtest volume is not mounted; no root-disk fallback')
    mounted=json.loads(subprocess.check_output(['findmnt','--json','--target',str(VOLUME),
        '--output','TARGET,UUID,FSTYPE'],text=True))['filesystems'][0]
    if mounted['target']!=str(VOLUME) or mounted['fstype']!='ext4':raise ValueError('wrong dedicated volume mount')
    common=Path(subprocess.check_output(['git','rev-parse','--git-common-dir'],cwd=root,text=True).strip())
    if not common.is_absolute():common=(root/common).resolve()
    paths={'data':root/'data','reports':root/'reports','lfs_objects':common/'lfs/objects',
        'lfs_temp':common/'lfs/tmp','scratch':VOLUME/'scratch','temp':VOLUME/'tmp','cache':VOLUME/'cache'}
    rows={}
    for label,path in paths.items():
        if not path.is_dir():raise ValueError('missing actual storage destination: '+label)
        rows[label]={'path':str(path),'device_id':path.stat().st_dev,'free_bytes':shutil.disk_usage(path).free}
    result=validate_measurements(rows,expected_device=VOLUME.stat().st_dev,uuid=mounted['uuid'],mounted=True)
    # Called before Pandas/Arrow temporary files are created by the entrypoint.
    os.environ.update(TMPDIR=str(VOLUME/'tmp'),TMP=str(VOLUME/'tmp'),TEMP=str(VOLUME/'tmp'),
        XDG_CACHE_HOME=str(VOLUME/'cache'),MPLCONFIGDIR=str(VOLUME/'cache/matplotlib'))
    import tempfile
    tempfile.tempdir=str(VOLUME/'tmp')
    return result
