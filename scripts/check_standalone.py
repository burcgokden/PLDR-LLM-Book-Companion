#!/usr/bin/env python3
"""Run book scientific checks in an export with original code paths denied."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from verify_release import payload_paths,verify


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--blocked-root',type=Path,action='append',default=[])
    p.add_argument('--data-repo',type=Path)
    a=p.parse_args();verify();records=[]
    with tempfile.TemporaryDirectory(prefix='pldr-book-export-') as tmp:
        base=Path(tmp);code=base/'code';guard=base/'guard';guard.mkdir()
        for src in [*payload_paths(),ROOT/'provenance/current-release.json']:
            dest=code/src.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        relocated_data=None
        if a.data_repo:
            source_data=a.data_repo.resolve();relocated_data=base/'data'
            names=list(json.loads((source_data/'manifest.json').read_text())['files'])+['manifest.json']
            for name in names:
                dest=relocated_data/name;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source_data/name,dest)
        blocked=[str(ROOT),*[str(x.resolve()) for x in a.blocked_root]]
        if a.data_repo:blocked.append(str(a.data_repo.resolve()))
        (guard/'sitecustomize.py').write_text('import os,sys\nB='+repr(blocked)+'\n'
          'def audit(event,args):\n'
          ' if event in {"open","os.listdir","os.scandir","os.chdir"} and args and isinstance(args[0],(str,bytes,os.PathLike)):\n'
          '  p=os.path.realpath(os.fsdecode(args[0]))\n'
          '  if any(p==b or p.startswith(b+os.sep) for b in B):raise PermissionError("Original source access blocked")\n'
          'sys.addaudithook(audit)\n')
        env={k:v for k,v in os.environ.items() if k not in {'PYTHONPATH','PLDR_DATA_ROOT','MODEL_RG_DATA_ROOT','PLDR_RG_DATA_ROOT','PLDR_ROW_DATA_ROOT','PLDR_INPUT_MANIFEST'}}
        env.update(PYTHONPATH=str(guard),PLDR_READ_GUARD=str(guard),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        jobs=[('guard',[sys.executable,'-c','from pathlib import Path\ntry:Path('+repr(str(ROOT/'README.md'))+').read_bytes()\nexcept PermissionError:print("blocked")\nelse:raise SystemExit(1)']),
              ('integrity',[sys.executable,'-B','scripts/check.py','integrity']),
              ('scientific',[sys.executable,'-B','scripts/check.py','scientific']),
              ('resources',[sys.executable,'-B','scripts/check.py','resources']),
              ('synthetic-rg',[sys.executable,'-B','companions/dynamics/scripts/run_source.py','rg','scripts/run_synthetic_rg.py','--output',str(base/'rg.json')]),
              ('worker-dispatch',[sys.executable,'-B','companions/dynamics/scripts/run_source.py','model','scripts/run_cache_state_transfer.py','--validate-worker'])]
        if a.data_repo:jobs.append(('evidence',[sys.executable,'-B','scripts/check.py','evidence','--data-repo',str(relocated_data)]))
        for name,cmd in jobs:
            result=subprocess.run(cmd,cwd=code,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=1200)
            records.append(dict(check=name,returncode=result.returncode,output=result.stdout.replace(str(base),'<export>')))
            print(name,result.returncode,flush=True)
            if result.returncode:print(result.stdout[-5000:],flush=True);break
    report=dict(status='passed' if len(records)==len(jobs) and all(r['returncode']==0 for r in records) else 'failed',
                checks=records,scope='Fresh source export, blocked original reads with negative control, scientific suites, resources and finite routes; no full acquisition.')
    (ROOT/'validation').mkdir(exist_ok=True)
    (ROOT/'validation/standalone.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['status']=='passed' else 1

if __name__=='__main__':raise SystemExit(main())
