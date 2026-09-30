#!/usr/bin/env python3
"""Record intentional source edits separately from the immutable import record.

Run only after reviewing changes. This records bytes; it does not run validation.
"""
import hashlib
import json
from pathlib import Path
from verify_release import ROOT,MANIFEST,payload_paths

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    baseline=json.loads((ROOT/'provenance/upstream-baseline.json').read_text());changes=[]
    for component,info in baseline['sources'].items():
        for name,old in info['files'].items():
            p=ROOT/info['destination']/name;new=digest(p) if p.is_file() else None
            if old!=new:changes.append(dict(path=str(p.relative_to(ROOT)),imported_sha256=old,current_sha256=new,status='removed' if new is None else 'modified'))
    (ROOT/'provenance/book-changes.json').write_text(json.dumps(dict(schema='pldr-book-changes-v1',changes=changes,scope='Intentional included-code edits; immutable import record retained.'),indent=2)+'\n')
    d=ROOT/'companions/dynamics';m=json.loads((d/'scientific-manifest.json').read_text())
    m['files']={n:digest(d/n) for n in m['files'] if (d/n).is_file()}
    for p in (d/'PldrTrainingDynamics').glob('*.lean'):m['files'][str(p.relative_to(d))]=digest(p)
    m['payload_sha256']=hashlib.sha256(json.dumps(m['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (d/'scientific-manifest.json').write_text(json.dumps(m,sort_keys=True,indent=2)+'\n')
    # The nested manifest update itself is included in the book change record.
    for c in changes:
        if c['path']=='companions/dynamics/scientific-manifest.json':c['current_sha256']=digest(d/'scientific-manifest.json')
    (ROOT/'provenance/book-changes.json').write_text(json.dumps(dict(schema='pldr-book-changes-v1',changes=changes,scope='Intentional included-code edits; immutable import record retained.'),indent=2)+'\n')
    files={str(p.relative_to(ROOT)):digest(p) for p in payload_paths()}
    payload=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (ROOT/MANIFEST).write_text(json.dumps(dict(schema='pldr-book-release-v1',files=files,payload_sha256=payload),sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(files=len(files),payload_sha256=payload)))

if __name__=='__main__':main()
