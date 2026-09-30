#!/usr/bin/env python3
"""Authenticate the entire standalone code payload and its allowed layout."""
from pathlib import Path
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[1]
IGNORED={'.git','.lake','build','validation','__pycache__','.pytest_cache','.venv'}
MANIFEST='provenance/current-release.json'
FORBIDDEN_SUFFIXES={'.tex','.pdf','.bib','.cls','.sty','.bbl','.aux','.blg','.toc','.synctex'}


def payload_paths(root=ROOT):
    return sorted(p for p in root.rglob('*') if p.is_file() and
                  not any(x in IGNORED for x in p.relative_to(root).parts) and
                  str(p.relative_to(root)) != MANIFEST and p.suffix not in {'.pyc','.olean','.ilean'})


def verify(root=ROOT):
    root=Path(root);m=json.loads((root/MANIFEST).read_text());errors=[]
    paths=payload_paths(root);actual={str(p.relative_to(root)):p for p in paths}
    if set(actual)!=set(m['files']):errors.append('payload file inventory differs')
    for name,p in actual.items():
        if p.is_symlink() or p.stat().st_nlink!=1:errors.append('non-independent file: '+name)
        if p.suffix.lower() in FORBIDDEN_SUFFIXES:errors.append('document artifact: '+name)
        if name in m['files'] and hashlib.sha256(p.read_bytes()).hexdigest()!=m['files'][name]:errors.append('hash differs: '+name)
        if p.suffix in {'.py','.sh','.md','.json','.lean','.toml','.yml','.txt'}:
            if re.search(r'/(?:tf/projects|home/(?!runner)[^/]+|root|mnt|Users)/',p.read_text(errors='replace')):
                errors.append('machine-local path: '+name)
    if (root/'.gitmodules').exists():errors.append('submodules are not standalone sources')
    if any(p.name=='.git' for p in (root/'companions').rglob('.git')):errors.append('nested git metadata')
    canonical=json.dumps(m['files'],sort_keys=True,separators=(',',':')).encode()
    if hashlib.sha256(canonical).hexdigest()!=m['payload_sha256']:errors.append('manifest payload digest differs')
    if errors:raise ValueError('\n'.join(errors))
    return dict(status='passed',files=len(paths),payload_sha256=m['payload_sha256'],scope='Standalone code bytes, document exclusions and portable paths.')

if __name__=='__main__':print(json.dumps(verify(),indent=2))
