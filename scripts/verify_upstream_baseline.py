"""Verify current imported-source lineage and optionally authenticate its original record."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]


def verify(root=ROOT,historical_record=None):
    root=Path(root)
    baseline=json.loads((root/'provenance/upstream-baseline.json').read_text())
    if baseline.get('schema')!='pldr-book-import-lineage-v2':raise ValueError('Unsupported import lineage')
    history=baseline['historical_record']
    if (not history['repository'].startswith('https://github.com/') or
            not re.fullmatch('[0-9a-f]{40}',history['commit']) or
            not re.fullmatch('[0-9a-f]{64}',history['sha256'])):
        raise ValueError('Incomplete historical source identity')
    changes=json.loads((root/'provenance/book-changes.json').read_text())['changes']
    modifications={r['path']:r for r in changes if r['status']=='modified'}
    records={}
    for component,source in baseline['sources'].items():
        folder=root/source['destination'];unchanged=0;modified=0
        if len(source['files'])+len(source['excluded_original_entries'])!=source['original_file_count']:
            raise ValueError('Incomplete original import accounting')
        if set(source['files'])!=set(source['git_modes']) or set(source['files'])!=set(source['original_path_sha256']):
            raise ValueError('Import identity inventory differs')
        for name,old in source['files'].items():
            relative=Path(name)
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe public source path')
            path=folder/relative
            if not path.is_file() or path.resolve()!=path.absolute() or path.stat().st_nlink!=1:
                raise ValueError('Missing or non-independent public source '+name)
            mode='100755' if path.stat().st_mode&0o111 else '100644'
            if mode!=source['git_modes'][name]:raise ValueError('Source mode differs '+name)
            current=hashlib.sha256(path.read_bytes()).hexdigest()
            if current==old:unchanged+=1
            else:
                change=modifications.get(path.relative_to(root).as_posix())
                if not change or change['imported_sha256']!=old or change['current_sha256']!=current:
                    raise ValueError('Unrecorded source modification '+name)
                modified+=1
        records[component]=dict(original_files=source['original_file_count'],unchanged=unchanged,
                                modified=modified,excluded=len(source['excluded_original_entries']),
                                source_commit=source['source_commit'])
    authenticated=False
    if historical_record is not None:
        raw=Path(historical_record).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=history['sha256']:raise ValueError('Historical record digest differs')
        original=json.loads(raw)
        for component,source in baseline['sources'].items():
            old=original['sources'][component]
            expected={(hashlib.sha256(p.encode()).hexdigest(),h,old['git_modes'][p]) for p,h in old['files'].items()}
            actual={(source['original_path_sha256'][p],h,source['git_modes'][p]) for p,h in source['files'].items()}
            actual|={(r['path_sha256'],r['imported_sha256'],r['git_mode']) for r in source['excluded_original_entries']}
            if actual!=expected:raise ValueError('Historical import lineage does not reconstruct')
        authenticated=True
    return dict(status='passed',sources=records,historical_record_authenticated=authenticated,
                scope='Current public files and recorded import changes; optional historical record is separately hash-authenticated. This is not a claim that modified files retain their imported bytes.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical-record',type=Path)
    args=parser.parse_args()
    print(json.dumps(verify(historical_record=args.historical_record),indent=2))
