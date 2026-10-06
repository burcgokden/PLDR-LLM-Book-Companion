#!/usr/bin/env python3
"""Record current source bytes separately from authenticated historical imports.

This records complete current inventories; it does not run validation.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
from verify_release import ROOT, MANIFEST, payload_paths


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refresh(root=ROOT):
    root = Path(root)
    nested = root / 'companions/dynamics'
    spec = importlib.util.spec_from_file_location('nested_scientific_manifest', nested / 'scripts/verify_scientific_manifest.py')
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    path = nested / verifier.MANIFEST
    manifest = json.loads(path.read_text())
    manifest['files'] = verifier.payload_files(nested)
    manifest['payload_sha256'] = verifier.canonical(manifest['files'])
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    verifier.verify(nested)

    baseline = json.loads((root / 'provenance/upstream-baseline.json').read_text())
    changes = []
    for info in baseline['sources'].values():
        folder = root / info['destination']
        for name, old in info['files'].items():
            path = folder / name
            current = digest(path) if path.is_file() else None
            if old != current:
                row = dict(imported_sha256=old, current_sha256=current,
                           status='removed' if current is None else 'modified')
                if current is None:
                    row['removed_path_sha256'] = hashlib.sha256(path.relative_to(root).as_posix().encode()).hexdigest()
                else:
                    row['path'] = path.relative_to(root).as_posix()
                changes.append(row)
        # New names, including rename destinations, are explicit provenance entries.
        for path in payload_paths(folder):
            name = path.relative_to(folder).as_posix()
            if name not in info['files']:
                changes.append(dict(path=path.relative_to(root).as_posix(), imported_sha256=None,
                                    current_sha256=digest(path), status='added'))
    changes.sort(key=lambda item: item.get('path', item.get('removed_path_sha256', '')))
    (root / 'provenance/book-changes.json').write_text(json.dumps(dict(
        schema='pldr-book-changes-v2', changes=changes,
        scope='Current public paths and removed-path digests compared with authenticated import lineage; original path spellings remain in the historical record.'), indent=2) + '\n')
    files = {path.relative_to(root).as_posix(): digest(path) for path in payload_paths(root)}
    payload = verifier.canonical(files)
    (root / MANIFEST).write_text(json.dumps(dict(schema='pldr-book-release-v1', files=files,
                                              payload_sha256=payload), sort_keys=True, indent=2) + '\n')
    return dict(files=len(files), payload_sha256=payload)


if __name__ == '__main__':
    print(json.dumps(refresh()))
