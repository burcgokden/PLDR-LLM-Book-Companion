#!/usr/bin/env python3
"""Check maintained command paths and safe help/syntax routes, without acquisition."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = 'provenance/documented-commands.json'


def shipped(root, name, kind):
    path = root / name
    if Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Nonlocal ' + kind + ': ' + name)
    if path.resolve() != path.absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Aliased ' + kind + ': ' + name)
    if not (path.is_dir() if kind == 'working directory' else path.is_file()):
        raise ValueError('Missing ' + kind + ': ' + name)
    return path


def documented_tokens(text):
    blocks = re.findall(r'```(?:sh|bash)\n(.*?)```', text, re.S)
    commands = re.findall(r'(?<!`)`([^`\n]+)`(?!`)', text)
    for block in blocks:
        commands.extend(block.replace('\\\n', '').splitlines())
    result = []
    for command in commands:
        try:
            result.append(shlex.split(command))
        except ValueError:
            pass  # A prose code span need not be a shell command.
    return result


def verify(root=ROOT, run_safe=True):
    root = Path(root).resolve()
    inventory = json.loads((root / INVENTORY).read_text())
    if inventory['schema'] != 'pldr-documented-commands-v1':
        raise ValueError('Unsupported command inventory')
    records = []
    for entry in inventory['commands']:
        cwd = shipped(root, entry['cwd'], 'working directory')
        argv = entry['argv']
        if argv[0] not in {'python3', 'sh'}:
            raise ValueError('Unsupported external interpreter: ' + argv[0])
        index = entry['entry_point_index']
        if index < 1 or argv[index].startswith('-'):
            raise ValueError('Invalid local entry point')
        path = shipped(root, str(Path(entry['cwd']) / argv[index]), 'entry point')
        for placeholder in entry.get('placeholders', []):
            if placeholder not in argv or placeholder == argv[index]:
                raise ValueError('Invalid external-input placeholder')
        documents = []
        for name in entry['documents']:
            doc = shipped(root, name, 'documentation')
            if argv not in documented_tokens(doc.read_text()):
                raise ValueError('Documented command differs: ' + entry['id'] + ' in ' + name)
            documents.append({'path': name, 'sha256': hashlib.sha256(doc.read_bytes()).hexdigest()})
        safe = None
        if run_safe:
            if argv[0] == 'sh':
                safe = ['sh', '-n', argv[index]]
            elif entry.get('safe_help'):
                # Pass no acquisition arguments or placeholders to a producer.
                safe = [sys.executable, '-B', argv[index], '--help']
        route = {'status': 'not requested'}
        if safe:
            result = subprocess.run(safe, cwd=cwd, capture_output=True, text=True, timeout=30)
            route = {'kind': 'syntax' if argv[0] == 'sh' else 'help',
                     'returncode': result.returncode, 'output': result.stdout + result.stderr}
            if result.returncode:
                raise ValueError('Safe route failed: ' + entry['id'] + '\n' + route['output'])
        records.append({'id': entry['id'], 'cwd': entry['cwd'], 'argv': argv,
                        'external_interpreter': argv[0],
                        'entry_point': str(path.relative_to(root)),
                        'entry_point_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'external_input_placeholders': entry.get('placeholders', []),
                        'documents': documents, 'safe_check': route})
    return {'status': 'passed', 'commands': records,
            'scope': 'Maintained documentation, shipped paths and safe help/syntax only; no acquisition executed.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--static-only', action='store_true', help='Check paths and documentation without help/syntax calls')
    args = parser.parse_args()
    report = verify(run_safe=not args.static_only)
    output = ROOT / 'validation/documented-commands.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'commands': len(report['commands']), 'scope': report['scope']}))


if __name__ == '__main__':
    main()
