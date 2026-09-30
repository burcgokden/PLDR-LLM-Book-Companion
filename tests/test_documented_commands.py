"""Documentation failures must be detected before any acquisition can run."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('documented_commands', Path(__file__).resolve().parents[1] / 'scripts/check_documented_commands.py')
commands = importlib.util.module_from_spec(spec)
spec.loader.exec_module(commands)


def fixture(root):
    (root / 'provenance').mkdir()
    (root / 'component/scripts').mkdir(parents=True)
    (root / 'docs').mkdir()
    (root / 'component/scripts/reconstruct.sh').write_text('exit 99\n')
    (root / 'docs/run.md').write_text('From `component/`:\n```sh\nsh scripts/reconstruct.sh \\\n  RAW FRESH\n```\n')
    inventory = {'schema': 'pldr-documented-commands-v1', 'commands': [{
        'id': 'reconstruction', 'cwd': 'component',
        'argv': ['sh', 'scripts/reconstruct.sh', 'RAW', 'FRESH'],
        'entry_point_index': 1, 'placeholders': ['RAW', 'FRESH'],
        'documents': ['docs/run.md']}]}
    (root / commands.INVENTORY).write_text(json.dumps(inventory))


def test_shell_example_is_only_syntax_checked_and_placeholders_are_not_files(tmp_path):
    fixture(tmp_path)
    report = commands.verify(tmp_path)
    assert report['status'] == 'passed'
    assert report['commands'][0]['safe_check']['kind'] == 'syntax'
    assert not (tmp_path / 'component/RAW').exists()
    assert not (tmp_path / 'component/FRESH').exists()


def test_missing_shipped_entry_point_is_rejected(tmp_path):
    fixture(tmp_path)
    (tmp_path / 'component/scripts/reconstruct.sh').unlink()
    with pytest.raises(ValueError, match='Missing entry point'):
        commands.verify(tmp_path)


def test_documentation_typo_is_rejected(tmp_path):
    fixture(tmp_path)
    doc = tmp_path / 'docs/run.md'
    doc.write_text(doc.read_text().replace('reconstruct.sh', 'absent.sh'))
    with pytest.raises(ValueError, match='Documented command differs'):
        commands.verify(tmp_path)


def test_wrong_working_directory_is_rejected(tmp_path):
    fixture(tmp_path)
    path = tmp_path / commands.INVENTORY
    inventory = json.loads(path.read_text())
    inventory['commands'][0]['cwd'] = '.'
    path.write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match='Missing entry point'):
        commands.verify(tmp_path)


def test_symlink_is_not_a_standalone_entry_point(tmp_path):
    fixture(tmp_path)
    script = tmp_path / 'component/scripts/reconstruct.sh'
    script.unlink()
    script.symlink_to(tmp_path / 'docs/run.md')
    with pytest.raises(ValueError, match='Aliased entry point'):
        commands.verify(tmp_path)
