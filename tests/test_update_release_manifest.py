"""Exercise nested additions, renames, deletions and manifest-driven export."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from update_release_manifest import refresh
from verify_release import verify


class ManifestUpdateTests(unittest.TestCase):
    def test_complete_nested_inventory_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / 'companions/dynamics'
            (nested / 'scripts').mkdir(parents=True)
            (root / 'provenance').mkdir()
            shutil.copyfile(ROOT / 'companions/dynamics/scripts/verify_scientific_manifest.py', nested / 'scripts/verify_scientific_manifest.py')
            (nested / 'scientific-manifest.json').write_text(json.dumps({'files': {}, 'payload_sha256': ''}))
            (nested / 'old.py').write_text('print(1)\n')
            import hashlib
            old = hashlib.sha256((nested / 'old.py').read_bytes()).hexdigest()
            (root / 'provenance/upstream-baseline.json').write_text(json.dumps({'sources': {'dynamics': {'destination': 'companions/dynamics', 'files': {'old.py': old}}}}))
            refresh(root)
            (nested / 'old.py').rename(nested / 'renamed.py')
            (nested / 'OtherProofs').mkdir()
            (nested / 'OtherProofs/New.lean').write_text('example : True := True.intro\n')
            (nested / 'new_test.py').write_text('def test_new(): assert True\n')
            refresh(root)
            self.assertEqual(verify(root)['status'], 'passed')
            manifest = json.loads((nested / 'scientific-manifest.json').read_text())
            self.assertIn('renamed.py', manifest['files'])
            self.assertIn('new_test.py', manifest['files'])
            self.assertIn('OtherProofs/New.lean', manifest['files'])
            self.assertNotIn('old.py', manifest['files'])
            rows = json.loads((root / 'provenance/book-changes.json').read_text())['changes']
            changes = {r['path']: r['status'] for r in rows if 'path' in r}
            removed = [r for r in rows if r['status'] == 'removed']
            self.assertEqual(len(removed), 1)
            self.assertEqual(removed[0]['removed_path_sha256'], hashlib.sha256(b'companions/dynamics/old.py').hexdigest())
            self.assertEqual(changes['companions/dynamics/renamed.py'], 'added')
            export = root / 'validation/export'
            for name in [*manifest['files'], 'scientific-manifest.json']:
                out = export / name
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(nested / name, out)
            spec = importlib.util.spec_from_file_location('export_verifier', export / 'scripts/verify_scientific_manifest.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertEqual(module.verify(export)['status'], 'passed')
            (nested / 'new_test.py').unlink()
            refresh(root)
            self.assertNotIn('new_test.py', json.loads((nested / 'scientific-manifest.json').read_text())['files'])
            before = (root / 'provenance/current-release.json').read_bytes()
            refresh(root)
            self.assertEqual(before, (root / 'provenance/current-release.json').read_bytes())
