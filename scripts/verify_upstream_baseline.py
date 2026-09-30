"""Verify the original imported payloads without reading the source repositories."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "provenance/upstream-baseline.json").read_text())
    errors = []
    records = {}
    for component, source in manifest["sources"].items():
        folder = ROOT / source["destination"]
        expected = source["files"]
        for name, digest in expected.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                errors.append(f"{component}: invalid path {name}")
                continue
            path = folder / relative
            if path.is_symlink() or not path.is_file():
                errors.append(f"{component}: missing or non-regular {name}")
                continue
            if any(parent.is_symlink() for parent in path.parents if parent != ROOT):
                errors.append(f"{component}: symlinked parent of {name}")
            if path.stat().st_nlink != 1:
                errors.append(f"{component}: multiply linked {name}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                errors.append(f"{component}: changed {name}")
            mode = "100755" if path.stat().st_mode & 0o111 else "100644"
            if mode != source["git_modes"][name]:
                errors.append(f"{component}: changed executable mode {name}")
        nested_git = list(folder.rglob(".git"))
        if nested_git:
            errors.append(f"{component}: nested Git metadata")
        records[component] = {
            "files_checked": len(expected),
            "lean_source_files": source["lean_source_files"],
            "source_commit": source["source_commit"],
        }
    if (ROOT / ".gitmodules").exists():
        errors.append("Root Git submodules are not part of the standalone baseline")
    report = {
        "status": "failed" if errors else "passed",
        "sources": records,
        "errors": errors,
        "scope": "Original imported bytes and independent-file layout only; no formal build or scientific audit is claimed.",
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
