"""Build a source-only or evidence-inclusive ZIP from reviewed allowlists."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def checked_path(name):
    rel = PurePosixPath(name)
    prohibited = {"__pycache__", ".pytest_cache", ".venv", "build", "dist", ".git"}
    if rel.is_absolute() or ".." in rel.parts or prohibited.intersection(rel.parts) or ':' in name or '\\' in name:
        raise ValueError('Unsafe allowlist entry: '+name)
    path = ROOT/name
    if not path.is_file() or not path.resolve().is_relative_to(ROOT):
        raise ValueError('Missing or external artifact: '+name)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--include-evidence',action='store_true',
                        help='Add verified v6/v5 results and recovered historical assets; never add model weights.')
    args = parser.parse_args()
    cfg = json.loads((ROOT / "CODE_INDEX.json").read_text(encoding="utf-8"))
    names = sorted(cfg["included_files"])
    if len(names) != len(set(names)):
        raise ValueError("Duplicate allowlist entry")
    prohibited = {"__pycache__", ".pytest_cache", ".venv", "results", "artifacts", "build", "dist"}
    for name in names:
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts or prohibited.intersection(rel.parts):
            raise ValueError("Unsafe/non-source allowlist entry: " + name)
        checked_path(name)
        if name in cfg["retained_locally_excluded_from_delivery"]:
            raise ValueError("Excluded legacy file in allowlist: " + name)
    manifest = json.loads((ROOT / "oral_planner_v6/manifest.json").read_text())
    for name, expected in manifest["source_hashes"].items():
        if digest(ROOT / "oral_planner_v6" / name) != expected:
            raise ValueError("Frozen source changed: " + name)
    for name in names:
        if name.endswith(".sha256"):
            p = ROOT / name
            if digest(p.with_suffix(".json")) != p.read_text().strip().split()[0]:
                raise ValueError("Checksum mismatch: " + name)
    recovered_sources = json.loads((ROOT/'RECOVERED_SOURCES.json').read_text(encoding='utf-8'))
    for row in recovered_sources['files']:
        if row['path'] not in names or digest(checked_path(row['path']))!=row['sha256']:
            raise ValueError('Recovered source mismatch: '+row['path'])
    if args.include_evidence:
        evidence = json.loads((ROOT/'RECOVERED_ASSETS.json').read_text(encoding='utf-8'))['files']
        bundles = json.loads((ROOT/'RESULT_BUNDLES.json').read_text(encoding='utf-8'))
        evidence += [dict(path=name,sha256=h) for name,h in bundles['files'].items()]
        evidence += json.loads((ROOT/'FIGURE_ASSETS.json').read_text(encoding='utf-8'))['files']
        for row in evidence:
            name = row['path']
            if digest(checked_path(name))!=row['sha256']:
                raise ValueError('Evidence hash mismatch: '+name)
            if name not in names:
                names.append(name)
        names.sort()
    payload = {name: digest(ROOT / name) for name in names}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(ROOT / name, "code/" + name)
        archive.writestr("code/DELIVERY_SHA256.json", json.dumps(payload, indent=2))
    print(json.dumps({"files": len(names) + 1, "includes_evidence":args.include_evidence,"bytes": args.output.stat().st_size,
                      "sha256": digest(args.output), "output": str(args.output)}))


if __name__ == "__main__":
    main()
