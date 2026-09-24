"""Build the public source-only code.zip without results or model assets."""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = (
    "planning", "pairing_analysis", "certificate_efficiency",
    "sampled_interventions", "shared_anchor", "finite_catalog",
    "identification_audit", "language_models", "scripts",
)
ROOT_FILES = (
    ".gitignore", "README.md", "EXPERIMENTS.md", "requirements.txt",
    "requirements-model.txt", "requirements-figures.txt",
)
SOURCE_SUFFIXES = {".py", ".md", ".json", ".txt", ".sha256", ".yaml", ".yml"}
EXCLUDED_PARTS = {
    "__pycache__", ".pytest_cache", ".venv", "results", "outputs",
    "artifacts", "checkpoints", "raw", "data", "build", "dist",
}
SECRET_PATTERNS = (
    re.compile(rb"C:[\\/]Users[\\/]", re.I),
    re.compile(rb"ghp_[A-Za-z0-9]{20,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
)


def select_files() -> list[Path]:
    files = [ROOT / name for name in ROOT_FILES]
    for dirname in SOURCE_DIRS:
        directory = ROOT / dirname
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT)
            if any(part in EXCLUDED_PARTS or part.startswith("model_files_")
                   for part in rel.parts[:-1]):
                continue
            if path.suffix in SOURCE_SUFFIXES:
                files.append(path)
    for path in files:
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f"Missing or unsafe source: {path}")
    return sorted(set(files), key=lambda p: p.relative_to(ROOT).as_posix())


def build(output: Path) -> tuple[int, int]:
    paths = select_files()
    output = output.resolve()
    if output.is_relative_to(ROOT) or output.name.lower() != "code.zip":
        raise ValueError("Write code.zip outside the source tree")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for path in paths:
            data = path.read_bytes()
            if any(pattern.search(data) for pattern in SECRET_PATTERNS):
                raise ValueError(f"Potential private path or credential: {path}")
            rel = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo("code/" + rel, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=9)
    with zipfile.ZipFile(temp) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(paths):
            raise ValueError("Archive integrity check failed")
    temp.replace(output)
    return len(paths), output.stat().st_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "code.zip")
    args = parser.parse_args()
    count, size = build(args.output)
    print(f"Built {args.output}: {count} source files, {size} bytes")
