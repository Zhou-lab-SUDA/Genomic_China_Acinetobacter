"""Create a size and SHA-256 manifest for the rebuttal-figure package."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "FILE_MANIFEST.tsv"


def category(path: Path) -> str:
    """Classify a packaged file by its role."""
    if path.suffix.lower() == ".pdf":
        return "figure"
    if "code" in path.parts or path.suffix.lower() == ".py":
        return "code"
    if "data" in path.parts or path.suffix.lower() in {".tsv", ".csv", ".xlsx", ".nexus"}:
        return "data"
    return "documentation"


def sha256(path: Path) -> str:
    """Hash a file without loading the complete file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    rows = []
    for path in sorted(item for item in ROOT.rglob("*") if item.is_file()):
        if path == OUTPUT or any(part in {"__pycache__", ".mplconfig"} for part in path.parts):
            continue
        rows.append(
            {
                "file": path.relative_to(ROOT).as_posix(),
                "category": category(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

