"""Validate every packaged TSV/CSV table and write a compact audit report."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "TABLE_VALIDATION.tsv"


def delimiter_for(path: Path) -> str:
    """Return the expected delimiter from the file extension."""
    return "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","


def main() -> None:
    records: list[dict[str, object]] = []
    candidates = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".tsv", ".csv"}
        and path.name not in {OUTPUT.name, "DATA_PROVENANCE.tsv"}
    )
    for path in candidates:
        record: dict[str, object] = {
            "file": path.relative_to(ROOT).as_posix(),
            "status": "ok",
            "rows": "",
            "columns": "",
            "column_names": "",
            "completely_empty_cells": "",
            "duplicate_rows": "",
            "error": "",
        }
        try:
            frame = pd.read_csv(path, sep=delimiter_for(path), low_memory=False)
            record.update(
                rows=len(frame),
                columns=len(frame.columns),
                column_names=" | ".join(map(str, frame.columns)),
                completely_empty_cells=int(frame.isna().sum().sum()),
                duplicate_rows=int(frame.duplicated().sum()),
            )
        except Exception as exc:  # The report records failures instead of stopping early.
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)

    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()

