"""Standardize published recombination and mutation event data for RF11.

The source worksheet uses one row per 10-kb chromosome window. This script
renames fields to unambiguous snake_case names, makes the coordinate convention
explicit, preserves missing values, and exports wide, long, block-segment and
block-summary TSV tables.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FIGURE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = FIGURE_DIR / "data"
INPUT = DATA_DIR / "source_recombination_mutation_events.xlsx"
OUTPUT_WIDE = DATA_DIR / "recombination_mutation_events_10kb.tsv"
OUTPUT_LONG = DATA_DIR / "recombination_mutation_events_10kb_long.tsv"
OUTPUT_SEGMENTS = DATA_DIR / "recombination_block_segments.tsv"
OUTPUT_SUMMARY = DATA_DIR / "recombination_mutation_block_summary.tsv"
OUTPUT_QUALITY = DATA_DIR / "source_data_quality.tsv"

WINDOW_SIZE_BP = 10_000
SERIES = {
    "rechmm_recombination_events": ("recombination", "RecHMM"),
    "gubbins_recombination_events": ("recombination", "Gubbins"),
    "clonalframeml_recombination_events": ("recombination", "ClonalFrameML"),
    "mutation_events": ("mutation", "Published mutation-event count"),
}


def load_source() -> pd.DataFrame:
    """Read the six populated columns from the `Rec&Mut` worksheet."""
    raw = pd.read_excel(INPUT, sheet_name="Rec&Mut", header=1, usecols="A:F")
    raw.columns = [
        "window_index_10kb",
        "rechmm_recombination_events",
        "mutation_events",
        "recombination_block",
        "gubbins_recombination_events",
        "clonalframeml_recombination_events",
    ]
    numeric_columns = ["window_index_10kb", *SERIES]
    for column in numeric_columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    if raw["window_index_10kb"].isna().any():
        raise ValueError("Window index contains missing or non-numeric values")
    raw["window_index_10kb"] = raw["window_index_10kb"].astype(int)
    return raw


def validate_windows(data: pd.DataFrame) -> None:
    """Require unique, sorted and consecutive 10-kb window indices."""
    if data["window_index_10kb"].duplicated().any():
        raise ValueError("Duplicate 10-kb window indices found")
    observed = data["window_index_10kb"].tolist()
    expected = list(range(observed[0], observed[-1] + 1))
    if observed != expected:
        raise ValueError("Window indices are not sorted and consecutive")
    if not data["recombination_block"].notna().all():
        raise ValueError("Recombination-block labels contain missing values")


def add_coordinates(data: pd.DataFrame) -> pd.DataFrame:
    """Add explicit zero-based, half-open chromosome coordinates."""
    data = data.copy()
    data["window_start_bp_0based"] = data["window_index_10kb"] * WINDOW_SIZE_BP
    data["window_end_bp_exclusive"] = data["window_start_bp_0based"] + WINDOW_SIZE_BP
    data["window_midpoint_mb"] = (
        data["window_start_bp_0based"] + WINDOW_SIZE_BP / 2
    ) / 1_000_000
    columns = [
        "window_index_10kb",
        "window_start_bp_0based",
        "window_end_bp_exclusive",
        "window_midpoint_mb",
        "recombination_block",
        "rechmm_recombination_events",
        "gubbins_recombination_events",
        "clonalframeml_recombination_events",
        "mutation_events",
    ]
    return data[columns]


def make_long_table(data: pd.DataFrame) -> pd.DataFrame:
    """Convert the event columns to one row per window and event series."""
    id_columns = [
        "window_index_10kb",
        "window_start_bp_0based",
        "window_end_bp_exclusive",
        "window_midpoint_mb",
        "recombination_block",
    ]
    long = data.melt(
        id_vars=id_columns,
        value_vars=list(SERIES),
        var_name="series_id",
        value_name="event_count",
    )
    long["event_class"] = long["series_id"].map(lambda value: SERIES[value][0])
    long["method_or_measure"] = long["series_id"].map(lambda value: SERIES[value][1])
    return long[
        [*id_columns, "event_class", "method_or_measure", "series_id", "event_count"]
    ]


def make_segments(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize contiguous runs, retaining the split HRB1 segments."""
    run_id = data["recombination_block"].ne(data["recombination_block"].shift()).cumsum()
    segments = (
        data.assign(_run_id=run_id)
        .groupby("_run_id", sort=True)
        .agg(
            recombination_block=("recombination_block", "first"),
            start_window_index_10kb=("window_index_10kb", "min"),
            end_window_index_10kb_inclusive=("window_index_10kb", "max"),
            start_bp_0based=("window_start_bp_0based", "min"),
            end_bp_exclusive=("window_end_bp_exclusive", "max"),
            n_windows=("window_index_10kb", "size"),
        )
        .reset_index(drop=True)
    )
    segments.insert(0, "segment_id", np.arange(1, len(segments) + 1))
    return segments


def make_block_summary(long: pd.DataFrame) -> pd.DataFrame:
    """Calculate descriptive event-count statistics by block and series."""
    summary = (
        long.groupby(
            ["recombination_block", "event_class", "method_or_measure", "series_id"],
            sort=False,
            dropna=False,
        )["event_count"]
        .agg(
            n_nonmissing="count",
            total_events="sum",
            mean_events_per_10kb="mean",
            median_events_per_10kb="median",
            sd_events_per_10kb="std",
            minimum_events_per_10kb="min",
            maximum_events_per_10kb="max",
            q1_events_per_10kb=lambda values: values.quantile(0.25),
            q3_events_per_10kb=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )
    window_counts = long.groupby(
        ["recombination_block", "series_id"], sort=False
    ).size()
    summary["n_missing"] = [
        int(window_counts[(row.recombination_block, row.series_id)] - row.n_nonmissing)
        for row in summary.itertuples()
    ]
    return summary


def make_quality_table(data: pd.DataFrame) -> pd.DataFrame:
    """Record completeness and range checks for every standardized field."""
    rows = []
    for column in data.columns:
        series = data[column]
        rows.append(
            {
                "field": column,
                "rows": len(series),
                "nonmissing": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "unique_nonmissing": int(series.nunique(dropna=True)),
                "minimum": series.min() if pd.api.types.is_numeric_dtype(series) else "",
                "maximum": series.max() if pd.api.types.is_numeric_dtype(series) else "",
                "notes": (
                    "Source workbook contains two blank mutation-event cells"
                    if column == "mutation_events" and series.isna().sum() == 2
                    else ""
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    data = load_source()
    validate_windows(data)
    data = add_coordinates(data)
    long = make_long_table(data)
    segments = make_segments(data)
    summary = make_block_summary(long)
    quality = make_quality_table(data)

    data.to_csv(OUTPUT_WIDE, sep="\t", index=False, float_format="%.8g")
    long.to_csv(OUTPUT_LONG, sep="\t", index=False, float_format="%.8g")
    segments.to_csv(OUTPUT_SEGMENTS, sep="\t", index=False)
    summary.to_csv(OUTPUT_SUMMARY, sep="\t", index=False, float_format="%.8g")
    quality.to_csv(OUTPUT_QUALITY, sep="\t", index=False)


if __name__ == "__main__":
    main()

