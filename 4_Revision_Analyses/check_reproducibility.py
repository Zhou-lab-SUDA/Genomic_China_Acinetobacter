"""Compare frozen result tables with outputs regenerated from packaged code."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "REPRODUCIBILITY_CHECKS.tsv"

PAIRS = [
    ("RF1 resistance tests", "Rebuttal_Figure_1/data/SIR_resistance_rate_tests.tsv", "Rebuttal_Figure_1/data/SIR_resistance_rate_tests_recalculated.tsv"),
    ("RF2 date observed subsamples", "Rebuttal_Figure_2/data/date_randomisation_observed_subsamples.tsv", "Rebuttal_Figure_2/reproduced_date_randomisation/date_randomisation_observed_subsamples.tsv"),
    ("RF2 date replicates", "Rebuttal_Figure_2/data/date_randomisation_replicates.tsv", "Rebuttal_Figure_2/reproduced_date_randomisation/date_randomisation_replicates.tsv"),
    ("RF2 date summaries", "Rebuttal_Figure_2/data/date_randomisation_summary.tsv", "Rebuttal_Figure_2/reproduced_date_randomisation/date_randomisation_summary.tsv"),
    ("RF3 adhesion summary", "Rebuttal_Figure_3/data/selected_strains_MOI10_MOI20_logCFU_summary.tsv", "Rebuttal_Figure_3/data/selected_strains_MOI10_MOI20_logCFU_summary_recalculated.tsv"),
    ("RF5 date replicates", "Rebuttal_Figure_5/data/date_randomisation_replicates.tsv", "Rebuttal_Figure_5/reproduced_date_randomisation/date_randomisation_replicates.tsv"),
    ("RF5 date summaries", "Rebuttal_Figure_5/data/date_randomisation_summary.tsv", "Rebuttal_Figure_5/reproduced_date_randomisation/date_randomisation_summary.tsv"),
    ("RF7 hotspot figure table", "Rebuttal_Figure_7/data/mutation_hotspots_10kb_figure_data.tsv", "Rebuttal_Figure_7/data/mutation_hotspots_10kb_figure_data_recalculated.tsv"),
    ("RF8 Storey FDR", "Rebuttal_Figure_8/data/storey_fdr_results.tsv", "Rebuttal_Figure_8/data/storey_fdr_results_recalculated.tsv"),
    ("RF9 sampling replicates", "Rebuttal_Figure_9/data/sampling_bias_replicates.tsv", "Rebuttal_Figure_9/reproduced_sampling_bias/rttd_sampling_bias_replicates.tsv"),
    ("RF9 sampling summary", "Rebuttal_Figure_9/data/sampling_bias_summary.tsv", "Rebuttal_Figure_9/reproduced_sampling_bias/rttd_sampling_bias_summary.tsv"),
    ("RF10 degree statistics", "Rebuttal_Figure_10/data/degree_scatter_all_replicates.tsv", "Rebuttal_Figure_10/data/degree_scatter_recalculated.tsv"),
    ("RF12 annual replicates", "Rebuttal_Figure_12/data/annual_clade_proportion_replicates.tsv", "Rebuttal_Figure_12/reproduced_clade_composition/annual_clade_proportion_replicates.tsv"),
    ("RF12 annual summary", "Rebuttal_Figure_12/data/annual_clade_proportion_summary.tsv", "Rebuttal_Figure_12/reproduced_clade_composition/annual_clade_proportion_summary.tsv"),
    ("RF12 total summary", "Rebuttal_Figure_12/data/total_clade_proportion_summary.tsv", "Rebuttal_Figure_12/reproduced_clade_composition/total_clade_proportion_summary.tsv"),
]


def compare(reference: Path, regenerated: Path) -> tuple[str, int, int, float, str]:
    """Compare dimensions, columns, text cells and numeric values."""
    if not reference.exists() or not regenerated.exists():
        return "missing", 0, 0, float("nan"), "One or both files are absent"
    left = pd.read_csv(reference, sep="\t", low_memory=False)
    right = pd.read_csv(regenerated, sep="\t", low_memory=False)
    if left.shape != right.shape:
        return "different", len(left), len(right), float("nan"), f"Shape {left.shape} versus {right.shape}"
    if list(left.columns) != list(right.columns):
        return "different", len(left), len(right), float("nan"), "Column names or order differ"

    max_difference = 0.0
    for column in left.columns:
        left_numeric = pd.to_numeric(left[column], errors="coerce")
        right_numeric = pd.to_numeric(right[column], errors="coerce")
        numeric_mask = left_numeric.notna() | right_numeric.notna()
        if numeric_mask.any():
            left_values = left_numeric[numeric_mask].astype(float).to_numpy()
            right_values = right_numeric[numeric_mask].astype(float).to_numpy()
            if not np.allclose(
                left_values, right_values,
                rtol=1e-5, atol=5e-6, equal_nan=True,
            ):
                diff = np.nanmax(np.abs(left_values - right_values))
                return "different", len(left), len(right), float(diff), f"Numeric mismatch in {column}"
            finite_pairs = np.isfinite(left_values) & np.isfinite(right_values)
            if finite_pairs.any():
                diff = np.max(np.abs(left_values[finite_pairs] - right_values[finite_pairs]))
                max_difference = max(max_difference, float(diff))
        text_mask = ~numeric_mask
        if text_mask.any():
            a = left.loc[text_mask, column].fillna("").astype(str)
            b = right.loc[text_mask, column].fillna("").astype(str)
            if not a.equals(b):
                return "different", len(left), len(right), max_difference, f"Text mismatch in {column}"
    return "match", len(left), len(right), max_difference, ""


def check_rf11_workbook() -> dict[str, object]:
    """Verify that RF11 standardization preserves all six source fields."""
    reference = ROOT / "Rebuttal_Figure_11/data/source_recombination_mutation_events.xlsx"
    standardized = ROOT / "Rebuttal_Figure_11/data/recombination_mutation_events_10kb.tsv"
    source = pd.read_excel(reference, sheet_name="Rec&Mut", header=1, usecols="A:F")
    output = pd.read_csv(standardized, sep="\t")
    source.columns = [
        "window_index_10kb", "rechmm_recombination_events", "mutation_events",
        "recombination_block", "gubbins_recombination_events",
        "clonalframeml_recombination_events",
    ]
    columns = list(source.columns)
    status = "match"
    note = ""
    max_difference = 0.0
    if len(source) != len(output):
        status = "different"
        note = f"Row count {len(source)} versus {len(output)}"
    else:
        for column in columns:
            if column == "recombination_block":
                if not source[column].fillna("").astype(str).equals(
                    output[column].fillna("").astype(str)
                ):
                    status, note = "different", "Block labels differ"
                    break
            else:
                left = pd.to_numeric(source[column], errors="coerce").to_numpy(float)
                right = pd.to_numeric(output[column], errors="coerce").to_numpy(float)
                if not np.allclose(left, right, rtol=0, atol=0, equal_nan=True):
                    status, note = "different", f"Values differ in {column}"
                    break
        expected_start = output["window_index_10kb"] * 10_000
        expected_end = expected_start + 10_000
        if status == "match" and not (
            output["window_start_bp_0based"].equals(expected_start)
            and output["window_end_bp_exclusive"].equals(expected_end)
        ):
            status, note = "different", "Derived coordinate columns are inconsistent"
    return {
        "check": "RF11 workbook standardization",
        "status": status,
        "reference_file": reference.relative_to(ROOT).as_posix(),
        "regenerated_file": standardized.relative_to(ROOT).as_posix(),
        "reference_rows": len(source),
        "regenerated_rows": len(output),
        "maximum_numeric_difference": max_difference,
        "note": note,
    }


def main() -> None:
    rows = []
    for label, reference, regenerated in PAIRS:
        status, reference_rows, regenerated_rows, max_difference, note = compare(
            ROOT / reference, ROOT / regenerated
        )
        rows.append(
            {
                "check": label,
                "status": status,
                "reference_file": reference,
                "regenerated_file": regenerated,
                "reference_rows": reference_rows,
                "regenerated_rows": regenerated_rows,
                "maximum_numeric_difference": max_difference,
                "note": note,
            }
        )
    rows.append(check_rf11_workbook())
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
