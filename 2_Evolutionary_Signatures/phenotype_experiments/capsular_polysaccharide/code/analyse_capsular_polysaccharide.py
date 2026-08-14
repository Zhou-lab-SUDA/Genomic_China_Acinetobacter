"""Summarize capsular-polysaccharide measurements at isolate level."""

from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"


def comparison(frame: pd.DataFrame, column: str, left: str, right: str, name: str) -> dict:
    a = frame.loc[frame[column] == left, "strain_mean"]
    b = frame.loc[frame[column] == right, "strain_mean"]
    result = mannwhitneyu(a, b, alternative="two-sided", method="auto")
    return {
        "contrast": name,
        "left_group": left,
        "right_group": right,
        "n_left": len(a),
        "n_right": len(b),
        "mann_whitney_u": result.statistic,
        "two_sided_p": result.pvalue,
        "method": "scipy_auto",
    }


def main() -> None:
    raw = pd.read_csv(DATA / "capsular_polysaccharide_measurements.tsv", sep="\t")
    long = raw.melt(
        id_vars=["lineage", "strain", "kl_type"],
        value_vars=["replicate_1", "replicate_2"],
        var_name="assay_replicate",
        value_name="capsular_polysaccharide",
    )
    long.to_csv(DATA / "capsular_polysaccharide_long.tsv", sep="\t", index=False)
    strains = (
        long.groupby(["lineage", "strain", "kl_type"], as_index=False)
        .agg(n_measurements=("capsular_polysaccharide", "size"),
             strain_mean=("capsular_polysaccharide", "mean"),
             strain_sd=("capsular_polysaccharide", "std"))
    )
    strains.to_csv(DATA / "capsular_polysaccharide_strain_summary.tsv", sep="\t", index=False)
    summaries = []
    for grouping in ["lineage", "kl_type"]:
        for group, values in strains.groupby(grouping):
            summaries.append({
                "grouping": grouping,
                "group": group,
                "n_strains": len(values),
                "mean": values["strain_mean"].mean(),
                "sd": values["strain_mean"].std(),
                "median": values["strain_mean"].median(),
            })
    pd.DataFrame(summaries).to_csv(DATA / "capsular_polysaccharide_group_summary.tsv", sep="\t", index=False)
    tests = [
        comparison(strains, "lineage", "ESL2.4", "ESL2.5", "lineage"),
        comparison(strains, "kl_type", "KL2", "KL3", "KL2_vs_KL3"),
    ]
    pd.DataFrame(tests).to_csv(DATA / "capsular_polysaccharide_tests.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
