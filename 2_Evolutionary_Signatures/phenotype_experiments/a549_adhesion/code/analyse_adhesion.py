"""Summarize A549 adhesion with isolate as the independent unit."""

from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"


def main() -> None:
    measurements = pd.read_csv(
        DATA / "selected_strains_MOI10_MOI20_adhesion.tsv", sep="\t",
        dtype={"strain": str},
    ).rename(columns={"MOI": "moi"})
    strain_summary = (
        measurements.groupby(["moi", "lineage", "strain"], as_index=False)
        .agg(
            n_experiments=("adhesion_percent", "size"),
            mean_adhesion_percent=("adhesion_percent", "mean"),
            sd_adhesion_percent=("adhesion_percent", "std"),
        )
    )
    strain_summary.to_csv(DATA / "adhesion_strain_summary.tsv", sep="\t", index=False)

    lineage_summary = (
        strain_summary.groupby(["moi", "lineage"], as_index=False)
        .agg(
            n_strains=("strain", "size"),
            mean_of_strain_means=("mean_adhesion_percent", "mean"),
            sd_of_strain_means=("mean_adhesion_percent", "std"),
            median_of_strain_means=("mean_adhesion_percent", "median"),
        )
    )
    lineage_summary.to_csv(DATA / "adhesion_lineage_summary.tsv", sep="\t", index=False)

    tests = []
    for moi in sorted(strain_summary["moi"].unique()):
        subset = strain_summary.loc[strain_summary["moi"] == moi]
        values24 = subset.loc[subset["lineage"] == "ESL2.4", "mean_adhesion_percent"]
        values25 = subset.loc[subset["lineage"] == "ESL2.5", "mean_adhesion_percent"]
        result = mannwhitneyu(values24, values25, alternative="two-sided", method="exact")
        tests.append(
            {
                "moi": moi,
                "n_strains_esl2_4": len(values24),
                "n_strains_esl2_5": len(values25),
                "mann_whitney_u": result.statistic,
                "two_sided_p": result.pvalue,
                "method": "scipy_exact",
            }
        )
    pd.DataFrame(tests).to_csv(DATA / "adhesion_lineage_tests.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
