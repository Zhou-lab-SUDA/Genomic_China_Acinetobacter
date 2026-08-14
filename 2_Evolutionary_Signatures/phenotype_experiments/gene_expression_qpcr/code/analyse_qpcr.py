"""Calculate isolate-level delta-Ct summaries from technical qPCR replicates."""

from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"


def bh_adjust(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [1.0] * len(values)
    running = 1.0
    for rank_index in range(len(values) - 1, -1, -1):
        index = order[rank_index]
        running = min(running, values[index] * len(values) / (rank_index + 1))
        adjusted[index] = min(1.0, running)
    return adjusted


def main() -> None:
    raw = pd.read_csv(DATA / "qpcr_ct_measurements.tsv", sep="\t")
    means = (
        raw.groupby(["target_as_reported", "target_standardized", "strain", "lineage"], as_index=False)
        .agg(n_technical_replicates=("ct", "size"), mean_ct=("ct", "mean"), sd_ct=("ct", "std"))
    )
    reference = means.loc[means["target_standardized"] == "16s", ["strain", "mean_ct"]].rename(columns={"mean_ct": "reference_16s_ct"})
    targets = means.loc[means["target_standardized"] != "16s"].merge(reference, on="strain", validate="many_to_one")
    targets["delta_ct"] = targets["mean_ct"] - targets["reference_16s_ct"]
    targets["relative_expression_2_neg_delta_ct"] = 2.0 ** (-targets["delta_ct"])
    targets.to_csv(DATA / "qpcr_isolate_expression.tsv", sep="\t", index=False)
    summary = (
        targets.groupby(["target_standardized", "lineage"], as_index=False)
        .agg(n_strains=("strain", "size"), mean_delta_ct=("delta_ct", "mean"),
             sd_delta_ct=("delta_ct", "std"), median_delta_ct=("delta_ct", "median"))
    )
    summary.to_csv(DATA / "qpcr_lineage_summary.tsv", sep="\t", index=False)
    tests = []
    for target in sorted(targets["target_standardized"].unique()):
        current = targets.loc[targets["target_standardized"] == target]
        a = current.loc[current["lineage"] == "ESL2.4", "delta_ct"]
        b = current.loc[current["lineage"] == "ESL2.5", "delta_ct"]
        result = mannwhitneyu(a, b, alternative="two-sided", method="exact")
        tests.append({"target": target, "n_esl2_4": len(a), "n_esl2_5": len(b),
                      "mann_whitney_u": result.statistic, "p_value": result.pvalue,
                      "method": "scipy_exact"})
    adjusted = bh_adjust([row["p_value"] for row in tests])
    for row, q_value in zip(tests, adjusted):
        row["bh_adjusted_p"] = q_value
    pd.DataFrame(tests).to_csv(DATA / "qpcr_lineage_tests.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
