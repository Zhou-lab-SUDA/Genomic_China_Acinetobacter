"""Summarize longitudinal mouse nasal-colonisation measurements."""

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
        rank = rank_index + 1
        running = min(running, values[index] * len(values) / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def main() -> None:
    data = pd.read_csv(DATA / "nasal_colonisation_measurements.tsv", sep="\t", na_values=["NA"])
    summary = (
        data.groupby(["lineage", "day"], as_index=False)
        .agg(n_observations=("log10_cfu", "count"), mean=("log10_cfu", "mean"),
             sd=("log10_cfu", "std"), median=("log10_cfu", "median"),
             q1=("log10_cfu", lambda x: x.quantile(0.25)),
             q3=("log10_cfu", lambda x: x.quantile(0.75)))
    )
    summary.to_csv(DATA / "nasal_colonisation_summary.tsv", sep="\t", index=False)
    tests = []
    for day in sorted(data["day"].unique()):
        current = data.loc[data["day"] == day]
        a = current.loc[current["lineage"] == "ESL2.4", "log10_cfu"].dropna()
        b = current.loc[current["lineage"] == "ESL2.5", "log10_cfu"].dropna()
        result = mannwhitneyu(a, b, alternative="two-sided", method="auto")
        tests.append({"day": day, "n_esl2_4": len(a), "n_esl2_5": len(b),
                      "mann_whitney_u": result.statistic, "p_value": result.pvalue,
                      "method": "scipy_auto"})
    adjusted = bh_adjust([row["p_value"] for row in tests])
    for row, q_value in zip(tests, adjusted):
        row["bh_adjusted_p"] = q_value
    pd.DataFrame(tests).to_csv(DATA / "nasal_colonisation_tests.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
