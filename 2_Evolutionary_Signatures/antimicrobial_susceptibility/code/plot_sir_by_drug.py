"""Shared parsing, colour and multiple-testing helpers for susceptibility data.

The module also generates the S/I/R stacked-bar panel and is imported by
``plot_sir_resistance_rate.py`` for resistance-rate comparisons.
"""

from pathlib import Path
import os
import re

FIGURE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = FIGURE_DIR / "data"
os.environ.setdefault("MPLCONFIGDIR", str(FIGURE_DIR / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT = DATA_DIR / "SIR_raw_table.tsv"
OUTPUT_LONG = DATA_DIR / "SIR_cleaned_long_from_script.tsv"
OUTPUT_SUMMARY = DATA_DIR / "SIR_percentage_summary_from_script.tsv"
OUTPUT_TESTS = DATA_DIR / "SIR_lineage_tests_from_script.tsv"
OUTPUT_PDF = FIGURE_DIR / "antimicrobial_SIR_profiles.pdf"
OUTPUT_PREVIEW = FIGURE_DIR / "antimicrobial_SIR_profiles.png"

N_MONTE_CARLO = 200_000
SEED = 20260802
ORDER = ["S", "I", "R"]
COLORS = {"S": "#55A868", "I": "#F5BA1A", "R": "#EB6564"}


def pearson_stat(table: np.ndarray) -> float:
    expected = (
        table.sum(axis=1, keepdims=True)
        * table.sum(axis=0, keepdims=True)
        / table.sum()
    )
    return float(np.sum((table - expected) ** 2 / expected))


def conditional_monte_carlo_p(
    table: np.ndarray, rng: np.random.Generator
) -> tuple[float, float, int]:
    keep = table.sum(axis=0) > 0
    observed = table[:, keep].astype(float)
    statistic = pearson_stat(observed)
    column_totals = observed.sum(axis=0).astype(int)
    row1_n = int(observed[0].sum())
    sampled_row1 = rng.multivariate_hypergeometric(
        column_totals, row1_n, size=N_MONTE_CARLO
    ).astype(float)
    sampled_row2 = column_totals[None, :] - sampled_row1
    expected_row1 = row1_n * column_totals / column_totals.sum()
    expected_row2 = (column_totals.sum() - row1_n) * column_totals / column_totals.sum()
    simulated = (
        ((sampled_row1 - expected_row1) ** 2 / expected_row1).sum(axis=1)
        + ((sampled_row2 - expected_row2) ** 2 / expected_row2).sum(axis=1)
    )
    p_value = (1 + np.count_nonzero(simulated >= statistic - 1e-12)) / (
        N_MONTE_CARLO + 1
    )
    return statistic, float(p_value), observed.shape[1] - 1


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def stars(q: float) -> str:
    if q < 1e-4:
        return "****"
    if q < 1e-3:
        return "***"
    if q < 1e-2:
        return "**"
    if q < 0.05:
        return "*"
    return "n.s."


def short_name(drug: str) -> str:
    match = re.search(r"\(([^)]+)\)$", drug)
    return match.group(1) if match else drug


def main() -> None:
    wide = pd.read_csv(INPUT, sep="\t", dtype=str).fillna("")
    duplicate_rows = int(wide.duplicated().sum())
    wide = wide.drop_duplicates().copy()
    if set(wide["ESL_lineage"]) != {"2.4", "2.5"}:
        raise ValueError("Expected only ESL lineages 2.4 and 2.5")

    drug_columns = list(wide.columns[2:])
    long = wide.melt(
        id_vars=["Genome", "ESL_lineage"],
        value_vars=drug_columns,
        var_name="drug",
        value_name="SIR",
    )
    long = long.loc[long["SIR"].isin(ORDER)].copy()
    long.to_csv(OUTPUT_LONG, sep="\t", index=False)

    summary = (
        long.groupby(["drug", "ESL_lineage", "SIR"], observed=False)
        .size().rename("count").reset_index()
    )
    complete = pd.MultiIndex.from_product(
        [drug_columns, ["2.4", "2.5"], ORDER],
        names=["drug", "ESL_lineage", "SIR"],
    )
    summary = (
        summary.set_index(["drug", "ESL_lineage", "SIR"])
        .reindex(complete, fill_value=0).reset_index()
    )
    summary["n_valid"] = summary.groupby(["drug", "ESL_lineage"])["count"].transform("sum")
    summary["percent"] = 100 * summary["count"] / summary["n_valid"]
    summary.to_csv(OUTPUT_SUMMARY, sep="\t", index=False, float_format="%.8g")

    rng = np.random.default_rng(SEED)
    test_rows = []
    for drug in drug_columns:
        subset = summary.loc[summary["drug"] == drug]
        table = np.array([
            [int(subset.loc[(subset["ESL_lineage"] == lineage) &
                            (subset["SIR"] == category), "count"].iloc[0])
             for category in ORDER]
            for lineage in ["2.4", "2.5"]
        ])
        statistic, p_value, df = conditional_monte_carlo_p(table, rng)
        test_rows.append(
            {
                "drug": drug,
                "n_ESL2.4": int(table[0].sum()),
                "n_ESL2.5": int(table[1].sum()),
                "S_ESL2.4": table[0, 0], "I_ESL2.4": table[0, 1], "R_ESL2.4": table[0, 2],
                "S_ESL2.5": table[1, 0], "I_ESL2.5": table[1, 1], "R_ESL2.5": table[1, 2],
                "pearson_chi_square": statistic,
                "df_for_statistic": df,
                "monte_carlo_p": p_value,
            }
        )
    tests = pd.DataFrame(test_rows)
    tests["BH_q"] = bh_adjust(tests["monte_carlo_p"].to_numpy())
    tests["significance"] = tests["BH_q"].map(stars)
    tests["monte_carlo_replicates"] = N_MONTE_CARLO
    tests["seed"] = SEED
    tests["exact_duplicate_rows_removed"] = duplicate_rows
    tests.to_csv(OUTPUT_TESTS, sep="\t", index=False, float_format="%.8g")

    fig, axes = plt.subplots(4, 4, figsize=(11.2, 10.2), sharey=True)
    axes = axes.ravel()
    for ax, drug in zip(axes, drug_columns):
        bottoms = np.zeros(2)
        for category in ORDER:
            values = np.array([
                summary.loc[
                    (summary["drug"] == drug)
                    & (summary["ESL_lineage"] == lineage)
                    & (summary["SIR"] == category),
                    "percent",
                ].iloc[0]
                for lineage in ["2.4", "2.5"]
            ])
            ax.bar([0, 1], values, bottom=bottoms, width=0.64,
                   color=COLORS[category], edgecolor="white", linewidth=0.7,
                   label=category)
            bottoms += values

        result = tests.loc[tests["drug"] == drug].iloc[0]
        ax.plot([0, 0, 1, 1], [103, 105, 105, 103], color="#222222", lw=0.9)
        ax.text(0.5, 106.2, result["significance"], ha="center", va="bottom",
                fontsize=11, fontweight="bold")
        ax.set_title(short_name(drug), fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["ESL2.4", "ESL2.5"], rotation=20)
        ax.set_ylim(0, 113)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", width=0.9)

    for ax in axes[len(drug_columns):]:
        ax.axis("off")
    for row in range(4):
        axes[row * 4].set_ylabel("Percentage (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[key]) for key in ORDER]
    fig.legend(handles, ORDER, title="Phenotype", frameon=False, ncol=3,
               loc="upper center", bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=[0, 0, 1, 0.965], h_pad=1.5, w_pad=1.2)
    fig.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_PREVIEW, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
