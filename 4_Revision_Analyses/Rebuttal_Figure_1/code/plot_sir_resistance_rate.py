"""Plot antimicrobial S/I/R profiles and compare resistance rates.

Input records are deduplicated isolate-level susceptibility calls.  For each
drug, the statistical comparison is a two-sided Fisher exact test of resistant
versus non-resistant isolates (R versus S + I), followed by Benjamini-Hochberg
correction across the 14 drugs.
"""

from pathlib import Path
import math
import os

FIGURE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = FIGURE_DIR / "data"
os.environ.setdefault("MPLCONFIGDIR", str(FIGURE_DIR / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_sir_by_drug import ORDER, COLORS, bh_adjust, short_name, stars


INPUT = DATA_DIR / "SIR_raw_table.tsv"
OUTPUT_TESTS = DATA_DIR / "SIR_resistance_rate_tests_recalculated.tsv"
OUTPUT_PDF = FIGURE_DIR / "RF1_susceptibility_panel_reproduced.pdf"
OUTPUT_PREVIEW = FIGURE_DIR / "RF1_susceptibility_panel_reproduced.png"


def log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    row1 = a + b
    resistant = a + c
    total = a + b + c + d
    lower = max(0, row1 - (total - resistant))
    upper = min(row1, resistant)
    support = np.arange(lower, upper + 1, dtype=int)
    log_probs = np.array([
        log_comb(resistant, int(x))
        + log_comb(total - resistant, row1 - int(x))
        - log_comb(total, row1)
        for x in support
    ])
    probs = np.exp(log_probs - log_probs.max())
    probs /= probs.sum()
    observed_probability = probs[np.where(support == a)[0][0]]
    return float(probs[probs <= observed_probability * (1 + 1e-12)].sum())


def odds_ratio(a: int, b: int, c: int, d: int) -> float:
    if b * c == 0:
        if a * d == 0:
            return float("nan")
        return float("inf")
    return a * d / (b * c)


def main() -> None:
    wide = pd.read_csv(INPUT, sep="\t", dtype=str).fillna("").drop_duplicates()
    drug_columns = list(wide.columns[2:])
    long = wide.melt(
        id_vars=["Genome", "ESL_lineage"], value_vars=drug_columns,
        var_name="drug", value_name="SIR"
    )
    long = long.loc[long["SIR"].isin(ORDER)].copy()

    summary = (
        long.groupby(["drug", "ESL_lineage", "SIR"]).size()
        .rename("count").reset_index()
    )
    complete = pd.MultiIndex.from_product(
        [drug_columns, ["2.4", "2.5"], ORDER],
        names=["drug", "ESL_lineage", "SIR"]
    )
    summary = (
        summary.set_index(["drug", "ESL_lineage", "SIR"])
        .reindex(complete, fill_value=0).reset_index()
    )
    summary["n_valid"] = summary.groupby(["drug", "ESL_lineage"])["count"].transform("sum")
    summary["percent"] = 100 * summary["count"] / summary["n_valid"]

    rows = []
    for drug in drug_columns:
        counts = {}
        for lineage in ["2.4", "2.5"]:
            part = summary.loc[
                (summary["drug"] == drug) & (summary["ESL_lineage"] == lineage)
            ].set_index("SIR")["count"]
            counts[lineage] = {key: int(part[key]) for key in ORDER}
        a = counts["2.4"]["R"]
        b = counts["2.4"]["S"] + counts["2.4"]["I"]
        c = counts["2.5"]["R"]
        d = counts["2.5"]["S"] + counts["2.5"]["I"]
        rows.append(
            {
                "drug": drug,
                "R_ESL2.4": a, "nonR_ESL2.4": b,
                "R_ESL2.5": c, "nonR_ESL2.5": d,
                "resistance_percent_ESL2.4": 100 * a / (a + b),
                "resistance_percent_ESL2.5": 100 * c / (c + d),
                "difference_pp_ESL2.5_minus_ESL2.4": 100 * c / (c + d) - 100 * a / (a + b),
                "odds_ratio_ESL2.4_vs_ESL2.5": odds_ratio(a, b, c, d),
                "fisher_exact_p": fisher_two_sided(a, b, c, d),
            }
        )
    tests = pd.DataFrame(rows)
    tests["BH_q"] = bh_adjust(tests["fisher_exact_p"].to_numpy())
    tests["significance"] = tests["BH_q"].map(stars)
    tests.to_csv(OUTPUT_TESTS, sep="\t", index=False, float_format="%.8g")

    fig, axes = plt.subplots(2, 7, figsize=(15.2, 5.2), sharey=True)
    axes = axes.ravel()
    for ax, drug in zip(axes, drug_columns):
        bottoms = np.zeros(2)
        for category in ORDER:
            values = np.array([
                summary.loc[
                    (summary["drug"] == drug)
                    & (summary["ESL_lineage"] == lineage)
                    & (summary["SIR"] == category), "percent"
                ].iloc[0]
                for lineage in ["2.4", "2.5"]
            ])
            ax.bar([0, 1], values, bottom=bottoms, width=0.64,
                   color=COLORS[category], edgecolor="white", linewidth=0.7)
            bottoms += values
        result = tests.loc[tests["drug"] == drug].iloc[0]
        ax.plot([0, 0, 1, 1], [103, 105, 105, 103], color="#222222", lw=0.9)
        ax.text(0.5, 106.2, result["significance"], ha="center", va="bottom",
                fontsize=11, fontweight="bold")
        ax.set_title(short_name(drug), fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["ESL2.4", "ESL2.5"], rotation=0)
        ax.set_ylim(0, 113)
        ax.set_box_aspect(1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", width=0.9)

    for ax in axes[len(drug_columns):]:
        ax.axis("off")
    for row in range(2):
        axes[row * 7].set_ylabel("Percentage (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[key]) for key in ORDER]
    fig.legend(handles, ORDER, title="Phenotype", frameon=False, ncol=3,
               loc="upper center", bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=[0, 0, 1, 0.93], h_pad=1.2, w_pad=0.9)
    fig.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_PREVIEW, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
