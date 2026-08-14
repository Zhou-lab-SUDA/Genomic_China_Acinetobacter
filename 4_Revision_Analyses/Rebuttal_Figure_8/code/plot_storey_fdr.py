"""Recalculate Storey FDR q-values and plot RF8 city associations.

Nominal permutation P values are adjusted with a fixed lambda of 0.5.  The
recalculated q-values are exported so every plotted value is auditable.
"""

from pathlib import Path
import os

FIGURE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = FIGURE_DIR / "data"
os.environ.setdefault("MPLCONFIGDIR", str(FIGURE_DIR / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT = DATA_DIR / "city_ESL25_spearman_permutation.tsv"
OUTPUT_PDF = FIGURE_DIR / "RF8_storey_fdr_reproduced.pdf"
OUTPUT_TABLE = DATA_DIR / "storey_fdr_results_recalculated.tsv"
COMPARISON_TABLE = DATA_DIR / "city_association_multiple_testing_recalculated.tsv"


def storey_qvalues(pvalues: np.ndarray, lam: float = 0.5) -> tuple[np.ndarray, float]:
    """Storey FDR q-values with a fixed lambda and monotone step-up adjustment."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    pi0 = min(1.0, np.sum(p > lam) / (m * (1.0 - lam)))
    order = np.argsort(p)
    ranked = p[order]
    adjusted = pi0 * m * ranked / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    q = np.empty(m, dtype=float)
    q[order] = adjusted
    return q, pi0


def main() -> None:
    df = pd.read_csv(INPUT, sep="\t")
    df = df.loc[df["filter"].eq("all_cities")].copy()
    df = df.rename(columns={"permutation_p": "nominal_p"})
    if len(df) != 21 or df["Variable"].duplicated().any():
        raise ValueError("Expected exactly 21 unique all-city variables")
    df["Storey_FDR_q_recalculated"], pi0 = storey_qvalues(
        df["nominal_p"].to_numpy(), lam=0.5
    )
    df["raw_significant"] = df["nominal_p"] < 0.05
    df["storey_significant"] = df["Storey_FDR_q_recalculated"] < 0.05
    df.to_csv(OUTPUT_TABLE, sep="\t", index=False, float_format="%.6g")

    # Repair the earlier figure-comparison table while retaining its manually
    # curated display correlations and variable order.
    comparison = pd.read_csv(DATA_DIR / "city_association_multiple_testing.tsv", sep="\t")
    corrected = df[
        ["Variable", "nominal_p", "BH_q", "Storey_FDR_q_recalculated"]
    ].rename(
        columns={
            "BH_q": "BH_adjusted_q_corrected",
            "Storey_FDR_q_recalculated": "Storey_FDR_q_lambda0.5_corrected",
        }
    )
    comparison = comparison.drop(
        columns=["nominal_p", "BH_adjusted_q", "Storey_FDR_q_lambda0.5"],
        errors="ignore",
    ).merge(corrected, on="Variable", how="left", validate="one_to_one")
    comparison = comparison.rename(
        columns={
            "BH_adjusted_q_corrected": "BH_adjusted_q",
            "Storey_FDR_q_lambda0.5_corrected": "Storey_FDR_q_lambda0.5",
        }
    )
    comparison.to_csv(COMPARISON_TABLE, sep="\t", index=False, float_format="%.6g")

    x = np.arange(len(df))
    raw_y = -np.log10(df["nominal_p"].to_numpy())
    adj_y = -np.log10(df["Storey_FDR_q_recalculated"].to_numpy())

    fig, ax = plt.subplots(figsize=(12.0, 5.2))
    for xi, y_raw, y_adj in zip(x, raw_y, adj_y):
        ax.plot([xi, xi], [y_adj, y_raw], color="#D7D7D7", lw=1.4, zorder=1)

    ax.scatter(x, raw_y, s=48, color="#69C1C5", edgecolor="none",
               label=r"$P_{\mathrm{raw}}$", zorder=3)
    ax.scatter(x, adj_y, s=48, color="#F39A18", edgecolor="none",
               label=r"$q_{\mathrm{Storey\ FDR}}$", zorder=3)
    ax.axhline(-np.log10(0.05), color="red", lw=1.0, ls="--")
    ax.text(-0.45, -np.log10(0.05), r"$P/q=0.05$", color="red",
            va="center", ha="right", fontsize=10)

    for xi, y_raw, significant in zip(x, raw_y, df["raw_significant"]):
        if significant:
            ax.text(xi, y_raw + 0.08, "*", ha="center", va="bottom",
                    fontsize=15, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(df["Variable"], rotation=48, ha="right")
    ax.set_ylabel(r"$-\log_{10}(P\ \mathrm{or}\ q)$")
    ax.set_xlim(-0.9, len(df) - 0.1)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=1.2)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.55))
    fig.text(
        0.995, 0.02, rf"Storey FDR: $\lambda=0.5$, $\hat{{\pi}}_0={pi0:.3f}$",
        ha="right", va="bottom", fontsize=8, color="#555555"
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
