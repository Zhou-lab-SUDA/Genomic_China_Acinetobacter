"""Plot A549 adhesion measurements from the tidy source table.

Each bar is the strain-level mean log10(CFU), error bars are sample standard
deviations, and points are the three independent experiments.  The script
does not re-parse the source workbook; that conversion is represented by the
TSV documented in the analysis README.
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


INPUT = DATA_DIR / "selected_strains_MOI10_MOI20_logCFU.tsv"
OUTPUT_SUMMARY = DATA_DIR / "selected_strains_MOI10_MOI20_logCFU_summary_from_script.tsv"
OUTPUT_PDF = FIGURE_DIR / "A549_adhesion_MOI10_MOI20.pdf"
OUTPUT_PREVIEW = FIGURE_DIR / "A549_adhesion_MOI10_MOI20.png"

LINEAGES = {
    "ESL2.4": ["34", "35", "160"],
    "ESL2.5": ["42", "135", "239"],
}
COLORS = {"ESL2.4": "#EB6564", "ESL2.5": "#F5BA1A"}


def load_data() -> pd.DataFrame:
    """Load and validate the isolate-level adhesion measurements."""
    data = pd.read_csv(INPUT, sep="\t", dtype={"strain": str})
    required = {"lineage", "strain", "MOI", "replicate", "log10_CFU"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    expected = pd.MultiIndex.from_product(
        [sum(LINEAGES.values(), []), [10, 20]], names=["strain", "MOI"]
    )
    counts = data.groupby(["strain", "MOI"]).size().reindex(expected)
    if counts.isna().any() or not (counts == 3).all():
        raise ValueError(f"Expected three experiments per strain and MOI:\n{counts}")
    return data


def main() -> None:
    data = load_data()
    summary = (
        data.groupby(["MOI", "lineage", "strain"], sort=False)["log10_CFU"]
        .agg(n="count", mean="mean", sd="std")
        .reset_index()
    )
    summary.to_csv(OUTPUT_SUMMARY, sep="\t", index=False, float_format="%.6g")

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.3), sharey=True)
    strains = LINEAGES["ESL2.4"] + LINEAGES["ESL2.5"]
    positions = np.array([0, 1, 2, 4, 5, 6], dtype=float)
    jitter = np.array([-0.055, 0.0, 0.055])

    for ax, moi in zip(axes, [10, 20]):
        for lineage, lineage_strains in LINEAGES.items():
            indices = [strains.index(strain) for strain in lineage_strains]
            x_positions = positions[indices]
            values_by_strain = [
                data.loc[
                    (data["strain"] == strain) & (data["MOI"] == moi),
                    "log10_CFU",
                ].to_numpy()
                for strain in lineage_strains
            ]
            means = [values.mean() for values in values_by_strain]
            standard_deviations = [values.std(ddof=1) for values in values_by_strain]
            color = COLORS[lineage]
            ax.bar(x_positions, means, width=0.68, color=color, edgecolor="none",
                   label=lineage, zorder=1)
            ax.errorbar(x_positions, means, yerr=standard_deviations, fmt="none",
                        ecolor="#333333", elinewidth=1.1, capsize=3,
                        capthick=1.1, zorder=2)
            for position, values in zip(x_positions, values_by_strain):
                ax.scatter(position + jitter, values, s=28, facecolor="white",
                           edgecolor=color, linewidth=1.25, zorder=3)

        ax.set_title(f"MOI = {moi}", fontsize=14, fontweight="bold")
        ax.set_xticks(positions)
        ax.set_xticklabels(strains)
        ax.set_xlabel("Strain")
        ax.axvline(3, color="#D0D0D0", lw=1.0, zorder=0)
        ax.set_ylim(0, 6.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out", width=1.1)
        ax.legend(frameon=False, fontsize=9)

    axes[0].set_ylabel(r"$\log_{10}(\mathrm{CFU})$")
    fig.tight_layout(w_pad=2.2)
    fig.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_PREVIEW, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
