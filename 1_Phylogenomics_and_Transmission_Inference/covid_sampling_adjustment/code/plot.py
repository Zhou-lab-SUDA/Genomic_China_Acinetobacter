"""Draw Supplementary Fig. 7 panels from the recomputed model tables."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "2.4": "#2878B5",
    "2.5": "#D95F59",
    "Difference-in-change": "#6B6B6B",
}


def assemble_plot_values(directory) -> pd.DataFrame:
    summary = pd.read_csv(directory / "changes.tsv", sep="\t")
    interactions = pd.read_csv(directory / "interactions.tsv", sep="\t")
    rows: list[dict[str, object]] = []
    specifications = [
        ("c", "actual_covid", "2011–2019 vs 2020–2023"),
        ("d", "prepandemic_placebo", "2011–2013 vs 2014–2017"),
    ]
    for panel, analysis, comparison in specifications:
        subset = summary.loc[summary["analysis"].eq(analysis)]
        for lineage in ("2.4", "2.5"):
            record = subset.loc[subset["lineage"].astype(str).eq(lineage)].iloc[0]
            rows.append({
                "panel": panel,
                "analysis": analysis,
                "comparison": comparison,
                "category": f"ESL{lineage}",
                "lineage": lineage,
                "estimate_km_per_year": record["delta_km_per_year"],
                "ci95_low": record["delta_ci95_low"],
                "ci95_high": record["delta_ci95_high"],
                "p_value": record["p_value_delta"],
                "adjusted_pre_km_per_year": record["adjusted_pre_km_per_year"],
                "adjusted_post_km_per_year": record["adjusted_post_km_per_year"],
                "reduction_percent": record["reduction_percent"],
                "conversion_status": "fixed_lineage_clock_conditional_conversion",
            })
        interaction = interactions.loc[interactions["analysis"].eq(analysis)].iloc[0]
        rows.append({
            "panel": panel,
            "analysis": analysis,
            "comparison": comparison,
            "category": "Difference-in-change",
            "lineage": "difference",
            "estimate_km_per_year": interaction["estimate_km_per_year"],
            "ci95_low": interaction["ci95_low"],
            "ci95_high": interaction["ci95_high"],
            "p_value": interaction["p_value"],
            "adjusted_pre_km_per_year": np.nan,
            "adjusted_post_km_per_year": np.nan,
            "reduction_percent": np.nan,
            "conversion_status": "approximate_difference_using_lineage_specific_fixed_clocks",
        })
    result = pd.DataFrame(rows)
    return result


def style_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", width=0.9, length=4)
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.7, zorder=0)


def plot_panel_a(ax: plt.Axes, converted: pd.DataFrame) -> None:
    for lineage, group in converted.groupby("lineage"):
        color = COLORS[lineage]
        ax.fill_between(group.year, group.ci_lower_km_per_year, group.ci_upper_km_per_year, color=color, alpha=.18)
        ax.plot(group["year"], group["estimate_km_per_year"], color=color, linewidth=1.8,
                label=f"ESL{lineage}")
        sizes = 22 + 34 * np.sqrt(group["n_genomes"] / converted["n_genomes"].max())
        ax.scatter(group["year"], group["estimate_km_per_year"], s=sizes, color=color,
                   edgecolor="white", linewidth=0.55, zorder=3)
    ax.axvspan(2019.5, 2023.5, color="#B9B9B9", alpha=0.16, zorder=0)
    ax.axvline(2019.5, color="#555555", linestyle="--", linewidth=1.0)
    ax.set_ylabel("Annual spatial dissemination (km/year)")
    ax.set_xlabel("Year")
    ax.set_xticks(range(2011, 2024, 2))
    ax.set_ylim(0, float(converted["ci_upper_km_per_year"].max()) * 1.08)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Fitted annual spatial dissemination")
    ax.text(-0.10, 1.04, "a", transform=ax.transAxes, fontsize=15, fontweight="bold")
    style_axes(ax)


def plot_panel_b(ax: plt.Axes, master: pd.DataFrame) -> None:
    twin = ax.twinx()
    for lineage, group in master.groupby("lineage"):
        color = COLORS[lineage]
        ax.plot(group["year"], group["n_genomes"], color=color, linewidth=1.5,
                marker="o", markersize=4, label=f"ESL{lineage} genomes")
        twin.plot(group["year"], group["n_cities"], color=color, linewidth=1.2,
                  linestyle="--", marker="s", markersize=3.5, alpha=0.72,
                  label=f"ESL{lineage} cities")
    ax.axvline(2019.5, color="#555555", linestyle="--", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_ylabel("Genomes sampled (log scale)")
    twin.set_ylabel("Cities sampled")
    ax.set_xlabel("Year")
    ax.set_xticks(range(2011, 2024, 2))
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = twin.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, frameon=False,
              fontsize=7.5, ncol=2, loc="upper right")
    ax.set_title("Sampling intensity and geographic coverage")
    ax.text(-0.10, 1.04, "b", transform=ax.transAxes, fontsize=15, fontweight="bold")
    ax.spines["top"].set_visible(False)
    twin.spines["top"].set_visible(False)
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.7)


def plot_forest(
    ax: plt.Axes,
    values: pd.DataFrame,
    panel: str,
    title: str,
    xlim: tuple[float, float],
    annotation_x: float,
) -> None:
    order = ["ESL2.4", "ESL2.5", "Difference-in-change"]
    table = values.loc[values["panel"].eq(panel)].set_index("category").loc[order]
    y = np.arange(3)[::-1]
    ax.axvline(0, color="#444444", linewidth=1.0, zorder=1)
    for yi, category, record in zip(y, order, table.itertuples()):
        key = record.lineage if record.lineage in {"2.4", "2.5"} else "Difference-in-change"
        color = COLORS[key]
        marker = "D" if category == "Difference-in-change" else "o"
        ax.errorbar(
            record.estimate_km_per_year,
            yi,
            xerr=[[record.estimate_km_per_year - record.ci95_low],
                  [record.ci95_high - record.estimate_km_per_year]],
            fmt=marker,
            markersize=6.5,
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=3.2,
            markeredgecolor="white",
            markeredgewidth=0.55,
            zorder=3,
        )
        if panel == "c" and category != "Difference-in-change":
            label = (
                f"{record.adjusted_pre_km_per_year:.2f} → "
                f"{record.adjusted_post_km_per_year:.2f} km/year\n"
                f"{record.reduction_percent:.1f}%; P={record.p_value:.3f}"
            )
        elif category == "Difference-in-change":
            label = f"Approx. converted\nP={record.p_value:.3f}"
        else:
            label = f"P={record.p_value:.3f}"
        ax.text(annotation_x, yi, label, ha="left", va="center", fontsize=7.4,
                color="#333333", linespacing=1.15)

    ax.set_xlim(*xlim)
    ax.set_ylim(-0.45, 2.45)
    ax.set_yticks(y, order)
    ax.set_xlabel("Adjusted change (km/year, 95% CI)")
    ax.set_title(title)
    ax.text(-0.10, 1.04, panel, transform=ax.transAxes, fontsize=15, fontweight="bold")
    style_axes(ax)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    annual = pd.read_csv(args.results / "annual.tsv", sep="\t", dtype={"lineage": str})
    values = assemble_plot_values(args.results)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), layout="constrained")
    plot_panel_a(axes[0, 0], annual)
    plot_panel_b(axes[0, 1], annual)
    plot_forest(axes[1, 0], values, "c", "Pandemic-period contrasts", (-52, 33), 10.5)
    plot_forest(axes[1, 1], values, "d", "Pre-pandemic placebo contrasts", (-96, 110), 77)
    fig.savefig(args.figure, dpi=400)
