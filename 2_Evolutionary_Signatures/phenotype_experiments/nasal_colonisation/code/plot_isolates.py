#!/usr/bin/env python3
"""Generate a six-panel isolate-level nasal colonisation trajectory figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"lineage", "genome", "mouse_id", "day", "log10_cfu"}
ISOLATE_LAYOUT = [
    ("ESL2.4", "Ab34"),
    ("ESL2.4", "Ab35"),
    ("ESL2.4", "Ab160"),
    ("ESL2.5", "Ab42"),
    ("ESL2.5", "Ab135"),
    ("ESL2.5", "Ab239"),
]
LINEAGE_COLOURS = {"ESL2.4": "#EAB426", "ESL2.5": "#D7322A"}
DAY_TICKS = [1, 2, 3, 5, 7, 14]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot isolate-level nasal colonisation by isolate."
    )
    parser.add_argument("--input", required=True, type=Path, help="Long-format TSV file")
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Directory for PNG and PDF"
    )
    return parser.parse_args()


def load_and_validate(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, sep="\t", na_values=["NA", "NaN", ""])
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    data = data.loc[:, ["lineage", "genome", "mouse_id", "day", "log10_cfu"]].copy()
    data["day"] = pd.to_numeric(data["day"], errors="raise")
    data["log10_cfu"] = pd.to_numeric(data["log10_cfu"], errors="coerce")
    data = data.dropna(subset=["log10_cfu"])

    expected = set(ISOLATE_LAYOUT)
    observed = set(map(tuple, data[["lineage", "genome"]].drop_duplicates().to_numpy()))
    if observed != expected:
        raise ValueError(
            "Unexpected lineage/isolate combinations. "
            f"Expected {sorted(expected)}, observed {sorted(observed)}"
        )

    unexpected_days = sorted(set(data["day"]) - set(DAY_TICKS))
    if unexpected_days:
        raise ValueError(f"Unexpected observation days: {unexpected_days}")

    duplicated = data.duplicated(["lineage", "genome", "mouse_id", "day"])
    if duplicated.any():
        raise ValueError("Duplicate lineage/genome/mouse_id/day rows detected")

    return data.sort_values(["lineage", "genome", "mouse_id", "day"])


def style_defaults() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.labelsize": 12,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_panel(ax: plt.Axes, subset: pd.DataFrame, lineage: str, genome: str) -> None:
    colour = LINEAGE_COLOURS[lineage]

    ax.scatter(subset["day"], subset["log10_cfu"], color=colour, alpha=0.38,
               s=20, linewidths=0, zorder=2)

    summary = (
        subset.groupby("day", as_index=False)["log10_cfu"]
        .agg(mean="mean", sd="std", n="count")
        .sort_values("day")
    )
    summary["sem"] = summary["sd"] / np.sqrt(summary["n"])

    ax.errorbar(
        summary["day"],
        summary["mean"],
        yerr=summary["sem"],
        fmt="none",
        ecolor=colour,
        elinewidth=1.25,
        capsize=2.8,
        capthick=1.1,
        alpha=0.95,
        zorder=3,
    )
    ax.plot(
        summary["day"],
        summary["mean"],
        color=colour,
        linewidth=2.7,
        solid_capstyle="round",
        zorder=4,
    )
    ax.scatter(
        summary["day"],
        summary["mean"],
        s=52,
        facecolor=colour,
        edgecolor="white",
        linewidth=1.0,
        zorder=5,
    )

    ax.set_title(f"{lineage}  {genome}", color=colour, fontweight="bold", pad=8)
    ax.set_xlim(0.45, 14.55)
    ax.set_ylim(-0.15, 4.70)
    # Lock the physical plotting region of every small-multiple panel to 1:1.
    # This is independent of the numerical x/y data ranges.
    ax.set_box_aspect(1)
    ax.set_xticks(DAY_TICKS)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.65, alpha=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(colors="#333333", labelbottom=True)


def make_figure(data: pd.DataFrame) -> plt.Figure:
    style_defaults()
    fig, axes = plt.subplots(2, 3, figsize=(10, 10), sharex=True, sharey=True)

    for ax, (lineage, genome) in zip(axes.flat, ISOLATE_LAYOUT):
        subset = data[(data["lineage"] == lineage) & (data["genome"] == genome)]
        plot_panel(ax, subset, lineage, genome)

    fig.suptitle(
        "Isolate-level nasal colonisation trajectories",
        x=0.5,
        y=0.973,
        fontsize=17,
        fontweight="bold",
        color="#202020",
    )
    fig.supxlabel("Days post-inoculation", x=0.5, y=0.125, fontsize=13)
    fig.supylabel(r"log$_{10}$(CFU)", x=0.035, y=0.51, fontsize=13)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#777777",
            linestyle="none",
            alpha=0.50,
            marker="o",
            markersize=4,
            markerfacecolor="#777777",
            markeredgecolor="white",
            label="Individual observation",
        ),
        Line2D(
            [0],
            [0],
            color="#333333",
            linewidth=2.7,
            marker="o",
            markersize=6,
            markerfacecolor="#333333",
            markeredgecolor="white",
            label="Mean ± SEM",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=2,
        frameon=False,
        handlelength=2.3,
        columnspacing=2.2,
    )
    fig.subplots_adjust(left=0.105, right=0.98, bottom=0.235, top=0.875, wspace=0.18, hspace=0.16)
    return fig


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_and_validate(args.input)
    figure = make_figure(data)

    png_path = args.output_dir / "isolate_level_nasal_colonisation.png"
    pdf_path = args.output_dir / "isolate_level_nasal_colonisation.pdf"
    figure.savefig(png_path, dpi=300, facecolor="white")
    figure.savefig(pdf_path, facecolor="white")
    plt.close(figure)

    print(f"Rows plotted: {len(data)}")
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    main()
