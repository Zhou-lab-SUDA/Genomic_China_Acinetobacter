"""Plot the three recombination tracks and RecHMM block distributions for RF11.

The mutation-event column in the standardized source table is intentionally
not plotted because it is not part of the submitted RF11 panel.
"""

from __future__ import annotations

from pathlib import Path
import os

FIGURE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = FIGURE_DIR / "data"
os.environ.setdefault("MPLCONFIGDIR", str(FIGURE_DIR / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


INPUT = DATA_DIR / "recombination_mutation_events_10kb.tsv"
OUTPUT_PDF = FIGURE_DIR / "RF11_recombination_landscape_reproduced.pdf"
OUTPUT_PNG = FIGURE_DIR / "RF11_recombination_landscape_reproduced.png"

TRACKS = [
    ("rechmm_recombination_events", "RecHMM recombination events", "#355C7D"),
    ("gubbins_recombination_events", "Gubbins recombination events", "#4C956C"),
    ("clonalframeml_recombination_events", "ClonalFrameML recombination events", "#7A5195"),
]
BLOCK_ORDER = ["HRB1", "MRB1", "HRB2", "LRB", "HRB3", "MRB2"]
BLOCK_COLORS = {
    "HRB1": "#F9D5D3", "MRB1": "#FCE8C3", "HRB2": "#F9D5D3",
    "LRB": "#D9EAD3", "HRB3": "#F9D5D3", "MRB2": "#FCE8C3",
}


def contiguous_segments(data: pd.DataFrame) -> list[tuple[str, float, float]]:
    """Return block name and chromosome span for each contiguous run."""
    run_id = data["recombination_block"].ne(data["recombination_block"].shift()).cumsum()
    segments = []
    for _, group in data.groupby(run_id, sort=True):
        segments.append(
            (
                str(group["recombination_block"].iloc[0]),
                float(group["window_start_bp_0based"].min()) / 1_000_000,
                float(group["window_end_bp_exclusive"].max()) / 1_000_000,
            )
        )
    return segments


def main() -> None:
    data = pd.read_csv(INPUT, sep="\t")
    required = {
        "window_midpoint_mb", "window_start_bp_0based", "window_end_bp_exclusive",
        "recombination_block", *(track[0] for track in TRACKS),
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    fig, axes = plt.subplots(
        4, 1, figsize=(11, 9),
        gridspec_kw={"height_ratios": [1, 1, 1, 1.5]},
        constrained_layout=True,
    )
    segments = contiguous_segments(data)
    bar_width_mb = 0.0095

    for axis, (column, label, color) in zip(axes[:3], TRACKS):
        for block, start, end in segments:
            axis.axvspan(start, end, color=BLOCK_COLORS[block], alpha=0.22, lw=0)
        axis.bar(
            data["window_midpoint_mb"], data[column], width=bar_width_mb,
            color=color, edgecolor="none",
        )
        axis.set_ylabel("Events per 10 kb", fontsize=9)
        axis.text(
            0.01, 0.92, label, transform=axis.transAxes,
            ha="left", va="top", fontsize=10, fontweight="bold",
        )
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(0, data["window_end_bp_exclusive"].max() / 1_000_000)
        axis.tick_params(direction="out")

    axes[0].set_title("Recombination events in 10-kb chromosome windows")
    axes[2].set_xlabel("Chromosome position (Mb)")

    values = [
        data.loc[data["recombination_block"] == block, "rechmm_recombination_events"].dropna()
        for block in BLOCK_ORDER
    ]
    box = axes[3].boxplot(values, tick_labels=BLOCK_ORDER, patch_artist=True, showfliers=False)
    for patch, block in zip(box["boxes"], BLOCK_ORDER):
        patch.set_facecolor(BLOCK_COLORS[block])
        patch.set_edgecolor("#444444")
    axes[3].set_ylabel("RecHMM events per 10 kb")
    axes[3].set_xlabel("Recombination-frequency block")
    axes[3].spines[["top", "right"]].set_visible(False)
    axes[3].tick_params(direction="out")

    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
