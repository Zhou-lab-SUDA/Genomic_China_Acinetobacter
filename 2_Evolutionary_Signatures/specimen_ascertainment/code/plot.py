"""Render the specimen ascertainment summaries from public aggregate counts."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
GROUPS = ["<20", "20-49", ">=50"]


def draw(output):
    groups = pd.read_csv(DATA / "02_invasive_by_sampling_group.tsv", sep="\t").set_index("depth_group").loc[GROUPS]
    cities = pd.read_csv(DATA / "10_bloodstream_city_descriptive.tsv", sep="\t")
    lineages = pd.read_csv(DATA / "06_bloodstream_sensitivity_summary.tsv", sep="\t")
    counts = lineages.pivot(index="depth_group", columns="lineage", values="n_bloodstream").loc[GROUPS]
    composition = 100 * counts.div(counts.sum(axis=1), axis=0)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    bars = ax[0].bar(GROUPS, groups.invasive_percent, color="#43819A")
    ax[0].bar_label(bars, labels=[f"{a}/{b}" for a, b in zip(groups.n_invasive, groups.n_binary)], padding=3)
    ax[0].set(ylabel="Invasive specimens (%)", xlabel="Major-lineage genomes per city", ylim=(0, 30))
    for group, color in zip(GROUPS, ["#43819A", "#CF804B", "#79569C"]):
        data = cities.loc[cities.depth_group.eq(group)]
        ax[1].scatter(data.degree, 100 * data.esl25_fraction_bloodstream,
                      s=15 + data.bloodstream_total, color=color, label=group, alpha=.8)
    ax[1].legend(frameon=False)
    ax[1].set(xlabel="City connectivity (specimen-analysis snapshot)", ylabel="ESL2.5 among bloodstream isolates (%)")
    ax[2].bar(GROUPS, composition["ESL2.4"], label="ESL2.4", color="#43819A")
    ax[2].bar(GROUPS, composition["ESL2.5"], bottom=composition["ESL2.4"], label="ESL2.5", color="#CF804B")
    ax[2].set(ylabel="Bloodstream lineage composition (%)", xlabel="Major-lineage genomes per city")
    ax[2].legend(frameon=False)
    for axis, label in zip(ax, "def"):
        axis.set_title(label, loc="left", fontweight="bold")
    fig.savefig(output, dpi=400)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", type=Path, required=True)
    draw(parser.parse_args().figure)
