"""Plot fitted estimates and contrasts used for the revised main panels."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

COLORS = {"ESL2.4": "#C96B6A", "ESL2.5": "#E8AA28"}
FILLS = {"ESL2.4": "#EAB6B5", "ESL2.5": "#F8D483"}
PALE = {lineage.removeprefix("ESL"): shade for lineage, shade in FILLS.items()}


def period_row(summary, lineage, period):
    return summary.loc[summary.lineage.eq(lineage) & summary.period.eq(period)].iloc[0]


def bracket(ax: plt.Axes, x: float, y0: float, y1: float, label: str) -> None:
    width = 1.25
    ax.plot([x - width, x, x, x - width], [y0, y0, y1, y1],
            color="#222222", lw=1.05, clip_on=False)
    ax.text(x + 0.8, (y0 + y1) / 2, label, va="center", ha="left", fontsize=8.5)


def make_figure(yearly: pd.DataFrame, summary: pd.DataFrame, pairwise: pd.DataFrame, output: Path) -> None:
    rows = [
        ("ESL2.4", "npi", 3, "2020-2023"),
        ("ESL2.4", "pre", 2, "2011-2019"),
        ("ESL2.5", "npi", 1, "2020-2023"),
        ("ESL2.5", "pre", 0, "2011-2019"),
    ]
    rng = np.random.default_rng(20261026)
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 9.5, "axes.linewidth": 1.0,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(8.8, 5.25))
    ax.axhspan(-0.48, 1.48, color="#F1F1F1", zorder=0)

    for lineage, period, y, _ in rows:
        lineage_key = lineage.replace("ESL", "")
        annual = yearly.loc[yearly.lineage.eq(lineage) & yearly.period.eq(period)]
        jitter = rng.uniform(-0.105, 0.105, len(annual))
        ax.scatter(annual.estimate_km_per_year, y + jitter, s=18,
                   color=PALE[lineage_key], alpha=0.80, edgecolor="none", zorder=2)

        point = summary.loc[summary.lineage.eq(lineage) & summary.period.eq(period)].iloc[0]
        estimate = point.estimate_km_per_year
        range_low = float(annual.estimate_km_per_year.min())
        range_high = float(annual.estimate_km_per_year.max())
        ax.hlines(y, range_low, range_high, color="#111111", lw=1.4, zorder=3)
        ax.scatter([estimate], [y], s=58, color=COLORS[lineage],
                   edgecolor="#111111", linewidth=0.7, zorder=4)
        ax.text(range_high + 1.8, y, f"{estimate:.1f} km/year", va="center", ha="left",
                fontsize=9.2)

    p = pairwise.set_index("comparison_id")
    bracket(ax, 70.0, 2, 3, f"{p.loc['within_ESL2.4', 'significance_label']}\nP={p.loc['within_ESL2.4', 'p_value']:.3f}")
    bracket(ax, 75.0, 1, 3, f"{p.loc['between_npi', 'significance_label']}\nP={p.loc['between_npi', 'p_value']:.3f}")
    bracket(ax, 80.0, 0, 1, f"{p.loc['within_ESL2.5', 'significance_label']}\nP={p.loc['within_ESL2.5', 'p_value']:.3f}")
    bracket(ax, 85.0, 0, 2, f"{p.loc['between_pre', 'significance_label']}\nP={p.loc['between_pre', 'p_value']:.3f}")

    ax.set_xlim(-2, 98)
    ax.set_ylim(-0.5, 3.5)
    ax.set_yticks([3, 2, 1, 0], ["2020-2023", "2011-2019", "2020-2023", "2011-2019"])
    ax.set_xlabel("Sampling-adjusted spatial dissemination (km/year)", fontsize=11)
    ax.set_title("Sampling-adjusted pandemic-period spatial dissemination", fontsize=12, pad=10)
    ax.text(-0.17, 0.75, "ESL2.4", transform=ax.transAxes, fontsize=12,
            ha="right", va="center")
    ax.text(-0.17, 0.25, "ESL2.5", transform=ax.transAxes, fontsize=12,
            ha="right", va="center")
    ax.text(-0.12, 1.02, "c", transform=ax.transAxes, fontsize=18, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(True)
    ax.tick_params(direction="out", length=4, width=0.9)
    ax.grid(False)

    interaction = p.loc["difference_in_change"]
    fig.text(
        0.52, 0.045,
        "Approximate difference-in-change: "
        f"{interaction.estimate_difference_km_per_year:.1f} km/year; "
        f"P={interaction.p_value:.3f}.",
        ha="center", fontsize=8.2, color="#444444",
    )
    fig.text(
        0.52, 0.018,
        "Dots are fitted lineage-year means; large points are standardised period means; horizontal lines span annual min-max (not CIs).",
        ha="center", fontsize=7.5, color="#555555",
    )
    fig.subplots_adjust(left=0.22, right=0.97, top=0.88, bottom=0.17)
    fig.savefig(output, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)



def draw_figure(yearly: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 10,
        "axes.linewidth": 1.1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    ax.axvspan(2020, 2023.45, color="#F2F2F2", zorder=0)
    ax.axhline(0, color="#777777", lw=0.8, ls=(0, (2, 2)), zorder=0.5)

    for lineage in ("ESL2.4", "ESL2.5"):
        data = yearly.loc[yearly.lineage.eq(lineage)].sort_values("year")
        years = data.year.to_numpy(dtype=float)
        estimates = data.estimate_km_per_year.to_numpy(dtype=float)
        lower = data.ci_lower_km_per_year.to_numpy(dtype=float)
        upper = data.ci_upper_km_per_year.to_numpy(dtype=float)

        ax.fill_between(
            years, lower, upper, color=FILLS[lineage], alpha=0.38,
            linewidth=0, zorder=1,
        )
        ax.plot(
            years, estimates, color=COLORS[lineage], lw=2.6,
            marker="o", ms=4.2, markeredgecolor="white", markeredgewidth=0.6,
            zorder=3,
        )

        pre = period_row(summary, lineage, "pre")
        npi = period_row(summary, lineage, "npi")
        ax.hlines(
            pre.estimate_km_per_year, 2010.75, 2019.35,
            color=COLORS[lineage], lw=1.6, ls=(0, (6, 4)), zorder=2,
        )
        ax.hlines(
            npi.estimate_km_per_year, 2019.65, 2023.35,
            color=COLORS[lineage], lw=1.6, ls=(0, (6, 4)), zorder=2,
        )

    ax.axvline(2020, color="#555555", lw=1.2, ls=(0, (1.5, 2.0)), zorder=4)
    ax.text(2020, 110.5, "COVID-19 / NPI", ha="center", va="bottom", fontsize=10.5)

    legend_handles = [
        Line2D([0], [0], color=COLORS["ESL2.4"], lw=2.6, marker="o", ms=4.2,
               label="ESL2.4"),
        Line2D([0], [0], color=COLORS["ESL2.5"], lw=2.6, marker="o", ms=4.2,
               label="ESL2.5"),
        Line2D([0], [0], color="#555555", lw=1.6, ls=(0, (6, 4)),
               label="Adjusted period mean"),
    ]
    ax.legend(
        handles=legend_handles, title="Lineage", loc="upper left",
        bbox_to_anchor=(1.01, 0.94), frameon=False, borderaxespad=0,
        fontsize=9.5, title_fontsize=10.5, handlelength=3.0,
    )

    pre24 = period_row(summary, "ESL2.4", "pre")
    npi24 = period_row(summary, "ESL2.4", "npi")
    pre25 = period_row(summary, "ESL2.5", "pre")
    npi25 = period_row(summary, "ESL2.5", "npi")
    reduction24 = 100.0 * (npi24.estimate_km_per_year / pre24.estimate_km_per_year - 1.0)
    reduction25 = 100.0 * (npi25.estimate_km_per_year / pre25.estimate_km_per_year - 1.0)
    fig.text(0.79, 0.49, "Adjusted period means (95% CI)", fontsize=9.2,
             fontweight="bold", ha="left", va="top")
    fig.text(
        0.79, 0.445,
        f"ESL2.4\n{pre24.estimate_km_per_year:.1f} "
        f"[{pre24.ci_lower_km_per_year:.1f}, {pre24.ci_upper_km_per_year:.1f}] to\n"
        f"{npi24.estimate_km_per_year:.1f} "
        f"[{npi24.ci_lower_km_per_year:.1f}, {npi24.ci_upper_km_per_year:.1f}]\n"
        f"{reduction24:.1f}%",
        color=COLORS["ESL2.4"], fontsize=8.4, ha="left", va="top", linespacing=1.25,
    )
    fig.text(
        0.79, 0.315,
        f"ESL2.5\n{pre25.estimate_km_per_year:.1f} "
        f"[{pre25.ci_lower_km_per_year:.1f}, {pre25.ci_upper_km_per_year:.1f}] to\n"
        f"{npi25.estimate_km_per_year:.1f} "
        f"[{npi25.ci_lower_km_per_year:.1f}, {npi25.ci_upper_km_per_year:.1f}]\n"
        f"{reduction25:.1f}%",
        color=COLORS["ESL2.5"], fontsize=8.4, ha="left", va="top", linespacing=1.25,
    )

    ax.set_xlim(2010.65, 2023.45)
    ax.set_ylim(-26, 116)
    ax.set_xticks([2011, 2013, 2015, 2017, 2019, 2021, 2023])
    ax.set_yticks([-20, 0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Sampling-adjusted spatial dissemination (km/year)", fontsize=12)
    ax.set_title("Sampling-adjusted annual spatial dissemination", fontsize=13, pad=10)
    ax.text(-0.13, 1.03, "j", transform=ax.transAxes, fontsize=23, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=4, width=1.0)
    ax.grid(False)

    fig.text(
        0.47, 0.025,
        "Lines and points: annual adjusted fitted means; shading: annual 95% CI; "
        "dashed segments: standardised period means.",
        ha="center", fontsize=7.9, color="#555555",
    )
    fig.subplots_adjust(left=0.14, right=0.77, top=0.89, bottom=0.16)
    fig.savefig(output, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    annual = pd.read_csv(args.results / "annual.tsv", sep="\t", dtype={"lineage": str})
    summary = pd.read_csv(args.results / "period_estimates.tsv", sep="\t", dtype={"lineage": str})
    summary["period"] = summary.period_label.map({"2011-2019": "pre", "2020-2023": "npi"})
    for table in (annual, summary):
        table["lineage"] = "ESL" + table.lineage
    pairs = pd.read_csv(args.results / "main_figure_contrasts.tsv", sep="\t")
    pairs["significance_label"] = np.select([pairs.p_value.lt(.01), pairs.p_value.lt(.05)], ["**", "*"], default="n.s.")
    make_figure(annual, summary, pairs, args.out / "fig_2c.svg")
    draw_figure(annual, summary, args.out / "fig_5j.svg")
