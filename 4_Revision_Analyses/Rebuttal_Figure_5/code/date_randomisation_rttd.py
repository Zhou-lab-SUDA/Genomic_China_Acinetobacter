"""Date-randomisation test for root-to-tip temporal signal.

This script uses precomputed root-to-tip distances and sampling years. For each
group, it calculates the observed Pearson R and compares it with R values from
date-randomised, 50%-subsampled replicates.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev


@dataclass
class Row:
    genome: str
    year: float
    root_to_tip: float
    group: str


def pearson_r(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    mx = mean(xs)
    my = mean(ys)
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sx * sy)


def read_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for rec in reader:
            rows.append(
                Row(
                    genome=rec["genome"],
                    year=float(rec["year"]),
                    root_to_tip=float(rec["root_to_tip"]),
                    group=rec["group"],
                )
            )
    return rows


def randomised_replicates(
    rows: list[Row],
    *,
    randomisations: int,
    subsamples_per_randomisation: int,
    sample_fraction: float,
    rng: random.Random,
) -> list[dict[str, object]]:
    n = len(rows)
    sample_n = max(3, int(round(n * sample_fraction)))
    years = [r.year for r in rows]
    out: list[dict[str, object]] = []

    for randomisation in range(1, randomisations + 1):
        shuffled_years = years[:]
        rng.shuffle(shuffled_years)
        permuted = [
            Row(r.genome, shuffled_years[i], r.root_to_tip, r.group)
            for i, r in enumerate(rows)
        ]
        for subsample in range(1, subsamples_per_randomisation + 1):
            selected = rng.sample(permuted, sample_n)
            r_value = pearson_r(
                [r.year for r in selected],
                [r.root_to_tip for r in selected],
            )
            out.append(
                {
                    "randomisation": randomisation,
                    "subsample": subsample,
                    "n": sample_n,
                    "r": r_value,
                }
            )
    return out


def observed_subsample_replicates(
    rows: list[Row],
    *,
    subsamples: int,
    sample_fraction: float,
    rng: random.Random,
) -> list[dict[str, object]]:
    n = len(rows)
    sample_n = max(3, int(round(n * sample_fraction)))
    out: list[dict[str, object]] = []
    for subsample in range(1, subsamples + 1):
        selected = rng.sample(rows, sample_n)
        r_value = pearson_r(
            [r.year for r in selected],
            [r.root_to_tip for r in selected],
        )
        out.append({"subsample": subsample, "n": sample_n, "r": r_value})
    return out


def summarise_values(values: list[float]) -> dict[str, float]:
    mu = mean(values)
    sd = stdev(values) if len(values) > 1 else 0.0
    se = sd / math.sqrt(len(values)) if values else 0.0
    return {
        "mean": mu,
        "sd": sd,
        "se": se,
        "ci_low": mu - 1.96 * se,
        "ci_high": mu + 1.96 * se,
        "spread_95_low": mu - 1.96 * sd,
        "spread_95_high": mu + 1.96 * sd,
    }


def summarise_randomisations(reps: list[dict[str, object]]) -> list[dict[str, object]]:
    by_randomisation: dict[int, list[float]] = {}
    by_n: dict[int, int] = {}
    for rec in reps:
        rid = int(rec["randomisation"])
        by_randomisation.setdefault(rid, []).append(float(rec["r"]))
        by_n[rid] = int(rec["n"])
    out: list[dict[str, object]] = []
    for rid in sorted(by_randomisation):
        vals = by_randomisation[rid]
        stats = summarise_values(vals)
        out.append(
            {
                "randomisation": rid,
                "n": by_n[rid],
                "replicates": len(vals),
                "mean_r": stats["mean"],
                "sd_r": stats["sd"],
                "se_r": stats["se"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
            }
        )
    return out


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def make_plot(
    groups: list[str],
    labels: dict[str, str],
    observed: dict[str, dict[str, float]],
    randomised: dict[str, list[dict[str, object]]],
    output_svg: Path,
    output_pdf: Path | None,
) -> None:
    colors = {
        "All": "#4A6FB5",
        "2.3": "#134E5A",
        "2.4": "#EB6564",
        "2.5": "#F5BA1A",
    }
    pale_blue = "#8ecae6"
    grey = "#9aa0a6"

    def nice_ticks(ymin: float, ymax: float, nticks: int = 5) -> list[float]:
        span = ymax - ymin
        if span <= 0:
            return [ymin]
        raw_step = span / max(1, nticks - 1)
        exponent = math.floor(math.log10(raw_step))
        fraction = raw_step / (10 ** exponent)
        if fraction <= 1:
            nice_fraction = 1
        elif fraction <= 2:
            nice_fraction = 2
        elif fraction <= 5:
            nice_fraction = 5
        else:
            nice_fraction = 10
        step = nice_fraction * (10 ** exponent)
        start = math.floor(ymin / step) * step
        vals = []
        v = start
        while v <= ymax + step * 0.5:
            if v >= ymin - step * 0.2:
                vals.append(v)
            v += step
        return vals

    def fmt_tick(v: float) -> str:
        if abs(v) >= 0.1:
            return f"{v:.2f}".rstrip("0").rstrip(".")
        if abs(v) >= 0.01:
            return f"{v:.2f}".rstrip("0").rstrip(".")
        return f"{v:.3f}".rstrip("0").rstrip(".")

    panel_w = 155
    panel_h = 205
    left_pad = 50
    right_pad = 28
    top_pad = 58
    bottom_pad = 28
    gap = 28
    width = left_pad + right_pad + len(groups) * panel_w + (len(groups) - 1) * gap
    height = top_pad + panel_h + bottom_pad
    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="16" y="27" font-family="Arial" font-size="18" font-weight="700">Date-randomisation test</text>',
    ]

    for idx, group in enumerate(groups):
        px = left_pad + idx * (panel_w + gap)
        py = top_pad
        reps = randomised[group]
        ys = [float(r["mean_r"]) for r in reps]
        obs = observed[group]["point_r"]

        low = min([float(r["ci_low"]) for r in reps] + [observed[group]["plot_ci_low"]])
        high = max([float(r["ci_high"]) for r in reps] + [observed[group]["plot_ci_high"]])
        pad = max((high - low) * 0.18, 0.006)
        ymin = low - pad
        ymax = high + pad

        random_mean_stats = summarise_values(ys)
        ci_low = random_mean_stats["spread_95_low"]
        ci_high = random_mean_stats["spread_95_high"]
        mu = random_mean_stats["mean"]

        def sx(x: float) -> float:
            return px + (x - 0.4) / (13.45 - 0.4) * panel_w

        def sy(y: float) -> float:
            return py + panel_h - (y - ymin) / (ymax - ymin) * panel_h

        out.append(f'<text x="{px + panel_w / 2:.1f}" y="{py - 14}" text-anchor="middle" font-family="Arial" font-size="13" font-weight="700">{labels[group]}</text>')
        out.append(f'<rect x="{px:.1f}" y="{sy(ci_high):.1f}" width="{panel_w}" height="{max(0.5, sy(ci_low) - sy(ci_high)):.1f}" fill="#e9ecef"/>')
        out.append(f'<line x1="{px:.1f}" y1="{sy(mu):.1f}" x2="{px + panel_w:.1f}" y2="{sy(mu):.1f}" stroke="#c9cdd2" stroke-width="0.8"/>')
        for yline in [ci_low, ci_high]:
            out.append(f'<line x1="{px:.1f}" y1="{sy(yline):.1f}" x2="{px + panel_w:.1f}" y2="{sy(yline):.1f}" stroke="#b8bcc2" stroke-width="0.7" stroke-dasharray="3 2"/>')

        for tick in nice_ticks(ymin, ymax):
            y = sy(tick)
            out.append(f'<line x1="{px - 5:.1f}" y1="{y:.1f}" x2="{px:.1f}" y2="{y:.1f}" stroke="black" stroke-width="0.8"/>')
            out.append(f'<text x="{px - 8:.1f}" y="{y + 3.5:.1f}" text-anchor="end" font-family="Arial" font-size="10">{fmt_tick(tick)}</text>')

        out.append(f'<text x="{px - 36:.1f}" y="{py + panel_h / 2:.1f}" text-anchor="middle" dominant-baseline="middle" font-family="Arial" font-size="20">R</text>')
        sorted_reps = sorted(reps, key=lambda rec: float(rec["mean_r"]))
        for rank, rec in enumerate(sorted_reps, start=1):
            x = float(rank)
            y = float(rec["mean_r"])
            out.append(f'<line x1="{sx(x):.1f}" y1="{sy(float(rec["ci_low"])):.1f}" x2="{sx(x):.1f}" y2="{sy(float(rec["ci_high"])):.1f}" stroke="{pale_blue}" stroke-width="1.4" stroke-opacity="0.95"/>')
            out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="1.8" fill="{pale_blue}" fill-opacity="0.95"/>')

        obs_x = 12.0
        obs_color = colors.get(group, "#4A6FB5")
        out.append(f'<line x1="{sx(obs_x):.1f}" y1="{sy(observed[group]["plot_ci_low"]):.1f}" x2="{sx(obs_x):.1f}" y2="{sy(observed[group]["plot_ci_high"]):.1f}" stroke="{obs_color}" stroke-width="2.2" stroke-opacity="0.95"/>')
        out.append(f'<circle cx="{sx(obs_x):.1f}" cy="{sy(obs):.1f}" r="2.2" fill="{obs_color}" fill-opacity="0.95"/>')

        text_y = py + 44
        out.append(f'<text x="{px + 20:.1f}" y="{text_y:.1f}" font-family="Arial" font-size="11" fill="{grey}">Randomly</text>')
        out.append(f'<text x="{px + 20:.1f}" y="{text_y + 12:.1f}" font-family="Arial" font-size="11" fill="{grey}">simulated</text>')
        out.append(f'<text x="{px + 20:.1f}" y="{text_y + 24:.1f}" font-family="Arial" font-size="11" fill="{grey}">data</text>')

        true_y = sy(obs) + (32 if obs > mu else -22)
        out.append(f'<text x="{sx(11.65):.1f}" y="{true_y:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="{grey}">True data</text>')
        out.append(f'<text x="{sx(11.65):.1f}" y="{true_y + 12:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="{grey}">({labels[group]})</text>')

        out.append(f'<text x="{sx(13.1):.1f}" y="{sy((ci_low + ci_high) / 2) + 3.5:.1f}" font-family="Arial" font-size="11" fill="#333333">95%</text>')
        out.append(f'<line x1="{sx(12.9):.1f}" y1="{sy(ci_low):.1f}" x2="{sx(12.9):.1f}" y2="{sy(ci_high):.1f}" stroke="#555555" stroke-width="0.8"/>')
        out.append(f'<line x1="{sx(12.75):.1f}" y1="{sy(ci_low):.1f}" x2="{sx(13.05):.1f}" y2="{sy(ci_low):.1f}" stroke="#555555" stroke-width="0.8"/>')
        out.append(f'<line x1="{sx(12.75):.1f}" y1="{sy(ci_high):.1f}" x2="{sx(13.05):.1f}" y2="{sy(ci_high):.1f}" stroke="#555555" stroke-width="0.8"/>')

        out.append(f'<rect x="{px:.1f}" y="{py:.1f}" width="{panel_w}" height="{panel_h}" fill="none" stroke="black" stroke-width="1.0"/>')

    out.append("</svg>")
    output_svg.write_text("\n".join(out), encoding="utf-8")
    if output_pdf is not None:
        # PDF export is intentionally skipped in the dependency-free SVG backend.
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    figure_dir = Path(__file__).resolve().parents[1]
    parser.add_argument("--input", default=str(figure_dir / "data" / "root_to_tip_distances.tsv"))
    parser.add_argument("--outdir", default=str(figure_dir / "reproduced_date_randomisation"))
    parser.add_argument("--randomisations", type=int, default=10)
    parser.add_argument("--subsamples", type=int, default=10)
    parser.add_argument("--sample-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260605)
    args = parser.parse_args()

    rows = read_rows(Path(args.input))
    groups = ["All", "2.3", "2.4", "2.5"]
    labels = {"All": "China A. baumannii", "2.3": "ESL 2.3", "2.4": "ESL 2.4", "2.5": "ESL 2.5"}
    grouped = {g: [r for r in rows if g == "All" or r.group == g] for g in groups}
    rng = random.Random(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_reps: list[dict[str, object]] = []
    all_randomisation_summaries: list[dict[str, object]] = []
    all_observed_reps: list[dict[str, object]] = []
    randomised: dict[str, list[dict[str, object]]] = {}
    observed: dict[str, dict[str, float]] = {}
    for g in groups:
        obs_reps = observed_subsample_replicates(
            grouped[g],
            subsamples=args.subsamples,
            sample_fraction=args.sample_fraction,
            rng=rng,
        )
        for rec in obs_reps:
            all_observed_reps.append({"group": g, **rec})
        obs_stats = summarise_values([float(r["r"]) for r in obs_reps])
        full_r = pearson_r([r.year for r in grouped[g]], [r.root_to_tip for r in grouped[g]])
        observed[g] = {
            **obs_stats,
            "full_r": full_r,
            "point_r": full_r,
            "plot_ci_low": full_r - 1.96 * obs_stats["se"],
            "plot_ci_high": full_r + 1.96 * obs_stats["se"],
            "n": max(3, int(round(len(grouped[g]) * args.sample_fraction))),
        }

        reps = randomised_replicates(
            grouped[g],
            randomisations=args.randomisations,
            subsamples_per_randomisation=args.subsamples,
            sample_fraction=args.sample_fraction,
            rng=rng,
        )
        summaries = summarise_randomisations(reps)
        randomised[g] = summaries
        for rec in reps:
            all_reps.append({"group": g, **rec})
        for rec in summaries:
            all_randomisation_summaries.append({"group": g, **rec})

    summary = []
    for g in groups:
        ys = [float(r["mean_r"]) for r in randomised[g]]
        random_stats = summarise_values(ys)
        summary.append(
            {
                "group": g,
                "label": labels[g],
                "n_true": len(grouped[g]),
                "sample_fraction": args.sample_fraction,
                "n_subsample": max(3, int(round(len(grouped[g]) * args.sample_fraction))),
                "observed_full_r": observed[g]["full_r"],
                "observed_subsample_mean_r": observed[g]["mean"],
                "observed_subsample_ci_low": observed[g]["ci_low"],
                "observed_subsample_ci_high": observed[g]["ci_high"],
                "observed_plot_ci_low": observed[g]["plot_ci_low"],
                "observed_plot_ci_high": observed[g]["plot_ci_high"],
                "randomised_mean_of_means_r": random_stats["mean"],
                "randomised_10x_mean_sd_r": random_stats["sd"],
                "randomised_10x_mean_95_spread_low": random_stats["spread_95_low"],
                "randomised_10x_mean_95_spread_high": random_stats["spread_95_high"],
                "outside_randomised_10x_95_spread": observed[g]["full_r"] < random_stats["spread_95_low"]
                or observed[g]["full_r"] > random_stats["spread_95_high"],
            }
        )

    write_tsv(
        outdir / "date_randomisation_observed_subsamples.tsv",
        all_observed_reps,
        ["group", "subsample", "n", "r"],
    )
    write_tsv(
        outdir / "date_randomisation_replicates.tsv",
        all_reps,
        ["group", "randomisation", "subsample", "n", "r"],
    )
    write_tsv(
        outdir / "date_randomisation_randomisation_summaries.tsv",
        all_randomisation_summaries,
        ["group", "randomisation", "n", "replicates", "mean_r", "sd_r", "se_r", "ci_low", "ci_high"],
    )
    write_tsv(
        outdir / "date_randomisation_summary.tsv",
        summary,
        [
            "group",
            "label",
            "n_true",
            "sample_fraction",
            "n_subsample",
            "observed_full_r",
            "observed_subsample_mean_r",
            "observed_subsample_ci_low",
            "observed_subsample_ci_high",
            "observed_plot_ci_low",
            "observed_plot_ci_high",
            "randomised_mean_of_means_r",
            "randomised_10x_mean_sd_r",
            "randomised_10x_mean_95_spread_low",
            "randomised_10x_mean_95_spread_high",
            "outside_randomised_10x_95_spread",
        ],
    )
    make_plot(
        groups,
        labels,
        observed,
        randomised,
        outdir / "date_randomisation_four_panel.svg",
        outdir / "date_randomisation_four_panel.pdf",
    )


if __name__ == "__main__":
    main()
