"""Plot selected network sensitivity metrics with 95% confidence intervals."""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METRICS = [
    ("n_nodes", "No. nodes"),
    ("n_edges", "No. edges"),
    ("module_nmi", "NMI"),
    ("module_ari", "ARI"),
    ("modularity_q", "Modularity Q"),
]

STRATEGIES = [
    ("full", "Original"),
    ("random_50", "50%"),
    ("city_weighted", "City\nweighted"),
    ("year_balanced", "Year\nbalanced"),
    ("city_cap20", "City\nbalanced"),
    ("city_cap20_year_balanced", "City+year\nbalanced"),
]


def svg_text(x: float, y: float, text: str, size: int = 12, weight: str = "normal", colour: str = "#222", anchor: str = "start") -> str:
    out = []
    for i, line in enumerate(text.split("\n")):
        out.append(
            f'<text x="{x:.1f}" y="{y + i * (size + 2):.1f}" font-size="{size}" '
            f'font-family="Arial, Helvetica, sans-serif" font-weight="{weight}" '
            f'fill="{colour}" text-anchor="{anchor}">{html.escape(line)}</text>'
        )
    return "\n".join(out)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def summarize(rows: list[dict[str, str]], metric: str) -> dict[str, dict[str, float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        try:
            values[row["strategy"]].append(float(row[metric]))
        except (KeyError, TypeError, ValueError):
            continue
    out = {}
    for strategy, vals in values.items():
        vals = [v for v in vals if not math.isnan(v)]
        if not vals:
            continue
        mu = mean(vals)
        sd = stdev(vals) if len(vals) > 1 else 0.0
        sem = sd / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
        out[strategy] = {
            "mean": mu,
            "sd": sd,
            "ci_low": mu - 1.96 * sem,
            "ci_high": mu + 1.96 * sem,
            "n": len(vals),
        }
    return out


def nice_ticks(low: float, high: float, target: int = 4) -> list[float]:
    if high <= low:
        return [low]
    raw = (high - low) / target
    mag = 10 ** math.floor(math.log10(raw))
    step = min([1, 2, 2.5, 5, 10], key=lambda x: abs(raw - x * mag)) * mag
    start = math.floor(low / step) * step
    ticks = []
    value = start
    while value <= high + step * 0.5:
        if value >= low - step * 0.5:
            ticks.append(value)
        value += step
    return ticks


def fmt_tick(value: float) -> str:
    if abs(value) >= 10:
        return f"{value:.0f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def read_summary(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Read the exact metric values supplied with the submitted figure.

    The selected-metric table is already a figure-ready summary.  Reading it
    directly prevents a later version of the upstream network workflow from
    silently changing values used in the submitted panel.
    """
    grouped: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            strategy = row["strategy"]
            grouped[row["metric"]][strategy] = {
                "mean": float(row["mean"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
                "n": 1 if strategy == "full" else 10,
            }
    return grouped


def render_panel(metric: str, title: str, stats: dict[str, dict[str, float]], x0: int, y0: int, width: int, height: int) -> str:
    left, right, top, bottom = 55, 18, 42, 62
    vals = []
    for strategy, _ in STRATEGIES:
        if strategy in stats:
            vals.extend([stats[strategy]["ci_low"], stats[strategy]["ci_high"], stats[strategy]["mean"]])
    if metric in {"module_nmi", "module_ari"}:
        y_min, y_max = 0.0, 1.05
    elif metric == "modularity_q":
        y_min, y_max = 0.0, max(0.45, max(vals) * 1.15)
    else:
        y_min = max(0.0, min(vals) * 0.9)
        y_max = max(vals) * 1.12

    def sx(i: int) -> float:
        return left + i * (width - left - right) / (len(STRATEGIES) - 1)

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * (height - top - bottom)

    lines = [f'<g transform="translate({x0},{y0})">']
    lines.append(svg_text(0, 18, title, 16, "bold"))
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222" stroke-width="1.3"/>')
    lines.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222" stroke-width="1.3"/>')
    for tick in nice_ticks(y_min, y_max):
        y = sy(tick)
        lines.append(f'<line x1="{left-4}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#222"/>')
        lines.append(svg_text(left - 8, y + 4, fmt_tick(tick), 9, "normal", "#222", "end"))
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#ECEFF2" stroke-width="0.8"/>')
    full = stats.get("full")
    if full:
        y = sy(full["mean"])
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#CDD2D8" stroke-width="4" stroke-opacity="0.8"/>')
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#87919B" stroke-width="1.0" stroke-dasharray="4 4"/>')
    for i, (strategy, label) in enumerate(STRATEGIES):
        if strategy not in stats:
            continue
        item = stats[strategy]
        x = sx(i)
        colour = "#315FAE" if strategy == "full" else "#8EC7E6"
        if item["n"] > 1:
            lo = max(y_min, item["ci_low"])
            hi = min(y_max, item["ci_high"])
            lines.append(f'<line x1="{x:.1f}" y1="{sy(lo):.1f}" x2="{x:.1f}" y2="{sy(hi):.1f}" stroke="{colour}" stroke-width="2.0"/>')
            lines.append(f'<line x1="{x-4:.1f}" y1="{sy(lo):.1f}" x2="{x+4:.1f}" y2="{sy(lo):.1f}" stroke="{colour}" stroke-width="1.5"/>')
            lines.append(f'<line x1="{x-4:.1f}" y1="{sy(hi):.1f}" x2="{x+4:.1f}" y2="{sy(hi):.1f}" stroke="{colour}" stroke-width="1.5"/>')
        lines.append(f'<circle cx="{x:.1f}" cy="{sy(item["mean"]):.1f}" r="4.2" fill="{colour}" stroke="{colour}" stroke-width="1"/>')
        lines.append(svg_text(x, height - 38, label, 9, "normal", "#222", "middle"))
    lines.append("</g>")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    figure_dir = Path(__file__).resolve().parents[1]
    data_dir = figure_dir / "data"
    parser.add_argument("--summary-input", default=str(data_dir / "selected_network_metrics_ci.tsv"))
    parser.add_argument("--out", default=str(figure_dir / "selected_network_metrics.svg"))
    args = parser.parse_args()

    summaries = read_summary(Path(args.summary_input))
    panel_w, panel_h = 340, 245
    gap_x, gap_y = 44, 48
    width = panel_w * 3 + gap_x * 2 + 36
    height = panel_h * 2 + gap_y + 30
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    for i, (metric, title) in enumerate(METRICS):
        stats = summaries[metric]
        x = 18 + (i % 3) * (panel_w + gap_x)
        y = 16 + (i // 3) * (panel_h + gap_y)
        svg.append(render_panel(metric, title, stats, x, y, panel_w, panel_h))
    legend_x = width - 270
    legend_y = height - 18
    svg.append(f'<circle cx="{legend_x}" cy="{legend_y}" r="4.2" fill="#315FAE" stroke="#315FAE"/>')
    svg.append(svg_text(legend_x + 10, legend_y + 4, "Original network", 11))
    svg.append(f'<circle cx="{legend_x + 130}" cy="{legend_y}" r="4.2" fill="#8EC7E6" stroke="#8EC7E6"/>')
    svg.append(svg_text(legend_x + 140, legend_y + 4, "Mean ± 95% CI", 11))
    svg.append("</svg>")
    Path(args.out).write_text("\n".join(svg), encoding="utf-8")



if __name__ == "__main__":
    main()
