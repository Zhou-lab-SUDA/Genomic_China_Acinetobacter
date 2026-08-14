"""Draw network downsampling sensitivity metrics in a dot-and-CI layout."""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METRICS = [
    ("n_nodes", "No. nodes", None),
    ("n_edges", "No. edges", None),
    ("modularity_q", "Modularity Q", None),
    ("module_nmi", "Community NMI", (0.0, 1.05)),
    ("module_ari", "Community ARI", (0.0, 1.05)),
    ("top_edge_overlap", "Top-edge overlap", (0.0, 1.05)),
    ("weighted_degree_spearman", "Weighted-degree\nSpearman rho", (0.0, 1.05)),
    ("within_module_ratio", "Within-community\nedge ratio", (0.0, 1.05)),
    ("weighted_long_distance_ratio_gt800km", ">800-km edge\nweight ratio", (0.0, 1.0)),
]

STRATEGY_ORDER = [
    "full",
    "random_50",
    "city_weighted",
    "year_balanced",
    "city_cap20",
    "city_cap20_year_balanced",
]

STRATEGY_LABELS = {
    "full": "Original",
    "random_50": "50%",
    "city_weighted": "City\nweighted",
    "year_balanced": "Year\nbalanced",
    "city_cap20": "City\nbalanced",
    "city_cap20_year_balanced": "City+year\nbalanced",
}


def read_replicates(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def summarise(rows: list[dict[str, str]], metric: str) -> dict[str, dict[str, float]]:
    by_strategy: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        try:
            value = float(row[metric])
        except (TypeError, ValueError):
            continue
        if not math.isnan(value):
            by_strategy[row["strategy"]].append(value)
    out = {}
    for strategy, values in by_strategy.items():
        mu = mean(values)
        sd = stdev(values) if len(values) > 1 else 0.0
        out[strategy] = {
            "mean": mu,
            "low": mu - 1.96 * sd,
            "high": mu + 1.96 * sd,
            "n": len(values),
        }
    return out


def nice_ticks(low: float, high: float, n: int = 4) -> list[float]:
    if high <= low:
        return [low]
    raw = (high - low) / max(1, n)
    mag = 10 ** math.floor(math.log10(raw))
    step = min([1, 2, 2.5, 5, 10], key=lambda x: abs(raw - x * mag)) * mag
    start = math.floor(low / step) * step
    ticks = []
    x = start
    while x <= high + step * 0.5:
        if x >= low - step * 0.5:
            ticks.append(x)
        x += step
    return ticks


def fmt_tick(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def svg_text(x: float, y: float, text: str, size: int = 12, weight: str = "normal", colour: str = "#222", anchor: str = "start") -> str:
    parts = []
    for i, line in enumerate(text.split("\n")):
        parts.append(
            f'<text x="{x:.1f}" y="{y + i * (size + 2):.1f}" font-size="{size}" '
            f'font-family="Arial, Helvetica, sans-serif" font-weight="{weight}" '
            f'fill="{colour}" text-anchor="{anchor}">{html.escape(line)}</text>'
        )
    return "\n".join(parts)


def render_panel(
    metric: str,
    title: str,
    fixed_ylim: tuple[float, float] | None,
    stats: dict[str, dict[str, float]],
    origin_x: int,
    origin_y: int,
    panel_w: int,
    panel_h: int,
) -> str:
    left, right, top, bottom = 48, 18, 40, 62
    values = []
    for strategy in STRATEGY_ORDER:
        if strategy in stats:
            values.extend([stats[strategy]["low"], stats[strategy]["high"], stats[strategy]["mean"]])
    if fixed_ylim:
        y_min, y_max = fixed_ylim
    else:
        y_min, y_max = min(values), max(values)
        pad = (y_max - y_min) * 0.16 if y_max > y_min else max(abs(y_max) * 0.1, 1.0)
        y_min -= pad
        y_max += pad
    if metric not in {"n_nodes", "n_edges"}:
        y_min = max(0.0, y_min)
        y_max = min(1.05, max(y_max, 0.2))

    def sx(i: int) -> float:
        usable = panel_w - left - right
        return left + i * usable / (len(STRATEGY_ORDER) - 1)

    def sy(value: float) -> float:
        usable = panel_h - top - bottom
        return top + (y_max - value) / (y_max - y_min) * usable

    lines = [f'<g transform="translate({origin_x},{origin_y})">']
    lines.append(svg_text(0, 18, title, 16, "bold"))
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{panel_h - bottom}" stroke="#222" stroke-width="1.4"/>')
    lines.append(f'<line x1="{left}" y1="{panel_h - bottom}" x2="{panel_w - right}" y2="{panel_h - bottom}" stroke="#222" stroke-width="1.4"/>')
    for tick in nice_ticks(y_min, y_max, 4):
        y = sy(tick)
        lines.append(f'<line x1="{left - 4}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#222" stroke-width="1"/>')
        lines.append(svg_text(left - 8, y + 4, fmt_tick(tick), 10, "normal", "#222", "end"))
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{panel_w - right}" y2="{y:.1f}" stroke="#E5E7EB" stroke-width="0.8"/>')
    full = stats.get("full")
    if full:
        y = sy(full["mean"])
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{panel_w - right}" y2="{y:.1f}" stroke="#CDD2D8" stroke-width="5" stroke-opacity="0.75"/>')
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{panel_w - right}" y2="{y:.1f}" stroke="#9EA7B0" stroke-width="1.2" stroke-dasharray="4 4"/>')
    for i, strategy in enumerate(STRATEGY_ORDER):
        if strategy not in stats:
            continue
        x = sx(i)
        item = stats[strategy]
        colour = "#315FAE" if strategy == "full" else "#8EC7E6"
        stroke = "#315FAE" if strategy == "full" else "#8EC7E6"
        if item["n"] > 1:
            low = max(y_min, item["low"])
            high = min(y_max, item["high"])
            lines.append(f'<line x1="{x:.1f}" y1="{sy(low):.1f}" x2="{x:.1f}" y2="{sy(high):.1f}" stroke="{stroke}" stroke-width="2.0"/>')
        lines.append(f'<circle cx="{x:.1f}" cy="{sy(item["mean"]):.1f}" r="4.3" fill="{colour}" stroke="{stroke}" stroke-width="1"/>')
        lines.append(svg_text(x, panel_h - 38, STRATEGY_LABELS[strategy], 9, "normal", "#222", "middle"))
    lines.append("</g>")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    figure_dir = Path(__file__).resolve().parents[1]
    parser.add_argument("--replicates", default=str(figure_dir / "data" / "genome_network_replicates.tsv"))
    parser.add_argument("--out", default=str(figure_dir / "network_metric_sensitivity.svg"))
    args = parser.parse_args()

    rows = read_replicates(Path(args.replicates))
    panel_w, panel_h = 330, 230
    gap_x, gap_y = 44, 48
    width = panel_w * 3 + gap_x * 2 + 36
    height = panel_h * 3 + gap_y * 2 + 30
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for i, (metric, title, fixed_ylim) in enumerate(METRICS):
        x = 18 + (i % 3) * (panel_w + gap_x)
        y = 16 + (i // 3) * (panel_h + gap_y)
        lines.append(render_panel(metric, title, fixed_ylim, summarise(rows, metric), x, y, panel_w, panel_h))
    legend_x = width - 260
    legend_y = height - 18
    lines.append(f'<circle cx="{legend_x}" cy="{legend_y}" r="4.3" fill="#315FAE" stroke="#315FAE"/>')
    lines.append(svg_text(legend_x + 10, legend_y + 4, "Original network", 11))
    lines.append(f'<circle cx="{legend_x + 130}" cy="{legend_y}" r="4.3" fill="#8EC7E6" stroke="#8EC7E6"/>')
    lines.append(svg_text(legend_x + 140, legend_y + 4, "Downsampled networks", 11))
    lines.append("</svg>")
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
