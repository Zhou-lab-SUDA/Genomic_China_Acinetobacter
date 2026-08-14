"""Scatter plots comparing original and downsampled city weighted degrees."""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections import defaultdict
from pathlib import Path

from network_resampling_sensitivity import ranks


STRATEGIES = [
    ("random_50", "50% size downsampling"),
    ("city_weighted", "City-weighted sampling"),
    ("year_balanced", "Year-balanced sampling"),
    ("city_cap20", "City-balanced sampling"),
    ("city_cap20_year_balanced", "City-balanced +\nyear-balanced sampling"),
]

NETWORK_FILE_STRATEGIES = [
    ("random_edge_50", "50% edge downsampling"),
    ("city_weighted_node", "Population-weighted\nnode sampling"),
    ("city_cap20_node", "High-degree balanced\nnode sampling"),
    ("city_cap20_edge_50", "High-degree edge\nperturbation"),
]


def read_edges(path: Path) -> dict[str, list[dict[str, object]]]:
    by_strategy: dict[str, list[dict[str, object]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            by_strategy[row["strategy"]].append({
                "source": row["source"],
                "target": row["target"],
                "weight": float(row["mean_weight"]),
            })
    return dict(by_strategy)


def read_degree_replicates(path: Path) -> dict[str, list[tuple[str, float, float]]]:
    if not path.exists():
        return {}
    by_strategy: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            strategy = row["strategy"]
            if strategy == "full":
                continue
            by_strategy[strategy].append((row["city"], float(row["original_degree"]), float(row["degree"])))
    return dict(by_strategy)


def weighted_degree(edges: list[dict[str, object]]) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for edge in edges:
        out[str(edge["source"])] += float(edge["weight"])
        out[str(edge["target"])] += float(edge["weight"])
    return dict(out)


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx <= 0 or sy <= 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sx * sy)


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(ranks(xs), ranks(ys))


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    if len(xs) < 2:
        return 0.0, 0.0, float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom <= 0:
        return 0.0, my, float("nan")
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    intercept = my - slope * mx
    r = pearson(xs, ys)
    return slope, intercept, r * r if not math.isnan(r) else float("nan")


def fixed_ticks(max_value: float, step: float = 100.0) -> list[float]:
    end = max(step, math.ceil(max_value / step) * step)
    ticks = []
    v = 0.0
    while v <= end + 1e-9:
        ticks.append(v)
        v += step
    return ticks


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
    title: str,
    points: list[tuple[str, float, float]],
    replicate_points: list[tuple[str, float, float]],
    rho: float,
    pearson_r: float,
    ox: int,
    oy: int,
    w: int,
    h: int,
) -> str:
    left, top = 58, 48
    plot_size = min(w - 86, h - 92)
    axis_right = left + plot_size
    axis_bottom = top + plot_size
    max_x = max([p[1] for p in points] + [1.0])
    max_y = max([p[2] for p in points] + [rp[2] for rp in replicate_points] + [1.0])
    x_lim = max(100.0, math.ceil(max_x / 100.0) * 100.0)
    y_lim = max_y * 1.12

    def sx(v: float) -> float:
        return left + v / x_lim * plot_size

    def sy(v: float) -> float:
        return top + (y_lim - v) / y_lim * plot_size

    lines = [f'<g transform="translate({ox},{oy})">']
    lines.append(svg_text(0, 18, title, 15, "bold"))
    slope, intercept, r2 = linear_fit([p[1] for p in points], [p[2] for p in points])
    lines.append(svg_text(w - 18, 34, f"R² = {r2:.2f}\nPearson r = {pearson_r:.2f}", 11, "normal", "#444", "end"))
    lines.append(f'<line x1="{left}" y1="{axis_bottom}" x2="{axis_right}" y2="{axis_bottom}" stroke="#222" stroke-width="1.3"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{axis_bottom}" stroke="#222" stroke-width="1.3"/>')
    fit_y0 = max(0.0, min(y_lim, intercept))
    fit_y1 = max(0.0, min(y_lim, slope * x_lim + intercept))
    lines.append(f'<line x1="{sx(0):.1f}" y1="{sy(fit_y0):.1f}" x2="{sx(x_lim):.1f}" y2="{sy(fit_y1):.1f}" stroke="#2F6F9F" stroke-width="1.6"/>')
    x_ticks = fixed_ticks(x_lim, 100.0)
    y_ticks = fixed_ticks(y_lim, 100.0)
    for val in x_ticks:
        x = sx(val)
        lines.append(f'<line x1="{x:.1f}" y1="{axis_bottom}" x2="{x:.1f}" y2="{axis_bottom+4}" stroke="#222"/>')
        label = f"{val:.0f}"
        lines.append(svg_text(x, axis_bottom + 18, label, 9, "normal", "#222", "middle"))
        if val not in [0, x_ticks[-1]]:
            lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{axis_bottom}" stroke="#ECEFF2" stroke-width="0.8"/>')
    for val in y_ticks:
        y = sy(val)
        lines.append(f'<line x1="{left-4}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#222"/>')
        lines.append(svg_text(left - 8, y + 4, f"{val:.0f}", 9, "normal", "#222", "end"))
        if val not in [0, y_ticks[-1]]:
            lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{axis_right}" y2="{y:.1f}" stroke="#ECEFF2" stroke-width="0.8"/>')
    jitter = x_lim * 0.003
    for idx, (_, xval, yval) in enumerate(replicate_points):
        x_jittered = min(x_lim, max(0.0, xval + ((idx % 9) - 4) * jitter))
        lines.append(
            f'<circle cx="{sx(x_jittered):.1f}" cy="{sy(yval):.1f}" r="3.0" '
            f'fill="#111111" fill-opacity="0.22" stroke="none"/>'
        )
    for city, xval, yval in points:
        lines.append(f'<circle cx="{sx(xval):.1f}" cy="{sy(yval):.1f}" r="4.2" fill="#8EC7E6" stroke="#2F6F9F" stroke-width="0.65"/>')
    lines.append(svg_text((left + axis_right) / 2, axis_bottom + 43, "Original weighted degree", 11, "normal", "#222", "middle"))
    axis_mid = (top + axis_bottom) / 2
    lines.append(f'<text x="13" y="{axis_mid:.1f}" font-size="11" font-family="Arial, Helvetica, sans-serif" fill="#222" text-anchor="middle" transform="rotate(-90 13 {axis_mid:.1f})">Downsampled weighted degree</text>')
    lines.append("</g>")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    figure_dir = Path(__file__).resolve().parents[1]
    data_dir = figure_dir / "data"
    source_dir = data_dir / "network_file_resampling"
    parser.add_argument("--edges", default=str(source_dir / "average_network_edges.tsv"))
    parser.add_argument("--replicate-degrees", default="")
    parser.add_argument("--out", default=str(figure_dir / "RF10_degree_scatter_reproduced.svg"))
    parser.add_argument("--summary", default=str(data_dir / "degree_scatter_recalculated.tsv"))
    parser.add_argument("--strategy-set", choices=["genome", "network_file"], default="network_file")
    args = parser.parse_args()

    edges_by_strategy = read_edges(Path(args.edges))
    replicate_path = Path(args.replicate_degrees) if args.replicate_degrees else Path(args.edges).with_name("degree_replicates.tsv")
    replicate_degrees = read_degree_replicates(replicate_path)
    original = weighted_degree(edges_by_strategy["full"])
    all_cities = sorted(original)
    rows = []
    panel_w, panel_h = 340, 340
    gap_x, gap_y = 46, 58
    width = panel_w * 3 + gap_x * 2 + 30
    height = panel_h * 2 + gap_y + 20
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    strategies = STRATEGIES if args.strategy_set == "genome" else NETWORK_FILE_STRATEGIES
    for i, (strategy, title) in enumerate(strategies):
        degree = weighted_degree(edges_by_strategy[strategy])
        points = [(city, original.get(city, 0.0), degree.get(city, 0.0)) for city in all_cities]
        pearson_r = pearson([p[1] for p in points], [p[2] for p in points])
        rho = spearman([p[1] for p in points], [p[2] for p in points])
        ox = 16 + (i % 3) * (panel_w + gap_x)
        oy = 12 + (i // 3) * (panel_h + gap_y)
        lines.append(render_panel(title, points, replicate_degrees.get(strategy, []), rho, pearson_r, ox, oy, panel_w, panel_h))
        _, _, r2 = linear_fit([p[1] for p in points], [p[2] for p in points])
        rows.append({"strategy": strategy, "pearson_r": pearson_r, "spearman_rho": rho, "r2": r2, "n_cities": len(points)})
    lines.append("</svg>")
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    with Path(args.summary).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=["strategy", "pearson_r", "spearman_rho", "r2", "n_cities"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
