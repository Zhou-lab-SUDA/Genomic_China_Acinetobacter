"""Plot paired network metrics with bidirectional 95% CI error bars."""

from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path


STRATEGIES = [
    ("full", "Original", "#315FAE"),
    ("random_50", "50%", "#8EC7E6"),
    ("city_weighted", "City-weighted", "#5CA4C8"),
    ("year_balanced", "Year-balanced", "#F0B35A"),
    ("city_cap20", "City-balanced", "#E56B6F"),
    ("city_cap20_year_balanced", "City+year-balanced", "#7E6AAD"),
]


def read_summary(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    out = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            out[(row["metric"], row["strategy"])] = {
                "mean": float(row["mean"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
            }
    return out


def svg_text(x: float, y: float, text: str, size: int = 12, weight: str = "normal", colour: str = "#222", anchor: str = "start") -> str:
    out = []
    for i, line in enumerate(text.split("\n")):
        out.append(
            f'<text x="{x:.1f}" y="{y + i * (size + 2):.1f}" font-size="{size}" '
            f'font-family="Arial, Helvetica, sans-serif" font-weight="{weight}" fill="{colour}" '
            f'text-anchor="{anchor}">{html.escape(line)}</text>'
        )
    return "\n".join(out)


def ticks(low: float, high: float, step_hint: float | None = None) -> list[float]:
    if step_hint:
        start = math.floor(low / step_hint) * step_hint
        end = math.ceil(high / step_hint) * step_hint
        vals = []
        v = start
        while v <= end + 1e-9:
            vals.append(v)
            v += step_hint
        return vals
    raw = (high - low) / 4
    mag = 10 ** math.floor(math.log10(raw))
    step = min([1, 2, 2.5, 5, 10], key=lambda x: abs(raw - x * mag)) * mag
    start = math.floor(low / step) * step
    end = math.ceil(high / step) * step
    vals = []
    v = start
    while v <= end + 1e-9:
        vals.append(v)
        v += step
    return vals


def fmt(value: float) -> str:
    if abs(value) >= 10:
        return f"{value:.0f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def render_plot(
    summary: dict[tuple[str, str], dict[str, float]],
    x_metric: str,
    y_metric: str,
    x_label: str,
    y_label: str,
    title: str,
    out: Path,
    tick_step_x: float | None = None,
    tick_step_y: float | None = None,
    force_zero_x: bool = False,
    force_zero_y: bool = False,
) -> None:
    size = 620
    left, right, top, bottom = 86, 160, 58, 82
    plot = min(size - left - right, size - top - bottom)
    width = left + plot + right
    height = top + plot + bottom
    points = []
    for strategy, label, colour in STRATEGIES:
        if (x_metric, strategy) not in summary or (y_metric, strategy) not in summary:
            continue
        x = summary[(x_metric, strategy)]
        y = summary[(y_metric, strategy)]
        points.append((strategy, label, colour, x, y))
    x_min = min(p[3]["ci_low"] for p in points)
    x_max = max(p[3]["ci_high"] for p in points)
    y_min = min(p[4]["ci_low"] for p in points)
    y_max = max(p[4]["ci_high"] for p in points)
    x_pad = (x_max - x_min) * 0.12 if x_max > x_min else 1
    y_pad = (y_max - y_min) * 0.12 if y_max > y_min else 1
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad
    if x_metric in {"module_nmi", "module_ari"}:
        x_min, x_max = max(0.0, x_min), min(1.02, x_max)
    if y_metric in {"module_nmi", "module_ari"}:
        y_min, y_max = max(0.0, y_min), min(1.02, y_max)
    if force_zero_x:
        x_min = 0.0
    if force_zero_y:
        y_min = 0.0

    def sx(v: float) -> float:
        return left + (v - x_min) / (x_max - x_min) * plot

    def sy(v: float) -> float:
        return top + (y_max - v) / (y_max - y_min) * plot

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(left, 26, title, 18, "bold"),
        f'<line x1="{left}" y1="{top+plot}" x2="{left+plot}" y2="{top+plot}" stroke="#222" stroke-width="1.4"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot}" stroke="#222" stroke-width="1.4"/>',
    ]
    for t in ticks(x_min, x_max, tick_step_x):
        if t < x_min - 1e-9 or t > x_max + 1e-9:
            continue
        x = sx(t)
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot}" stroke="#ECEFF2" stroke-width="0.8"/>')
        lines.append(f'<line x1="{x:.1f}" y1="{top+plot}" x2="{x:.1f}" y2="{top+plot+5}" stroke="#222"/>')
        lines.append(svg_text(x, top + plot + 22, fmt(t), 10, "normal", "#222", "middle"))
    for t in ticks(y_min, y_max, tick_step_y):
        if t < y_min - 1e-9 or t > y_max + 1e-9:
            continue
        y = sy(t)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot}" y2="{y:.1f}" stroke="#ECEFF2" stroke-width="0.8"/>')
        lines.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#222"/>')
        lines.append(svg_text(left - 10, y + 4, fmt(t), 10, "normal", "#222", "end"))
    for _, label, colour, x, y in points:
        xm, ym = x["mean"], y["mean"]
        lines.append(f'<line x1="{sx(x["ci_low"]):.1f}" y1="{sy(ym):.1f}" x2="{sx(x["ci_high"]):.1f}" y2="{sy(ym):.1f}" stroke="{colour}" stroke-width="2"/>')
        lines.append(f'<line x1="{sx(xm):.1f}" y1="{sy(y["ci_low"]):.1f}" x2="{sx(xm):.1f}" y2="{sy(y["ci_high"]):.1f}" stroke="{colour}" stroke-width="2"/>')
        lines.append(f'<circle cx="{sx(xm):.1f}" cy="{sy(ym):.1f}" r="5.2" fill="{colour}" stroke="#222" stroke-width="0.5"/>')
    lines.append(svg_text(left + plot / 2, height - 24, x_label, 13, "normal", "#222", "middle"))
    lines.append(f'<text x="20" y="{top + plot/2:.1f}" font-size="13" font-family="Arial, Helvetica, sans-serif" fill="#222" text-anchor="middle" transform="rotate(-90 20 {top + plot/2:.1f})">{html.escape(y_label)}</text>')
    legend_x = left + plot + 28
    legend_y = top + 12
    lines.append(svg_text(legend_x, legend_y, "Sampling strategy", 12, "bold"))
    y0 = legend_y + 24
    for _, label, colour, _, _ in points:
        lines.append(f'<circle cx="{legend_x+6}" cy="{y0-4}" r="5.0" fill="{colour}" stroke="#222" stroke-width="0.5"/>')
        lines.append(svg_text(legend_x + 18, y0, label, 11))
        y0 += 22
    lines.append("</svg>")
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    figure_dir = Path(__file__).resolve().parents[1]
    parser.add_argument("--summary", default=str(figure_dir / "data" / "selected_network_metrics_ci.tsv"))
    parser.add_argument("--outdir", default=str(figure_dir / "reproduced_metric_pairs"))
    args = parser.parse_args()
    summary = read_summary(Path(args.summary))
    outdir = Path(args.outdir)
    render_plot(
        summary,
        "n_nodes",
        "n_edges",
        "No. nodes",
        "No. edges",
        "Nodes and edges after downsampling",
        outdir / "nodes_edges_ci_scatter.svg",
        tick_step_x=1,
        tick_step_y=25,
    )
    render_plot(
        summary,
        "module_nmi",
        "module_ari",
        "NMI",
        "ARI",
        "Community similarity after downsampling",
        outdir / "nmi_ari_ci_scatter.svg",
        tick_step_x=0.1,
        tick_step_y=0.1,
        force_zero_x=True,
        force_zero_y=True,
    )


if __name__ == "__main__":
    main()
