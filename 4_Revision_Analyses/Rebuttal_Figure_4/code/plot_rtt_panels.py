#!/usr/bin/env python3
"""Draw four-panel root-to-tip temporal-signal scatter plots as SVG."""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path
from typing import Dict, List

import pandas as pd


COLORS = {
    "All": "#9E9E9E",
    "2.3": "#134E5A",
    "2.4": "#EB6564",
    "2.5": "#F5BA1A",
}
OUTLIER_COLOR = "#D62728"
BASE_RADIUS = 2.4
POINT_RADIUS = BASE_RADIUS * math.sqrt(2.5)  # 150% larger area than the original.
POINT_OPACITY = 0.1
OUTLIER_OPACITY = 0.1


def fmt_p(p: float) -> str:
    if math.isnan(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def load_inputs(distances: Path, results: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(distances, sep="\t")
    r = pd.read_csv(results, sep="\t")
    return d, r


def panel_svg(
    rows: pd.DataFrame,
    result: Dict[str, float],
    group: str,
    x: float,
    y: float,
    w: float,
    h: float,
) -> List[str]:
    color = COLORS[group]
    left, right, top, bottom = 58, 18, 34, 50
    px, py = x + left, y + top
    pw, ph = w - left - right, h - top - bottom

    xs = rows["year"].astype(float).tolist()
    ys = rows["root_to_tip"].astype(float).tolist()
    slope = float(result["slope"])
    intercept = float(result["intercept"])
    residuals = [yv - (intercept + slope * xv) for xv, yv in zip(xs, ys)]
    if len(residuals) >= 4:
        q1, q3 = statistics.quantiles(residuals, n=4)[0], statistics.quantiles(residuals, n=4)[2]
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        is_outlier = [rv < lo or rv > hi for rv in residuals]
    else:
        is_outlier = [False for _ in residuals]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmin == xmax:
        xmin -= 0.5
        xmax += 0.5
    if ymin == ymax:
        ymin -= 0.5
        ymax += 0.5
    ypad = (ymax - ymin) * 0.08
    ymin -= ypad
    ymax += ypad

    def sx(v: float) -> float:
        return px + (v - xmin) / (xmax - xmin) * pw

    def sy(v: float) -> float:
        return py + (ymax - v) / (ymax - ymin) * ph

    out = []
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="white"/>')
    out.append(f'<text x="{x + w/2}" y="{y + 20}" text-anchor="middle" font-family="Arial" font-size="17" font-weight="bold">{group}</text>')
    out.append(f'<line x1="{px}" y1="{py+ph}" x2="{px+pw}" y2="{py+ph}" stroke="black" stroke-width="1"/>')
    out.append(f'<line x1="{px}" y1="{py}" x2="{px}" y2="{py+ph}" stroke="black" stroke-width="1"/>')

    for i in range(4):
        xv = xmin + (xmax - xmin) * i / 3
        xx = sx(xv)
        out.append(f'<line x1="{xx:.2f}" y1="{py+ph}" x2="{xx:.2f}" y2="{py+ph+5}" stroke="black"/>')
        out.append(f'<text x="{xx:.2f}" y="{py+ph+20}" text-anchor="middle" font-family="Arial" font-size="11">{xv:.0f}</text>')
    for i in range(4):
        yv = ymin + (ymax - ymin) * i / 3
        yy = sy(yv)
        out.append(f'<line x1="{px-5}" y1="{yy:.2f}" x2="{px}" y2="{yy:.2f}" stroke="black"/>')
        out.append(f'<text x="{px-8}" y="{yy+4:.2f}" text-anchor="end" font-family="Arial" font-size="10">{yv:.3f}</text>')

    for xv, yv, outlier in zip(xs, ys, is_outlier):
        fill = OUTLIER_COLOR if outlier else color
        opacity = OUTLIER_OPACITY if outlier else POINT_OPACITY
        stroke = ' stroke="black" stroke-width="0.35"' if outlier else ""
        out.append(f'<circle cx="{sx(xv):.2f}" cy="{sy(yv):.2f}" r="{POINT_RADIUS:.2f}" fill="{fill}" fill-opacity="{opacity}"{stroke}/>')

    if not math.isnan(slope):
        x1, x2 = xmin, xmax
        y1 = intercept + slope * x1
        y2 = intercept + slope * x2
        out.append(f'<line x1="{sx(x1):.2f}" y1="{sy(y1):.2f}" x2="{sx(x2):.2f}" y2="{sy(y2):.2f}" stroke="{color}" stroke-width="2.6"/>')

    r = float(result["r"])
    p = float(result["permutation_p_two_sided"])
    stat = f'R = {r:.3f}; P = {fmt_p(p)}'
    out.append(f'<text x="{px + 12}" y="{py + 22}" font-family="Arial" font-size="13">{stat}</text>')
    out.append(f'<text x="{px + 12}" y="{py + 40}" font-family="Arial" font-size="12">n = {int(result["n"])}</text>')
    out.append(f'<text x="{px + 12}" y="{py + 56}" font-family="Arial" font-size="11" fill="{OUTLIER_COLOR}">outliers = {sum(is_outlier)}</text>')

    out.append(f'<text x="{px + pw/2}" y="{y+h-13}" text-anchor="middle" font-family="Arial" font-size="13">Sampling year</text>')
    out.append(
        f'<text x="{x+15}" y="{py + ph/2}" transform="rotate(-90 {x+15},{py + ph/2})" text-anchor="middle" font-family="Arial" font-size="13">Root-to-tip distance</text>'
    )
    return out


def main() -> None:
    figure_dir = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distances", default=str(figure_dir / "data" / "root_to_tip_distances.tsv"))
    ap.add_argument("--results", default=str(figure_dir / "data" / "temporal_signal_results.tsv"))
    ap.add_argument("--output", default=str(figure_dir / "RF4_root_to_tip_panels_reproduced.svg"))
    args = ap.parse_args()

    distances, results = load_inputs(Path(args.distances), Path(args.results))
    result_map = {str(row["group"]): row for _, row in results.iterrows()}

    panel_w, panel_h = 260, 190
    gap_x, gap_y = 18, 18
    width = panel_w * 2 + gap_x
    height = panel_h * 2 + gap_y
    groups = ["All", "2.3", "2.4", "2.5"]
    coords = {
        "All": (0, 0),
        "2.3": (panel_w + gap_x, 0),
        "2.4": (0, panel_h + gap_y),
        "2.5": (panel_w + gap_x, panel_h + gap_y),
    }

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    svg.append('<rect width="100%" height="100%" fill="white"/>')
    for g in groups:
        if g == "All":
            rows = distances.copy()
        else:
            rows = distances[distances["group"].astype(str) == g].copy()
        if rows.empty:
            continue
        x, y = coords[g]
        svg.extend(panel_svg(rows, result_map[g], g, x, y, panel_w, panel_h))
    svg.append("</svg>")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(svg), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
