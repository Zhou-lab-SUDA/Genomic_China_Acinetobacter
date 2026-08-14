"""Plot nominal and adjusted city-association P values for RF8.

The script reads the frozen comparison table and writes both an SVG panel and
the exact values displayed in that panel.
"""

from pathlib import Path
import html
import math

import pandas as pd


FIGURE_DIR = Path(__file__).resolve().parents[1]
INFILE = FIGURE_DIR / "data" / "city_association_multiple_testing.tsv"
OUTBASE = FIGURE_DIR / "RF8_city_association_multiple_testing_reproduced"


def sx(i: int, left: float, plot_w: float, n: int) -> float:
    return left + (i + 0.5) * plot_w / n


def sy(v: float, top: float, plot_h: float, ymin: float, ymax: float) -> float:
    return top + (ymax - v) / (ymax - ymin) * plot_h


def fmt(v: float) -> str:
    return f"{v:.3f}".rstrip("0").rstrip(".")


def p_label(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    return fmt(p)


def text(x, y, body, size=12, anchor="middle", weight="normal", rotate=None, fill="#222"):
    body = html.escape(str(body))
    trans = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}"{trans}>{body}</text>'
    )


def line(x1, y1, x2, y2, stroke="#222", width=1, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'


def main() -> None:
    df = pd.read_csv(INFILE, sep="\t")
    if "figure_order" in df.columns:
        df = df.sort_values("figure_order")
    df = df.reset_index(drop=True)
    if "rho_to_report" in df.columns:
        df["rho_plot"] = df["rho_to_report"]
    elif "spearman_rho" in df.columns:
        df["rho_plot"] = df["spearman_rho"]
    else:
        raise ValueError("Input table must contain rho_to_report or spearman_rho")
    df["raw_p_lt_0.05"] = df["nominal_p"] < 0.05
    df["BH_q_lt_0.05"] = df["BH_adjusted_q"] < 0.05
    df["neglog10_p"] = df["nominal_p"].clip(lower=1e-300).map(lambda x: -math.log10(x))
    df["neglog10_q"] = df["BH_adjusted_q"].clip(lower=1e-300).map(lambda x: -math.log10(x))

    n = len(df)
    width, height = 1380, 840
    left, right = 92, 24
    top1, h1 = 70, 360
    gap = 55
    top2, h2 = top1 + h1 + gap, 160
    bottom = 170
    plot_w = width - left - right
    ymin1, ymax1 = -0.43, 0.33
    ymax2 = max(2.6, float(df[["neglog10_p", "neglog10_q"]].max().max()) + 0.35)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(22, 48, "e", size=44, anchor="start", weight="bold"),
    ]

    # Panel 1 grid and y axis.
    for tick in [-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]:
        y = sy(tick, top1, h1, ymin1, ymax1)
        parts.append(line(left, y, width - right, y, stroke="#E0E0E0", width=2))
        parts.append(text(left - 15, y + 5, fmt(tick), size=24, anchor="end"))
    y0 = sy(0, top1, h1, ymin1, ymax1)
    parts.append(line(left, y0, width - right, y0, stroke="#111", width=2.2))
    parts.append(line(left, top1, left, top1 + h1, stroke="#111", width=2))
    parts.append(text(35, top1 + h1 / 2, "Spearman rho", size=34, rotate=-90))

    bar_w = min(31, plot_w / n * 0.48)
    for i, row in df.iterrows():
        x = sx(i, left, plot_w, n)
        rho = float(row["rho_plot"])
        if rho < 0:
            color = "#7EA6D8" if bool(row["raw_p_lt_0.05"]) else "#C7DFF2"
        else:
            color = "#D84A63" if bool(row["raw_p_lt_0.05"]) else "#F5BAC5"
        if bool(row["BH_q_lt_0.05"]):
            color = "#2C5AA0" if rho < 0 else "#B91E3A"
        y_val, y_base = sy(rho, top1, h1, ymin1, ymax1), sy(0, top1, h1, ymin1, ymax1)
        y_rect, h_rect = min(y_val, y_base), abs(y_val - y_base)
        parts.append(f'<rect x="{x - bar_w/2:.2f}" y="{y_rect:.2f}" width="{bar_w:.2f}" height="{h_rect:.2f}" fill="{color}"/>')
        if bool(row["BH_q_lt_0.05"]) or bool(row["raw_p_lt_0.05"]):
            marker = "**" if bool(row["BH_q_lt_0.05"]) else "*"
            my = y_val - 10 if rho >= 0 else y_val + 25
            parts.append(text(x, my, marker, size=18, weight="bold"))

    parts.append(text(width - right, top1 + 18, "* raw P < 0.05; ** BH-adjusted q < 0.05", size=14, anchor="end", fill="#444"))

    # Panel 2: -log10 P/q.
    for tick in [0, 0.5, 1.0, 1.30103, 1.5, 2.0, 2.5]:
        if tick > ymax2:
            continue
        y = sy(tick, top2, h2, 0, ymax2)
        color = "#B85C5C" if abs(tick - 1.30103) < 1e-5 else "#E8E8E8"
        dash = "6 5" if abs(tick - 1.30103) < 1e-5 else None
        parts.append(line(left, y, width - right, y, stroke=color, width=1.2, dash=dash))
        label = "0.05" if abs(tick - 1.30103) < 1e-5 else fmt(tick)
        parts.append(text(left - 12, y + 4, label, size=14, anchor="end", fill="#8A3F3F" if label == "0.05" else "#222"))
    parts.append(line(left, top2, left, top2 + h2, stroke="#111", width=1.8))
    parts.append(line(left, top2 + h2, width - right, top2 + h2, stroke="#111", width=1.8))
    parts.append(text(36, top2 + h2 / 2, "-log10(P or q)", size=19, rotate=-90))

    p_points, q_points = [], []
    for i, row in df.iterrows():
        x = sx(i, left, plot_w, n)
        yp = sy(float(row["neglog10_p"]), top2, h2, 0, ymax2)
        yq = sy(float(row["neglog10_q"]), top2, h2, 0, ymax2)
        p_points.append((x, yp))
        q_points.append((x, yq))
    for points, color in [(p_points, "#333333"), (q_points, "#8A8A8A")]:
        d = " ".join([("M" if i == 0 else "L") + f" {x:.2f} {y:.2f}" for i, (x, y) in enumerate(points)])
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.8"/>')
    for x, y in p_points:
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.6" fill="#333333"/>')
    for x, y in q_points:
        parts.append(f'<rect x="{x-4.1:.2f}" y="{y-4.1:.2f}" width="8.2" height="8.2" fill="#8A8A8A"/>')

    lx, ly = width - 250, top2 + 20
    parts.append(f'<circle cx="{lx:.2f}" cy="{ly:.2f}" r="4.6" fill="#333333"/>')
    parts.append(text(lx + 15, ly + 5, "raw P", size=14, anchor="start"))
    parts.append(f'<rect x="{lx-4.1:.2f}" y="{ly+20-4.1:.2f}" width="8.2" height="8.2" fill="#8A8A8A"/>')
    parts.append(text(lx + 15, ly + 25, "BH-adjusted q", size=14, anchor="start"))

    for i, row in df.iterrows():
        x = sx(i, left, plot_w, n)
        parts.append(text(x, top2 + h2 + 33, row["Variable"], size=22, rotate=-50, anchor="end"))

    parts.append("</svg>")
    OUTBASE.with_suffix(".svg").write_text("\n".join(parts), encoding="utf-8")

    table = df[
        [
            "Variable",
            "rho_plot",
            "nominal_p",
            "BH_adjusted_q",
            "raw_p_lt_0.05",
            "BH_q_lt_0.05",
        ]
    ].rename(columns={"rho_plot": "Spearman_rho_display"})
    table["nominal_p_label"] = table["nominal_p"].map(p_label)
    table["BH_adjusted_q_label"] = table["BH_adjusted_q"].map(p_label)
    table.to_csv(OUTBASE.with_suffix(".tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()
