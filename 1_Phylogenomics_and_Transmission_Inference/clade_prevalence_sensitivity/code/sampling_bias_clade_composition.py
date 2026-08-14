"""Assess clade-composition robustness under four resampling strategies.

Annual and overall ESL2.3/2.4/2.5 proportions are calculated for 100 fixed-
seed replicates per strategy.  The script exports replicate-level and summary
tables before drawing the sensitivity panel.
"""

from __future__ import annotations

from pathlib import Path
import html
import math

import numpy as np
import pandas as pd


FIGURE_DIR = Path(__file__).resolve().parents[1]
MODULE_DIR = Path(__file__).resolve().parents[2]
INPUT = MODULE_DIR / "sample_metadata.xlsx"
OUTDIR = FIGURE_DIR / "clade_composition_output"
N_REPS = 100
SEED = 20260609
CLADES = ["2.3", "2.4", "2.5"]
CLade_LABELS = {"2.3": "ESL2.3", "2.4": "ESL2.4", "2.5": "ESL2.5"}
COLORS = {"2.3": "#134E5A", "2.4": "#EB6564", "2.5": "#F5BA1A"}
STRATEGIES = [
    ("random_50", "50% random downsampling"),
    ("year_balanced", "Year-balanced sampling"),
    ("city_balanced", "City-balanced sampling"),
    ("city_year_balanced", "City + year-balanced sampling"),
]


def load_metadata() -> pd.DataFrame:
    meta = pd.read_excel(INPUT, sheet_name="S1 Metadata")
    df = meta[(meta["Country"].astype(str) == "China") & (meta["IC"].astype(str) == "IC2")].copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["City"] = df["City"].astype(str)
    df["clade"] = df["ESL cluster"].astype(str).str.strip()
    df = df[df["Year"].between(2010, 2024) & df["clade"].isin(CLADES)].copy()
    df = df.dropna(subset=["Year"])
    df["Year"] = df["Year"].astype(int)
    return df.reset_index(drop=True)


def sample_year_balanced(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    parts = []
    counts = df.groupby("Year").size()
    # Use the median yearly sample size as the cap; sparse years are kept in full.
    cap = int(np.median(counts))
    for _, g in df.groupby("Year", sort=False):
        if len(g) <= cap:
            parts.append(g)
        else:
            parts.append(g.sample(n=cap, replace=False, random_state=int(rng.integers(1, 2**31 - 1))))
    return pd.concat(parts, ignore_index=True)


def sample_city_balanced(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    parts = []
    for _, g in df.groupby("City", sort=False):
        if len(g) <= 20:
            parts.append(g)
        else:
            parts.append(g.sample(n=20, replace=False, random_state=int(rng.integers(1, 2**31 - 1))))
    return pd.concat(parts, ignore_index=True)


def sample_strategy(df: pd.DataFrame, strategy: str, rng: np.random.Generator) -> pd.DataFrame:
    if strategy == "random_50":
        return df.sample(frac=0.5, replace=False, random_state=int(rng.integers(1, 2**31 - 1))).reset_index(drop=True)
    if strategy == "year_balanced":
        return sample_year_balanced(df, rng)
    if strategy == "city_balanced":
        return sample_city_balanced(df, rng)
    if strategy == "city_year_balanced":
        return sample_year_balanced(sample_city_balanced(df, rng), rng)
    raise ValueError(strategy)


def proportions(df: pd.DataFrame, strategy: str, replicate: int) -> tuple[list[dict], list[dict]]:
    annual = []
    for year, gy in df.groupby("Year"):
        denom = len(gy)
        counts = gy["clade"].value_counts()
        for clade in CLADES:
            annual.append(
                {
                    "strategy": strategy,
                    "replicate": replicate,
                    "year": int(year),
                    "clade": clade,
                    "n_clade": int(counts.get(clade, 0)),
                    "n_total": int(denom),
                    "proportion": float(counts.get(clade, 0) / denom) if denom else np.nan,
                }
            )
    total = []
    denom = len(df)
    counts = df["clade"].value_counts()
    for clade in CLADES:
        total.append(
            {
                "strategy": strategy,
                "replicate": replicate,
                "clade": clade,
                "n_clade": int(counts.get(clade, 0)),
                "n_total": int(denom),
                "proportion": float(counts.get(clade, 0) / denom) if denom else np.nan,
            }
        )
    return annual, total


def summarize_annual(rep: pd.DataFrame) -> pd.DataFrame:
    out = (
        rep.groupby(["strategy", "year", "clade"])
        .agg(
            mean_proportion=("proportion", "mean"),
            lower95=("proportion", lambda x: np.quantile(x, 0.025)),
            upper95=("proportion", lambda x: np.quantile(x, 0.975)),
            mean_n_total=("n_total", "mean"),
        )
        .reset_index()
    )
    return out


def summarize_total(rep: pd.DataFrame) -> pd.DataFrame:
    return (
        rep.groupby(["strategy", "clade"])
        .agg(
            mean_proportion=("proportion", "mean"),
            lower95=("proportion", lambda x: np.quantile(x, 0.025)),
            upper95=("proportion", lambda x: np.quantile(x, 0.975)),
            mean_n_total=("n_total", "mean"),
        )
        .reset_index()
    )


def scale(x, lo, hi, px_lo, px_hi):
    return px_lo + (x - lo) / (hi - lo) * (px_hi - px_lo)


def text(x, y, s, size=11, anchor="middle", weight="normal", rotate=None, fill="#222"):
    t = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate is not None else ""
    return f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}"{t}>{html.escape(str(s))}</text>'


def line(x1, y1, x2, y2, stroke="#222", width=1, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width}"{d}/>'


def path(points, color, width=2):
    if not points:
        return ""
    d = " ".join([("M" if i == 0 else "L") + f" {x:.2f} {y:.2f}" for i, (x, y) in enumerate(points)])
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>'


def catmull_rom(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    if len(points) < 3:
        return " ".join([("M" if i == 0 else "L") + f" {x:.2f} {y:.2f}" for i, (x, y) in enumerate(points)])
    d = f"M {points[0][0]:.2f} {points[0][1]:.2f}"
    for i in range(len(points) - 1):
        p0 = points[max(i - 1, 0)]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[min(i + 2, len(points) - 1)]
        c1x = p1[0] + (p2[0] - p0[0]) / 6
        c1y = p1[1] + (p2[1] - p0[1]) / 6
        c2x = p2[0] - (p3[0] - p1[0]) / 6
        c2y = p2[1] - (p3[1] - p1[1]) / 6
        d += f" C {c1x:.2f} {c1y:.2f}, {c2x:.2f} {c2y:.2f}, {p2[0]:.2f} {p2[1]:.2f}"
    return d


def smooth_path(points: list[tuple[float, float]], color: str, width=2.2, opacity=1.0) -> str:
    d = catmull_rom(points)
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round" opacity="{opacity}"/>'


def smooth_band(upper: list[tuple[float, float]], lower: list[tuple[float, float]], color: str, opacity=0.16) -> str:
    if not upper or not lower:
        return ""
    d_upper = catmull_rom(upper)
    # Convert the lower curve into straight/smooth reverse path without an initial M.
    lower_rev = list(reversed(lower))
    d_lower = catmull_rom(lower_rev)
    if d_lower.startswith("M"):
        d_lower = "L" + d_lower[1:]
    d = f"{d_upper} {d_lower} Z"
    return f'<path d="{d}" fill="{color}" opacity="{opacity}" stroke="none"/>'


def draw_strategy_panel(strategy: str, title: str, annual: pd.DataFrame, total: pd.DataFrame, x0: int, y0: int) -> list[str]:
    parts = []
    w1, h = 500, 260
    w2 = 110
    gap = 52
    left = x0 + 56
    top = y0 + 42
    years = list(range(2010, 2025))
    y_min, y_max = 0, 1
    parts.append(text(x0, y0 + 4, title, size=17, anchor="start", weight="bold"))
    # annual axis
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        yy = scale(tick, y_min, y_max, top + h, top)
        parts.append(line(left, yy, left + w1, yy, stroke="#E6E6E6", width=1))
        parts.append(text(left - 9, yy + 4, f"{int(tick*100)}", size=10, anchor="end"))
    parts.append(line(left, top, left, top + h, width=1.2))
    parts.append(line(left, top + h, left + w1, top + h, width=1.2))
    parts.append(text(left - 42, top + h / 2, "Annual proportion (%)", size=12, rotate=-90))
    for yr in [2010, 2014, 2018, 2022, 2024]:
        xx = scale(yr, 2010, 2024, left, left + w1)
        parts.append(text(xx, top + h + 18, yr, size=10))
    parts.append(text(left + w1 / 2, top + h + 38, "Sampling year", size=12))

    for clade in CLADES:
        dat = annual[(annual["strategy"] == strategy) & (annual["clade"] == clade)].set_index("year")
        pts = []
        upper_pts = []
        lower_pts = []
        for yr in years:
            if yr not in dat.index:
                continue
            row = dat.loc[yr]
            xx = scale(yr, 2010, 2024, left, left + w1)
            yy = scale(row["mean_proportion"], y_min, y_max, top + h, top)
            ylo = scale(row["lower95"], y_min, y_max, top + h, top)
            yhi = scale(row["upper95"], y_min, y_max, top + h, top)
            pts.append((xx, yy))
            upper_pts.append((xx, yhi))
            lower_pts.append((xx, ylo))
        parts.append(smooth_band(upper_pts, lower_pts, COLORS[clade], opacity=0.24))
        parts.append(smooth_path(pts, COLORS[clade], width=2.4))

    # total axis
    left2 = left + w1 + gap
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        yy = scale(tick, y_min, y_max, top + h, top)
        parts.append(line(left2, yy, left2 + w2, yy, stroke="#E6E6E6", width=1))
        parts.append(text(left2 - 9, yy + 4, f"{int(tick*100)}", size=10, anchor="end"))
    parts.append(line(left2, top, left2, top + h, width=1.2))
    parts.append(line(left2, top + h, left2 + w2, top + h, width=1.2))
    parts.append(text(left2 + w2 / 2, top - 16, "Overall", size=13, weight="bold"))
    parts.append(text(left2 - 42, top + h / 2, "Total proportion (%)", size=12, rotate=-90))
    bar_w = 22
    for i, clade in enumerate(CLADES):
        row = total[(total["strategy"] == strategy) & (total["clade"] == clade)].iloc[0]
        xx = left2 + 24 + i * 34
        yy = scale(row["mean_proportion"], y_min, y_max, top + h, top)
        base = scale(0, y_min, y_max, top + h, top)
        parts.append(f'<rect x="{xx - bar_w/2:.2f}" y="{yy:.2f}" width="{bar_w:.2f}" height="{base - yy:.2f}" fill="{COLORS[clade]}"/>')
        parts.append(text(xx, top + h + 20, CLade_LABELS[clade], size=9, rotate=-35, anchor="end"))
    return parts


def draw_svg(annual: pd.DataFrame, total: pd.DataFrame):
    width, height = 1540, 760
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    positions = [(30, 28), (800, 28), (30, 390), (800, 390)]
    for (strategy, title), (x, y) in zip(STRATEGIES, positions):
        parts.extend(draw_strategy_panel(strategy, title, annual, total, x, y))
    # legend
    lx, ly = 1190, 708
    for i, clade in enumerate(CLADES):
        x = lx + i * 95
        parts.append(f'<circle cx="{x:.2f}" cy="{ly:.2f}" r="5.5" fill="{COLORS[clade]}"/>')
        parts.append(text(x + 12, ly + 4, CLade_LABELS[clade], size=13, anchor="start"))
    parts.append("</svg>")
    (OUTDIR / "clade_composition_sampling_bias.svg").write_text("\n".join(parts), encoding="utf-8")


def main():
    OUTDIR.mkdir(exist_ok=True)
    df = load_metadata()
    rng = np.random.default_rng(SEED)
    annual_rows, total_rows = [], []

    # Include the full data as a reference table, but not in the four-panel figure.
    a, t = proportions(df, "full", 0)
    pd.DataFrame(a).to_csv(OUTDIR / "full_annual_clade_proportions.tsv", sep="\t", index=False)
    pd.DataFrame(t).to_csv(OUTDIR / "full_total_clade_proportions.tsv", sep="\t", index=False)

    for strategy, _ in STRATEGIES:
        for rep in range(1, N_REPS + 1):
            sampled = sample_strategy(df, strategy, rng)
            a, t = proportions(sampled, strategy, rep)
            annual_rows.extend(a)
            total_rows.extend(t)

    annual_rep = pd.DataFrame(annual_rows)
    total_rep = pd.DataFrame(total_rows)
    annual_sum = summarize_annual(annual_rep)
    total_sum = summarize_total(total_rep)
    annual_rep.to_csv(OUTDIR / "annual_clade_proportion_replicates.tsv", sep="\t", index=False)
    total_rep.to_csv(OUTDIR / "total_clade_proportion_replicates.tsv", sep="\t", index=False)
    annual_sum.to_csv(OUTDIR / "annual_clade_proportion_summary.tsv", sep="\t", index=False)
    total_sum.to_csv(OUTDIR / "total_clade_proportion_summary.tsv", sep="\t", index=False)
    draw_svg(annual_sum, total_sum)

    # Compact values for the selected years.
    esl25 = annual_sum[(annual_sum["clade"] == "2.5") & (annual_sum["year"].isin([2017, 2018, 2019, 2020, 2021, 2022, 2023]))]
    esl25.to_csv(OUTDIR / "ESL25_key_year_summary.tsv", sep="\t", index=False)
    print("Input genomes:", len(df))
    print("Wrote:", (OUTDIR / "clade_composition_sampling_bias.svg").resolve())
    print("ESL2.5 key years:")
    print(esl25.to_string(index=False))


if __name__ == "__main__":
    main()
