"""RTTD-based sampling-bias sensitivity analysis.

This is a fast screening analysis for two metrics:
1. clock-rate proxy: slope of root-to-tip distance against sampling year
2. temporal signal: Pearson R of root-to-tip distance against sampling year

It compares full data with random, population-weighted, year-balanced,
city-balanced and combined city/year-balanced sampling.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

import openpyxl


def norm_id(value: str) -> str:
    value = str(value).strip()
    for suffix in [".fna.gz", ".fasta.gz", ".fa.gz", ".fna", ".fasta", ".fa"]:
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value


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


def regression_slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    mx = mean(xs)
    my = mean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def read_rttd(path: Path) -> list[dict[str, object]]:
    out = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            key = norm_id(row["genome"])
            out.append({
                "genome": row["genome"],
                "key": key,
                "year": float(row["year"]),
                "root_to_tip": float(row["root_to_tip"]),
                "group": row["group"],
            })
    return out


def read_metadata(path: Path, sheet: str) -> dict[str, dict[str, object]]:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet]
    headers = [ws.cell(1, c).value for c in range(1, ws.max_column + 1)]
    idx = {name: i for i, name in enumerate(headers) if name}
    out = {}
    for cells in ws.iter_rows(min_row=2, values_only=True):
        genome = cells[idx["genome"]]
        if genome is None:
            continue
        key = norm_id(str(genome))
        out[key] = {
            "metadata_genome": str(genome),
            "year_meta": cells[idx["Year"]],
            "city": cells[idx["City"]] or "",
            "province": cells[idx["China Province/Europe Countries"]] or "",
            "country": cells[idx["Country"]] or "",
            "esl": cells[idx["ESL cluster"]] or "",
        }
    return out


def read_city_populations(path: Path, sheet: str = "S2 CityAttr") -> dict[str, float]:
    """Read city populations used to allocate population-weighted samples."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet]
    headers = [ws.cell(1, c).value for c in range(1, ws.max_column + 1)]
    idx = {name: i for i, name in enumerate(headers) if name}
    populations: dict[str, float] = {}
    for cells in ws.iter_rows(min_row=2, values_only=True):
        city = cells[idx["City"]]
        population = cells[idx["Population (10K)"]]
        if city is None or population is None:
            continue
        try:
            populations[str(city).strip()] = float(population)
        except (TypeError, ValueError):
            continue
    if "Tai'an" in populations:
        populations["Taian"] = populations["Tai'an"]
    return populations


def join_rows(rttd: list[dict[str, object]], metadata: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for rec in rttd:
        key = str(rec["key"])
        meta = metadata.get(key)
        if meta:
            rows.append({**rec, **meta, "has_city_match": True})
        else:
            rows.append({**rec, "city": "", "province": "", "country": "", "esl": "", "has_city_match": False})
    return rows


def weighted_sample_without_replacement(
    rows: list[dict[str, object]], n: int, weights: list[float], rng: random.Random
) -> list[dict[str, object]]:
    remaining = list(range(len(rows)))
    selected = []
    current_weights = weights[:]
    for _ in range(min(n, len(rows))):
        total = sum(current_weights[i] for i in remaining)
        if total <= 0:
            pick = rng.choice(remaining)
        else:
            threshold = rng.random() * total
            acc = 0.0
            pick = remaining[-1]
            for i in remaining:
                acc += current_weights[i]
                if acc >= threshold:
                    pick = i
                    break
        selected.append(rows[pick])
        remaining.remove(pick)
    return selected


def downsample(
    rows: list[dict[str, object]],
    strategy: str,
    fraction: float,
    rng: random.Random,
    city_populations: dict[str, float],
) -> list[dict[str, object]]:
    n = max(3, int(round(len(rows) * fraction)))
    if strategy == "random":
        return rng.sample(rows, n)
    if strategy == "city_weighted":
        counts = Counter(str(r["city"]) or "NA" for r in rows)
        known = [value for value in city_populations.values() if value > 0]
        fallback = sorted(known)[len(known) // 2] if known else 1.0
        weights = [
            max(city_populations.get(str(r["city"]), fallback), 1.0)
            / counts[str(r["city"]) or "NA"]
            for r in rows
        ]
        return weighted_sample_without_replacement(rows, n, weights, rng)
    if strategy == "year_balanced":
        return year_balanced_sample(rows, n, rng)
    if strategy == "city_cap20":
        return city_cap_sample(rows, cap=20, rng=rng)
    if strategy == "city_cap20_year_balanced":
        capped = city_cap_sample(rows, cap=20, rng=rng)
        return year_balanced_sample(capped, max(3, int(round(len(capped) * fraction))), rng)
    raise ValueError(f"Unknown strategy: {strategy}")


def city_cap_sample(rows: list[dict[str, object]], cap: int, rng: random.Random) -> list[dict[str, object]]:
    """Retain sparse cities in full and cap dense cities at a fixed number."""
    by_city: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        city = str(row.get("city") or "NA")
        by_city.setdefault(city, []).append(row)
    selected: list[dict[str, object]] = []
    for city in sorted(by_city):
        bucket = by_city[city]
        if len(bucket) <= cap:
            selected.extend(bucket)
        else:
            selected.extend(rng.sample(bucket, cap))
    return selected


def year_balanced_sample(rows: list[dict[str, object]], target_n: int, rng: random.Random) -> list[dict[str, object]]:
    """Sample toward an even year distribution.

    Sparse years are preferentially retained in full; dense years are capped.
    This is intentionally different from simple inverse-probability sampling.
    """
    by_year: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        by_year.setdefault(int(float(row["year"])), []).append(row)
    years = sorted(by_year)
    if not years:
        return []

    cap = 1
    while True:
        expected = sum(min(len(by_year[y]), cap) for y in years)
        next_expected = sum(min(len(by_year[y]), cap + 1) for y in years)
        if next_expected > target_n or cap >= max(len(v) for v in by_year.values()):
            break
        cap += 1

    selected: list[dict[str, object]] = []
    leftovers: list[dict[str, object]] = []
    for year in years:
        bucket = by_year[year]
        if len(bucket) <= cap:
            selected.extend(bucket)
        else:
            picked = rng.sample(bucket, cap)
            selected.extend(picked)
            picked_ids = {id(x) for x in picked}
            leftovers.extend([x for x in bucket if id(x) not in picked_ids])

    if len(selected) > target_n:
        return rng.sample(selected, target_n)
    if len(selected) < target_n and leftovers:
        selected.extend(rng.sample(leftovers, min(target_n - len(selected), len(leftovers))))
    return selected


def metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    xs = [float(r["year"]) for r in rows]
    ys = [float(r["root_to_tip"]) for r in rows]
    return {
        "n": len(rows),
        "min_year": min(xs) if xs else "",
        "max_year": max(xs) if xs else "",
        "slope_clock_rate_proxy": regression_slope(xs, ys),
        "r": pearson_r(xs, ys),
        "n_cities": len(set(str(r["city"]) for r in rows if str(r["city"]))),
        "n_years": len(set(int(float(r["year"])) for r in rows)),
        "n_city_matched": sum(1 for r in rows if r.get("has_city_match")),
    }


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def summarise(values: list[float]) -> tuple[float, float, float, float]:
    mu = mean(values)
    sd = stdev(values) if len(values) > 1 else 0.0
    return mu, sd, mu - 1.96 * sd, mu + 1.96 * sd


def metric_value(rows: list[dict[str, object]], metric: str) -> float:
    xs = [float(r["year"]) for r in rows]
    ys = [float(r["root_to_tip"]) for r in rows]
    if metric == "r":
        return pearson_r(xs, ys)
    if metric == "slope_clock_rate_proxy":
        return regression_slope(xs, ys)
    raise ValueError(metric)


def subsampling_interval(
    rows: list[dict[str, object]],
    metric: str,
    rng: random.Random,
    *,
    subsamples: int = 10,
    fraction: float = 0.5,
    center: float | None = None,
) -> dict[str, float]:
    if len(rows) < 3:
        val = float("nan")
        return {"point": val, "ci_low": val, "ci_high": val, "se": val}
    sample_n = max(3, int(round(len(rows) * fraction)))
    vals = []
    for _ in range(subsamples):
        sample = rng.sample(rows, min(sample_n, len(rows)))
        vals.append(metric_value(sample, metric))
    vals = [v for v in vals if not math.isnan(v)]
    point = metric_value(rows, metric) if center is None else center
    sd = stdev(vals) if len(vals) > 1 else 0.0
    se = sd / math.sqrt(len(vals)) if vals else 0.0
    return {"point": point, "ci_low": point - 1.96 * se, "ci_high": point + 1.96 * se, "se": se}


def source_for_strategy(rows: list[dict[str, object]], strategy: str) -> tuple[list[dict[str, object]], bool]:
    city_strategies = {"city_weighted", "city_cap20", "city_cap20_year_balanced"}
    if strategy in city_strategies:
        return [r for r in rows if r.get("has_city_match") and str(r.get("city") or "")], True
    return rows, False


def plot_strategy_grid(
    rows: list[dict[str, object]],
    *,
    strategies: list[str],
    out: Path,
    seed: int,
    metric: str = "r",
    replicates: int = 10,
    sample_fraction: float = 0.5,
    city_populations: dict[str, float],
) -> None:
    rng = random.Random(seed + 991)
    groups = ["All", "2.4", "2.5"]
    group_labels = {"All": "China ESL2 (ST2)", "2.3": "ESL 2.3", "2.4": "ESL 2.4", "2.5": "ESL 2.5"}
    strategy_labels = {
        "random": "Random 50%",
        "city_weighted": "City-weighted",
        "year_balanced": "Year-balanced",
        "city_cap20": "City cap-20",
        "city_cap20_year_balanced": "City cap-20 + year",
    }
    colors = {"All": "#4A6FB5", "2.3": "#134E5A", "2.4": "#EB6564", "2.5": "#F5BA1A"}
    pale_blue = "#8ECAE6"
    grey = "#9AA0A6"

    panel_data: dict[tuple[str, str], dict[str, object]] = {}
    all_y: list[float] = []
    for strategy in strategies:
        for group in groups:
            group_rows = rows if group == "All" else [r for r in rows if str(r["group"]) == group]
            src_rows, is_city_subset = source_for_strategy(group_rows, strategy)
            if len(src_rows) < 3:
                continue
            true_interval = subsampling_interval(src_rows, metric, rng, subsamples=10, fraction=sample_fraction)
            reps = []
            for rep in range(1, replicates + 1):
                sampled = downsample(src_rows, strategy, sample_fraction, rng, city_populations)
                interval = subsampling_interval(sampled, metric, rng, subsamples=10, fraction=sample_fraction)
                reps.append({"replicate": rep, "n": len(sampled), **interval})
            reps = sorted(reps, key=lambda x: float(x["point"]))
            rep_points = [float(x["point"]) for x in reps if not math.isnan(float(x["point"]))]
            mu, sd, spread_low, spread_high = summarise(rep_points)
            panel_data[(strategy, group)] = {
                "true": true_interval,
                "reps": reps,
                "spread_low": spread_low,
                "spread_high": spread_high,
                "spread_mean": mu,
                "city_subset": is_city_subset,
                "n_true": len(src_rows),
            }
            all_y.extend([spread_low, spread_high, float(true_interval["ci_low"]), float(true_interval["ci_high"])])
            for r in reps:
                all_y.extend([float(r["ci_low"]), float(r["ci_high"])])

    all_y = [v for v in all_y if not math.isnan(v)]
    if metric == "r":
        ymin, ymax = -0.1, 0.6
    else:
        ymin, ymax = min(all_y), max(all_y)
        pad = max((ymax - ymin) * 0.08, 2e-8)
        ymin -= pad
        ymax += pad
    if metric == "r":
        ymin = max(ymin, -0.1)
        ymax = min(ymax, 0.6)

    panel_w = 118
    panel_h = 178
    left_pad = 100
    top_pad = 56
    row_gap = 52
    col_gap = 30
    right_pad = 140
    bottom_pad = 48
    width = left_pad + len(groups) * panel_w + (len(groups) - 1) * col_gap + right_pad
    height = top_pad + len(strategies) * panel_h + (len(strategies) - 1) * row_gap + bottom_pad

    def fmt(v: float) -> str:
        if metric == "r":
            if abs(v) >= 0.1:
                return f"{v:.1f}"
            return f"{v:.2f}"
        return f"{v:.1e}"

    def nice_ticks() -> list[float]:
        if metric == "r":
            return [-0.1, 0.0, 0.2, 0.4, 0.6]
        return [ymin + (ymax - ymin) * i / 4 for i in range(5)]

    def sx(px: float, x: float) -> float:
        return px + (x - 0.4) / (13.1 - 0.4) * panel_w

    def sy(py: float, y: float) -> float:
        return py + panel_h - (y - ymin) / (ymax - ymin) * panel_h

    ylabel = "Coefficient (R)" if metric == "r" else "RTTD slope"
    title = "Sampling-bias sensitivity of root-to-tip temporal signal"
    out_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="18" y="30" font-family="Arial" font-size="20" font-weight="700">{title}</text>',
    ]

    for gi, group in enumerate(groups):
        px = left_pad + gi * (panel_w + col_gap)
        out_lines.append(f'<text x="{px + panel_w / 2:.1f}" y="{top_pad - 18}" text-anchor="middle" font-family="Arial" font-size="14" font-weight="700">{group_labels[group]}</text>')

    ticks = nice_ticks()
    for si, strategy in enumerate(strategies):
        py = top_pad + si * (panel_h + row_gap)
        out_lines.append(f'<text x="12" y="{py + panel_h / 2:.1f}" text-anchor="start" dominant-baseline="middle" font-family="Arial" font-size="13" font-weight="700">{strategy_labels[strategy]}</text>')
        for gi, group in enumerate(groups):
            px = left_pad + gi * (panel_w + col_gap)
            panel = panel_data.get((strategy, group))
            if not panel:
                continue
            out_lines.append(f'<rect x="{px}" y="{py}" width="{panel_w}" height="{panel_h}" fill="none" stroke="black" stroke-width="1.1"/>')
            if gi == 0:
                for t in ticks:
                    y = sy(py, t)
                    out_lines.append(f'<line x1="{px - 4}" y1="{y:.1f}" x2="{px}" y2="{y:.1f}" stroke="black" stroke-width="0.9"/>')
                    out_lines.append(f'<text x="{px - 7}" y="{y + 3.5:.1f}" text-anchor="end" font-family="Arial" font-size="9">{fmt(t)}</text>')
                out_lines.append(f'<text x="{px - 52}" y="{py + panel_h / 2:.1f}" text-anchor="middle" dominant-baseline="middle" transform="rotate(-90 {px - 52} {py + panel_h / 2:.1f})" font-family="Arial" font-size="13">{ylabel}</text>')

            spread_low = float(panel["spread_low"])
            spread_high = float(panel["spread_high"])
            spread_mean = float(panel["spread_mean"])
            out_lines.append(f'<rect x="{px}" y="{sy(py, spread_high):.1f}" width="{panel_w}" height="{max(0.5, sy(py, spread_low) - sy(py, spread_high)):.1f}" fill="#E9ECEF"/>')
            out_lines.append(f'<line x1="{px}" y1="{sy(py, spread_mean):.1f}" x2="{px + panel_w}" y2="{sy(py, spread_mean):.1f}" stroke="#C9CDD2" stroke-width="1.1"/>')
            out_lines.append(f'<line x1="{px}" y1="{sy(py, spread_low):.1f}" x2="{px + panel_w}" y2="{sy(py, spread_low):.1f}" stroke="#B8BCC2" stroke-width="0.9" stroke-dasharray="3 2"/>')
            out_lines.append(f'<line x1="{px}" y1="{sy(py, spread_high):.1f}" x2="{px + panel_w}" y2="{sy(py, spread_high):.1f}" stroke="#B8BCC2" stroke-width="0.9" stroke-dasharray="3 2"/>')

            for idx, rep in enumerate(panel["reps"], start=1):
                x = sx(px, float(idx))
                out_lines.append(f'<line x1="{x:.1f}" y1="{sy(py, float(rep["ci_low"])):.1f}" x2="{x:.1f}" y2="{sy(py, float(rep["ci_high"])):.1f}" stroke="{pale_blue}" stroke-width="1.5"/>')
                out_lines.append(f'<circle cx="{x:.1f}" cy="{sy(py, float(rep["point"])):.1f}" r="2" fill="{pale_blue}"/>')

            true_x = sx(px, 12.0)
            true = panel["true"]
            out_lines.append(f'<line x1="{true_x:.1f}" y1="{sy(py, float(true["ci_low"])):.1f}" x2="{true_x:.1f}" y2="{sy(py, float(true["ci_high"])):.1f}" stroke="{colors[group]}" stroke-width="2.2"/>')
            out_lines.append(f'<circle cx="{true_x:.1f}" cy="{sy(py, float(true["point"])):.1f}" r="2.8" fill="{colors[group]}"/>')
            if si == 0:
                out_lines.append(f'<text x="{true_x - 3:.1f}" y="{sy(py, float(true["point"])) - 12:.1f}" text-anchor="end" font-family="Arial" font-size="9" fill="{grey}">True data</text>')

    lx = left_pad + len(groups) * (panel_w + col_gap) - col_gap + 28
    ly = top_pad + 20
    out_lines.append(f'<circle cx="{lx}" cy="{ly}" r="2.4" fill="{pale_blue}"/>')
    out_lines.append(f'<line x1="{lx}" y1="{ly - 15}" x2="{lx}" y2="{ly + 15}" stroke="{pale_blue}" stroke-width="1.5"/>')
    out_lines.append(f'<text x="{lx + 18}" y="{ly - 5}" font-family="Arial" font-size="12">sampling</text>')
    out_lines.append(f'<text x="{lx + 18}" y="{ly + 9}" font-family="Arial" font-size="12">replicates</text>')
    out_lines.append(f'<rect x="{lx - 6}" y="{ly + 54}" width="14" height="38" fill="#E9ECEF"/>')
    out_lines.append(f'<line x1="{lx - 6}" y1="{ly + 73}" x2="{lx + 8}" y2="{ly + 73}" stroke="#C9CDD2" stroke-width="1.2"/>')
    out_lines.append(f'<line x1="{lx - 6}" y1="{ly + 54}" x2="{lx + 8}" y2="{ly + 54}" stroke="#B8BCC2" stroke-width="0.9" stroke-dasharray="3 2"/>')
    out_lines.append(f'<line x1="{lx - 6}" y1="{ly + 92}" x2="{lx + 8}" y2="{ly + 92}" stroke="#B8BCC2" stroke-width="0.9" stroke-dasharray="3 2"/>')
    out_lines.append(f'<text x="{lx + 18}" y="{ly + 68}" font-family="Arial" font-size="12">95% spread</text>')
    out_lines.append(f'<text x="{lx + 18}" y="{ly + 82}" font-family="Arial" font-size="12">of 10 replicates</text>')
    out_lines.append("</svg>")
    out.write_text("\n".join(out_lines), encoding="utf-8")


def plot_metric(summary_rows: list[dict[str, object]], metric: str, out: Path) -> None:
    groups = ["All", "2.3", "2.4", "2.5"]
    group_labels = {"All": "China ESL2 (ST2)", "2.3": "ESL 2.3", "2.4": "ESL 2.4", "2.5": "ESL 2.5"}
    strategies = ["full", "random", "city_weighted", "year_balanced", "city_cap20", "city_cap20_year_balanced"]
    strategy_labels = {
        "full": "Full dataset",
        "random": "Random 50% downsampling",
        "city_weighted": "City-weighted downsampling",
        "year_balanced": "Year-balanced downsampling",
        "city_cap20": "City cap-20 downsampling",
        "city_cap20_year_balanced": "City cap-20 + year-balanced",
    }
    colors = {
        "All": "#4A6FB5",
        "2.3": "#134E5A",
        "2.4": "#EB6564",
        "2.5": "#F5BA1A",
    }
    ylabel = "Coefficient (R)" if metric == "r" else "RTTD slope"
    title = "Temporal signal after sampling-bias correction" if metric == "r" else "RTTD-derived clock-rate proxy after sampling-bias correction"

    data = {
        (str(r["group"]), str(r["strategy"])): r
        for r in summary_rows
        if str(r["metric"]) == metric
    }
    vals = []
    for g in groups:
        for s in strategies:
            row = data.get((g, s))
            if row:
                vals.extend([float(row["spread_95_low"]), float(row["spread_95_high"]), float(row["mean"])])
    ymin, ymax = min(vals), max(vals)
    pad = max((ymax - ymin) * 0.15, 0.02 if metric == "r" else 2e-8)
    ymin -= pad
    ymax += pad

    panel_w = 170
    panel_h = 210
    left_pad = 70
    top_pad = 62
    bottom_pad = 58
    gap = 38
    width = left_pad + len(strategies) * panel_w + (len(strategies) - 1) * gap + 22
    height = top_pad + panel_h + bottom_pad

    def sy(v: float) -> float:
        return top_pad + panel_h - (v - ymin) / (ymax - ymin) * panel_h

    def fmt(v: float) -> str:
        if metric == "slope_clock_rate_proxy":
            return f"{v:.1e}"
        if abs(v) >= 0.1:
            return f"{v:.1f}"
        return f"{v:.2f}"

    out_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="14" y="30" font-family="Arial" font-size="19" font-weight="700">{title}</text>',
    ]
    ticks = [ymin + (ymax - ymin) * i / 4 for i in range(5)]

    for si, strategy in enumerate(strategies):
        px = left_pad + si * (panel_w + gap)
        out_lines.append(f'<text x="{px + panel_w / 2:.1f}" y="{top_pad - 18}" text-anchor="middle" font-family="Arial" font-size="13" font-weight="700">{strategy_labels[strategy]}</text>')
        out_lines.append(f'<rect x="{px}" y="{top_pad}" width="{panel_w}" height="{panel_h}" fill="none" stroke="black" stroke-width="1.2"/>')
        if si == 0:
            out_lines.append(f'<text x="26" y="{top_pad + panel_h / 2:.1f}" text-anchor="middle" dominant-baseline="middle" transform="rotate(-90 26 {top_pad + panel_h / 2:.1f})" font-family="Arial" font-size="16">{ylabel}</text>')
            for t in ticks:
                y = sy(t)
                out_lines.append(f'<line x1="{px - 5}" y1="{y:.1f}" x2="{px}" y2="{y:.1f}" stroke="black" stroke-width="1"/>')
                out_lines.append(f'<text x="{px - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="10">{fmt(t)}</text>')

        for gi, group in enumerate(groups):
            row = data.get((group, strategy))
            if not row:
                continue
            x = px + 25 + gi * 39
            lo = float(row["spread_95_low"])
            hi = float(row["spread_95_high"])
            mu = float(row["mean"])
            if strategy != "full":
                out_lines.append(f'<line x1="{x:.1f}" y1="{sy(lo):.1f}" x2="{x:.1f}" y2="{sy(hi):.1f}" stroke="{colors[group]}" stroke-width="2" stroke-opacity="0.9"/>')
            out_lines.append(f'<circle cx="{x:.1f}" cy="{sy(mu):.1f}" r="3" fill="{colors[group]}"/>')
            out_lines.append(f'<text x="{x:.1f}" y="{top_pad + panel_h + 18}" text-anchor="middle" font-family="Arial" font-size="9">{group_labels[group].replace("China ESL2 (ST2)", "All")}</text>')

    legend_x = width - 155
    legend_y = height - 22
    out_lines.append(f'<text x="{legend_x - 80}" y="{legend_y}" font-family="Arial" font-size="9" fill="#555">Lines indicate 95% spread across 10 downsampling replicates.</text>')
    out_lines.append("</svg>")
    out.write_text("\n".join(out_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    figure_dir = Path(__file__).resolve().parents[1]
    module_dir = Path(__file__).resolve().parents[3]
    parser.add_argument("--rttd", default=str(figure_dir / "data" / "root_to_tip_distances.tsv"))
    parser.add_argument("--metadata", default=str(module_dir / "sample_metadata.xlsx"))
    parser.add_argument("--sheet", default="S1 Metadata")
    parser.add_argument("--outdir", default=str(figure_dir / "sampling_sensitivity_output"))
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260606)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(args.metadata)
    rows = join_rows(read_rttd(Path(args.rttd)), read_metadata(metadata_path, args.sheet))
    city_populations = read_city_populations(metadata_path)
    groups = ["All", "2.3", "2.4", "2.5"]
    strategies = ["full", "random", "city_weighted", "year_balanced", "city_cap20", "city_cap20_year_balanced"]

    replicate_rows = []
    for group in groups:
        group_rows = rows if group == "All" else [r for r in rows if str(r["group"]) == group]
        full_metrics = metrics(group_rows)
        replicate_rows.append({"group": group, "strategy": "full", "replicate": 0, "city_matched_subset": False, **full_metrics})
        for strategy in strategies:
            if strategy == "full":
                continue
            source_rows = group_rows
            city_matched_subset = False
            if strategy in {"city_weighted", "city_cap20", "city_cap20_year_balanced"}:
                source_rows = [r for r in group_rows if r.get("has_city_match") and str(r["city"])]
                city_matched_subset = True
            if len(source_rows) < 3:
                continue
            for rep in range(1, args.replicates + 1):
                sampled = downsample(source_rows, strategy, args.fraction, rng, city_populations)
                replicate_rows.append({"group": group, "strategy": strategy, "replicate": rep, "city_matched_subset": city_matched_subset, **metrics(sampled)})

    fields = [
        "group",
        "strategy",
        "replicate",
        "n",
        "min_year",
        "max_year",
        "slope_clock_rate_proxy",
        "r",
        "n_cities",
        "n_years",
        "n_city_matched",
        "city_matched_subset",
    ]
    write_tsv(outdir / "rttd_sampling_bias_replicates.tsv", replicate_rows, fields)

    summary_rows = []
    for group in groups:
        for strategy in strategies:
            subset = [r for r in replicate_rows if r["group"] == group and r["strategy"] == strategy]
            if not subset:
                continue
            for metric_name in ["slope_clock_rate_proxy", "r"]:
                values = [float(r[metric_name]) for r in subset if not math.isnan(float(r[metric_name]))]
                mu, sd, lo, hi = summarise(values)
                summary_rows.append(
                    {
                        "group": group,
                        "strategy": strategy,
                        "metric": metric_name,
                        "replicates": len(values),
                        "mean": mu,
                        "sd": sd,
                        "spread_95_low": lo,
                        "spread_95_high": hi,
                    }
                )
    write_tsv(
        outdir / "rttd_sampling_bias_summary.tsv",
        summary_rows,
        ["group", "strategy", "metric", "replicates", "mean", "sd", "spread_95_low", "spread_95_high"],
    )
    plot_metric(summary_rows, "r", outdir / "rttd_sampling_bias_temporal_signal_R.svg")
    plot_metric(summary_rows, "slope_clock_rate_proxy", outdir / "rttd_sampling_bias_clock_rate_proxy.svg")
    plot_strategy_grid(
        rows,
        strategies=[s for s in strategies if s != "full"],
        out=outdir / "rttd_sampling_bias_R_strategy_grid.svg",
        seed=args.seed,
        metric="r",
        replicates=args.replicates,
        sample_fraction=args.fraction,
        city_populations=city_populations,
    )


if __name__ == "__main__":
    main()
