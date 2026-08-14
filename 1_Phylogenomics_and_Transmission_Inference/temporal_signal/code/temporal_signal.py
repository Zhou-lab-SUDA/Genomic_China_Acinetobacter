#!/usr/bin/env python3
"""
Root-to-tip temporal-signal analysis for dated bacterial phylogenies.

This script intentionally avoids heavy dependencies such as Biopython,
matplotlib and scipy. It parses a Newick tree with the Python standard
library, reads metadata with pandas, and writes:

  - temporal_signal_results.tsv
  - root_to_tip_distances.tsv
  - unmatched_tree_tips.tsv
  - unmatched_dates.tsv
  - temporal_signal.svg

By default it analyses all dated tips together. If a metadata table and group
column are supplied, it also runs separate regressions for each group, for
example ESL2.3, ESL2.4 and ESL2.5.
"""

from __future__ import annotations

import argparse
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


@dataclass
class Node:
    name: str = ""
    length: float = 0.0
    children: List["Node"] = field(default_factory=list)

    @property
    def is_tip(self) -> bool:
        return not self.children


def _skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i].isspace():
        i += 1
    return i


def _parse_label(s: str, i: int) -> Tuple[str, int]:
    i = _skip_ws(s, i)
    if i >= len(s):
        return "", i
    if s[i] in "'\"":
        quote = s[i]
        i += 1
        out = []
        while i < len(s):
            if s[i] == quote:
                i += 1
                break
            out.append(s[i])
            i += 1
        return "".join(out), i
    start = i
    while i < len(s) and s[i] not in ":,();":
        i += 1
    return s[start:i].strip(), i


def _parse_length(s: str, i: int) -> Tuple[float, int]:
    i = _skip_ws(s, i)
    if i >= len(s) or s[i] != ":":
        return 0.0, i
    i += 1
    i = _skip_ws(s, i)
    start = i
    while i < len(s) and s[i] not in ",();":
        i += 1
    raw = s[start:i].strip()
    try:
        return float(raw), i
    except ValueError as exc:
        raise ValueError(f"Could not parse branch length {raw!r}") from exc


def _parse_subtree(s: str, i: int) -> Tuple[Node, int]:
    i = _skip_ws(s, i)
    node = Node()
    if i < len(s) and s[i] == "(":
        i += 1
        while True:
            child, i = _parse_subtree(s, i)
            node.children.append(child)
            i = _skip_ws(s, i)
            if i >= len(s):
                raise ValueError("Unexpected end of Newick string")
            if s[i] == ",":
                i += 1
                continue
            if s[i] == ")":
                i += 1
                break
            raise ValueError(f"Unexpected Newick character {s[i]!r} at {i}")
        node.name, i = _parse_label(s, i)
        node.length, i = _parse_length(s, i)
    else:
        node.name, i = _parse_label(s, i)
        node.length, i = _parse_length(s, i)
    return node, i


def parse_newick(path: Path) -> Node:
    text = path.read_text(encoding="utf-8").strip()
    if not text.endswith(";"):
        raise ValueError("Newick file must end with ';'")
    root, i = _parse_subtree(text, 0)
    i = _skip_ws(text, i)
    if i < len(text) and text[i] == ";":
        return root
    raise ValueError(f"Unexpected trailing text at position {i}: {text[i:i+50]!r}")


def collect_tip_distances(root: Node) -> Dict[str, float]:
    tips: Dict[str, float] = {}

    def walk(node: Node, dist: float) -> None:
        here = dist + node.length
        if node.is_tip:
            if not node.name:
                raise ValueError("Encountered an unnamed tip")
            tips[node.name] = here
            return
        for child in node.children:
            walk(child, here)

    # Root branch length is usually zero/absent; include it if present.
    walk(root, 0.0)
    return tips


def normalise_id(x: object) -> str:
    return str(x).strip()


def match_key(x: object) -> str:
    """A relaxed sample key for matching tree names to metadata names."""
    s = normalise_id(x)
    for suffix in (
        ".result.fastq.gz",
        ".result.fasta",
        ".result.fa",
        ".result.fna",
        ".fastq.gz",
        ".fasta",
        ".fa",
        ".fna",
    ):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s


def unique_lookup(pairs: Iterable[Tuple[str, object]]) -> Dict[str, object]:
    """Return key->value only for keys that map to one unique value."""
    values: Dict[str, set] = {}
    for key, value in pairs:
        if not key:
            continue
        values.setdefault(key, set()).add(value)
    return {k: next(iter(v)) for k, v in values.items() if len(v) == 1}


def infer_separator(path: Path) -> str:
    if path.suffix.lower() in {".tsv", ".tab"}:
        return "\t"
    if path.suffix.lower() == ".csv":
        return ","
    return "\t"


def load_table(path: Path, sheet: Optional[str] = None) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet or 0, dtype=str)
    return pd.read_csv(path, sep=infer_separator(path), dtype=str)


def decimal_year(value: object) -> Optional[float]:
    """Parse year-like values: 2022, 2022.5, 2022-06-15 or 2022/06/15."""
    if value is None:
        return None
    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "na", "none", "unknown"}:
        return None
    try:
        return float(raw)
    except ValueError:
        pass
    m = re.match(r"^(\d{4})[-/](\d{1,2})(?:[-/](\d{1,2}))?$", raw)
    if not m:
        return None
    year = int(m.group(1))
    month = int(m.group(2))
    day = int(m.group(3) or 15)
    # Good enough for temporal-signal plots; avoids datetime leap-year details.
    return year + ((month - 1) * 30.4375 + (day - 1)) / 365.25


def pearson_r(x: List[float], y: List[float]) -> float:
    if len(x) < 2:
        return float("nan")
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    sx = math.sqrt(sum(v * v for v in dx))
    sy = math.sqrt(sum(v * v for v in dy))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum(a * b for a, b in zip(dx, dy)) / (sx * sy)


def linear_regression(x: List[float], y: List[float]) -> Dict[str, float]:
    n = len(x)
    if n < 2:
        return {
            "n": n,
            "slope": float("nan"),
            "intercept": float("nan"),
            "r": float("nan"),
            "r2": float("nan"),
            "rmse": float("nan"),
        }
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((v - mx) ** 2 for v in x)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    slope = sxy / sxx if sxx else float("nan")
    intercept = my - slope * mx if not math.isnan(slope) else float("nan")
    pred = [intercept + slope * v for v in x]
    rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(y, pred)) / n)
    r = pearson_r(x, y)
    return {
        "n": n,
        "slope": slope,
        "intercept": intercept,
        "r": r,
        "r2": r * r if not math.isnan(r) else float("nan"),
        "rmse": rmse,
    }


def permutation_p_value(
    x: List[float], y: List[float], observed_r: float, n_perm: int, seed: int
) -> Optional[float]:
    if n_perm <= 0 or len(x) < 3 or math.isnan(observed_r):
        return None
    rng = random.Random(seed)
    y_perm = list(y)
    count = 0
    target = abs(observed_r)
    for _ in range(n_perm):
        rng.shuffle(y_perm)
        r = pearson_r(x, y_perm)
        if not math.isnan(r) and abs(r) >= target:
            count += 1
    return (count + 1) / (n_perm + 1)


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.8g}"
    return str(value)


def analyse_group(
    rows: List[Dict[str, object]], group_name: str, permutations: int, seed: int
) -> Dict[str, object]:
    x = [float(r["year"]) for r in rows]
    y = [float(r["root_to_tip"]) for r in rows]
    stats = linear_regression(x, y)
    p_perm = permutation_p_value(x, y, float(stats["r"]), permutations, seed)
    return {
        "group": group_name,
        **stats,
        "permutation_p_two_sided": p_perm,
        "min_year": min(x) if x else float("nan"),
        "max_year": max(x) if x else float("nan"),
    }


def write_tsv(path: Path, rows: Iterable[Dict[str, object]], columns: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(columns) + "\n")
        for row in rows:
            fh.write("\t".join(fmt(row.get(c)) for c in columns) + "\n")


def write_svg(path: Path, rows: List[Dict[str, object]], results: List[Dict[str, object]]) -> None:
    if not rows:
        return
    width, height = 1000, 720
    left, right, top, bottom = 90, 40, 45, 90
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [float(r["year"]) for r in rows]
    ys = [float(r["root_to_tip"]) for r in rows]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmin == xmax:
        xmin -= 0.5
        xmax += 0.5
    if ymin == ymax:
        ymin -= 0.5
        ymax += 0.5
    ypad = (ymax - ymin) * 0.06
    ymin -= ypad
    ymax += ypad

    groups = sorted({str(r["group"]) for r in rows})
    palette = [
        "#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    colors = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

    def sx(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * plot_w

    def sy(y: float) -> float:
        return top + (ymax - y) / (ymax - ymin) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="25" text-anchor="middle" font-family="Arial" font-size="20">Root-to-tip temporal signal</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="black"/>',
    ]
    for i in range(6):
        xv = xmin + (xmax - xmin) * i / 5
        px = sx(xv)
        lines.append(f'<line x1="{px:.2f}" y1="{top+plot_h}" x2="{px:.2f}" y2="{top+plot_h+6}" stroke="black"/>')
        lines.append(f'<text x="{px:.2f}" y="{top+plot_h+25}" text-anchor="middle" font-family="Arial" font-size="12">{xv:.0f}</text>')
        yv = ymin + (ymax - ymin) * i / 5
        py = sy(yv)
        lines.append(f'<line x1="{left-6}" y1="{py:.2f}" x2="{left}" y2="{py:.2f}" stroke="black"/>')
        lines.append(f'<text x="{left-10}" y="{py+4:.2f}" text-anchor="end" font-family="Arial" font-size="12">{yv:.2g}</text>')
    lines.append(f'<text x="{left + plot_w/2}" y="{height-35}" text-anchor="middle" font-family="Arial" font-size="15">Sampling year</text>')
    lines.append(f'<text x="22" y="{top + plot_h/2}" transform="rotate(-90 22,{top + plot_h/2})" text-anchor="middle" font-family="Arial" font-size="15">Root-to-tip distance</text>')

    for row in rows:
        g = str(row["group"])
        lines.append(
            f'<circle cx="{sx(float(row["year"])):.2f}" cy="{sy(float(row["root_to_tip"])):.2f}" r="2.6" fill="{colors[g]}" fill-opacity="0.55"/>'
        )

    result_by_group = {str(r["group"]): r for r in results}
    for g in groups:
        res = result_by_group.get(g)
        if not res or math.isnan(float(res["slope"])):
            continue
        x1, x2 = xmin, xmax
        y1 = float(res["intercept"]) + float(res["slope"]) * x1
        y2 = float(res["intercept"]) + float(res["slope"]) * x2
        lines.append(
            f'<line x1="{sx(x1):.2f}" y1="{sy(y1):.2f}" x2="{sx(x2):.2f}" y2="{sy(y2):.2f}" stroke="{colors[g]}" stroke-width="2"/>'
        )

    lx = left + 20
    ly = top + 20
    for i, g in enumerate(groups):
        res = result_by_group.get(g, {})
        label = f'{g}: n={fmt(res.get("n"))}, R={fmt(res.get("r"))}, R2={fmt(res.get("r2"))}'
        lines.append(f'<circle cx="{lx}" cy="{ly + i*22}" r="5" fill="{colors[g]}"/>')
        lines.append(f'<text x="{lx+12}" y="{ly + i*22 + 5}" font-family="Arial" font-size="13">{label}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", default="China_Ab_GC2.labelled.nwk", help="Newick tree file")
    parser.add_argument("--dates", default="Dates.tsv", help="Table with genome and sampling year/date")
    parser.add_argument("--date-column", default="Year", help="Sampling date/year column")
    parser.add_argument("--id-column", default="genome", help="Sample ID column in dates/metadata")
    parser.add_argument("--metadata", default=None, help="Optional table with group assignments")
    parser.add_argument("--metadata-sheet", default=None, help="Sheet name for Excel metadata")
    parser.add_argument("--group-column", default=None, help="Optional group column, e.g. ESL clade")
    parser.add_argument("--groups", default=None, help="Comma-separated groups to include")
    parser.add_argument("--outdir", default="temporal_signal", help="Output directory")
    parser.add_argument("--permutations", type=int, default=999, help="Permutation p-value replicates; 0 disables")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for permutations")
    args = parser.parse_args()

    tree_path = Path(args.tree)
    dates_path = Path(args.dates)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    root = parse_newick(tree_path)
    tip_dist = collect_tip_distances(root)

    dates = load_table(dates_path)
    if args.id_column not in dates.columns:
        raise SystemExit(f"ID column {args.id_column!r} not found in {dates_path}")
    if args.date_column not in dates.columns:
        raise SystemExit(f"Date column {args.date_column!r} not found in {dates_path}")

    date_exact: Dict[str, float] = {}
    date_pairs_for_relaxed = []
    for _, row in dates.iterrows():
        sid = normalise_id(row[args.id_column])
        year = decimal_year(row[args.date_column])
        if sid and year is not None:
            date_exact[sid] = year
            date_pairs_for_relaxed.append((match_key(sid), year))
    date_relaxed = unique_lookup(date_pairs_for_relaxed)

    group_exact: Dict[str, str] = {}
    group_pairs_for_relaxed = []
    if args.metadata and args.group_column:
        meta = load_table(Path(args.metadata), sheet=args.metadata_sheet)
        if args.id_column not in meta.columns:
            raise SystemExit(f"ID column {args.id_column!r} not found in {args.metadata}")
        if args.group_column not in meta.columns:
            raise SystemExit(f"Group column {args.group_column!r} not found in {args.metadata}")
        for _, row in meta.iterrows():
            sid = normalise_id(row[args.id_column])
            group = normalise_id(row[args.group_column])
            if sid and group and group.lower() not in {"nan", "none", "na"}:
                group_exact[sid] = group
                group_pairs_for_relaxed.append((match_key(sid), group))
    group_relaxed = unique_lookup(group_pairs_for_relaxed)

    allowed_groups = None
    if args.groups:
        allowed_groups = {g.strip() for g in args.groups.split(",") if g.strip()}

    rows: List[Dict[str, object]] = []
    unmatched_tips = []
    for tip, dist in tip_dist.items():
        year = date_exact.get(tip)
        if year is None:
            year = date_relaxed.get(match_key(tip))
        if year is None:
            unmatched_tips.append({"genome": tip, "root_to_tip": dist})
            continue
        group = group_exact.get(tip)
        if group is None:
            group = group_relaxed.get(match_key(tip), "All")
        if allowed_groups and group not in allowed_groups:
            continue
        rows.append({
            "genome": tip,
            "year": year,
            "root_to_tip": dist,
            "group": group,
        })

    tip_keys = {match_key(tip) for tip in tip_dist}
    unmatched_dates = [
        {"genome": sid, "year": year}
        for sid, year in sorted(date_exact.items())
        if sid not in tip_dist and match_key(sid) not in tip_keys
    ]

    results: List[Dict[str, object]] = []
    results.append(analyse_group(rows, "All", args.permutations, args.seed))
    if args.metadata and args.group_column:
        for group in sorted({str(r["group"]) for r in rows}):
            group_rows = [r for r in rows if str(r["group"]) == group]
            if len(group_rows) >= 3:
                results.append(analyse_group(group_rows, group, args.permutations, args.seed))

    write_tsv(
        outdir / "root_to_tip_distances.tsv",
        rows,
        ["genome", "year", "root_to_tip", "group"],
    )
    write_tsv(
        outdir / "temporal_signal_results.tsv",
        results,
        [
            "group", "n", "slope", "intercept", "r", "r2", "rmse",
            "permutation_p_two_sided", "min_year", "max_year",
        ],
    )
    write_tsv(outdir / "unmatched_tree_tips.tsv", unmatched_tips, ["genome", "root_to_tip"])
    write_tsv(outdir / "unmatched_dates.tsv", unmatched_dates, ["genome", "year"])
    write_svg(outdir / "temporal_signal.svg", rows, results)

    print(f"Tree tips: {len(tip_dist)}")
    print(f"Dated matched tips used: {len(rows)}")
    print(f"Tree tips without dates: {len(unmatched_tips)}")
    print(f"Date rows not in tree: {len(unmatched_dates)}")
    print(f"Results: {outdir / 'temporal_signal_results.tsv'}")
    print(f"Distances: {outdir / 'root_to_tip_distances.tsv'}")
    print(f"Plot: {outdir / 'temporal_signal.svg'}")


if __name__ == "__main__":
    main()
