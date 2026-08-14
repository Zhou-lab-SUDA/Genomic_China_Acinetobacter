"""Genome-level downsampling sensitivity from TreeTime annotated Nexus.

The script parses a TreeTime mugration Nexus tree, extracts city-state changes
on annotated branches, and rebuilds city-level transmission networks after
tip/genome downsampling.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median, stdev

from openpyxl import load_workbook

sys.path.append(str(Path(__file__).resolve().parent))
from network_resampling_sensitivity import (  # noqa: E402
    ari,
    label_propagation,
    nmi,
    read_edges,
    spearman_from_maps,
    top_edges,
    weighted_degrees,
    within_module_ratio,
    write_tsv,
)


@dataclass
class Node:
    label: str = ""
    city: str = ""
    length: float = 0.0
    children: list["Node"] = field(default_factory=list)
    parent: "Node | None" = None
    tips: set[str] = field(default_factory=set)


def normalise_id(value: object) -> str:
    s = str(value).strip().strip("'\"")
    return s


def read_taxlabels(text: str) -> dict[str, str]:
    m = re.search(r"Begin taxa;(.+?)End;", text, re.S | re.I)
    if not m:
        raise ValueError("No taxa block found")
    block = m.group(1)
    start = block.lower().find("taxlabels")
    labels_text = block[start + len("taxlabels") :]
    labels_text = labels_text.split(";")[0]
    labels = [normalise_id(x) for x in labels_text.split() if x.strip()]
    return {str(i + 1): label for i, label in enumerate(labels)}


def read_tree_text(text: str) -> str:
    idx = text.find("tree TREE1")
    if idx < 0:
        idx = text.lower().find("tree tree1")
    if idx < 0:
        raise ValueError("No TREE1 found")
    eq = text.find("=", idx)
    end_block = text.find("End;", eq)
    tree = text[eq + 1 : end_block].strip()
    if tree.endswith(";"):
        tree = tree[:-1]
    if tree.startswith("[&R]"):
        tree = tree[4:].strip()
    return tree


class NewickParser:
    def __init__(self, text: str):
        self.text = text
        self.i = 0

    def peek(self) -> str:
        return self.text[self.i] if self.i < len(self.text) else ""

    def consume_ws(self) -> None:
        while self.peek() and self.peek().isspace():
            self.i += 1

    def parse(self) -> Node:
        node = self.parse_subtree()
        return node

    def parse_subtree(self) -> Node:
        self.consume_ws()
        node = Node()
        if self.peek() == "(":
            self.i += 1
            while True:
                child = self.parse_subtree()
                child.parent = node
                node.children.append(child)
                self.consume_ws()
                if self.peek() == ",":
                    self.i += 1
                    continue
                if self.peek() == ")":
                    self.i += 1
                    break
                raise ValueError(f"Unexpected character {self.peek()!r} at {self.i}")
        self.parse_label_annotation_length(node)
        return node

    def parse_label_annotation_length(self, node: Node) -> None:
        self.consume_ws()
        label_chars = []
        while self.peek() and self.peek() not in "[:,);[":
            label_chars.append(self.peek())
            self.i += 1
        node.label = "".join(label_chars).strip()
        self.consume_ws()
        if self.peek() == "[":
            start = self.i
            depth = 0
            in_quote = False
            while self.i < len(self.text):
                ch = self.text[self.i]
                if ch == '"':
                    in_quote = not in_quote
                elif not in_quote and ch == "[":
                    depth += 1
                elif not in_quote and ch == "]":
                    depth -= 1
                    if depth == 0:
                        self.i += 1
                        break
                self.i += 1
            annot = self.text[start : self.i]
            m = re.search(r'city="([^"]+)"', annot)
            if m:
                node.city = m.group(1)
        self.consume_ws()
        if self.peek() == ":":
            self.i += 1
            num = []
            while self.peek() and self.peek() not in ",);":
                num.append(self.peek())
                self.i += 1
            try:
                node.length = float("".join(num).strip())
            except ValueError:
                node.length = 0.0


def assign_tips(node: Node, taxlabels: dict[str, str]) -> set[str]:
    if not node.children:
        node.tips = {taxlabels.get(node.label, node.label)}
    else:
        tips = set()
        for child in node.children:
            tips.update(assign_tips(child, taxlabels))
        node.tips = tips
    return node.tips


def iter_nodes(node: Node):
    yield node
    for child in node.children:
        yield from iter_nodes(child)


def nearest_city(node: Node) -> str:
    cur = node
    while cur is not None:
        if cur.city:
            return cur.city
        cur = cur.parent
    return ""


def tip_metadata(root: Node) -> dict[str, dict[str, str]]:
    out = {}
    for node in iter_nodes(root):
        if not node.children:
            tip = next(iter(node.tips))
            out[tip] = {"city": nearest_city(node)}
    return out


def read_dates(path: Path) -> dict[str, int]:
    out = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            try:
                out[row["genome"]] = int(float(row["Year"]))
            except Exception:
                pass
    return out


def read_distances(path: Path) -> dict[tuple[str, str], float]:
    out = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            a, b, d = parts[0], parts[1], float(parts[2])
            out[(a, b)] = d
            out[(b, a)] = d
    return out


def extract_events(root: Node) -> list[dict[str, object]]:
    events = []
    for node in iter_nodes(root):
        if node.parent is None:
            continue
        pc = node.parent.city
        cc = node.city
        if not pc or not cc or pc == cc or pc == "Unknown" or cc == "Unknown":
            continue
        events.append({"source": pc, "target": cc, "tips": node.tips})
    return events


def network_from_events(
    events: list[dict[str, object]],
    sampled: set[str],
    distances: dict[tuple[str, str], float],
    weight_mode: str = "sample_fraction",
    rng: random.Random | None = None,
) -> tuple[list[dict[str, object]], list[float]]:
    weights: dict[tuple[str, str], float] = defaultdict(float)
    dists = []
    for event in events:
        event_tips = event["tips"]
        sampled_tips = event_tips & sampled
        if not sampled_tips:
            continue
        s, t = str(event["source"]), str(event["target"])
        if weight_mode == "binary":
            contribution = 1.0
        elif weight_mode == "sample_fraction":
            contribution = len(sampled_tips) / len(event_tips)
        elif weight_mode == "bernoulli_sample_fraction":
            probability = len(sampled_tips) / len(event_tips)
            if (rng or random).random() > probability:
                continue
            contribution = 1.0
        else:
            raise ValueError(weight_mode)
        weights[(s, t)] += contribution
        if (s, t) in distances:
            dists.extend([distances[(s, t)]] * max(1, round(contribution)))
    return [{"source": s, "target": t, "weight": w} for (s, t), w in weights.items()], dists


def modularity_q(edges: list[dict[str, object]], labels: dict[str, int]) -> float:
    pair_weights: dict[tuple[str, str], float] = defaultdict(float)
    degree: dict[str, float] = defaultdict(float)
    total = 0.0
    for edge in edges:
        s, t = sorted([str(edge["source"]), str(edge["target"])])
        if s == t:
            continue
        pair_weights[(s, t)] += float(edge["weight"])
    for (s, t), weight in pair_weights.items():
        degree[s] += weight
        degree[t] += weight
        total += weight
    if total <= 0:
        return float("nan")
    q = 0.0
    for (s, t), weight in pair_weights.items():
        if labels.get(s) == labels.get(t):
            q += weight - (degree[s] * degree[t]) / (2 * total)
    return q / (2 * total)


def weighted_long_distance_ratio(edges: list[dict[str, object]], distances: dict[tuple[str, str], float], boundary: float = 800.0) -> float:
    total = 0.0
    long = 0.0
    for edge in edges:
        s, t = str(edge["source"]), str(edge["target"])
        if (s, t) not in distances:
            continue
        weight = float(edge["weight"])
        total += weight
        if distances[(s, t)] > boundary:
            long += weight
    return long / total if total else float("nan")


def year_balanced(items: list[str], years: dict[str, int], target_n: int, rng: random.Random) -> list[str]:
    by_year: dict[int, list[str]] = defaultdict(list)
    for item in items:
        by_year[years.get(item, 0)].append(item)
    by_year.pop(0, None)
    if not by_year:
        return rng.sample(items, min(target_n, len(items)))
    cap = 1
    while sum(min(len(v), cap + 1) for v in by_year.values()) <= target_n:
        cap += 1
        if cap >= max(len(v) for v in by_year.values()):
            break
    selected = []
    leftovers = []
    for year in sorted(by_year):
        bucket = by_year[year]
        if len(bucket) <= cap:
            selected.extend(bucket)
        else:
            pick = rng.sample(bucket, cap)
            selected.extend(pick)
            picked = set(pick)
            leftovers.extend([x for x in bucket if x not in picked])
    if len(selected) > target_n:
        selected = rng.sample(selected, target_n)
    elif len(selected) < target_n and leftovers:
        selected.extend(rng.sample(leftovers, min(target_n - len(selected), len(leftovers))))
    return selected


def city_cap20(items: list[str], cities: dict[str, str], rng: random.Random) -> list[str]:
    by_city: dict[str, list[str]] = defaultdict(list)
    for item in items:
        city = cities.get(item, "")
        if city and city != "Unknown":
            by_city[city].append(item)
    selected = []
    for city in sorted(by_city):
        bucket = by_city[city]
        selected.extend(bucket if len(bucket) <= 20 else rng.sample(bucket, 20))
    return selected


def read_city_populations(path: Path) -> dict[str, float]:
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb["S2 CityAttr"]
    header = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    city_i = header.index("City")
    pop_i = header.index("Population (10K)")
    pops = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        city = row[city_i]
        pop = row[pop_i]
        if not city:
            continue
        try:
            pops[str(city).strip()] = float(pop)
        except (TypeError, ValueError):
            pass
    if "Tai'an" in pops:
        pops["Taian"] = pops["Tai'an"]
    if "Jinan" not in pops and pops:
        pops["Jinan"] = sorted(pops.values())[len(pops) // 2]
    return pops


def city_weighted_sample(items: list[str], cities: dict[str, str], city_populations: dict[str, float], target_n: int, rng: random.Random) -> set[str]:
    by_city: dict[str, list[str]] = defaultdict(list)
    for item in items:
        city = cities.get(item, "")
        if city and city != "Unknown":
            by_city[city].append(item)
    if not by_city:
        return set(rng.sample(items, min(target_n, len(items))))
    known_pops = [v for v in city_populations.values() if v > 0]
    fallback = sorted(known_pops)[len(known_pops) // 2] if known_pops else 1.0
    weights = {city: max(city_populations.get(city, fallback), 1.0) for city in by_city}
    total_weight = sum(weights.values())
    quotas = {}
    fractions = []
    selected = []
    for city in sorted(by_city):
        raw = target_n * weights[city] / total_weight
        quota = min(len(by_city[city]), int(math.floor(raw)))
        quotas[city] = quota
        fractions.append((raw - math.floor(raw), city))
    remaining = target_n - sum(quotas.values())
    for _, city in sorted(fractions, reverse=True):
        if remaining <= 0:
            break
        if quotas[city] < len(by_city[city]):
            quotas[city] += 1
            remaining -= 1
    for city in sorted(by_city):
        quota = min(len(by_city[city]), quotas[city])
        if quota:
            selected.extend(rng.sample(by_city[city], quota))
    if len(selected) < target_n:
        selected_set = set(selected)
        leftovers = [item for item in items if item not in selected_set]
        selected.extend(rng.sample(leftovers, min(target_n - len(selected), len(leftovers))))
    return set(selected)


def sample_tips(items: list[str], strategy: str, years: dict[str, int], cities: dict[str, str], rng: random.Random, fraction: float) -> set[str]:
    target_n = max(3, round(len(items) * fraction))
    if strategy == "random_50":
        return set(rng.sample(items, target_n))
    if strategy == "year_balanced":
        return set(year_balanced(items, years, target_n, rng))
    if strategy == "city_cap20":
        return set(city_cap20(items, cities, rng))
    if strategy == "city_cap20_year_balanced":
        capped = city_cap20(items, cities, rng)
        return set(year_balanced(capped, years, max(3, round(len(capped) * fraction)), rng))
    raise ValueError(strategy)


def summarise(values: list[float]) -> tuple[float, float, float, float]:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mu = mean(vals)
    sd = stdev(vals) if len(vals) > 1 else 0.0
    return mu, sd, mu - 1.96 * sd, mu + 1.96 * sd


def main() -> None:
    parser = argparse.ArgumentParser()
    figure_dir = Path(__file__).resolve().parents[1]
    data_dir = figure_dir / "data"
    parser.add_argument("--nexus", default=str(data_dir / "treetime_annotated.nexus"))
    parser.add_argument("--network", default=str(data_dir / "China_Transmission_Normalization.tsv"))
    parser.add_argument("--distances", default=str(data_dir / "PairwiseChinaCities_distance.tsv"))
    parser.add_argument("--dates", default=str(data_dir / "Dates.tsv"))
    parser.add_argument("--metadata", default=str(data_dir / "Supplementary_Tablesv2.xlsx"))
    parser.add_argument("--outdir", default=str(figure_dir / "reproduced_analysis"))
    parser.add_argument("--reference-mode", choices=["network", "nexus"], default="network")
    parser.add_argument("--event-weight-mode", choices=["sample_fraction", "binary", "bernoulli_sample_fraction"], default="bernoulli_sample_fraction")
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260607)
    args = parser.parse_args()

    text = Path(args.nexus).read_text(encoding="utf-8")
    taxlabels = read_taxlabels(text)
    root = NewickParser(read_tree_text(text)).parse()
    assign_tips(root, taxlabels)
    tips = sorted(root.tips)
    meta = tip_metadata(root)
    years = read_dates(Path(args.dates))
    cities = {tip: meta.get(tip, {}).get("city", "") for tip in tips}
    city_populations = read_city_populations(Path(args.metadata))
    distances = read_distances(Path(args.distances))
    events = extract_events(root)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    if args.reference_mode == "network":
        full_edges = read_edges(Path(args.network))
        full_dists = [distances[(str(e["source"]), str(e["target"]))] for e in full_edges if (str(e["source"]), str(e["target"])) in distances]
    else:
        full_edges, full_dists = network_from_events(events, set(tips), distances, args.event_weight_mode, rng)
    full_labels = label_propagation(full_edges)
    full_degree = weighted_degrees(full_edges)
    full_top = top_edges(full_edges, args.top_k)

    rows = []
    degree_rows = []
    strategies = ["full", "random_50", "city_weighted", "year_balanced", "city_cap20", "city_cap20_year_balanced"]
    for strategy in strategies:
        reps = [0] if strategy == "full" else range(1, args.replicates + 1)
        for rep in reps:
            if strategy == "full":
                sampled = set(tips)
                edges, dists = full_edges, full_dists
            elif strategy == "city_weighted":
                sampled = city_weighted_sample(tips, cities, city_populations, max(3, round(len(tips) * args.fraction)), rng)
                edges, dists = network_from_events(events, sampled, distances, args.event_weight_mode, rng)
            else:
                sampled = sample_tips(tips, strategy, years, cities, rng, args.fraction)
                edges, dists = network_from_events(events, sampled, distances, args.event_weight_mode, rng)
            labels = label_propagation(edges)
            degree = weighted_degrees(edges)
            top = top_edges(edges, args.top_k)
            if rep:
                replicate_edge_rows = [
                    {
                        "strategy": strategy,
                        "replicate": rep,
                        "source": edge["source"],
                        "target": edge["target"],
                        "weight": edge["weight"],
                    }
                    for edge in edges
                ]
                write_tsv(
                    outdir / f"replicate_edges_{strategy}_{rep}.tsv",
                    replicate_edge_rows,
                    ["strategy", "replicate", "source", "target", "weight"],
                )
            for city in sorted(full_degree):
                degree_rows.append({
                    "strategy": strategy,
                    "replicate": rep,
                    "city": city,
                    "original_degree": full_degree.get(city, 0.0),
                    "degree": degree.get(city, 0.0),
                })
            rows.append({
                "strategy": strategy,
                "replicate": rep,
                "n_tips": len(sampled),
                "n_edges": len(edges),
                "n_nodes": len(set([e["source"] for e in edges] + [e["target"] for e in edges])),
                "n_modules": len(set(labels.values())),
                "modularity_q": modularity_q(edges, labels),
                "module_nmi": nmi(full_labels, labels),
                "module_ari": ari(full_labels, labels),
                "top_edge_overlap": len(full_top & top) / len(full_top) if full_top else float("nan"),
                "weighted_degree_spearman": spearman_from_maps(full_degree, degree),
                "within_module_ratio": within_module_ratio(edges, labels),
                "weighted_long_distance_ratio_gt800km": weighted_long_distance_ratio(edges, distances, 800.0),
                "mean_distance": mean(dists) if dists else float("nan"),
                "median_distance": median(dists) if dists else float("nan"),
                "prop_long_distance_gt800": sum(1 for d in dists if d > 800) / len(dists) if dists else float("nan"),
            })

    fields = [
        "strategy", "replicate", "n_tips", "n_edges", "n_nodes", "n_modules",
        "modularity_q", "module_nmi", "module_ari", "top_edge_overlap", "weighted_degree_spearman",
        "within_module_ratio", "weighted_long_distance_ratio_gt800km", "mean_distance", "median_distance", "prop_long_distance_gt800",
    ]
    write_tsv(outdir / "genome_network_replicates.tsv", rows, fields)
    write_tsv(outdir / "degree_replicates.tsv", degree_rows, ["strategy", "replicate", "city", "original_degree", "degree"])

    edge_rows = []
    for edge in sorted(full_edges, key=lambda e: (str(e["source"]), str(e["target"]))):
        edge_rows.append({
            "strategy": "full",
            "source": edge["source"],
            "target": edge["target"],
            "mean_weight": float(edge["weight"]),
        })
    avg_rng = random.Random(args.seed)
    for strategy in strategies[1:]:
        totals: dict[tuple[str, str], float] = defaultdict(float)
        for _ in range(args.replicates):
            if strategy == "city_weighted":
                sampled = city_weighted_sample(tips, cities, city_populations, max(3, round(len(tips) * args.fraction)), avg_rng)
            else:
                sampled = sample_tips(tips, strategy, years, cities, avg_rng, args.fraction)
            edges, _ = network_from_events(events, sampled, distances, args.event_weight_mode, avg_rng)
            for edge in edges:
                totals[(str(edge["source"]), str(edge["target"]))] += float(edge["weight"])
        for (source, target), weight in sorted(totals.items()):
            edge_rows.append({
                "strategy": strategy,
                "source": source,
                "target": target,
                "mean_weight": weight / args.replicates,
            })
    write_tsv(outdir / "average_network_edges.tsv", edge_rows, ["strategy", "source", "target", "mean_weight"])

    summary = []
    metrics = fields[2:]
    for strategy in strategies:
        subset = [r for r in rows if r["strategy"] == strategy]
        for metric in metrics:
            vals = [float(r[metric]) for r in subset]
            mu, sd, lo, hi = summarise(vals)
            summary.append({"strategy": strategy, "metric": metric, "replicates": len(vals), "mean": mu, "sd": sd, "spread_95_low": lo, "spread_95_high": hi})
    write_tsv(outdir / "genome_network_summary.tsv", summary, ["strategy", "metric", "replicates", "mean", "sd", "spread_95_low", "spread_95_high"])


if __name__ == "__main__":
    main()
