"""Network resampling sensitivity for city-level transmission networks.

Input is a directed weighted edge list with columns Source, Target, Weight.
The script compares resampled networks against the full network using:
1. module NMI and ARI
2. top-k edge overlap
3. Spearman correlation of city weighted degree
4. within-module transmission ratio

Community detection uses deterministic weighted label propagation to avoid
external dependencies.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev


def read_edges(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append({
                "source": row["Source"],
                "target": row["Target"],
                "weight": float(row["Weight"]),
            })
    return rows


def nodes_from_edges(edges: list[dict[str, object]]) -> list[str]:
    nodes = set()
    for edge in edges:
        nodes.add(str(edge["source"]))
        nodes.add(str(edge["target"]))
    return sorted(nodes)


def undirected_adjacency(edges: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    adj: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for edge in edges:
        s = str(edge["source"])
        t = str(edge["target"])
        w = float(edge["weight"])
        if s == t:
            continue
        adj[s][t] += w
        adj[t][s] += w
    return {k: dict(v) for k, v in adj.items()}


def label_propagation(edges: list[dict[str, object]], max_iter: int = 100) -> dict[str, int]:
    nodes = nodes_from_edges(edges)
    labels = {node: node for node in nodes}
    adj = undirected_adjacency(edges)
    for _ in range(max_iter):
        changed = False
        for node in nodes:
            scores: dict[str, float] = defaultdict(float)
            for nb, weight in adj.get(node, {}).items():
                scores[labels[nb]] += weight
            if not scores:
                continue
            best = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            if labels[node] != best:
                labels[node] = best
                changed = True
        if not changed:
            break
    unique = {label: idx + 1 for idx, label in enumerate(sorted(set(labels.values())))}
    return {node: unique[label] for node, label in labels.items()}


def weighted_degrees(edges: list[dict[str, object]]) -> dict[str, float]:
    degree: dict[str, float] = defaultdict(float)
    for edge in edges:
        degree[str(edge["source"])] += float(edge["weight"])
        degree[str(edge["target"])] += float(edge["weight"])
    return dict(degree)


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j + 2) / 2
        for k in range(i, j + 1):
            out[order[k]] = rank
        i = j + 1
    return out


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sx * sy)


def spearman_from_maps(a: dict[str, float], b: dict[str, float]) -> float:
    keys = sorted(set(a) | set(b))
    av = [a.get(k, 0.0) for k in keys]
    bv = [b.get(k, 0.0) for k in keys]
    return pearson(ranks(av), ranks(bv))


def top_edges(edges: list[dict[str, object]], k: int) -> set[tuple[str, str]]:
    ordered = sorted(edges, key=lambda e: (-float(e["weight"]), str(e["source"]), str(e["target"])))
    return {(str(e["source"]), str(e["target"])) for e in ordered[:k]}


def nmi(labels_a: dict[str, int], labels_b: dict[str, int]) -> float:
    keys = sorted(set(labels_a) & set(labels_b))
    n = len(keys)
    if n == 0:
        return float("nan")
    ca = Counter(labels_a[k] for k in keys)
    cb = Counter(labels_b[k] for k in keys)
    cab = Counter((labels_a[k], labels_b[k]) for k in keys)
    mi = 0.0
    for (la, lb), count in cab.items():
        mi += count / n * math.log((count * n) / (ca[la] * cb[lb]))
    ha = -sum((v / n) * math.log(v / n) for v in ca.values())
    hb = -sum((v / n) * math.log(v / n) for v in cb.values())
    if ha + hb == 0:
        return 1.0
    return 2 * mi / (ha + hb)


def comb2(x: int) -> float:
    return x * (x - 1) / 2


def ari(labels_a: dict[str, int], labels_b: dict[str, int]) -> float:
    keys = sorted(set(labels_a) & set(labels_b))
    n = len(keys)
    if n < 2:
        return float("nan")
    ca = Counter(labels_a[k] for k in keys)
    cb = Counter(labels_b[k] for k in keys)
    cab = Counter((labels_a[k], labels_b[k]) for k in keys)
    sum_ab = sum(comb2(v) for v in cab.values())
    sum_a = sum(comb2(v) for v in ca.values())
    sum_b = sum(comb2(v) for v in cb.values())
    total = comb2(n)
    expected = sum_a * sum_b / total if total else 0.0
    max_index = 0.5 * (sum_a + sum_b)
    denom = max_index - expected
    if denom == 0:
        return 1.0
    return (sum_ab - expected) / denom


def within_module_ratio(edges: list[dict[str, object]], labels: dict[str, int]) -> float:
    total = 0.0
    within = 0.0
    for edge in edges:
        s = str(edge["source"])
        t = str(edge["target"])
        w = float(edge["weight"])
        total += w
        if labels.get(s) == labels.get(t):
            within += w
    return within / total if total else float("nan")


def resample_edges(edges: list[dict[str, object]], strategy: str, rng: random.Random, fraction: float) -> list[dict[str, object]]:
    if strategy == "random_edge_50":
        n = max(1, round(len(edges) * fraction))
        return rng.sample(edges, n)
    if strategy == "weight_bootstrap":
        # Sample edges with probability proportional to weight and aggregate.
        n = max(1, round(len(edges) * fraction))
        weights = [float(e["weight"]) for e in edges]
        total = sum(weights)
        picked: dict[tuple[str, str], float] = defaultdict(float)
        for _ in range(n):
            threshold = rng.random() * total
            acc = 0.0
            edge = edges[-1]
            for candidate, weight in zip(edges, weights):
                acc += weight
                if acc >= threshold:
                    edge = candidate
                    break
            picked[(str(edge["source"]), str(edge["target"]))] += float(edge["weight"])
        return [{"source": s, "target": t, "weight": w} for (s, t), w in picked.items()]
    if strategy == "node_balanced":
        # Retain all sparse-node edges and cap dense source/target participation.
        node_counts: dict[str, int] = defaultdict(int)
        for edge in edges:
            node_counts[str(edge["source"])] += 1
            node_counts[str(edge["target"])] += 1
        cap = max(3, int(round(mean(node_counts.values()))))
        shuffled = edges[:]
        rng.shuffle(shuffled)
        kept = []
        kept_counts: dict[str, int] = defaultdict(int)
        for edge in sorted(shuffled, key=lambda e: float(e["weight"]), reverse=True):
            s, t = str(edge["source"]), str(edge["target"])
            if node_counts[s] <= cap or node_counts[t] <= cap or (kept_counts[s] < cap and kept_counts[t] < cap):
                kept.append(edge)
                kept_counts[s] += 1
                kept_counts[t] += 1
        return kept
    raise ValueError(strategy)


def summarise(values: list[float]) -> dict[str, float]:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return {"mean": float("nan"), "sd": float("nan"), "low": float("nan"), "high": float("nan")}
    mu = mean(vals)
    sd = stdev(vals) if len(vals) > 1 else 0.0
    return {"mean": mu, "sd": sd, "low": mu - 1.96 * sd, "high": mu + 1.96 * sd}


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    figure_dir = Path(__file__).resolve().parents[1]
    parser.add_argument("--edges", default=str(figure_dir / "data" / "China_Transmission_Normalization.tsv"))
    parser.add_argument("--outdir", default=str(figure_dir / "reproduced_network_resampling"))
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260607)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    edges = read_edges(Path(args.edges))
    full_labels = label_propagation(edges)
    full_degrees = weighted_degrees(edges)
    full_top = top_edges(edges, args.top_k)
    full_within = within_module_ratio(edges, full_labels)

    strategies = ["random_edge_50", "weight_bootstrap", "node_balanced"]
    rows = []
    full_metrics = {
        "strategy": "full",
        "replicate": 0,
        "n_nodes": len(nodes_from_edges(edges)),
        "n_edges": len(edges),
        "n_modules": len(set(full_labels.values())),
        "module_nmi": 1.0,
        "module_ari": 1.0,
        "top_edge_overlap": 1.0,
        "weighted_degree_spearman": 1.0,
        "within_module_ratio": full_within,
    }
    rows.append(full_metrics)
    for strategy in strategies:
        for rep in range(1, args.replicates + 1):
            sampled = resample_edges(edges, strategy, rng, args.fraction)
            labels = label_propagation(sampled)
            deg = weighted_degrees(sampled)
            top = top_edges(sampled, args.top_k)
            rows.append({
                "strategy": strategy,
                "replicate": rep,
                "n_nodes": len(nodes_from_edges(sampled)),
                "n_edges": len(sampled),
                "n_modules": len(set(labels.values())),
                "module_nmi": nmi(full_labels, labels),
                "module_ari": ari(full_labels, labels),
                "top_edge_overlap": len(full_top & top) / len(full_top) if full_top else float("nan"),
                "weighted_degree_spearman": spearman_from_maps(full_degrees, deg),
                "within_module_ratio": within_module_ratio(sampled, labels),
            })

    fields = [
        "strategy", "replicate", "n_nodes", "n_edges", "n_modules",
        "module_nmi", "module_ari", "top_edge_overlap",
        "weighted_degree_spearman", "within_module_ratio",
    ]
    write_tsv(outdir / "network_resampling_replicates.tsv", rows, fields)

    summary = []
    metrics = ["module_nmi", "module_ari", "top_edge_overlap", "weighted_degree_spearman", "within_module_ratio", "n_modules"]
    for strategy in ["full"] + strategies:
        subset = [r for r in rows if r["strategy"] == strategy]
        for metric in metrics:
            vals = [float(r[metric]) for r in subset]
            stats = summarise(vals)
            summary.append({
                "strategy": strategy,
                "metric": metric,
                "replicates": len(vals),
                "mean": stats["mean"],
                "sd": stats["sd"],
                "spread_95_low": stats["low"],
                "spread_95_high": stats["high"],
            })
    write_tsv(
        outdir / "network_resampling_summary.tsv",
        summary,
        ["strategy", "metric", "replicates", "mean", "sd", "spread_95_low", "spread_95_high"],
    )


if __name__ == "__main__":
    main()
