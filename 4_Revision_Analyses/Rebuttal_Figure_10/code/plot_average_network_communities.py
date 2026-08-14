"""Plot average resampled transmission networks with community colours.

The script rebuilds city-level transmission networks from the TreeTime
annotated Nexus, averages edges across genome-downsampling replicates, detects
communities in each average network, and maps those communities to the
manuscript geography-cluster colour scheme where possible.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from openpyxl import load_workbook

sys.path.append(str(Path(__file__).resolve().parent))
from genome_network_sensitivity import (  # noqa: E402
    NewickParser,
    assign_tips,
    extract_events,
    network_from_events,
    read_dates,
    read_distances,
    read_taxlabels,
    read_tree_text,
    sample_tips,
    tip_metadata,
)
from network_resampling_sensitivity import (  # noqa: E402
    label_propagation,
    nodes_from_edges,
    read_edges,
    weighted_degrees,
    within_module_ratio,
    write_tsv,
)


PALETTE = [
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#76B7B2",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
    "#1F77B4",
    "#FF7F0E",
]

STRATEGY_LABELS = {
    "full": "Original network\n(metadata clusters)",
    "random_50": "50% size downsampling",
    "city_weighted": "City-weighted sampling",
    "year_balanced": "Year-balanced sampling",
    "city_cap20": "City-balanced sampling",
    "city_cap20_year_balanced": "City-balanced +\nyear-balanced sampling",
}


def read_city_clusters(path: Path) -> dict[str, str]:
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb["S2 CityAttr"]
    header = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    city_i = header.index("City")
    cluster_i = header.index("Geography cluster")
    clusters = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        city = row[city_i]
        cluster = row[cluster_i]
        if city:
            clusters[str(city).strip()] = str(cluster).strip() if cluster else "Unknown"
    # TreeTime labels omit apostrophes for some Shandong cities. Jinan is not
    # listed in S2 CityAttr but belongs to the same North China geography group.
    aliases = {
        "Taian": "Tai'an",
        "Jinan": "North China",
    }
    for alias, target in aliases.items():
        clusters[alias] = clusters.get(target, target)
    return clusters


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


def city_weighted_sample(
    items: list[str],
    cities: dict[str, str],
    city_populations: dict[str, float],
    target_n: int,
    rng: random.Random,
) -> set[str]:
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
    selected: list[str] = []
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
        bucket = by_city[city]
        quota = min(len(bucket), quotas[city])
        if quota:
            selected.extend(rng.sample(bucket, quota))
    if len(selected) < target_n:
        leftovers = [item for item in items if item not in set(selected)]
        selected.extend(rng.sample(leftovers, min(target_n - len(selected), len(leftovers))))
    return set(selected)


def cluster_colours(city_clusters: dict[str, str]) -> dict[str, str]:
    names = sorted({v for v in city_clusters.values() if v and v != "Unknown"})
    colours = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(names)}
    colours["Unknown"] = "#BDBDBD"
    return colours


def average_edges(edge_sets: list[list[dict[str, object]]]) -> list[dict[str, object]]:
    totals: dict[tuple[str, str], float] = defaultdict(float)
    n = max(1, len(edge_sets))
    for edges in edge_sets:
        for edge in edges:
            totals[(str(edge["source"]), str(edge["target"]))] += float(edge["weight"])
    return [
        {"source": s, "target": t, "weight": w / n}
        for (s, t), w in sorted(totals.items())
        if w > 0
    ]


def community_to_geo_cluster(
    labels: dict[str, int],
    city_clusters: dict[str, str],
) -> dict[int, str]:
    members: dict[int, list[str]] = defaultdict(list)
    for city, label in labels.items():
        members[label].append(city)
    mapping = {}
    for label, cities in members.items():
        counts = Counter(city_clusters.get(city, "Unknown") for city in cities)
        if counts:
            mapping[label] = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        else:
            mapping[label] = "Unknown"
    return mapping


def undirected_weights(edges: list[dict[str, object]]) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = defaultdict(float)
    for edge in edges:
        s, t = sorted([str(edge["source"]), str(edge["target"])])
        if s != t:
            out[(s, t)] += float(edge["weight"])
    return dict(out)


def modularity_q(edges: list[dict[str, object]], labels: dict[str, int]) -> float:
    weights = undirected_weights(edges)
    if not weights:
        return float("nan")
    degree: dict[str, float] = defaultdict(float)
    total = 0.0
    for (a, b), weight in weights.items():
        degree[a] += weight
        degree[b] += weight
        total += weight
    if total <= 0:
        return float("nan")
    q = 0.0
    for (a, b), weight in weights.items():
        if labels.get(a) == labels.get(b):
            q += weight - (degree[a] * degree[b]) / (2 * total)
    return q / (2 * total)


def weighted_long_distance_ratio(
    edges: list[dict[str, object]],
    distances: dict[tuple[str, str], float],
    boundary: float = 800.0,
) -> float:
    total = 0.0
    long = 0.0
    for edge in edges:
        s, t = str(edge["source"]), str(edge["target"])
        weight = float(edge["weight"])
        if (s, t) not in distances:
            continue
        total += weight
        if distances[(s, t)] > boundary:
            long += weight
    return long / total if total else float("nan")


def force_layout(
    nodes: list[str],
    edges: list[dict[str, object]],
    seed: int = 20260607,
    iterations: int = 500,
) -> dict[str, tuple[float, float]]:
    rng = random.Random(seed)
    n = max(1, len(nodes))
    weights = undirected_weights(edges)
    max_w = max(weights.values()) if weights else 1.0
    pos = {}
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / n
        jitter = rng.uniform(-0.05, 0.05)
        pos[node] = (0.5 + 0.35 * math.cos(angle + jitter), 0.5 + 0.35 * math.sin(angle + jitter))
    k = math.sqrt(1.0 / n)
    temperature = 0.08
    for step in range(iterations):
        disp = {node: [0.0, 0.0] for node in nodes}
        for i, a in enumerate(nodes):
            ax, ay = pos[a]
            for b in nodes[i + 1 :]:
                bx, by = pos[b]
                dx, dy = ax - bx, ay - by
                dist = math.hypot(dx, dy) + 1e-6
                force = k * k / dist
                disp[a][0] += dx / dist * force
                disp[a][1] += dy / dist * force
                disp[b][0] -= dx / dist * force
                disp[b][1] -= dy / dist * force
        for (a, b), weight in weights.items():
            ax, ay = pos[a]
            bx, by = pos[b]
            dx, dy = ax - bx, ay - by
            dist = math.hypot(dx, dy) + 1e-6
            strength = 0.35 + 1.65 * math.sqrt(weight / max_w)
            force = (dist * dist / k) * strength
            disp[a][0] -= dx / dist * force
            disp[a][1] -= dy / dist * force
            disp[b][0] += dx / dist * force
            disp[b][1] += dy / dist * force
        t = temperature * (1 - step / iterations)
        for node in nodes:
            dx, dy = disp[node]
            length = math.hypot(dx, dy) + 1e-6
            x, y = pos[node]
            x += dx / length * min(length, t)
            y += dy / length * min(length, t)
            pos[node] = (min(0.98, max(0.02, x)), min(0.98, max(0.02, y)))
    xs = [pos[n][0] for n in nodes]
    ys = [pos[n][1] for n in nodes]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return {
        node: (
            (pos[node][0] - min_x) / (max_x - min_x + 1e-9),
            (pos[node][1] - min_y) / (max_y - min_y + 1e-9),
        )
        for node in nodes
    }


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
    strategy: str,
    edges: list[dict[str, object]],
    layout: dict[str, tuple[float, float]],
    node_colours: dict[str, str],
    degree_ref: dict[str, float],
    origin_x: int,
    origin_y: int,
    width: int,
    height: int,
    max_edge_weight: float,
    max_degree: float,
    max_edges: int,
) -> str:
    margin = 34
    lines = [
        f'<g transform="translate({origin_x},{origin_y})">',
        svg_text(0, 18, STRATEGY_LABELS[strategy], 18, "bold"),
    ]
    sorted_edges = sorted(edges, key=lambda e: float(e["weight"]), reverse=True)[:max_edges]
    for edge in sorted(sorted_edges, key=lambda e: float(e["weight"])):
        s, t = str(edge["source"]), str(edge["target"])
        if s not in layout or t not in layout:
            continue
        sx, sy = layout[s]
        tx, ty = layout[t]
        x1 = margin + sx * (width - 2 * margin)
        y1 = 48 + margin + sy * (height - 78 - 2 * margin)
        x2 = margin + tx * (width - 2 * margin)
        y2 = 48 + margin + ty * (height - 78 - 2 * margin)
        w = float(edge["weight"])
        stroke_w = 0.8 + 5.0 * math.sqrt(w / max_edge_weight)
        edge_colour = node_colours.get(s, "#8C8C8C")
        lines.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{edge_colour}" stroke-width="{stroke_w:.2f}" stroke-opacity="1.0" />'
        )
    nodes = sorted(layout, key=lambda n: degree_ref.get(n, 0))
    for node in nodes:
        x, y = layout[node]
        cx = margin + x * (width - 2 * margin)
        cy = 48 + margin + y * (height - 78 - 2 * margin)
        degree = degree_ref.get(node, 0.0)
        radius = 3.0 + 6.0 * math.sqrt(degree / max_degree) if max_degree else 3.0
        colour = node_colours.get(node, "#BDBDBD")
        lines.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.2f}" '
            f'fill="{colour}" fill-opacity="0.92" stroke="#333333" stroke-width="0.55" />'
        )
    lines.append("</g>")
    return "\n".join(lines)


def render_svg(
    path: Path,
    strategy_edges: dict[str, list[dict[str, object]]],
    strategy_colours: dict[str, dict[str, str]],
    layout: dict[str, tuple[float, float]],
    cluster_colour: dict[str, str],
    full_degree: dict[str, float],
    max_edges: int,
    network_stats: dict[str, dict[str, float]],
) -> None:
    panel_w, panel_h = 360, 300
    gap_x, gap_y = 34, 56
    width = panel_w * 3 + gap_x * 2 + 250
    height = panel_h * 2 + gap_y + 30
    max_edge_weight = max(float(e["weight"]) for edges in strategy_edges.values() for e in edges)
    max_degree = max(full_degree.values()) if full_degree else 1.0
    order = ["full", "random_50", "city_weighted", "year_balanced", "city_cap20", "city_cap20_year_balanced"]
    positions = {
        "full": (20, 18),
        "random_50": (20 + panel_w + gap_x, 18),
        "city_weighted": (20 + 2 * (panel_w + gap_x), 18),
        "year_balanced": (20, 18 + panel_h + gap_y),
        "city_cap20": (20 + panel_w + gap_x, 18 + panel_h + gap_y),
        "city_cap20_year_balanced": (20 + 2 * (panel_w + gap_x), 18 + panel_h + gap_y),
    }
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for strategy in order:
        lines.append(
            render_panel(
                strategy,
                strategy_edges[strategy],
                layout,
                strategy_colours[strategy],
                full_degree,
                positions[strategy][0],
                positions[strategy][1],
                panel_w,
                panel_h,
                max_edge_weight,
                max_degree,
                max_edges,
            )
        )
        stat = network_stats.get(strategy, {})
        if stat:
            x, y = positions[strategy]
            text = f'nodes={int(stat["n_nodes"])}, edges={int(stat["n_edges"])}, Q={stat["modularity_q"]:.2f}'
            lines.append(svg_text(x + 4, y + panel_h - 10, text, 10, "normal", "#555"))
    legend_x = 20 + 2 * (panel_w + gap_x)
    legend_y = 18 + panel_h + gap_y + 20
    lines.append(svg_text(legend_x, legend_y, "Geography-cluster colour scheme", 14, "bold"))
    y = legend_y + 28
    for cluster, colour in sorted(cluster_colour.items()):
        if cluster == "Unknown":
            continue
        lines.append(f'<rect x="{legend_x}" y="{y - 11}" width="14" height="14" fill="{colour}" stroke="#333" stroke-width="0.4"/>')
        lines.append(svg_text(legend_x + 22, y, cluster, 11))
        y += 22
    lines.append(svg_text(legend_x, y + 8, "Node size: original weighted degree\nEdge colour: source-node cluster", 11, "normal", "#555"))
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    figure_dir = Path(__file__).resolve().parents[1]
    data_dir = figure_dir / "data"
    parser.add_argument("--nexus", default=str(data_dir / "treetime_annotated.nexus"))
    parser.add_argument("--distances", default=str(data_dir / "PairwiseChinaCities_distance.tsv"))
    parser.add_argument("--dates", default=str(data_dir / "Dates.tsv"))
    parser.add_argument("--metadata", default=str(data_dir / "Supplementary_Tablesv2.xlsx"))
    parser.add_argument("--network", default=str(data_dir / "China_Transmission_Normalization.tsv"))
    parser.add_argument("--outdir", default=str(figure_dir / "reproduced_average_networks"))
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--max-edges", type=int, default=10000)
    parser.add_argument("--event-weight-mode", choices=["binary", "sample_fraction"], default="binary")
    args = parser.parse_args()

    text = Path(args.nexus).read_text(encoding="utf-8")
    taxlabels = read_taxlabels(text)
    root = NewickParser(read_tree_text(text)).parse()
    assign_tips(root, taxlabels)
    tips = sorted(root.tips)
    meta = tip_metadata(root)
    years = read_dates(Path(args.dates))
    cities = {tip: meta.get(tip, {}).get("city", "") for tip in tips}
    distances = read_distances(Path(args.distances))
    events = extract_events(root)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    city_clusters = read_city_clusters(Path(args.metadata))
    city_populations = read_city_populations(Path(args.metadata))
    colour_by_cluster = cluster_colours(city_clusters)

    full_edges = read_edges(Path(args.network))
    strategies = ["full", "random_50", "city_weighted", "year_balanced", "city_cap20", "city_cap20_year_balanced"]
    strategy_edges: dict[str, list[dict[str, object]]] = {"full": full_edges}
    for strategy in strategies[1:]:
        replicate_edges = []
        for _ in range(args.replicates):
            if strategy == "city_weighted":
                sampled = city_weighted_sample(
                    tips,
                    cities,
                    city_populations,
                    max(3, round(len(tips) * args.fraction)),
                    rng,
                )
            else:
                sampled = sample_tips(tips, strategy, years, cities, rng, args.fraction)
            edges, _ = network_from_events(events, sampled, distances, args.event_weight_mode)
            replicate_edges.append(edges)
        strategy_edges[strategy] = average_edges(replicate_edges)

    all_nodes = sorted({node for edges in strategy_edges.values() for node in nodes_from_edges(edges)})
    layout = force_layout(all_nodes, full_edges, seed=args.seed)
    full_degree = weighted_degrees(full_edges)

    strategy_colours: dict[str, dict[str, str]] = {}
    community_rows = []
    network_metric_rows = []
    network_stats = {}
    for strategy, edges in strategy_edges.items():
        labels = label_propagation(edges)
        if strategy == "full":
            cluster_ids = {cluster: idx + 1 for idx, cluster in enumerate(sorted(set(city_clusters.get(city, "Unknown") for city in all_nodes)))}
            displayed_labels = {city: cluster_ids[city_clusters.get(city, "Unknown")] for city in all_nodes}
            node_colours = {city: colour_by_cluster.get(city_clusters.get(city, "Unknown"), "#BDBDBD") for city in all_nodes}
            for city in all_nodes:
                community_rows.append({
                    "strategy": strategy,
                    "city": city,
                    "community": "",
                    "mapped_geography_cluster": city_clusters.get(city, "Unknown"),
                    "colour": node_colours[city],
                })
        else:
            mapped = community_to_geo_cluster(labels, city_clusters)
            displayed_labels = labels
            node_colours = {}
            for city in all_nodes:
                label = labels.get(city, 0)
                geo = mapped.get(label, "Unknown")
                colour = colour_by_cluster.get(geo, "#BDBDBD")
                node_colours[city] = colour
                community_rows.append({
                    "strategy": strategy,
                    "city": city,
                    "community": label if label else "",
                    "mapped_geography_cluster": geo,
                    "colour": colour,
                })
        strategy_colours[strategy] = node_colours
        nodes_present = set(nodes_from_edges(edges))
        stats = {
            "strategy": strategy,
            "n_nodes": len(nodes_present),
            "n_edges": len(edges),
            "n_modules": len(set(displayed_labels.get(node, 0) for node in nodes_present)),
            "modularity_q": modularity_q(edges, displayed_labels),
            "within_module_ratio": within_module_ratio(edges, displayed_labels),
            "weighted_long_distance_ratio_gt800km": weighted_long_distance_ratio(edges, distances, 800.0),
        }
        network_stats[strategy] = stats
        network_metric_rows.append(stats)

    edge_rows = []
    for strategy, edges in strategy_edges.items():
        for edge in sorted(edges, key=lambda e: (str(e["source"]), str(e["target"]))):
            edge_rows.append({
                "strategy": strategy,
                "source": edge["source"],
                "target": edge["target"],
                "mean_weight": float(edge["weight"]),
            })
    write_tsv(outdir / "average_network_edges.tsv", edge_rows, ["strategy", "source", "target", "mean_weight"])
    write_tsv(outdir / "average_network_communities.tsv", community_rows, ["strategy", "city", "community", "mapped_geography_cluster", "colour"])
    write_tsv(
        outdir / "displayed_network_metrics.tsv",
        network_metric_rows,
        ["strategy", "n_nodes", "n_edges", "n_modules", "modularity_q", "within_module_ratio", "weighted_long_distance_ratio_gt800km"],
    )
    with (outdir / "network_cluster_colour_map.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=["geography_cluster", "colour"])
        writer.writeheader()
        for cluster, colour in sorted(colour_by_cluster.items()):
            writer.writerow({"geography_cluster": cluster, "colour": colour})

    render_svg(
        outdir / "average_network_community_figure.svg",
        strategy_edges,
        strategy_colours,
        layout,
        colour_by_cluster,
        full_degree,
        args.max_edges,
        network_stats,
    )


if __name__ == "__main__":
    main()
