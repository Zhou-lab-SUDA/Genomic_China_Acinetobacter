"""Plot China normalized-network skeleton updated by nexus downsampling results."""

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
from network_resampling_sensitivity import label_propagation, nodes_from_edges, weighted_degrees, within_module_ratio, write_tsv
from plot_average_network_communities import cluster_colours, community_to_geo_cluster, force_layout, modularity_q, svg_text


STRATEGY_LABELS = {
    "full": "Original China network\n(52 nexus cities)",
    "random_50": "50% size downsampling",
    "city_weighted": "City-weighted sampling",
    "year_balanced": "Year-balanced sampling",
    "city_cap20": "City-balanced sampling",
    "city_cap20_year_balanced": "City-balanced +\nyear-balanced sampling",
}


def read_china_edges(path: Path) -> list[dict[str, object]]:
    weights: dict[tuple[str, str], float] = defaultdict(float)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            weights[(row["Source"], row["Target"])] += float(row["Weight"])
    return [{"source": s, "target": t, "weight": w} for (s, t), w in sorted(weights.items())]


def read_nexus_average_edges(path: Path) -> dict[str, dict[tuple[str, str], float]]:
    out: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            out[row["strategy"]][(row["source"], row["target"])] = float(row["mean_weight"])
    return dict(out)


def read_nexus_replicate_edges(directory: Path, replicate: int) -> dict[str, dict[tuple[str, str], float]]:
    out: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
    avg = read_nexus_average_edges(directory / "average_network_edges.tsv")
    out["full"] = avg["full"]
    for strategy in ["random_50", "city_weighted", "year_balanced", "city_cap20", "city_cap20_year_balanced"]:
        path = directory / f"replicate_edges_{strategy}_{replicate}.tsv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                out[strategy][(row["source"], row["target"])] = float(row["weight"])
    return dict(out)


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
    clusters["Taian"] = clusters.get("Tai'an", "North China")
    clusters["Jinan"] = "North China"
    return clusters


def strategy_display_edges(
    strategy: str,
    china_edges_52: list[dict[str, object]],
    nexus_edges: dict[str, dict[tuple[str, str], float]],
    mode: str,
) -> list[dict[str, object]]:
    nexus_full = nexus_edges["full"]
    current = nexus_edges[strategy]
    out = []
    for edge in china_edges_52:
        key = (str(edge["source"]), str(edge["target"]))
        base = float(edge["weight"])
        if strategy == "full":
            if mode == "all_china_full":
                weight = base
            else:
                weight = base if key in nexus_full else 0.0
        else:
            full_w = nexus_full.get(key, 0.0)
            cur_w = current.get(key, 0.0)
            if full_w <= 0 or cur_w <= 0:
                weight = 0.0
            else:
                weight = base * min(1.0, cur_w / full_w)
        if weight > 0:
            out.append({"source": key[0], "target": key[1], "weight": weight})
    return out


def node_colour_map(
    strategy: str,
    nodes: list[str],
    edges: list[dict[str, object]],
    city_clusters: dict[str, str],
    colours: dict[str, str],
) -> tuple[dict[str, str], dict[str, int | str]]:
    if strategy == "full":
        return (
            {node: colours.get(city_clusters.get(node, "Unknown"), "#BDBDBD") for node in nodes},
            {node: city_clusters.get(node, "Unknown") for node in nodes},
        )
    labels = label_propagation(edges) if edges else {}
    mapped = community_to_geo_cluster(labels, city_clusters) if labels else {}
    out = {}
    assignments = {}
    for node in nodes:
        if node in labels:
            geo = mapped.get(labels[node], city_clusters.get(node, "Unknown"))
            assignments[node] = labels[node]
        else:
            geo = city_clusters.get(node, "Unknown")
            assignments[node] = "isolated"
        out[node] = colours.get(geo, "#BDBDBD")
    return out, assignments


def render_panel(
    strategy: str,
    nodes: list[str],
    edges: list[dict[str, object]],
    layout: dict[str, tuple[float, float]],
    node_colours: dict[str, str],
    full_degree: dict[str, float],
    x0: int,
    y0: int,
    width: int,
    height: int,
    max_edge: float,
    max_degree: float,
    stats: dict[str, float],
) -> str:
    margin = 34
    lines = [f'<g transform="translate({x0},{y0})">', svg_text(0, 18, STRATEGY_LABELS[strategy], 18, "bold")]
    for edge in sorted(edges, key=lambda e: float(e["weight"])):
        s, t = str(edge["source"]), str(edge["target"])
        sx, sy = layout[s]
        tx, ty = layout[t]
        x1 = margin + sx * (width - 2 * margin)
        y1 = 48 + margin + sy * (height - 88 - 2 * margin)
        x2 = margin + tx * (width - 2 * margin)
        y2 = 48 + margin + ty * (height - 88 - 2 * margin)
        weight = float(edge["weight"])
        stroke_w = 0.55 + 4.3 * math.sqrt(weight / max_edge)
        colour = node_colours.get(s, "#888888")
        lines.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{colour}" stroke-width="{stroke_w:.2f}" stroke-opacity="1.0" />'
        )
    for node in sorted(nodes, key=lambda n: full_degree.get(n, 0.0)):
        x, y = layout[node]
        cx = margin + x * (width - 2 * margin)
        cy = 48 + margin + y * (height - 88 - 2 * margin)
        radius = 3.0 + 6.0 * math.sqrt(full_degree.get(node, 0.0) / max_degree) if max_degree else 3.0
        lines.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.2f}" fill="{node_colours.get(node, "#BDBDBD")}" '
            f'fill-opacity="0.95" stroke="#333333" stroke-width="0.55" />'
        )
    label = f'nodes=52, active={int(stats["active_nodes"])}, edges={int(stats["n_edges"])}, Q={stats["modularity_q"]:.2f}'
    lines.append(svg_text(4, height - 12, label, 10, "normal", "#555"))
    lines.append("</g>")
    return "\n".join(lines)


def render_svg(
    out: Path,
    nodes: list[str],
    display_edges: dict[str, list[dict[str, object]]],
    node_colours_by_strategy: dict[str, dict[str, str]],
    layout: dict[str, tuple[float, float]],
    colour_by_cluster: dict[str, str],
    full_degree: dict[str, float],
    stats_by_strategy: dict[str, dict[str, float]],
) -> None:
    panel_w, panel_h = 360, 300
    gap_x, gap_y = 34, 56
    width = panel_w * 3 + gap_x * 2 + 250
    height = panel_h * 2 + gap_y + 30
    order = ["full", "random_50", "city_weighted", "year_balanced", "city_cap20", "city_cap20_year_balanced"]
    positions = {
        "full": (20, 18),
        "random_50": (20 + panel_w + gap_x, 18),
        "city_weighted": (20 + 2 * (panel_w + gap_x), 18),
        "year_balanced": (20, 18 + panel_h + gap_y),
        "city_cap20": (20 + panel_w + gap_x, 18 + panel_h + gap_y),
        "city_cap20_year_balanced": (20 + 2 * (panel_w + gap_x), 18 + panel_h + gap_y),
    }
    max_edge = max(float(edge["weight"]) for edges in display_edges.values() for edge in edges)
    max_degree = max(full_degree.values()) if full_degree else 1.0
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    for strategy in order:
        lines.append(render_panel(strategy, nodes, display_edges[strategy], layout, node_colours_by_strategy[strategy], full_degree, positions[strategy][0], positions[strategy][1], panel_w, panel_h, max_edge, max_degree, stats_by_strategy[strategy]))
    legend_x = 20 + 2 * (panel_w + gap_x)
    legend_y = 18 + panel_h + gap_y + 18
    lines.append(svg_text(legend_x, legend_y, "Node colour scheme", 14, "bold"))
    y = legend_y + 28
    for cluster, colour in sorted(colour_by_cluster.items()):
        if cluster == "Unknown":
            continue
        lines.append(f'<rect x="{legend_x}" y="{y - 11}" width="14" height="14" fill="{colour}" stroke="#333" stroke-width="0.4"/>')
        lines.append(svg_text(legend_x + 22, y, cluster, 11))
        y += 22
    lines.append(svg_text(legend_x, y + 8, "Edges: China network skeleton\nThickness/presence: nexus sampling", 11, "normal", "#555"))
    lines.append("</svg>")
    out.write_text("\n".join(lines), encoding="utf-8")


def circular_layout(
    nodes: list[str],
    full_edges: list[dict[str, object]],
    city_clusters: dict[str, str],
) -> dict[str, tuple[float, float]]:
    degree = weighted_degrees(full_edges)
    ordered = sorted(nodes, key=lambda n: (city_clusters.get(n, "Unknown"), -degree.get(n, 0.0), n))
    n = max(1, len(ordered))
    out = {}
    for i, node in enumerate(ordered):
        angle = 2 * math.pi * i / n - math.pi / 2
        out[node] = (0.5 + 0.47 * math.cos(angle), 0.5 + 0.47 * math.sin(angle))
    return out


def forceatlas_like_layout(
    nodes: list[str],
    edges: list[dict[str, object]],
    city_clusters: dict[str, str],
    seed: int = 20260607,
    iterations: int = 1400,
) -> dict[str, tuple[float, float]]:
    rng = random.Random(seed)
    n = max(1, len(nodes))
    clusters = sorted({city_clusters.get(node, "Unknown") for node in nodes})
    cluster_angles = {cluster: 2 * math.pi * i / max(1, len(clusters)) for i, cluster in enumerate(clusters)}
    pos = {}
    for node in nodes:
        angle = cluster_angles.get(city_clusters.get(node, "Unknown"), 0.0)
        radius = rng.uniform(0.02, 0.16)
        pos[node] = (
            0.5 + radius * math.cos(angle) + rng.uniform(-0.07, 0.07),
            0.5 + radius * math.sin(angle) + rng.uniform(-0.07, 0.07),
        )

    undirected: dict[tuple[str, str], float] = defaultdict(float)
    for edge in edges:
        s, t = sorted([str(edge["source"]), str(edge["target"])])
        if s != t:
            undirected[(s, t)] += math.sqrt(float(edge["weight"]))
    max_w = max(undirected.values()) if undirected else 1.0
    degree = weighted_degrees(edges)
    max_degree = max(degree.values()) if degree else 1.0

    for step in range(iterations):
        disp = {node: [0.0, 0.0] for node in nodes}
        # Repulsion with hub damping, closer to the compact ForceAtlas visual.
        for i, a in enumerate(nodes):
            ax, ay = pos[a]
            for b in nodes[i + 1 :]:
                bx, by = pos[b]
                dx, dy = ax - bx, ay - by
                dist2 = dx * dx + dy * dy + 0.002
                dist = math.sqrt(dist2)
                hub_damping = 0.55 + 0.45 * (1 - min(degree.get(a, 0.0), degree.get(b, 0.0)) / max_degree)
                force = 0.0028 * hub_damping / dist2
                disp[a][0] += dx / dist * force
                disp[a][1] += dy / dist * force
                disp[b][0] -= dx / dist * force
                disp[b][1] -= dy / dist * force
        # Attraction along weighted edges.
        for (a, b), weight in undirected.items():
            ax, ay = pos[a]
            bx, by = pos[b]
            dx, dy = ax - bx, ay - by
            dist = math.sqrt(dx * dx + dy * dy) + 1e-6
            force = 0.018 * (weight / max_w) * dist
            disp[a][0] -= dx / dist * force
            disp[a][1] -= dy / dist * force
            disp[b][0] += dx / dist * force
            disp[b][1] += dy / dist * force
        # Mild gravity to keep the graph as a dense cloud.
        for node in nodes:
            x, y = pos[node]
            disp[node][0] += (0.5 - x) * 0.006
            disp[node][1] += (0.5 - y) * 0.006
        temp = 0.09 * (1 - step / iterations) + 0.002
        for node in nodes:
            dx, dy = disp[node]
            length = math.sqrt(dx * dx + dy * dy) + 1e-9
            x, y = pos[node]
            x += dx / length * min(length, temp)
            y += dy / length * min(length, temp)
            pos[node] = (x, y)

    xs = [pos[node][0] for node in nodes]
    ys = [pos[node][1] for node in nodes]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    scale = max(max_x - min_x, max_y - min_y) + 1e-9
    return {
        node: (
            0.5 + ((pos[node][0] - (min_x + max_x) / 2) / scale) * 0.88,
            0.5 + ((pos[node][1] - (min_y + max_y) / 2) / scale) * 0.88,
        )
        for node in nodes
    }


def fr_cloud_layout(
    nodes: list[str],
    edges: list[dict[str, object]],
    seed: int = 20260607,
    iterations: int = 1800,
) -> dict[str, tuple[float, float]]:
    rng = random.Random(seed)
    n = max(1, len(nodes))
    pos = {node: [rng.gauss(0.5, 0.18), rng.gauss(0.5, 0.18)] for node in nodes}
    undirected: dict[tuple[str, str], float] = defaultdict(float)
    for edge in edges:
        s, t = sorted([str(edge["source"]), str(edge["target"])])
        if s != t:
            undirected[(s, t)] += math.sqrt(float(edge["weight"]))
    max_w = max(undirected.values()) if undirected else 1.0
    k = math.sqrt(1.0 / n) * 0.52
    temperature = 0.14
    for step in range(iterations):
        disp = {node: [0.0, 0.0] for node in nodes}
        for i, a in enumerate(nodes):
            ax, ay = pos[a]
            for b in nodes[i + 1 :]:
                bx, by = pos[b]
                dx, dy = ax - bx, ay - by
                dist = math.hypot(dx, dy) + 1e-6
                force = (k * k / dist) * 0.62
                disp[a][0] += dx / dist * force
                disp[a][1] += dy / dist * force
                disp[b][0] -= dx / dist * force
                disp[b][1] -= dy / dist * force
        for (a, b), weight in undirected.items():
            ax, ay = pos[a]
            bx, by = pos[b]
            dx, dy = ax - bx, ay - by
            dist = math.hypot(dx, dy) + 1e-6
            strength = 0.8 + 3.2 * (weight / max_w)
            force = (dist * dist / k) * strength
            disp[a][0] -= dx / dist * force
            disp[a][1] -= dy / dist * force
            disp[b][0] += dx / dist * force
            disp[b][1] += dy / dist * force
        for node in nodes:
            x, y = pos[node]
            disp[node][0] += (0.5 - x) * 0.012
            disp[node][1] += (0.5 - y) * 0.012
        temp = temperature * (1 - step / iterations) ** 1.35 + 0.001
        for node in nodes:
            dx, dy = disp[node]
            length = math.hypot(dx, dy) + 1e-9
            x, y = pos[node]
            x += dx / length * min(length, temp)
            y += dy / length * min(length, temp)
            pos[node] = [x, y]
    xs = [pos[node][0] for node in nodes]
    ys = [pos[node][1] for node in nodes]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    scale = max(max_x - min_x, max_y - min_y, 1e-9)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    return {
        node: (
            0.5 + ((pos[node][0] - center_x) / scale) * 0.88,
            0.5 + ((pos[node][1] - center_y) / scale) * 0.88,
        )
        for node in nodes
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    figure_dir = Path(__file__).resolve().parents[1]
    data_dir = figure_dir / "data"
    parser.add_argument("--china-network", default=str(data_dir / "China_Transmission_Normalization.tsv"))
    parser.add_argument("--nexus-average-edges", default=str(data_dir / "average_network_edges.tsv"))
    parser.add_argument("--nexus-replicate-dir", default="")
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--metadata", default=str(Path(__file__).resolve().parents[2] / "sample_metadata.xlsx"))
    parser.add_argument("--outdir", default=str(figure_dir / "network_skeleton_output"))
    parser.add_argument("--full-mode", choices=["all_china_full", "nexus_supported_only"], default="all_china_full")
    parser.add_argument("--layout", choices=["forceatlas", "fr", "circular"], default="forceatlas")
    parser.add_argument("--seed", type=int, default=20260607)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    china_edges = read_china_edges(Path(args.china_network))
    if args.replicate and args.nexus_replicate_dir:
        nexus_edges = read_nexus_replicate_edges(Path(args.nexus_replicate_dir), args.replicate)
    else:
        nexus_edges = read_nexus_average_edges(Path(args.nexus_average_edges))
    nexus_nodes = sorted({node for key in nexus_edges["full"] for node in key})
    nexus_node_set = set(nexus_nodes)
    china_edges_52 = [edge for edge in china_edges if edge["source"] in nexus_node_set and edge["target"] in nexus_node_set]

    city_clusters = read_city_clusters(Path(args.metadata))
    colours = cluster_colours(city_clusters)
    if args.layout == "forceatlas":
        layout = forceatlas_like_layout(nexus_nodes, china_edges_52, city_clusters, seed=args.seed)
    elif args.layout == "circular":
        layout = circular_layout(nexus_nodes, china_edges_52, city_clusters)
    else:
        layout = fr_cloud_layout(nexus_nodes, china_edges_52, seed=args.seed)
    full_degree = weighted_degrees(china_edges_52)

    display_edges = {}
    node_colours_by_strategy = {}
    community_rows = []
    stats_rows = []
    stats_by_strategy = {}
    for strategy in STRATEGY_LABELS:
        edges = strategy_display_edges(strategy, china_edges_52, nexus_edges, args.full_mode)
        display_edges[strategy] = edges
        node_colours, assignments = node_colour_map(strategy, nexus_nodes, edges, city_clusters, colours)
        node_colours_by_strategy[strategy] = node_colours
        active_nodes = set(nodes_from_edges(edges))
        labels = label_propagation(edges) if edges else {}
        q = modularity_q(edges, labels) if labels else float("nan")
        within = within_module_ratio(edges, labels) if labels else float("nan")
        stats = {
            "strategy": strategy,
            "total_nodes": len(nexus_nodes),
            "active_nodes": len(active_nodes),
            "n_edges": len(edges),
            "n_modules": len(set(labels.values())) if labels else 0,
            "modularity_q": q,
            "within_module_ratio": within,
        }
        stats_rows.append(stats)
        stats_by_strategy[strategy] = stats
        for node in nexus_nodes:
            community_rows.append({"strategy": strategy, "city": node, "assignment": assignments[node], "colour": node_colours[node]})

    edge_rows = []
    for strategy, edges in display_edges.items():
        for edge in edges:
            edge_rows.append({"strategy": strategy, "source": edge["source"], "target": edge["target"], "display_weight": edge["weight"]})
    removed_nodes = sorted(({edge["source"] for edge in china_edges} | {edge["target"] for edge in china_edges}) - nexus_node_set)
    removed_rows = [{"removed_city": city} for city in removed_nodes]

    write_tsv(outdir / "china_skeleton_nexus_display_edges.tsv", edge_rows, ["strategy", "source", "target", "display_weight"])
    write_tsv(outdir / "china_skeleton_nexus_node_colours.tsv", community_rows, ["strategy", "city", "assignment", "colour"])
    write_tsv(outdir / "china_skeleton_nexus_network_metrics.tsv", stats_rows, ["strategy", "total_nodes", "active_nodes", "n_edges", "n_modules", "modularity_q", "within_module_ratio"])
    write_tsv(outdir / "china_skeleton_removed_nodes.tsv", removed_rows, ["removed_city"])
    suffix = f"_replicate{args.replicate}" if args.replicate else ""
    suffix = f"{suffix}_{args.layout}" if suffix else ""
    render_svg(outdir / f"china_skeleton_nexus_downsampling_network{suffix}.svg", nexus_nodes, display_edges, node_colours_by_strategy, layout, colours, full_degree, stats_by_strategy)


if __name__ == "__main__":
    main()
