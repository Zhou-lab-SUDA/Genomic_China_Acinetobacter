"""Plot average nexus downsampling networks from a precomputed edge table."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from network_resampling_sensitivity import label_propagation, nodes_from_edges, weighted_degrees, within_module_ratio, write_tsv  # noqa: E402
from plot_average_network_communities import (  # noqa: E402
    cluster_colours,
    community_to_geo_cluster,
    force_layout,
    modularity_q,
    read_city_clusters,
    render_svg,
    weighted_long_distance_ratio,
)
from genome_network_sensitivity import read_distances  # noqa: E402


def read_average_edge_table(path: Path) -> dict[str, list[dict[str, object]]]:
    out: dict[str, list[dict[str, object]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            out.setdefault(row["strategy"], []).append({
                "source": row["source"],
                "target": row["target"],
                "weight": float(row["mean_weight"]),
            })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    figure_dir = Path(__file__).resolve().parents[1]
    module_dir = Path(__file__).resolve().parents[2]
    data_dir = figure_dir / "data"
    parser.add_argument("--edges", default=str(data_dir / "average_network_edges.tsv"))
    parser.add_argument("--metadata", default=str(module_dir / "sample_metadata.xlsx"))
    parser.add_argument("--distances", default=str(data_dir / "PairwiseChinaCities_distance.tsv"))
    parser.add_argument("--outdir", default=str(figure_dir / "average_network_output"))
    parser.add_argument("--out", default="nexus_average_network_community_figure.svg")
    parser.add_argument("--max-edges", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260607)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    strategy_edges = read_average_edge_table(Path(args.edges))
    full_edges = strategy_edges["full"]
    full_nodes = set(nodes_from_edges(full_edges))

    # Enforce one visual universe: the 52 cities present in the nexus full network.
    for strategy, edges in list(strategy_edges.items()):
        strategy_edges[strategy] = [
            edge for edge in edges
            if str(edge["source"]) in full_nodes and str(edge["target"]) in full_nodes
        ]

    city_clusters = read_city_clusters(Path(args.metadata))
    colour_by_cluster = cluster_colours(city_clusters)
    distances = read_distances(Path(args.distances))
    layout = force_layout(sorted(full_nodes), full_edges, seed=args.seed)
    full_degree = weighted_degrees(full_edges)

    strategy_colours: dict[str, dict[str, str]] = {}
    community_rows = []
    network_rows = []
    network_stats = {}

    cluster_ids = {cluster: idx + 1 for idx, cluster in enumerate(sorted(set(city_clusters.get(city, "Unknown") for city in full_nodes)))}
    full_labels = {city: cluster_ids[city_clusters.get(city, "Unknown")] for city in full_nodes}

    for strategy, edges in strategy_edges.items():
        labels = label_propagation(edges)
        if strategy == "full":
            display_labels = full_labels
            metric_labels = labels
            node_colours = {
                city: colour_by_cluster.get(city_clusters.get(city, "Unknown"), "#BDBDBD")
                for city in full_nodes
            }
            for city in sorted(full_nodes):
                community_rows.append({
                    "strategy": strategy,
                    "city": city,
                    "community": "",
                    "mapped_geography_cluster": city_clusters.get(city, "Unknown"),
                    "colour": node_colours[city],
                })
        else:
            mapped = community_to_geo_cluster(labels, city_clusters)
            display_labels = labels
            metric_labels = labels
            node_colours = {}
            for city in sorted(full_nodes):
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
        row = {
            "strategy": strategy,
            "n_nodes": len(nodes_present),
            "n_edges": len(edges),
            "n_modules": len(set(metric_labels.get(node, 0) for node in nodes_present)),
            "modularity_q": modularity_q(edges, metric_labels),
            "within_module_ratio": within_module_ratio(edges, metric_labels),
            "weighted_long_distance_ratio_gt800km": weighted_long_distance_ratio(edges, distances, 800.0),
        }
        network_stats[strategy] = row
        network_rows.append(row)

    write_tsv(
        outdir / "nexus_average_network_communities.tsv",
        community_rows,
        ["strategy", "city", "community", "mapped_geography_cluster", "colour"],
    )
    write_tsv(
        outdir / "nexus_average_network_metrics.tsv",
        network_rows,
        ["strategy", "n_nodes", "n_edges", "n_modules", "modularity_q", "within_module_ratio", "weighted_long_distance_ratio_gt800km"],
    )
    render_svg(
        outdir / args.out,
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
