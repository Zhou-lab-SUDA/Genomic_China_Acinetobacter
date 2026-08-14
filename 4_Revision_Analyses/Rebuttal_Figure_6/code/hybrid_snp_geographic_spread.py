#!/usr/bin/env python3
"""Relate geographic spread to the unified observed-plus-proxy SNP distance."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["svg.hashsalt"] = "nm_ab_snp_geographic_spread_v1"


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "SNP"))

from compute_pairwise_snp_from_alignment import sample_key  # noqa: E402
from hybrid_snp_analysis import dated_time_matrix  # noqa: E402
from snp_time_validation import lca, load_tree  # noqa: E402


COLOURS = {
    "All ESL": "#A8A8A8",
    "ESL2.4": "#EB6564",
    "ESL2.5": "#F5BA1A",
}
LINEAGES = ("2.3", "2.4", "2.5")


def canonical(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = pd.Series(x).rank(method="average").to_numpy()
    y_rank = pd.Series(y).rank(method="average").to_numpy()
    return pearson_correlation(x_rank, y_rank)


def load_city_distances(path: Path) -> dict[tuple[str, str], float]:
    table = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["city1", "city2", "geographic_km", "road_km"],
    )
    result: dict[tuple[str, str], float] = {}
    for row in table.itertuples(index=False):
        result[canonical(str(row.city1), str(row.city2))] = float(
            row.geographic_km
        )
    return result


def geographic_matrix(
    cities: np.ndarray,
    distance_lookup: dict[tuple[str, str], float],
) -> np.ndarray:
    unique = sorted(set(cities))
    city_index = {city: i for i, city in enumerate(unique)}
    local = np.full((len(unique), len(unique)), np.nan, dtype=np.float32)
    np.fill_diagonal(local, 0.0)
    for (city1, city2), distance in distance_lookup.items():
        if city1 in city_index and city2 in city_index:
            i = city_index[city1]
            j = city_index[city2]
            local[i, j] = local[j, i] = distance
    codes = np.asarray([city_index[city] for city in cities], dtype=int)
    return local[np.ix_(codes, codes)]


def build_mrca_year_matrix(
    genomes: np.ndarray,
    groups: np.ndarray,
    nexus: Path,
    dates_path: Path,
    cache_path: Path,
) -> np.ndarray:
    if cache_path.exists():
        cached = np.load(cache_path)
        if np.array_equal(cached["genomes"].astype(str), genomes):
            return cached["mrca_year"]

    tips, parent, depth, generation, up = load_tree(nexus)
    dates = pd.read_csv(dates_path, sep="\t")
    dates["numeric_year"] = pd.to_numeric(dates.Year, errors="coerce")
    dates = dates.dropna(subset=["numeric_year"])
    date_lookup = {
        sample_key(genome): float(year)
        for genome, year in zip(dates.genome, dates.numeric_year)
    }
    nodes = np.asarray([tips[sample_key(genome)] for genome in genomes], dtype=int)
    dated_positions = np.asarray(
        [
            i
            for i, genome in enumerate(genomes)
            if sample_key(genome) in date_lookup
        ],
        dtype=int,
    )
    if len(dated_positions) < 10:
        raise ValueError("Too few dated tips to infer the dated-tree root year.")
    dated_years = np.asarray(
        [
            date_lookup[sample_key(genomes[i])]
            for i in dated_positions
        ],
        dtype=float,
    )
    root_date = float(
        np.median(dated_years - depth[nodes[dated_positions]])
    )
    node_dates = root_date + depth

    matrix = np.full((len(genomes), len(genomes)), np.nan, dtype=np.float32)
    np.fill_diagonal(matrix, node_dates[nodes])
    for lineage in LINEAGES:
        indices = np.flatnonzero(groups == lineage)
        for local_i, i in enumerate(indices[:-1]):
            node_i = int(nodes[i])
            for j in indices[local_i + 1 :]:
                ancestor = lca(
                    node_i,
                    int(nodes[j]),
                    parent,
                    generation,
                    up,
                )
                matrix[i, j] = matrix[j, i] = np.floor(node_dates[ancestor])
    np.savez_compressed(
        cache_path,
        genomes=genomes,
        mrca_year=matrix,
        inferred_root_date=root_date,
    )
    return matrix


def pair_vectors(
    indices: np.ndarray,
    snp_matrix: np.ndarray,
    geo_matrix: np.ndarray,
    observed_matrix: np.ndarray,
    mrca_year_matrix: np.ndarray | None,
    minimum_mrca_year: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(indices) < 2:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty
    upper = np.triu_indices(len(indices), 1)
    snp = snp_matrix[np.ix_(indices, indices)][upper]
    distance = geo_matrix[np.ix_(indices, indices)][upper]
    observed = observed_matrix[np.ix_(indices, indices)][upper]
    if mrca_year_matrix is None or minimum_mrca_year is None:
        return snp, distance, observed
    mrca_year = mrca_year_matrix[np.ix_(indices, indices)][upper]
    recent = mrca_year >= minimum_mrca_year
    return snp[recent], distance[recent], observed[recent]


def vectors_for_group(
    group: str,
    groups: np.ndarray,
    snp_matrix: np.ndarray,
    geo_matrix: np.ndarray,
    observed_matrix: np.ndarray,
    mrca_year_matrix: np.ndarray | None,
    minimum_mrca_year: int | None,
    selected: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lineage_values = LINEAGES if group == "All ESL" else (group.replace("ESL", ""),)
    chunks = []
    for lineage in lineage_values:
        if selected is None:
            indices = np.flatnonzero(groups == lineage)
        else:
            indices = selected[lineage]
        chunks.append(
            pair_vectors(
                indices,
                snp_matrix,
                geo_matrix,
                observed_matrix,
                mrca_year_matrix,
                minimum_mrca_year,
            )
        )
    return tuple(
        np.concatenate([chunk[column] for chunk in chunks])
        for column in range(3)
    )


def binned_means(
    snp: np.ndarray,
    distance: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(distance) & (snp >= edges[0]) & (snp <= edges[-1])
    snp = snp[valid]
    distance = distance[valid]
    bins = np.searchsorted(edges, snp, side="right") - 1
    bins[snp == edges[-1]] = len(edges) - 2
    count = np.bincount(bins, minlength=len(edges) - 1)
    total = np.bincount(bins, weights=distance, minlength=len(edges) - 1)
    mean = np.divide(
        total,
        count,
        out=np.full(len(edges) - 1, np.nan),
        where=count > 0,
    )
    return mean, count


def smooth_bins(values: np.ndarray, window: int = 3) -> np.ndarray:
    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def calibrate_shared_snp_rate(
    samples: pd.DataFrame,
    groups: np.ndarray,
    snp_matrix: np.ndarray,
    nexus: Path,
    cache_path: Path,
    maximum_calibration_snp: int = 100,
) -> tuple[float, int]:
    if cache_path.exists():
        cached = np.load(cache_path)
        time_matrix = cached["treetime_years"]
    else:
        time_matrix = dated_time_matrix(samples.reset_index(), nexus)
        np.savez_compressed(
            cache_path,
            genomes=samples.index.to_numpy(dtype=str),
            treetime_years=time_matrix,
        )

    snp_chunks = []
    time_chunks = []
    for lineage in LINEAGES:
        indices = np.flatnonzero(groups == lineage)
        upper = np.triu_indices(len(indices), 1)
        snp = snp_matrix[np.ix_(indices, indices)][upper].astype(float)
        time = time_matrix[np.ix_(indices, indices)][upper].astype(float)
        valid = (
            np.isfinite(time)
            & (time > 0)
            & (snp <= maximum_calibration_snp)
        )
        snp_chunks.append(snp[valid])
        time_chunks.append(time[valid])
    snp = np.concatenate(snp_chunks)
    time = np.concatenate(time_chunks)
    rate = float(np.dot(time, snp) / np.dot(time, time))
    return rate, len(snp)


def threshold_summary(
    vectors: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    bootstrap_threshold_means: dict[str, dict[int, list[float]]],
) -> pd.DataFrame:
    rows = []
    for threshold in (40, 50, 100):
        estimates = {}
        for group in ("ESL2.4", "ESL2.5", "All ESL"):
            snp, distance, source = vectors[group]
            valid = np.isfinite(distance) & (snp <= threshold)
            values = distance[valid]
            estimates[group] = float(np.mean(values))
            boot = np.asarray(
                bootstrap_threshold_means[group][threshold], dtype=float
            )
            rows.append(
                {
                    "threshold_snp": threshold,
                    "group": group,
                    "pairs": int(valid.sum()),
                    "observed_snp_pairs": int(np.sum(source[valid] == 1)),
                    "proxy_filled_pairs": int(np.sum(source[valid] == 0)),
                    "mean_geographic_km": estimates[group],
                    "median_geographic_km": float(np.median(values)),
                    "bootstrap_lower95_km": float(
                        np.nanpercentile(boot, 2.5)
                    ),
                    "bootstrap_upper95_km": float(
                        np.nanpercentile(boot, 97.5)
                    ),
                }
            )
        diff = np.asarray(
            bootstrap_threshold_means["ESL2.5"][threshold]
        ) - np.asarray(bootstrap_threshold_means["ESL2.4"][threshold])
        rows.append(
            {
                "threshold_snp": threshold,
                "group": "ESL2.5 minus ESL2.4",
                "pairs": np.nan,
                "observed_snp_pairs": np.nan,
                "proxy_filled_pairs": np.nan,
                "mean_geographic_km": (
                    estimates["ESL2.5"] - estimates["ESL2.4"]
                ),
                "median_geographic_km": np.nan,
                "bootstrap_lower95_km": float(np.nanpercentile(diff, 2.5)),
                "bootstrap_upper95_km": float(np.nanpercentile(diff, 97.5)),
            }
        )
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={
            "Creator": "hybrid_snp_geographic_spread.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        stem.with_suffix(".svg"),
        bbox_inches="tight",
        metadata={
            "Creator": "hybrid_snp_geographic_spread.py",
            "Date": None,
        },
    )
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        default="SNP/snp_time_network_validation/21_hybrid_snp_matrix.npz",
    )
    parser.add_argument(
        "--samples",
        default="SNP/snp_time_network_validation/13_complete_tree_proxy_samples.tsv",
    )
    parser.add_argument(
        "--city-distances",
        default="dating_sampling_bias/PairwiseChinaCities.distance",
    )
    parser.add_argument(
        "--nexus",
        default="dating_sampling_bias/treetime.annotated.nexus",
    )
    parser.add_argument("--dates", default="Analysis/Dates.tsv")
    parser.add_argument(
        "--outdir",
        default="SNP/snp_time_network_validation",
    )
    parser.add_argument(
        "--global-rttd-results",
        default="Analysis/temporal_signal/temporal_signal_results.tsv",
    )
    parser.add_argument("--core-genome-length", type=int, default=2_350_000)
    parser.add_argument("--max-snp", type=int, default=100)
    parser.add_argument("--display-max-snp", type=int, default=20)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--minimum-mrca-year", type=int)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cached = np.load(args.matrix)
    genomes = cached["genomes"].astype(str)
    snp_matrix = cached["snp"]
    observed_matrix = cached["observed_pair"]

    samples = pd.read_csv(args.samples, sep="\t", dtype={"group": str})
    samples = samples.set_index("genome").reindex(genomes)
    if samples[["city", "group"]].isna().any().any():
        raise ValueError("The SNP matrix and sample metadata are not aligned.")
    cities = samples.city.astype(str).to_numpy()
    groups = samples.group.astype(str).to_numpy()
    geo_matrix = geographic_matrix(
        cities,
        load_city_distances(Path(args.city_distances)),
    )
    mrca_year_matrix = None
    if args.minimum_mrca_year is not None:
        mrca_year_matrix = build_mrca_year_matrix(
            genomes,
            groups,
            Path(args.nexus),
            Path(args.dates),
            outdir / "27_mrca_year_matrix.npz",
        )
    snp_per_year, calibration_pairs = calibrate_shared_snp_rate(
        samples,
        groups,
        snp_matrix,
        Path(args.nexus),
        outdir / "21c_hybrid_treetime_matrix.npz",
        maximum_calibration_snp=args.max_snp,
    )
    rttd = pd.read_csv(args.global_rttd_results, sep="\t")
    global_rttd_slope = float(rttd.loc[rttd.group == "All", "slope"].iloc[0])
    global_rttd_snp_per_year = global_rttd_slope * args.core_genome_length
    pd.DataFrame(
        [
            {
                "model": "shared_SNP_per_year_through_origin",
                "calibration_pairs": calibration_pairs,
                "maximum_calibration_snp": args.max_snp,
                "snp_per_year": snp_per_year,
                "years_per_snp": 1.0 / snp_per_year,
                "global_rttd_substitutions_per_site_per_year": (
                    global_rttd_slope
                ),
                "core_genome_length": args.core_genome_length,
                "global_rttd_snp_per_year": global_rttd_snp_per_year,
                "rttd_vs_hybrid_relative_difference": (
                    global_rttd_snp_per_year - snp_per_year
                )
                / snp_per_year,
                "display_maximum_snp": args.display_max_snp,
            }
        ]
    ).to_csv(
        outdir / "27a_hybrid_snp_year_calibration.tsv",
        sep="\t",
        index=False,
    )

    plot_groups = ("All ESL", "ESL2.4", "ESL2.5")
    vectors = {
        group: vectors_for_group(
            group,
            groups,
            snp_matrix,
            geo_matrix,
            observed_matrix,
            mrca_year_matrix,
            args.minimum_mrca_year,
        )
        for group in plot_groups
    }

    centres = np.arange(0, args.display_max_snp + 1, dtype=float)
    edges = np.arange(-0.5, args.display_max_snp + 1.5, 1.0)

    full_means = {}
    full_counts = {}
    correlation_rows = []
    for group, (snp, distance, source) in vectors.items():
        means, counts = binned_means(snp, distance, edges)
        full_means[group] = smooth_bins(means)
        full_counts[group] = counts
        valid = np.isfinite(distance) & (snp <= args.max_snp)
        correlation_rows.append(
            {
                "group": group,
                "max_snp": args.max_snp,
                "pairs": int(valid.sum()),
                "observed_snp_pairs": int(np.sum(source[valid] == 1)),
                "proxy_filled_pairs": int(np.sum(source[valid] == 0)),
                "pearson_r": pearson_correlation(
                    snp[valid], distance[valid]
                ),
                "spearman_rho": spearman_correlation(
                    snp[valid], distance[valid]
                ),
            }
        )

    rng = np.random.default_rng(args.seed)
    bootstrap_bins = {
        group: np.full((args.replicates, len(centres)), np.nan)
        for group in plot_groups
    }
    bootstrap_threshold_means = {
        group: {threshold: [] for threshold in (40, 50, 100)}
        for group in plot_groups
    }
    lineage_indices = {
        lineage: np.flatnonzero(groups == lineage) for lineage in LINEAGES
    }
    for replicate in range(args.replicates):
        selected = {
            lineage: rng.choice(
                indices,
                size=max(2, int(np.ceil(len(indices) * 0.5))),
                replace=False,
            )
            for lineage, indices in lineage_indices.items()
        }
        for group in plot_groups:
            snp, distance, _ = vectors_for_group(
                group,
                groups,
                snp_matrix,
                geo_matrix,
                observed_matrix,
                mrca_year_matrix,
                args.minimum_mrca_year,
                selected=selected,
            )
            replicate_means, _ = binned_means(snp, distance, edges)
            bootstrap_bins[group][replicate] = smooth_bins(replicate_means)
            for threshold in (40, 50, 100):
                valid = np.isfinite(distance) & (snp <= threshold)
                bootstrap_threshold_means[group][threshold].append(
                    float(np.mean(distance[valid]))
                )

    binned_rows = []
    for group in plot_groups:
        lower = np.nanpercentile(bootstrap_bins[group], 2.5, axis=0)
        upper = np.nanpercentile(bootstrap_bins[group], 97.5, axis=0)
        observed_fraction = []
        snp, distance, source = vectors[group]
        valid = (
            np.isfinite(distance)
            & (snp >= edges[0])
            & (snp <= edges[-1])
        )
        bins = np.searchsorted(
            edges, snp[valid], side="right"
        ) - 1
        bins[snp[valid] == edges[-1]] = len(edges) - 2
        for bin_index in range(len(centres)):
            local = source[valid][bins == bin_index]
            observed_fraction.append(
                float(np.mean(local == 1)) if len(local) else np.nan
            )
            binned_rows.append(
                {
                    "group": group,
                    "snp_block": centres[bin_index],
                    "snp_bin_start": max(0.0, edges[bin_index]),
                    "snp_bin_end": edges[bin_index + 1],
                    "pairs": int(full_counts[group][bin_index]),
                    "observed_pair_fraction": observed_fraction[-1],
                    "mean_geographic_km": full_means[group][bin_index],
                    "subsample_lower95_km": lower[bin_index],
                    "subsample_upper95_km": upper[bin_index],
                }
            )

    pd.DataFrame(binned_rows).to_csv(
        outdir / "27_hybrid_snp_geographic_distance_binned.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame(correlation_rows).to_csv(
        outdir / "27b_hybrid_snp_geographic_distance_correlation.tsv",
        sep="\t",
        index=False,
    )
    summary = threshold_summary(vectors, bootstrap_threshold_means)
    summary.to_csv(
        outdir / "27c_hybrid_snp_geographic_distance_threshold_summary.tsv",
        sep="\t",
        index=False,
    )

    figure_table = pd.DataFrame(binned_rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for group in plot_groups:
        subset = figure_table[figure_table.group == group]
        ax.fill_between(
            subset.snp_block.to_numpy(float),
            subset.subsample_lower95_km.to_numpy(float),
            subset.subsample_upper95_km.to_numpy(float),
            color=COLOURS[group],
            alpha=0.20,
            linewidth=0,
        )
        ax.plot(
            subset.snp_block,
            subset.mean_geographic_km,
            color=COLOURS[group],
            linewidth=2.0,
            label=group,
            zorder=3 if group != "All ESL" else 2,
        )
    ax.set_xlabel("SNP distance")
    ax.set_ylabel("Mean geographic distance (km)")
    ax.set_xlim(0, args.display_max_snp)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(
        fig,
        outdir / "27_hybrid_snp_geographic_distance",
    )
    plt.close(fig)

    print(
        f"Wrote SNP-geographic spread analysis with {args.replicates} "
        f"subsampling replicates to {outdir}"
    )


if __name__ == "__main__":
    main()
