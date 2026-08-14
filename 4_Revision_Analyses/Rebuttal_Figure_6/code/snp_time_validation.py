#!/usr/bin/env python3
"""Compare pairwise SNP distances with TreeTime-derived divergence times."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sample_key(value: object) -> str:
    s = str(value).strip()
    if s.endswith("_1"):
        s = s[:-2]
    for suffix in (
        ".result.fastq.gz",
        ".result.fastq",
        ".result.fasta",
        ".result.fa",
        ".result.fna",
        ".fastq.gz",
        ".fastq",
        ".fasta",
        ".fa",
        ".fna",
    ):
        if s.endswith(suffix):
            return s[: -len(suffix)]
    return s


def correlation(x: np.ndarray, y: np.ndarray, rank: bool = False) -> tuple[float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if rank:
        x = pd.Series(x).rank(method="average").to_numpy()
        y = pd.Series(y).rank(method="average").to_numpy()
    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    z = abs(np.arctanh(np.clip(r, -0.999999999, 0.999999999))) * math.sqrt(len(x) - 3)
    p = math.erfc(z / math.sqrt(2.0))
    return r, p


def format_p(p: float) -> str:
    if not math.isfinite(p):
        return "NA"
    return "<0.001" if p < 0.001 else f"={p:.3f}"


def load_tree(nexus_path: Path):
    sys.path.insert(0, str(nexus_path.parent.resolve()))
    import genome_network_sensitivity as gns

    text = nexus_path.read_text(encoding="utf-8")
    taxlabels = gns.read_taxlabels(text)
    root = gns.NewickParser(gns.read_tree_text(text)).parse()
    gns.assign_tips(root, taxlabels)
    nodes = list(gns.iter_nodes(root))

    index = {id(node): i for i, node in enumerate(nodes)}
    parent = np.full(len(nodes), -1, dtype=np.int32)
    depth = np.zeros(len(nodes), dtype=float)
    generation = np.zeros(len(nodes), dtype=np.int32)
    tips: dict[str, int] = {}
    for node in nodes:
        i = index[id(node)]
        if node.parent is not None:
            parent[i] = index[id(node.parent)]
            depth[i] = depth[parent[i]] + float(node.length)
            generation[i] = generation[parent[i]] + 1
        if not node.children:
            tips[sample_key(next(iter(node.tips)))] = i

    levels = max(1, int(math.ceil(math.log2(len(nodes) + 1))))
    up = np.full((levels, len(nodes)), -1, dtype=np.int32)
    up[0] = parent
    for level in range(1, levels):
        prev = up[level - 1]
        valid = prev >= 0
        up[level, valid] = prev[prev[valid]]
    return tips, parent, depth, generation, up


def lca(a: int, b: int, parent: np.ndarray, generation: np.ndarray, up: np.ndarray) -> int:
    if generation[a] < generation[b]:
        a, b = b, a
    delta = int(generation[a] - generation[b])
    level = 0
    while delta:
        if delta & 1:
            a = int(up[level, a])
        delta >>= 1
        level += 1
    if a == b:
        return a
    for level in range(up.shape[0] - 1, -1, -1):
        aa = int(up[level, a])
        bb = int(up[level, b])
        if aa >= 0 and bb >= 0 and aa != bb:
            a, b = aa, bb
    return int(parent[a])


def add_tree_times(
    frame: pd.DataFrame,
    tips: dict[str, int],
    parent: np.ndarray,
    depth: np.ndarray,
    generation: np.ndarray,
    up: np.ndarray,
) -> pd.DataFrame:
    sample_names = sorted(set(frame.genome1) | set(frame.genome2))
    sample_nodes = []
    retained_names = []
    for name in sample_names:
        node = tips.get(sample_key(name))
        if node is not None:
            retained_names.append(name)
            sample_nodes.append(node)
    local = {name: i for i, name in enumerate(retained_names)}
    n = len(retained_names)
    patristic = np.zeros((n, n), dtype=np.float32)
    max_to_mrca = np.zeros((n, n), dtype=np.float32)
    for i in range(n - 1):
        a = sample_nodes[i]
        for j in range(i + 1, n):
            b = sample_nodes[j]
            ancestor = lca(a, b, parent, generation, up)
            da = depth[a] - depth[ancestor]
            db = depth[b] - depth[ancestor]
            patristic[i, j] = patristic[j, i] = da + db
            max_to_mrca[i, j] = max_to_mrca[j, i] = max(da, db)
        if (i + 1) % 200 == 0:
            print(f"Precomputed tree times for {i + 1}/{n} samples")

    idx1 = frame.genome1.map(local)
    idx2 = frame.genome2.map(local)
    keep = idx1.notna() & idx2.notna()
    out = frame.loc[keep].copy()
    a = idx1[keep].to_numpy(dtype=np.int32)
    b = idx2[keep].to_numpy(dtype=np.int32)
    out = out.rename(
        columns={
            "Num. SNPs": "snp_distance",
            "Compared sites": "compared_sites",
            "City1": "city1",
            "City2": "city2",
            "Year1": "year1",
            "Year2": "year2",
        }
    )
    out["patristic_time_years"] = patristic[a, b]
    out["max_time_to_mrca_years"] = max_to_mrca[a, b]
    out["sampling_interval_years"] = (
        pd.to_numeric(out.year1) - pd.to_numeric(out.year2)
    ).abs()
    return out


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    definitions = [
        ("All_pairs", frame),
        ("All_same_lineage", frame[frame.ESL1 == frame.ESL2]),
        ("ESL2.4", frame[(frame.ESL1 == 2.4) & (frame.ESL2 == 2.4)]),
        ("ESL2.5", frame[(frame.ESL1 == 2.5) & (frame.ESL2 == 2.5)]),
    ]
    rows = []
    for label, base in definitions:
        for threshold in (None, 100, 50, 40, 20, 10):
            data = base if threshold is None else base[base.snp_distance <= threshold]
            for time_col in ("patristic_time_years", "max_time_to_mrca_years"):
                x = data.snp_distance.to_numpy(float)
                y = data[time_col].to_numpy(float)
                pearson_r, pearson_p = correlation(x, y)
                spearman_rho, spearman_p = correlation(x, y, rank=True)
                rows.append(
                    {
                        "group": label,
                        "snp_threshold": "all" if threshold is None else threshold,
                        "time_definition": time_col,
                        "pairs": len(data),
                        "pearson_r": pearson_r,
                        "pearson_p": pearson_p,
                        "spearman_rho": spearman_rho,
                        "spearman_p": spearman_p,
                    }
                )
    return pd.DataFrame(rows)


def plot_correlations(
    frame: pd.DataFrame,
    output_base: Path,
    thresholds: tuple[int | None, ...] = (None, 40, 50),
    max_display_points: int = 12_000,
) -> None:
    panels = [
        ("All same-lineage pairs", frame[frame.ESL1 == frame.ESL2], "#777777"),
        ("ESL2.4", frame[(frame.ESL1 == 2.4) & (frame.ESL2 == 2.4)], "#EB6564"),
        ("ESL2.5", frame[(frame.ESL1 == 2.5) & (frame.ESL2 == 2.5)], "#F5BA1A"),
    ]
    fig, axes = plt.subplots(len(thresholds), 3, figsize=(10.8, 3.35 * len(thresholds)))
    axes = np.atleast_2d(axes)
    for row, threshold in enumerate(thresholds):
        for col, (title, base, colour) in enumerate(panels):
            ax = axes[row, col]
            data = base if threshold is None else base[base.snp_distance <= threshold]
            x = data.snp_distance.to_numpy(float)
            y = data.patristic_time_years.to_numpy(float)
            display_n = min(len(data), max_display_points)
            rng_seed = 20260727 + (0 if threshold is None else threshold * 10) + col
            rng = np.random.default_rng(rng_seed)
            display_idx = (
                np.arange(len(data))
                if display_n == len(data)
                else rng.choice(len(data), size=display_n, replace=False)
            )
            ax.scatter(
                x[display_idx],
                y[display_idx],
                s=4.5,
                c=colour,
                alpha=0.12,
                edgecolors="none",
            )
            x_max = float(np.nanmax(x)) if threshold is None else float(threshold)
            y_max = float(np.nanmax(y))
            ax.set_xlim(0, x_max * 1.01)
            ax.set_ylim(0, max(1.0, y_max * 1.02))
            if len(x) > 2:
                slope, intercept = np.polyfit(x, y, 1)
                xx = np.array([0.0, x_max])
                ax.plot(xx, intercept + slope * xx, color=colour, linewidth=1.8)
            pr, pp = correlation(x, y)
            sr, sp = correlation(x, y, rank=True)
            ax.text(
                0.04,
                0.96,
                f"Pearson r = {pr:.2f}, P {format_p(pp)}\n"
                f"Spearman rho = {sr:.2f}, P {format_p(sp)}\n"
                f"n = {len(data):,} pairs",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
            )
            prefix = (
                "All SNP distances"
                if threshold is None
                else f"Pairs within {threshold} SNPs"
            )
            ax.set_title(f"{title}\n{prefix}", fontsize=10, fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlabel("Pairwise SNP distance")
            if col == 0:
                ax.set_ylabel("TreeTime pairwise divergence time (years)")
    fig.text(
        0.5,
        0.002,
        "Points are randomly sampled for display; correlations use all eligible genome pairs.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(output_base.with_suffix(".pdf"))
    fig.savefig(output_base.with_suffix(".svg"))
    fig.savefig(output_base.with_suffix(".png"), dpi=600)
    plt.close(fig)


def write_network_checks(
    all_pairs: pd.DataFrame,
    original_path: Path,
    snp_edges_path: Path,
    output: Path,
    max_snp: int,
) -> None:
    rows = []
    for group in (2.4, 2.5):
        data = all_pairs[(all_pairs.ESL1 == group) & (all_pairs.ESL2 == group)].copy()
        between = data.City1 != data.City2
        close = data["Num. SNPs"] <= max_snp
        for label, mask in (("within_city", ~between), ("between_city", between)):
            subset = data[mask]
            n_close = int(close[mask].sum())
            rows.append(
                {
                    "metric": "close_pair_rate",
                    "group": f"ESL{group}",
                    "scope": label,
                    "numerator": n_close,
                    "denominator": len(subset),
                    "value": n_close / len(subset) if len(subset) else float("nan"),
                }
            )

    original = pd.read_csv(original_path, sep="\t")
    snp_edges = pd.read_csv(snp_edges_path, sep="\t")
    snp_edges = snp_edges[
        (snp_edges["threshold"] == max_snp)
        & (snp_edges["group"] == "All")
        & (snp_edges["city1"] != snp_edges["city2"])
    ]
    cities = set(snp_edges.city1) | set(snp_edges.city2)

    def edge(a: object, b: object) -> tuple[str, str]:
        return tuple(sorted((str(a), str(b))))

    original_edges = {
        edge(a, b)
        for a, b in zip(original.Source, original.Target)
        if a != b and a in cities and b in cities
    }
    snp_edge_set = {edge(a, b) for a, b in zip(snp_edges.city1, snp_edges.city2)}
    recovered = len(original_edges & snp_edge_set)
    rows.append(
        {
            "metric": "original_edge_recovery",
            "group": "All",
            "scope": "cities_covered_by_SNP_data",
            "numerator": recovered,
            "denominator": len(original_edges),
            "value": recovered / len(original_edges) if original_edges else float("nan"),
        }
    )
    pd.DataFrame(rows).to_csv(output, sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="SNP/HC1200_ESL235_city_date.snpairs.tsv")
    parser.add_argument("--nexus", default="dating_sampling_bias/treetime.annotated.nexus")
    parser.add_argument("--output-dir", default="SNP/snp_time_network_validation")
    parser.add_argument(
        "--original-network",
        default="dating_sampling_bias/China_Transmission_Normalization.tsv",
    )
    parser.add_argument(
        "--snp-city-edges",
        default="SNP/snp_time_network_validation/snp_network_le100/snp_city_edges.tsv",
    )
    parser.add_argument("--max-snp", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    all_pairs = pd.read_csv(args.pairs, sep="\t")
    tips, parent, depth, generation, up = load_tree(Path(args.nexus))
    mapped = add_tree_times(all_pairs, tips, parent, depth, generation, up)
    mapped.to_csv(
        outdir / "02_all_snp_vs_treetime_pairs.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    mapped[mapped.snp_distance <= 40].to_csv(
        outdir / "02b_snp_le40_vs_treetime_pairs.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    mapped[mapped.snp_distance <= 50].to_csv(
        outdir / "02c_snp_le50_vs_treetime_pairs.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    summary = summarise(mapped)
    summary.to_csv(outdir / "03_snp_treetime_correlation_summary.tsv", sep="\t", index=False)
    plot_correlations(mapped, outdir / "04_snp_vs_treetime_all_40_50")
    plot_correlations(
        mapped,
        outdir / "04a_snp_vs_treetime_all_distances",
        thresholds=(None,),
    )
    plot_correlations(
        mapped,
        outdir / "04b_snp_vs_treetime_le40",
        thresholds=(40,),
    )
    plot_correlations(
        mapped,
        outdir / "04c_snp_vs_treetime_le50",
        thresholds=(50,),
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
