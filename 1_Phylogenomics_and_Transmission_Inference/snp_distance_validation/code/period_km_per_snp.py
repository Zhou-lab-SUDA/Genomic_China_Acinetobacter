#!/usr/bin/env python3
"""Compare km/SNP distributions before and during the COVID-19 period.

Pairs are taken directly from the fixed hybrid SNP matrix. The earlier
isolation year of the two genomes defines the pair's starting year. Annual
means are used as the plotted observations so densely sampled years do not
dominate the period comparison.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["svg.hashsalt"] = "nm_ab_period_km_per_snp_v2"

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from hybrid_snp_geographic_spread import geographic_matrix, load_city_distances


COLOURS = {"ESL2.4": "#C96F6F", "ESL2.5": "#EDA83B"}
PERIODS = {
    "2010-2019": (2010, 2019),
    "2020-2023": (2020, 2023),
}
GCF_PATTERN = re.compile(r"^(GC[AF]_[0-9]+)")
SUFFIX_PATTERN = re.compile(
    r"\.(?:result\.(?:fastq|fasta|fa|fna)(?:\.gz)?|"
    r"fastq(?:\.gz)?|fasta|fa|fna)$"
)


def sample_alias(value: object) -> str:
    """Collapse assembly-version and local sequence suffix aliases."""
    text = str(value).strip()
    accession = GCF_PATTERN.match(text)
    if accession:
        return accession.group(1)
    return SUFFIX_PATTERN.sub("", text)


def unique_lookup(
    table: pd.DataFrame,
    key_column: str,
    value_column: str,
) -> dict[str, str]:
    values: dict[str, set[str]] = {}
    for key, value in zip(table[key_column], table[value_column]):
        if pd.isna(key) or pd.isna(value):
            continue
        clean = str(value).strip()
        if not clean or clean.lower() in {"nan", "none", "na", "tbd"}:
            continue
        values.setdefault(sample_alias(key), set()).add(clean)
    return {
        key: next(iter(options))
        for key, options in values.items()
        if len(options) == 1
    }


def rank_values(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(float)


def exact_mann_whitney_p(first: np.ndarray, second: np.ndarray) -> float:
    """Two-sided exact permutation P value for small annual samples."""
    values = np.concatenate([first, second])
    ranks = rank_values(values)
    n_first = len(first)
    observed = float(np.sum(ranks[:n_first]))
    centre = n_first * (len(values) + 1) / 2.0
    deviation = abs(observed - centre)
    extreme = 0
    total = 0
    for selected in combinations(range(len(values)), n_first):
        rank_sum = float(np.sum(ranks[list(selected)]))
        if abs(rank_sum - centre) >= deviation - 1e-12:
            extreme += 1
        total += 1
    return extreme / total


def p_to_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def load_sample_table(
    genomes: np.ndarray,
    samples_path: Path,
    dates_path: Path,
    metadata_path: Path,
    metadata_sheet: str,
) -> pd.DataFrame:
    base = pd.read_csv(samples_path, sep="\t", dtype=str)
    base["_alias"] = base["genome"].map(sample_alias)
    base = base.drop_duplicates("_alias").set_index("_alias")

    dates = pd.read_csv(dates_path, sep="\t", dtype=str)
    date_lookup = unique_lookup(dates, "genome", "Year")

    metadata = pd.read_excel(
        metadata_path,
        sheet_name=metadata_sheet,
        dtype=str,
    )
    group_lookup = unique_lookup(metadata, "genome", "ESL cluster")
    metadata_year_lookup = unique_lookup(metadata, "genome", "Year")

    rows = []
    for genome in genomes:
        alias = sample_alias(genome)
        if alias not in base.index:
            raise ValueError(f"Missing city metadata for {genome}")
        city = str(base.at[alias, "city"])
        group = group_lookup.get(alias, str(base.at[alias, "group"]))
        year_text = date_lookup.get(alias, metadata_year_lookup.get(alias))
        year = pd.to_numeric(year_text, errors="coerce")
        rows.append(
            {
                "genome": str(genome),
                "alias": alias,
                "city": city,
                "group": group,
                "year": year,
                "group_source": (
                    "supplementary_metadata"
                    if alias in group_lookup
                    else "existing_sample_table"
                ),
                "year_source": (
                    "dates_tsv"
                    if alias in date_lookup
                    else "supplementary_metadata"
                    if alias in metadata_year_lookup
                    else "missing"
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["year"] = pd.to_numeric(result["year"], errors="coerce")
    return result


def build_pair_table(
    matrix_path: Path,
    samples_path: Path,
    dates_path: Path,
    metadata_path: Path,
    metadata_sheet: str,
    city_distances_path: Path,
    minimum_snp: float,
    maximum_snp: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix_data = np.load(matrix_path)
    genomes = matrix_data["genomes"].astype(str)
    snp_matrix = matrix_data["snp"].astype(float)
    observed_matrix = matrix_data["observed_pair"].astype(bool)

    samples = load_sample_table(
        genomes,
        samples_path,
        dates_path,
        metadata_path,
        metadata_sheet,
    )
    cities = samples["city"].astype(str).to_numpy()
    years = samples["year"].to_numpy(float)
    groups = samples["group"].astype(str).to_numpy()
    geo_matrix = geographic_matrix(
        cities,
        load_city_distances(city_distances_path),
    ).astype(float)

    frames = []
    for group in ("2.4", "2.5"):
        indices = np.flatnonzero((groups == group) & np.isfinite(years))
        upper = np.triu_indices(len(indices), 1)
        first = indices[upper[0]]
        second = indices[upper[1]]
        snp = snp_matrix[first, second]
        distance = geo_matrix[first, second]
        start_year = np.minimum(years[first], years[second])
        valid = (
            np.isfinite(snp)
            & np.isfinite(distance)
            & (snp >= minimum_snp)
            & (snp <= maximum_snp)
            & (start_year >= 2010)
            & (start_year <= 2023)
        )
        first = first[valid]
        second = second[valid]
        snp = snp[valid]
        distance = distance[valid]
        start_year = start_year[valid].astype(int)
        period = np.where(start_year <= 2019, "2010-2019", "2020-2023")
        frames.append(
            pd.DataFrame(
                {
                    "lineage": f"ESL{group}",
                    "genome1": genomes[first],
                    "genome2": genomes[second],
                    "city1": cities[first],
                    "city2": cities[second],
                    "year1": years[first].astype(int),
                    "year2": years[second].astype(int),
                    "start_year": start_year,
                    "period": period,
                    "snp_distance": snp,
                    "geographic_km": distance,
                    "km_per_snp": distance / snp,
                    "observed_snp": observed_matrix[first, second],
                }
            )
        )
    return pd.concat(frames, ignore_index=True), samples


def annual_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    return (
        pairs.groupby(["lineage", "start_year", "period"], as_index=False)
        .agg(
            km_per_snp=("km_per_snp", "mean"),
            median_km_per_snp=("km_per_snp", "median"),
            pair_count=("km_per_snp", "size"),
            observed_snp_pair_fraction=("observed_snp", "mean"),
        )
        .rename(columns={"start_year": "year"})
    )


def period_summary(
    annual: pd.DataFrame,
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for lineage in ("ESL2.4", "ESL2.5"):
        for period in PERIODS:
            annual_values = annual.loc[
                (annual.lineage == lineage) & (annual.period == period),
                "km_per_snp",
            ].to_numpy(float)
            pair_values = pairs.loc[
                (pairs.lineage == lineage) & (pairs.period == period),
                "km_per_snp",
            ].to_numpy(float)
            sem = float(
                np.std(annual_values, ddof=1) / np.sqrt(len(annual_values))
            )
            rows.append(
                {
                    "lineage": lineage,
                    "period": period,
                    "annual_points": len(annual_values),
                    "pairs": len(pair_values),
                    "mean_of_annual_means_km_per_snp": float(
                        np.mean(annual_values)
                    ),
                    "median_of_annual_means_km_per_snp": float(
                        np.median(annual_values)
                    ),
                    "annual_standard_deviation": float(
                        np.std(annual_values, ddof=1)
                    ),
                    "annual_sem": sem,
                    "lower95_mean": max(
                        0.0,
                        float(np.mean(annual_values) - 1.96 * sem),
                    ),
                    "upper95_mean": float(
                        np.mean(annual_values) + 1.96 * sem
                    ),
                    "pooled_pair_mean_km_per_snp": float(
                        np.mean(pair_values)
                    ),
                    "pooled_pair_median_km_per_snp": float(
                        np.median(pair_values)
                    ),
                    "observed_snp_pair_fraction": float(
                        pairs.loc[
                            (pairs.lineage == lineage)
                            & (pairs.period == period),
                            "observed_snp",
                        ].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def comparison_table(annual: pd.DataFrame) -> pd.DataFrame:
    values = {
        (lineage, period): annual.loc[
            (annual.lineage == lineage) & (annual.period == period),
            "km_per_snp",
        ].to_numpy(float)
        for lineage in ("ESL2.4", "ESL2.5")
        for period in PERIODS
    }
    comparisons = [
        (
            "ESL2.4: 2010-2019 vs 2020-2023",
            ("ESL2.4", "2010-2019"),
            ("ESL2.4", "2020-2023"),
        ),
        (
            "ESL2.5: 2010-2019 vs 2020-2023",
            ("ESL2.5", "2010-2019"),
            ("ESL2.5", "2020-2023"),
        ),
        (
            "2010-2019: ESL2.4 vs ESL2.5",
            ("ESL2.4", "2010-2019"),
            ("ESL2.5", "2010-2019"),
        ),
        (
            "2020-2023: ESL2.4 vs ESL2.5",
            ("ESL2.4", "2020-2023"),
            ("ESL2.5", "2020-2023"),
        ),
    ]
    rows = []
    for label, first, second in comparisons:
        p_value = exact_mann_whitney_p(values[first], values[second])
        rows.append(
            {
                "comparison": label,
                "first_n_years": len(values[first]),
                "second_n_years": len(values[second]),
                "p_value": p_value,
                "significance": p_to_stars(p_value),
            }
        )
    return pd.DataFrame(rows)


def draw_bracket(
    ax: plt.Axes,
    x: float,
    y0: float,
    y1: float,
    text: str,
    width: float,
) -> None:
    ax.plot(
        [x, x + width, x + width, x],
        [y0, y0, y1, y1],
        color="black",
        linewidth=0.8,
        clip_on=False,
    )
    ax.text(
        x + width * 1.5,
        (y0 + y1) / 2,
        text,
        va="center",
        fontsize=9,
    )


def plot_distribution(
    annual: pd.DataFrame,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    output_stem: Path,
) -> None:
    positions = {
        ("ESL2.4", "2020-2023"): 3.0,
        ("ESL2.4", "2010-2019"): 2.0,
        ("ESL2.5", "2020-2023"): 1.0,
        ("ESL2.5", "2010-2019"): 0.0,
    }
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for (lineage, period), y_position in positions.items():
        values = annual.loc[
            (annual.lineage == lineage) & (annual.period == period),
            "km_per_snp",
        ].to_numpy(float)
        jitter = (
            np.linspace(-0.08, 0.08, len(values))
            if len(values) > 1
            else np.asarray([0.0])
        )
        ax.scatter(
            values,
            np.full(len(values), y_position) + jitter,
            s=18,
            color=COLOURS[lineage],
            alpha=0.70,
            edgecolor="none",
        )
        row = summary[
            (summary.lineage == lineage) & (summary.period == period)
        ].iloc[0]
        mean = row.mean_of_annual_means_km_per_snp
        ax.plot(
            [row.lower95_mean, row.upper95_mean],
            [y_position, y_position],
            color="black",
            linewidth=1.4,
        )
        ax.scatter(
            mean,
            y_position,
            s=78,
            color=COLOURS[lineage],
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        ax.text(
            row.upper95_mean + summary.mean_of_annual_means_km_per_snp.max() * 0.04,
            y_position,
            f"{mean:.1f} km/SNP",
            va="center",
            fontsize=10,
        )

    ax.set_yticks([3.0, 2.0, 1.0, 0.0])
    ax.set_yticklabels(
        ["2020-2023", "2010-2019", "2020-2023", "2010-2019"]
    )
    ax.set_xlabel("SNP-normalised spread (km/SNP)", fontsize=12)
    ax.set_ylim(-0.5, 3.5)
    maximum = max(
        float(summary.upper95_mean.max()),
        float(annual.km_per_snp.max()),
    )
    ax.set_xlim(0, maximum * 1.65)
    ax.text(
        -0.17,
        0.76,
        "ESL2.4",
        transform=ax.transAxes,
        fontsize=12,
        ha="right",
        va="center",
    )
    ax.text(
        -0.17,
        0.24,
        "ESL2.5",
        transform=ax.transAxes,
        fontsize=12,
        ha="right",
        va="center",
    )
    ax.axhspan(-0.5, 1.5, color="#EEEEEE", zorder=-1)

    p_lookup = dict(
        zip(comparisons.comparison, comparisons.significance)
    )
    bracket_start = maximum * 1.10
    width = maximum * 0.018
    draw_bracket(
        ax,
        bracket_start,
        2.0,
        3.0,
        p_lookup["ESL2.4: 2010-2019 vs 2020-2023"],
        width,
    )
    draw_bracket(
        ax,
        bracket_start + maximum * 0.10,
        0.0,
        1.0,
        p_lookup["ESL2.5: 2010-2019 vs 2020-2023"],
        width,
    )
    draw_bracket(
        ax,
        bracket_start + maximum * 0.20,
        1.0,
        3.0,
        p_lookup["2020-2023: ESL2.4 vs ESL2.5"],
        width,
    )
    draw_bracket(
        ax,
        bracket_start + maximum * 0.30,
        0.0,
        2.0,
        p_lookup["2010-2019: ESL2.4 vs ESL2.5"],
        width,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={
            "Creator": "period_km_per_snp.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        output_stem.with_suffix(".svg"),
        bbox_inches="tight",
        metadata={"Creator": "period_km_per_snp.py", "Date": None},
    )
    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)


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
    parser.add_argument("--dates", default="Analysis/Dates.tsv")
    parser.add_argument(
        "--metadata",
        default=str(Path(__file__).resolve().parents[2] / "sample_metadata.xlsx"),
    )
    parser.add_argument("--metadata-sheet", default="S1 Metadata")
    parser.add_argument(
        "--city-distances",
        default="dating_sampling_bias/PairwiseChinaCities.distance",
    )
    parser.add_argument(
        "--outdir",
        default="SNP/snp_geographic_spread_fixed_v1/period_km_per_snp",
    )
    parser.add_argument("--minimum-snp", type=float, default=1.0)
    parser.add_argument("--maximum-snp", type=float, default=20.0)
    args = parser.parse_args()

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs, samples = build_pair_table(
        Path(args.matrix),
        Path(args.samples),
        Path(args.dates),
        Path(args.metadata),
        args.metadata_sheet,
        Path(args.city_distances),
        args.minimum_snp,
        args.maximum_snp,
    )
    pairs.to_csv(
        output_dir / "period_km_per_snp_pairs.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    samples.to_csv(
        output_dir / "period_km_per_snp_sample_metadata.tsv",
        sep="\t",
        index=False,
    )

    annual = annual_summary(pairs)
    annual.to_csv(
        output_dir / "period_km_per_snp_annual.tsv",
        sep="\t",
        index=False,
    )
    summary = period_summary(annual, pairs)
    summary.to_csv(
        output_dir / "period_km_per_snp_summary.tsv",
        sep="\t",
        index=False,
    )
    comparisons = comparison_table(annual)
    comparisons.to_csv(
        output_dir / "period_km_per_snp_comparisons.tsv",
        sep="\t",
        index=False,
    )
    plot_distribution(
        annual,
        summary,
        comparisons,
        output_dir / "period_km_per_snp_distribution",
    )
    print(f"Wrote {len(pairs):,} pair observations to {output_dir}")
    print(summary.to_string(index=False))
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
