#!/usr/bin/env python3
"""Reproduce the Reviewer 5 specimen-ascertainment analyses.

Primary cohort
--------------
* Chinese genomes from the 49-city Figure 3 cohort.
* ESL2.4 and ESL2.5 only, because the sampling-depth variable used for the
  Figure 3 sensitivity analysis is the combined number of these two lineages.
* Invasive/non-invasive comparisons exclude records labelled Uncertain or
  Unknown. These records are retained in coverage summaries.
* Sampling-depth groups are <20, 20--49 and >=50 ESL2.4/ESL2.5 genomes/city.

The script never modifies the source workbook.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.stats.contingency_tables import StratifiedTable


KNOWN = {"Invasive", "Non-invasive"}
DEPTH_ORDER = ["<20", "20-49", ">=50"]
BLOODSTREAM_LABELS = {
    "blood",
    "venous blood",
    "whole blood",
    "venous ductal blood",
    "blood culture",
    "bloodstream",
}


def proportion_table(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    out = (
        frame.groupby(by, observed=True)
        .agg(
            n_binary=("genome", "size"),
            n_invasive=("is_invasive", "sum"),
        )
        .reset_index()
    )
    out["n_non_invasive"] = out["n_binary"] - out["n_invasive"]
    out["invasive_percent"] = 100 * out["n_invasive"] / out["n_binary"]
    return out


def fisher_record(
    name: str,
    a: pd.DataFrame,
    b: pd.DataFrame,
    a_label: str,
    b_label: str,
) -> dict:
    table = np.array(
        [
            [int(a["is_invasive"].sum()), int((~a["is_invasive"]).sum())],
            [int(b["is_invasive"].sum()), int((~b["is_invasive"]).sum())],
        ]
    )
    result = fisher_exact(table, alternative="two-sided")
    return {
        "analysis": name,
        "method": "two-sided Fisher exact test at isolate level",
        "group_a": a_label,
        "group_b": b_label,
        "a_invasive": table[0, 0],
        "a_non_invasive": table[0, 1],
        "b_invasive": table[1, 0],
        "b_non_invasive": table[1, 1],
        "effect": float(result.statistic),
        "effect_label": "odds ratio, group A / group B",
        "ci_low": np.nan,
        "ci_high": np.nan,
        "p_value": float(result.pvalue),
        "n_cities": int(pd.concat([a, b])["City"].nunique()),
        "notes": "Descriptive isolate-level test; genomes within a city are not independent.",
    }


def city_equal_record(
    name: str,
    city_table: pd.DataFrame,
    a_groups: list[str],
    b_groups: list[str],
    a_label: str,
    b_label: str,
) -> dict:
    a = city_table.loc[city_table["depth_group"].isin(a_groups), "invasive_fraction"]
    b = city_table.loc[city_table["depth_group"].isin(b_groups), "invasive_fraction"]
    result = mannwhitneyu(a, b, alternative="two-sided")
    return {
        "analysis": name,
        "method": "two-sided Mann-Whitney U test on city invasive fractions",
        "group_a": a_label,
        "group_b": b_label,
        "a_invasive": np.nan,
        "a_non_invasive": np.nan,
        "b_invasive": np.nan,
        "b_non_invasive": np.nan,
        "effect": float(result.statistic),
        "effect_label": "Mann-Whitney U",
        "ci_low": np.nan,
        "ci_high": np.nan,
        "p_value": float(result.pvalue),
        "n_cities": int(len(a) + len(b)),
        "notes": f"City-equal comparison; n(A)={len(a)}, n(B)={len(b)}.",
    }


def gee_record(
    name: str,
    frame: pd.DataFrame,
    predictor: str,
    group_a: str,
    group_b: str,
    notes: str,
) -> dict:
    model = GEE.from_formula(
        f"is_invasive_int ~ {predictor}",
        groups="City",
        data=frame,
        family=Binomial(),
        cov_struct=Exchangeable(),
    ).fit()
    beta = float(model.params[predictor])
    low, high = map(float, model.conf_int().loc[predictor])
    return {
        "analysis": name,
        "method": "logistic GEE with exchangeable within-city correlation",
        "group_a": group_a,
        "group_b": group_b,
        "a_invasive": np.nan,
        "a_non_invasive": np.nan,
        "b_invasive": np.nan,
        "b_non_invasive": np.nan,
        "effect": float(np.exp(beta)),
        "effect_label": "odds ratio, group A / group B",
        "ci_low": float(np.exp(low)),
        "ci_high": float(np.exp(high)),
        "p_value": float(model.pvalues[predictor]),
        "n_cities": int(frame["City"].nunique()),
        "notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--city-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_excel(args.workbook, sheet_name="S1 Metadata")
    cities = pd.read_csv(args.city_table, sep="\t")
    cities["depth_group"] = pd.cut(
        cities["n_major_lineages"],
        bins=[-np.inf, 19, 49, np.inf],
        labels=DEPTH_ORDER,
    )

    data = meta.loc[
        meta["Country"].eq("China")
        & meta["City"].isin(cities["city"])
        & meta["ESL cluster"].astype(str).isin(["2.4", "2.5"])
    ].copy()
    data["lineage"] = "ESL" + data["ESL cluster"].astype(str)
    data = data.merge(
        cities[["city", "degree", "n_major_lineages", "depth_group"]],
        left_on="City",
        right_on="city",
        validate="many_to_one",
    )
    data["binary_status_known"] = data["Invasive sample"].isin(KNOWN)
    data["is_invasive"] = data["Invasive sample"].eq("Invasive")
    data["is_invasive_int"] = data["is_invasive"].astype(int)
    data["source_normalized"] = data["Source"].astype(str).str.strip().str.lower()
    data["is_bloodstream"] = data["source_normalized"].isin(BLOODSTREAM_LABELS)

    # Coverage table includes all four manual classifications.
    coverage = (
        data.groupby(["depth_group", "lineage", "Invasive sample"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    coverage.to_csv(args.output_dir / "01_classification_coverage.tsv", sep="\t", index=False)

    binary = data.loc[data["binary_status_known"]].copy()
    group_summary = proportion_table(binary, ["depth_group"])
    lineage_summary = proportion_table(binary, ["depth_group", "lineage"])
    group_summary.to_csv(args.output_dir / "02_invasive_by_sampling_group.tsv", sep="\t", index=False)
    lineage_summary.to_csv(args.output_dir / "03_invasive_by_sampling_group_and_lineage.tsv", sep="\t", index=False)

    city_summary = (
        data.groupby(["City", "depth_group", "n_major_lineages"], observed=True)
        .agg(
            n_total_major=("genome", "size"),
            n_invasive=("is_invasive", "sum"),
            n_non_invasive=("Invasive sample", lambda x: int((x == "Non-invasive").sum())),
            n_uncertain=("Invasive sample", lambda x: int((x == "Uncertain").sum())),
            n_unknown=("Invasive sample", lambda x: int((x == "Unknown").sum())),
        )
        .reset_index()
    )
    city_summary["n_binary"] = city_summary["n_invasive"] + city_summary["n_non_invasive"]
    city_summary["invasive_fraction"] = city_summary["n_invasive"] / city_summary["n_binary"].replace(0, np.nan)
    city_summary.to_csv(args.output_dir / "04_city_level_invasive_summary.tsv", sep="\t", index=False)

    tests: list[dict] = []
    low = binary.loc[binary["depth_group"].eq("<20")]
    mid = binary.loc[binary["depth_group"].eq("20-49")]
    high = binary.loc[binary["depth_group"].eq(">=50")]
    ge20 = binary.loc[binary["depth_group"].isin(["20-49", ">=50"])]
    tests.extend(
        [
            fisher_record("sampling depth: <20 vs 20-49", low, mid, "<20", "20-49"),
            fisher_record("sampling depth: <20 vs >=50", low, high, "<20", ">=50"),
            fisher_record("sampling depth: <20 vs >=20", low, ge20, "<20", ">=20"),
            fisher_record("sampling depth: 20-49 vs >=50", mid, high, "20-49", ">=50"),
        ]
    )

    usable_city = city_summary.dropna(subset=["invasive_fraction"]).copy()
    tests.extend(
        [
            city_equal_record("city-equal: <20 vs 20-49", usable_city, ["<20"], ["20-49"], "<20", "20-49"),
            city_equal_record("city-equal: <20 vs >=50", usable_city, ["<20"], [">=50"], "<20", ">=50"),
            city_equal_record("city-equal: <20 vs >=20", usable_city, ["<20"], ["20-49", ">=50"], "<20", ">=20"),
            city_equal_record("city-equal: 20-49 vs >=50", usable_city, ["20-49"], [">=50"], "20-49", ">=50"),
        ]
    )

    # Cluster-aware binary comparisons. Coding makes exp(beta) the odds ratio
    # for the first named group versus the reference group.
    low_ge20 = binary.loc[binary["depth_group"].isin(["<20", "20-49", ">=50"])].copy()
    low_ge20["is_low"] = low_ge20["depth_group"].eq("<20").astype(int)
    tests.append(
        gee_record(
            "sampling depth: <20 vs >=20, city-clustered",
            low_ge20,
            "is_low",
            "<20",
            ">=20",
            "Primary cluster-aware check of preferential invasive-specimen ascertainment in sparsely sampled cities.",
        )
    )

    # Lineage comparison among highly sampled cities.
    high = high.copy()
    high["is_esl24"] = high["lineage"].eq("ESL2.4").astype(int)
    tests.append(fisher_record(
        "lineage within >=50 cities",
        high.loc[high["lineage"].eq("ESL2.4")],
        high.loc[high["lineage"].eq("ESL2.5")],
        "ESL2.4",
        "ESL2.5",
    ))
    tests.append(
        gee_record(
            "lineage within >=50 cities, city-clustered",
            high,
            "is_esl24",
            "ESL2.4",
            "ESL2.5",
            "Lineage contrast with within-city correlation handled by GEE.",
        )
    )

    strata = []
    strata_cities = []
    for city_name, block in high.groupby("City"):
        table = (
            pd.crosstab(block["lineage"], block["is_invasive"])
            .reindex(index=["ESL2.4", "ESL2.5"], columns=[True, False], fill_value=0)
        )
        if (table.sum(axis=1) > 0).all():
            strata.append(table.to_numpy())
            strata_cities.append(city_name)
    if strata:
        stratified = StratifiedTable(np.stack(strata, axis=2), shift_zeros=False)
        low_ci, high_ci = stratified.oddsratio_pooled_confint()
        tests.append(
            {
                "analysis": "lineage within >=50 cities, Cochran-Mantel-Haenszel",
                "method": "Cochran-Mantel-Haenszel common odds ratio stratified by city",
                "group_a": "ESL2.4",
                "group_b": "ESL2.5",
                "a_invasive": np.nan,
                "a_non_invasive": np.nan,
                "b_invasive": np.nan,
                "b_non_invasive": np.nan,
                "effect": float(stratified.oddsratio_pooled),
                "effect_label": "common odds ratio, ESL2.4 / ESL2.5",
                "ci_low": float(low_ci),
                "ci_high": float(high_ci),
                "p_value": float(stratified.test_null_odds().pvalue),
                "n_cities": len(strata_cities),
                "notes": "Cities with known binary specimen status in both lineages: " + ", ".join(strata_cities),
            }
        )

    rho = spearmanr(usable_city["n_major_lineages"], usable_city["invasive_fraction"])
    tests.append(
        {
            "analysis": "city sampling depth versus city invasive fraction",
            "method": "two-sided Spearman rank correlation",
            "group_a": "n_major_lineages",
            "group_b": "invasive_fraction",
            "a_invasive": np.nan,
            "a_non_invasive": np.nan,
            "b_invasive": np.nan,
            "b_non_invasive": np.nan,
            "effect": float(rho.statistic),
            "effect_label": "Spearman rho",
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": float(rho.pvalue),
            "n_cities": int(len(usable_city)),
            "notes": "Each city contributes equally; cities without any binary-classified specimen are excluded.",
        }
    )

    pd.DataFrame(tests).to_csv(args.output_dir / "05_statistical_tests.tsv", sep="\t", index=False)

    bloodstream = (
        data.groupby(["depth_group", "lineage"], observed=True)
        .agg(
            n_all_major=("genome", "size"),
            n_source_known=("source_normalized", lambda x: int((x != "unknown").sum())),
            n_bloodstream=("is_bloodstream", "sum"),
        )
        .reset_index()
    )
    bloodstream["bloodstream_percent_all"] = 100 * bloodstream["n_bloodstream"] / bloodstream["n_all_major"]
    bloodstream["bloodstream_percent_source_known"] = 100 * bloodstream["n_bloodstream"] / bloodstream["n_source_known"].replace(0, np.nan)
    bloodstream.to_csv(args.output_dir / "06_bloodstream_sensitivity_summary.tsv", sep="\t", index=False)

    # Sensitivity to two choices that are easy to conflate in prose:
    # (i) all genomes versus the two focal lineages in the specimen numerator;
    # (ii) focal-lineage versus all-genome counts for defining city sampling depth.
    figure3_all = meta.loc[
        meta["Country"].eq("China") & meta["City"].isin(cities["city"])
    ].copy()
    figure3_all["lineage"] = "ESL" + figure3_all["ESL cluster"].astype(str)
    all_counts = figure3_all.groupby("City").size().rename("n_all_genomes")
    figure3_all = figure3_all.join(all_counts, on="City").merge(
        cities[["city", "n_major_lineages"]], left_on="City", right_on="city", validate="many_to_one"
    )
    definition_rows = []
    definition_tests = []
    for specimen_scope, scope_mask in {
        "all_lineages": pd.Series(True, index=figure3_all.index),
        "ESL2.4_and_ESL2.5": figure3_all["ESL cluster"].astype(str).isin(["2.4", "2.5"]),
    }.items():
        for depth_basis, depth_values in {
            "focal_lineage_count": figure3_all["n_major_lineages"],
            "all_genome_count": figure3_all["n_all_genomes"],
        }.items():
            z = figure3_all.loc[
                scope_mask & figure3_all["Invasive sample"].isin(KNOWN)
            ].copy()
            z["is_invasive"] = z["Invasive sample"].eq("Invasive")
            z["depth_group"] = pd.cut(
                depth_values.loc[z.index], [-np.inf, 19, 49, np.inf], labels=DEPTH_ORDER
            )
            summary = proportion_table(z, ["depth_group"])
            summary.insert(0, "depth_basis", depth_basis)
            summary.insert(0, "specimen_scope", specimen_scope)
            definition_rows.append(summary)
            a = z.loc[z["depth_group"].eq("<20")]
            b = z.loc[z["depth_group"].isin(["20-49", ">=50"])]
            test = fisher_record(
                f"{specimen_scope}; {depth_basis}; <20 vs >=20", a, b, "<20", ">=20"
            )
            test["specimen_scope"] = specimen_scope
            test["depth_basis"] = depth_basis
            definition_tests.append(test)
    pd.concat(definition_rows, ignore_index=True).to_csv(
        args.output_dir / "07_definition_sensitivity_summary.tsv", sep="\t", index=False
    )
    pd.DataFrame(definition_tests).to_csv(
        args.output_dir / "08_definition_sensitivity_tests.tsv", sep="\t", index=False
    )

    # Directly address the reviewer's request to restrict the Figure 3 pattern
    # by specimen class. Prevalence here is the focal-lineage fraction
    # ESL2.5/(ESL2.4+ESL2.5), so it is labelled as a sensitivity outcome and is
    # not silently substituted for the original Figure 3 denominator.
    restricted_rows = []
    restrictions = {
        "invasive": data["Invasive sample"].eq("Invasive"),
        "non_invasive": data["Invasive sample"].eq("Non-invasive"),
        "bloodstream": data["is_bloodstream"],
        "binary_classified": data["binary_status_known"],
    }
    for restriction, mask in restrictions.items():
        z = (
            data.loc[mask]
            .pivot_table(index="City", columns="lineage", values="genome", aggfunc="count", fill_value=0)
            .reindex(columns=["ESL2.4", "ESL2.5"], fill_value=0)
        )
        z["denominator"] = z["ESL2.4"] + z["ESL2.5"]
        z["esl25_fraction_major_only"] = z["ESL2.5"] / z["denominator"]
        z = z.merge(
            cities[["city", "degree"]], left_index=True, right_on="city", validate="one_to_one"
        )
        for minimum in [1, 5, 10, 20]:
            q = z.loc[z["denominator"] >= minimum]
            if len(q) >= 3:
                stat = spearmanr(q["degree"], q["esl25_fraction_major_only"])
                rho, p_value = float(stat.statistic), float(stat.pvalue)
            else:
                rho, p_value = np.nan, np.nan
            restricted_rows.append(
                {
                    "restriction": restriction,
                    "minimum_specimen_count_per_city": minimum,
                    "n_cities": int(len(q)),
                    "n_focal_lineage_genomes": int(q["denominator"].sum()),
                    "spearman_rho_degree_vs_esl25_fraction": rho,
                    "p_value": p_value,
                    "outcome_definition": "ESL2.5/(ESL2.4+ESL2.5) within the restricted specimen class",
                }
            )
    pd.DataFrame(restricted_rows).to_csv(
        args.output_dir / "09_specimen_restricted_degree_associations.tsv", sep="\t", index=False
    )

    blood_city = (
        data.loc[data["is_bloodstream"]]
        .pivot_table(
            index=["City", "degree", "depth_group"],
            columns="lineage",
            values="genome",
            aggfunc="count",
            fill_value=0,
            observed=True,
        )
        .reindex(columns=["ESL2.4", "ESL2.5"], fill_value=0)
        .reset_index()
    )
    blood_city["bloodstream_total"] = blood_city["ESL2.4"] + blood_city["ESL2.5"]
    blood_city["esl25_fraction_bloodstream"] = blood_city["ESL2.5"] / blood_city["bloodstream_total"]
    blood_city["degree_tertile"] = pd.qcut(
        blood_city["degree"], 3, labels=["low", "middle", "high"], duplicates="drop"
    )
    blood_city.to_csv(
        args.output_dir / "10_bloodstream_city_descriptive.tsv", sep="\t", index=False
    )

    blood_depth = (
        blood_city.groupby("depth_group", observed=True)
        .agg(
            n_cities=("City", "size"),
            n_bloodstream=("bloodstream_total", "sum"),
            n_esl24=("ESL2.4", "sum"),
            n_esl25=("ESL2.5", "sum"),
            city_equal_mean_esl25=("esl25_fraction_bloodstream", "mean"),
            city_equal_median_esl25=("esl25_fraction_bloodstream", "median"),
        )
        .reset_index()
    )
    blood_depth["pooled_esl25_fraction"] = blood_depth["n_esl25"] / blood_depth["n_bloodstream"]
    blood_degree = (
        blood_city.groupby("degree_tertile", observed=True)
        .agg(
            n_cities=("City", "size"),
            n_bloodstream=("bloodstream_total", "sum"),
            n_esl24=("ESL2.4", "sum"),
            n_esl25=("ESL2.5", "sum"),
            city_equal_mean_esl25=("esl25_fraction_bloodstream", "mean"),
            city_equal_median_esl25=("esl25_fraction_bloodstream", "median"),
        )
        .reset_index()
    )
    blood_degree["pooled_esl25_fraction"] = blood_degree["n_esl25"] / blood_degree["n_bloodstream"]
    blood_depth.insert(0, "stratification", "sampling_depth")
    blood_depth = blood_depth.rename(columns={"depth_group": "stratum"})
    blood_degree.insert(0, "stratification", "degree_tertile")
    blood_degree = blood_degree.rename(columns={"degree_tertile": "stratum"})
    pd.concat([blood_depth, blood_degree], ignore_index=True).to_csv(
        args.output_dir / "11_bloodstream_trend_summary.tsv", sep="\t", index=False
    )

    audit = {
        "source_workbook": str(args.workbook.resolve()),
        "source_sheet": "S1 Metadata",
        "figure3_city_table": str(args.city_table.resolve()),
        "n_figure3_cities": int(cities["city"].nunique()),
        "n_major_lineage_genomes": int(len(data)),
        "n_binary_classified": int(data["binary_status_known"].sum()),
        "n_invasive": int(data["is_invasive"].sum()),
        "n_uncertain": int(data["Invasive sample"].eq("Uncertain").sum()),
        "n_unknown": int(data["Invasive sample"].eq("Unknown").sum()),
        "binary_definition": "Invasive versus Non-invasive; Uncertain and Unknown excluded",
        "sampling_depth_definition": "ESL2.4 + ESL2.5 genomes per Figure 3 city",
        "sampling_groups": DEPTH_ORDER,
        "bloodstream_labels": sorted(BLOODSTREAM_LABELS),
    }
    (args.output_dir / "00_analysis_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(json.dumps(audit, indent=2))
    print("\nSampling groups")
    print(group_summary.to_string(index=False))
    print("\nSampling groups by lineage")
    print(lineage_summary.to_string(index=False))
    print("\nTests")
    print(pd.DataFrame(tests)[["analysis", "effect", "effect_label", "ci_low", "ci_high", "p_value", "n_cities"]].to_string(index=False))


if __name__ == "__main__":
    main()
