"""Five city-equal fractional-logit models for Supplementary Fig. 3g."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

DATA = Path(__file__).resolve().parents[2] / "data"
URBAN = ["gdp_pc_z", "population_z", "urbanisation_z"]
CATEGORIES = ["sampling_mid", "sampling_high"]
MODELS = {
    "Degree only": ["degree_z"],
    "+ log(genome count)": ["degree_z", "sampling_z"],
    "+ sampling-depth categories": ["degree_z", *CATEGORIES],
    "+ log(genome count) + urban covariates": ["degree_z", "sampling_z", *URBAN],
    "+ depth categories + urban covariates": ["degree_z", *CATEGORIES, *URBAN],
}


def complete_cases():
    data = pd.read_csv(DATA / "cities.tsv", sep="\t")
    d = data.dropna(subset=["degree", "Population (10K)", "GDP (100M rmb)", "Urbanization rate(%)"]).copy()
    assert len(d) == 47 and d.complete_case_primary.all()
    pop = d["Population (10K)"]
    features = pd.DataFrame(dict(degree_z=d.degree, sampling_z=np.log1p(d.n_major_lineages),
                                 gdp_pc_z=np.log(d["GDP (100M rmb)"] / pop),
                                 population_z=np.log(pop), urbanisation_z=d["Urbanization rate(%)"]))
    d[features.columns] = (features - features.mean()) / features.std(ddof=1)
    d["sampling_mid"] = d.n_major_lineages.between(20, 49).astype(int)
    d["sampling_high"] = d.n_major_lineages.ge(50).astype(int)
    return d


def fit(d, terms, robust=True):
    model = sm.GLM(d.esl25_prevalence_fig3, sm.add_constant(d[terms], has_constant="add"),
                   family=sm.families.Binomial())
    return model.fit(cov_type="HC3") if robust else model.fit()


def run(out, draw):
    out.mkdir(parents=True, exist_ok=True)
    d = complete_cases()
    rows, influence, diagnostics = [], [], []
    for name, terms in MODELS.items():
        result = fit(d, terms)
        beta = result.params["degree_z"]
        low, high = result.conf_int().loc["degree_z"]
        rows.append((name, len(d), d.n_major_lineages.sum(), beta, result.bse["degree_z"],
                     np.exp(beta), np.exp(low), np.exp(high), result.pvalues["degree_z"], result.deviance))
        design = result.model.exog
        diagnostics.extend((name, term, variance_inflation_factor(design, i))
                           for i, term in enumerate(result.model.exog_names) if term != "const")
        influence.extend((name, city, fit(d.drop(i), terms, robust=False).params["degree_z"])
                         for i, city in d.city.items())
    table = pd.DataFrame(rows, columns=["model", "n_cities", "n_major_lineage_genomes",
                                        "beta", "hc3_se", "or", "ci_low", "ci_high", "p_value", "deviance"])
    table.to_csv(out / "models.tsv", sep="\t", index=False)
    pd.DataFrame(influence, columns=["model", "omitted_city", "beta"]).to_csv(out / "leave_one_city_out.tsv", sep="\t", index=False)
    pd.DataFrame(diagnostics, columns=["model", "term", "vif"]).to_csv(out / "vif.tsv", sep="\t", index=False)
    if draw:
        y = np.arange(len(table))[::-1]
        or_values = table["or"]
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        ax.errorbar(or_values, y, xerr=[or_values - table.ci_low, table.ci_high - or_values],
                    fmt="o", color="#79569C", capsize=4)
        ax.axvline(1, color="0.4", linestyle="--")
        ax.set(yticks=y, yticklabels=table.model, xscale="log",
               xlabel="Odds ratio per SD of network degree (95% HC3 CI)")
        fig.savefig(out / "supp_fig_3g.svg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    run(args.out, args.plot)
