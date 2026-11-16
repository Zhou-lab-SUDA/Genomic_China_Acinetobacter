"""Refit the annual dissemination model and apply fixed lineage clock factors."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DATA = Path(__file__).resolve().parents[1] / "data"
SEED = 20261026

def fit_hc3(data: pd.DataFrame):
    formula = "km_per_snp ~ post * lineage25 + log1p_n_genomes + n_cities"
    base = smf.ols(formula, data=data).fit()
    robust = base.get_robustcov_results(cov_type="HC3", use_t=True)
    return base, robust


def t_contrast(model, vector: np.ndarray) -> tuple[float, float, float, float]:
    test = model.t_test(vector)
    lo, hi = map(float, test.conf_int()[0])
    return float(test.effect.item()), lo, hi, float(test.pvalue.item())


def mv_t_draws(model, n_draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    beta = np.asarray(model.params, dtype=float)
    cov = np.asarray(model.cov_params(), dtype=float)
    z = rng.multivariate_normal(np.zeros(len(beta)), cov, size=n_draws)
    q = rng.chisquare(float(model.df_resid), size=n_draws)
    return beta + z * np.sqrt(float(model.df_resid) / q)[:, None]


def design(mean_log_n: float, mean_cities: float, lineage25: int, post: int) -> np.ndarray:
    return np.array([1.0, post, lineage25, post * lineage25, mean_log_n, mean_cities])


def converted_analysis(
    data: pd.DataFrame,
    rates: dict[str, float],
    analysis: str,
    pre_window: str,
    post_window: str,
    n_draws: int = 250_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, model = fit_hc3(data)
    draws = mv_t_draws(model, n_draws=n_draws, seed=SEED + len(data))
    mean_log_n = float(data["log1p_n_genomes"].mean())
    mean_cities = float(data["n_cities"].mean())
    summary_rows: list[dict[str, object]] = []
    change_vectors: dict[str, np.ndarray] = {}

    for lineage, lin25 in (("2.4", 0), ("2.5", 1)):
        rate = rates[lineage]
        x_pre = design(mean_log_n, mean_cities, lin25, 0)
        x_post = design(mean_log_n, mean_cities, lin25, 1)
        x_change = (x_post - x_pre) * rate
        change_vectors[lineage] = x_change

        pre_draw = (draws @ x_pre) * rate
        post_draw = (draws @ x_post) * rate
        delta_draw = post_draw - pre_draw
        valid = (pre_draw > 0) & (post_draw > 0)
        reduction_draw = 100.0 * (post_draw[valid] / pre_draw[valid] - 1.0)

        beta = np.asarray(model.params)
        pre = float(beta @ x_pre * rate)
        post = float(beta @ x_post * rate)
        delta, delta_lo, delta_hi, p_value = t_contrast(model, x_change)
        reduction = 100.0 * (post / pre - 1.0)
        summary_rows.append({
            "analysis": analysis,
            "status": "EXPLORATORY_CONDITIONAL_CONVERSION",
            "lineage": lineage,
            "pre_window": pre_window,
            "post_window": post_window,
            "adjusted_pre_km_per_year": pre,
            "adjusted_pre_ci95_low": float(np.quantile(pre_draw, 0.025)),
            "adjusted_pre_ci95_high": float(np.quantile(pre_draw, 0.975)),
            "adjusted_post_km_per_year": post,
            "adjusted_post_ci95_low": float(np.quantile(post_draw, 0.025)),
            "adjusted_post_ci95_high": float(np.quantile(post_draw, 0.975)),
            "delta_km_per_year": delta,
            "delta_ci95_low": delta_lo,
            "delta_ci95_high": delta_hi,
            "reduction_percent": reduction,
            "reduction_ci95_low": float(np.quantile(reduction_draw, 0.025)),
            "reduction_ci95_high": float(np.quantile(reduction_draw, 0.975)),
            "p_value_delta": p_value,
            "conditional_snp_per_year": rate,
            "positive_pre_post_draw_fraction": float(valid.mean()),
            "n_lineage_years": int(len(data)),
            "n_unique_years": int(data["year"].nunique()),
            "reference_log1p_genomes": mean_log_n,
            "reference_sampled_cities": mean_cities,
        })

    difference_vector = change_vectors["2.5"] - change_vectors["2.4"]
    estimate, lo, hi, p_value = t_contrast(model, difference_vector)
    interaction = pd.DataFrame([{
        "analysis": analysis,
        "status": "EXPLORATORY_CONDITIONAL_CONVERSION",
        "contrast": "ESL2.5 minus ESL2.4 difference-in-change",
        "estimate_km_per_year": estimate,
        "ci95_low": lo,
        "ci95_high": hi,
        "p_value": p_value,
        "note": "Lineage-specific clock scaling; not the raw km/SNP interaction coefficient",
    }])
    return pd.DataFrame(summary_rows), interaction



def run(out):
    out.mkdir(parents=True, exist_ok=True)
    annual = pd.read_csv(DATA / "S9a_Annual_speed_and_CI.tsv", sep="\t")
    annual["lineage"] = annual["lineage"].str.removeprefix("ESL")
    clock = pd.read_csv(DATA / "S9f_Clock_rates.tsv", sep="\t", dtype={"lineage": str})
    clock["lineage"] = clock["lineage"].str.removeprefix("ESL")
    rates = dict(zip(clock.lineage, clock.substitutions_per_site_per_year * clock.assumed_callable_block_bp))
    np.testing.assert_allclose(clock.lineage.map(rates), clock.conditional_snp_per_year)
    annual["lineage25"] = annual.lineage.eq("2.5").astype(int)
    annual["post"] = annual.year.ge(2020).astype(int)
    annual["log1p_n_genomes"] = np.log1p(annual.n_genomes)
    model, robust = fit_hc3(annual)
    tested = robust.t_test(model.model.exog * annual.lineage.map(rates).to_numpy()[:, None])
    annual["observed_km_per_year"] = annual.km_per_snp * annual.lineage.map(rates)
    annual["estimate_km_per_year"] = tested.effect
    annual["ci_lower_km_per_year"], annual["ci_upper_km_per_year"] = tested.conf_int().T
    annual.to_csv(out / "annual.tsv", sep="\t", index=False)
    mean_log_n = annual.log1p_n_genomes.mean()
    mean_cities = annual.n_cities.mean()
    period = {}
    rows = []
    for label, lineage25 in (("2.4", 0), ("2.5", 1)):
        for is_post, window in enumerate(["2011-2019", "2020-2023"]):
            vector = rates[label] * design(mean_log_n, mean_cities, lineage25, is_post)
            period[label, is_post] = vector
            estimate, low, high, _ = t_contrast(robust, vector)
            rows.append((label, window, estimate, low, high))
    columns = ["lineage", "period_label", "estimate_km_per_year", "ci_lower_km_per_year", "ci_upper_km_per_year"]
    pd.DataFrame(rows, columns=columns).to_csv(out / "period_estimates.tsv", sep="\t", index=False)
    p24, q24 = period["2.4", 0], period["2.4", 1]
    p25, q25 = period["2.5", 0], period["2.5", 1]
    comparisons = {
        "within_ESL2.4": q24 - p24, "within_ESL2.5": q25 - p25,
        "between_pre": p25 - p24, "between_npi": q25 - q24,
        "difference_in_change": (q25 - p25) - (q24 - p24),
    }
    columns = ["comparison_id", "estimate_difference_km_per_year", "ci_lower_km_per_year",
               "ci_upper_km_per_year", "p_value"]
    pd.DataFrame([(label, *t_contrast(robust, vector)) for label, vector in comparisons.items()],
                 columns=columns).to_csv(out / "main_figure_contrasts.tsv", sep="\t", index=False)
    placebo = annual.loc[annual.year.le(2017)].assign(post=lambda d: d.year.ge(2014).astype(int))
    actual = converted_analysis(annual, rates, "actual_covid", "2011-2019", "2020-2023")
    pseudo = converted_analysis(placebo, rates, "prepandemic_placebo", "2011-2013", "2014-2017")
    pd.concat([actual[0], pseudo[0]]).to_csv(out / "changes.tsv", sep="\t", index=False)
    pd.concat([actual[1], pseudo[1]]).to_csv(out / "interactions.tsv", sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
