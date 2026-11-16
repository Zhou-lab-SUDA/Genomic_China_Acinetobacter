#!/usr/bin/env python3
"""New, three-panel Supplementary Fig. X; fixed 18-city S1 historical cohort.

Run with the bundled Python runtime. Inputs and earlier figures are not modified.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.colors import Normalize
from scipy.stats import linregress, rankdata, t

import argparse


DATA = Path(__file__).resolve().parents[1] / "data"
PREFIX = "07_supplementary_fig_X"
SEED = 20261029
FEATURES = [
    "log_population", "log_gdp_per_capita", "primary_industry_percent",
    "tertiary_industry_percent", "log_hospitals_per_100k",
    "log_beds_per_100k", "log_doctors_per_100k",
]
LABELS = [
    "Population", "GDP per capita", "Primary industry (%)", "Tertiary industry (%)",
    "Hospitals /100,000", "Beds /100,000", "Doctors /100,000",
]
SHORT_LABELS = [
    "Population", "GDP per capita", "Primary industry", "Tertiary industry",
    "Hospitals /100k", "Beds /100k", "Doctors /100k",
]
HIST_COLOR = "#C57430"
CONT_COLOR = "#287F9A"


def read_matched_inputs():
    h = pd.read_csv(DATA / "S10a_Cities_and_PCA_scores.tsv", sep="\t")
    assert len(h) == 18 and h.total_genomes.sum() == 1477
    matches = pd.read_csv(DATA / "S10e_PCA_input_values.tsv", sep="\t")
    h = h.merge(matches.drop(columns="esl25_prevalence"), on=["canonical_city", "province"], validate="one_to_one")
    c = h[["canonical_city", "province"]].copy()
    for feature in FEATURES:
        h[feature] = h["historical_" + feature]
        c[feature] = h["contemporary_" + feature]
    return h, c


def fit_pca(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = d[FEATURES].to_numpy(float)
    z = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    scores = u * s
    loadings = vt.T
    # PC1 sign is arbitrary. Use one outcome-independent orientation for both
    # periods: higher GDPpc/tertiary share and lower primary-industry share.
    anchor = np.array([0., 1., -1., 1., 0., 0., 0.])
    if loadings[:, 0] @ anchor < 0:
        scores[:, 0] *= -1
        loadings[:, 0] *= -1
    # Preserve the previous historical PC2 orientation: population points down.
    if loadings[0, 1] > 0:
        scores[:, 1] *= -1
        loadings[:, 1] *= -1
    return scores, loadings, s**2 / np.sum(s**2)


def permutation_rho(x: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                    reps: int = 99999) -> tuple[float, float]:
    a, b = rankdata(x), rankdata(y)
    a -= a.mean(); b -= b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    rho = float(a @ b / denom)
    extreme = 0
    for start in range(0, reps, 10000):
        ix = np.array([rng.permutation(len(b)) for _ in range(min(10000, reps - start))])
        extreme += np.count_nonzero(np.abs(b[ix] @ a / denom) >= abs(rho) - 1e-12)
    return rho, (extreme + 1) / (reps + 1)


def bh_adjust(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    ranked = np.asarray(p)[order] * len(p) / np.arange(1, len(p) + 1)
    out = np.empty(len(p))
    out[order] = np.minimum(1, np.minimum.accumulate(ranked[::-1])[::-1])
    return out


def point_colors(h: pd.DataFrame) -> np.ndarray:
    colors = mpl.colormaps["plasma"](h["esl25_prevalence"].to_numpy())
    colors[h["esl25_genomes"].to_numpy() == 0] = mpl.colors.to_rgba("#BDBDBD")
    return colors


def city_labels(ax: plt.Axes, x: np.ndarray, y: np.ndarray, names: list[str],
                extra: list | None = None) -> None:
    texts = [ax.text(a, b, name, fontsize=7, color="#222222", zorder=6)
             for a, b, name in zip(x, y, names)]
    # Fixed random seed and iteration limit make label placement reproducible.
    np.random.seed(SEED)
    adjust_text(texts + (extra or []), x=x, y=y, ax=ax, iter_lim=800,
                expand=(1.10, 1.22), force_text=(0.14, 0.24),
                force_static=(0.12, 0.20), ensure_inside_axes=True,
                arrowprops={"arrowstyle": "-", "color": "#999999", "lw": 0.45})


def draw(h: pd.DataFrame, sh: np.ndarray, lh: np.ndarray, vh: np.ndarray,
         sc: np.ndarray, vc: np.ndarray, table: pd.DataFrame, stats: dict) -> None:
    mpl.rcParams.update({"font.family": "Arial", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .9, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(18.5, 7.0), facecolor="white")
    a = fig.add_axes([.045, .245, .29, .66])
    b = fig.add_axes([.392, .245, .235, .66])
    c = fig.add_axes([.762, .245, .222, .66])
    colors = point_colors(h)
    names = h["canonical_city"].tolist()

    # Panel a: original historical PCA geometry, all seven common arrows.
    a.set_box_aspect(1)
    a.set_anchor("N")
    a.set_xlim(-5.0, 4.8); a.set_ylim(-4.7, 5.1)
    a.axhline(0, color="#B0B0B0", lw=.8, ls=(0, (4, 4)))
    a.axvline(0, color="#B0B0B0", lw=.8, ls=(0, (4, 4)))
    a.scatter(sh[:, 0], sh[:, 1], c=colors, s=48, edgecolor="#333333", linewidth=.65, zorder=4)
    labels = []
    for i, short in enumerate(SHORT_LABELS):
        end = lh[i, :2] * 3.8
        a.annotate("", xy=end, xytext=(0, 0),
                   arrowprops={"arrowstyle": "-|>", "color": "#B74337", "lw": 1.1}, zorder=2)
        labels.append(a.text(*(end * 1.18), short, fontsize=7, color="#A5362B",
                             weight="bold", ha="center", va="center", zorder=5))
    a.set_xlabel(f"Historical PC1 ({vh[0]:.1%})")
    a.set_ylabel(f"Historical PC2 ({vh[1]:.1%})")
    a.set_title("Historical PCA\n2010–2017 observed means", loc="left", fontsize=11, pad=12)

    # Panel b: independently fitted common-variable PC1 on a common SD scale.
    x = sh[:, 0] / sh[:, 0].std(ddof=1)
    y = sc[:, 0] / sc[:, 0].std(ddof=1)
    b.set_box_aspect(1)
    b.set_anchor("N")
    b.set_xlim(-2.65, 2.65); b.set_ylim(-2.65, 2.65)
    b.axhline(0, color="#DDDDDD", lw=.7, zorder=0)
    b.axvline(0, color="#DDDDDD", lw=.7, zorder=0)
    fitted = stats["pc1_ols_intercept"] + stats["pc1_ols_slope"] * x
    grid = np.linspace(x.min(), x.max(), 200)
    predicted = stats["pc1_ols_intercept"] + stats["pc1_ols_slope"] * grid
    residual_sd = np.sqrt(np.sum((y - fitted)**2) / (len(x) - 2))
    mean_se = residual_sd * np.sqrt(1 / len(x) + (grid - x.mean())**2 / np.sum((x - x.mean())**2))
    half_width = t.ppf(.975, len(x) - 2) * mean_se
    b.fill_between(grid, predicted - half_width, predicted + half_width,
                   color="#515A67", alpha=.13, linewidth=0, zorder=1)
    b.plot(grid, predicted, color="#515A67", lw=1.5, zorder=2)
    b.scatter(x, y, c=colors, s=48, edgecolor="#333333", linewidth=.65, zorder=4)
    b.set_xlabel(f"Historical PC1 (SD; {vh[0]:.1%} variance)")
    b.set_ylabel(f"Contemporary PC1 (SD; {vc[0]:.1%} variance)")
    b.set_title("PC1 concordance\nSame 18 cities; same 7 predictors", loc="left", fontsize=11, pad=12)
    b.text(.04, .96, f"Spearman $\\rho$ = {stats['pc1_spearman_rho']:.2f}\n"
           f"OLS $R^2$ = {stats['pc1_ols_r_squared']:.2f}\n$n$ = 18",
           transform=b.transAxes, ha="left", va="top", fontsize=9,
           bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 2}, zorder=7)

    # Panel c: same city-years of outcome; only predictor vintage changes.
    ypos = np.arange(len(table))
    c.axvline(0, color="#909090", lw=.8, ls=(0, (4, 4)), zorder=0)
    for j, row in table.iterrows():
        c.plot([row["historical_rho"], row["contemporary_rho"]], [j, j],
               color="#A8A8A8", lw=1.6, zorder=1)
    c.scatter(table["historical_rho"], ypos, color=HIST_COLOR, s=52,
              edgecolor="white", linewidth=.5, label="Historical (2010–2017 mean)", zorder=3)
    c.scatter(table["contemporary_rho"], ypos, color=CONT_COLOR, marker="D", s=39,
              edgecolor="white", linewidth=.5, label="Contemporary (Table S2)", zorder=4)
    c.set_yticks(ypos, LABELS[:-1] + ["Doctors / health staff†"])
    c.set_ylim(len(table) - .45, -.55)
    c.set_xlim(-1, 1)
    c.set_xticks([-1, -.5, 0, .5, 1])
    c.grid(axis="x", color="#E8E8E8", lw=.6, zorder=0)
    c.tick_params(axis="y", length=0, pad=8)
    c.set_xlabel("Spearman $\\rho$ with ESL2.5 prevalence")
    c.set_title("Attribute–prevalence correlations\nFixed 2010–2017 outcome", loc="left", fontsize=11, pad=12)
    c.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(-.02, -.14),
             handletextpad=.5, labelspacing=.7)
    for ax, letter in zip((a, b, c), "abc"):
        ax.text(-.12, 1.075, letter, transform=ax.transAxes, weight="bold", fontsize=18,
                ha="left", va="bottom")
    fig.canvas.draw()
    city_labels(a, sh[:, 0], sh[:, 1], names, labels)
    city_labels(b, x, y, names)
    # Shared colour legend for a and b, plus an explicit observed-zero key.
    cbax = fig.add_axes([.072, .132, .19, .018])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(0, 1), cmap="plasma"), cax=cbax,
                      orientation="horizontal", ticks=[0, .5, 1])
    cb.ax.set_xticklabels(["0%", "50%", "100%"])
    cb.set_label("ESL2.5 prevalence (panels a and b)", fontsize=8, labelpad=2)
    zero_handle = mpl.lines.Line2D([], [], marker="o", linestyle="none", markerfacecolor="#BDBDBD",
                                  markeredgecolor="#333333", markersize=6)
    fig.legend([zero_handle], ["No ESL2.5 genomes"], loc="center left",
               bbox_to_anchor=(.29, .139), frameon=False, fontsize=8)
    fig.text(.045, .018,
             "Fixed S1 cohort: 18 cities; 1,477 genomes sampled in 2010–2017.  "
             "ESL2.5 prevalence = ESL2.5 / all sampled genomes.\n"
             "Seven shared predictors were specified without outcome screening; "
             "PCA was fitted separately by covariate vintage.\n"
             "† Historical licensed doctors versus contemporary HealthCarePro; source occupational definitions may differ.",
             fontsize=8, color="#4A4A4A", va="bottom", linespacing=1.6)
    DEST.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(DEST / f"Supplementary_Fig_X.{ext}", dpi=450 if ext == "png" else None,
                    facecolor="white")
    plt.close(fig)


def main() -> None:
    global OUT, DEST
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    OUT = DEST = args.out
    OUT.mkdir(parents=True, exist_ok=True)
    h, c = read_matched_inputs()
    sh, lh, vh = fit_pca(h)
    sc, lc, vc = fit_pca(c)
    rng = np.random.default_rng(SEED)
    rho, pval = permutation_rho(sh[:, 0], sc[:, 0], rng)
    regression = linregress(sh[:, 0] / sh[:, 0].std(ddof=1),
                            sc[:, 0] / sc[:, 0].std(ddof=1))
    stats = {"n_cities": 18, "n_genomes": 1477,
             "pc1_spearman_rho": rho, "pc1_permutation_p": pval,
             "pc1_ols_slope": float(regression.slope),
             "pc1_ols_intercept": float(regression.intercept),
             "pc1_ols_r_squared": float(regression.rvalue**2),
             "pc1_ols_slope_p": float(regression.pvalue),
             "historical_variance_explained": vh.tolist(),
             "contemporary_variance_explained": vc.tolist()}
    rows = []
    for feat, label in zip(FEATURES, LABELS):
        rh, ph = permutation_rho(h[feat].to_numpy(), h["esl25_prevalence"].to_numpy(), rng)
        rc, pc = permutation_rho(c[feat].to_numpy(), h["esl25_prevalence"].to_numpy(), rng)
        rows.append({"variable": feat, "label": label, "n_cities": 18,
                     "historical_rho": rh, "contemporary_rho": rc,
                     "rho_change_contemporary_minus_historical": rc - rh,
                     "historical_permutation_p": ph, "contemporary_permutation_p": pc})
    table = pd.DataFrame(rows)
    for vintage in ("historical", "contemporary"):
        table[f"{vintage}_BH_q"] = bh_adjust(table[f"{vintage}_permutation_p"].to_numpy())
    # Cross-check against the earlier comparison without overwriting it.
    previous = OUT / "06_historical_vs_2023_variable_comparison.tsv"
    if previous.exists():
        old = pd.read_csv(previous, sep="\t").set_index("variable").loc[FEATURES]
        np.testing.assert_allclose(table["historical_rho"], old["historical_outcome_rho"], atol=1e-12)
        np.testing.assert_allclose(table["contemporary_rho"], old["snapshot_2023_outcome_rho"], atol=1e-12)
    scores = h[["canonical_city", "province", "total_genomes", "esl25_genomes", "esl25_prevalence"]].copy()
    scores["historical_PC1"] = sh[:, 0]
    scores["historical_PC2"] = sh[:, 1]
    scores["contemporary_PC1"] = sc[:, 0]
    scores["contemporary_PC2"] = sc[:, 1]
    scores["historical_PC1_SD"] = sh[:, 0] / sh[:, 0].std(ddof=1)
    scores["contemporary_PC1_SD"] = sc[:, 0] / sc[:, 0].std(ddof=1)
    scores.to_csv(OUT / f"{PREFIX}_city_scores.tsv", sep="\t", index=False)
    table.to_csv(OUT / f"{PREFIX}_correlations.tsv", sep="\t", index=False)
    load = pd.DataFrame({"variable": FEATURES, "label": LABELS,
                         "historical_PC1_loading": lh[:, 0], "historical_PC2_loading": lh[:, 1],
                         "contemporary_PC1_loading": lc[:, 0], "contemporary_PC2_loading": lc[:, 1]})
    load.to_csv(OUT / f"{PREFIX}_loadings.tsv", sep="\t", index=False)
    data = scores[["canonical_city", "province", "esl25_prevalence"]].copy()
    for feat in FEATURES:
        data[f"historical_{feat}"] = h[feat].to_numpy()
        data[f"contemporary_{feat}"] = c[feat].to_numpy()
    data.to_csv(OUT / f"{PREFIX}_matched_predictors.tsv", sep="\t", index=False)
    raw = pd.read_csv(DATA / "S10b_Historical_covariates.tsv", sep="\t")
    wide = raw.pivot(index="City", columns="Indicator", values="Historical_value")
    wide["esl25_prevalence"] = h.set_index("canonical_city").esl25_prevalence
    reference = pd.read_csv(DATA / "S10c_Historical_Spearman.tsv", sep="\t")
    expanded = []
    indicator_names = dict(zip(
            ["total_population", "total_gdp", "gdp_per_capita", "primary_industry", "secondary_industry",
             "tertiary_industry", "hospitals_per_100k", "beds_per_100k", "doctors_per_100k",
             "road_km", "medical_insurance_per_100k", "water_pipe_km", "pension_insurance_per_100k"],
            ["Total population", "Total GDP", "GDP per capita", "Primary industry", "Secondary industry",
             "Tertiary industry", "Hospitals", "Hospital beds", "Licensed doctors",
             "Road mileage", "Medical insurance enrolment", "Drainage pipe length", "Pension insurance enrolment"],
    ))
    rng = np.random.default_rng(20261024)
    order = ["total_population", "total_gdp", "gdp_per_capita", "primary_industry", "secondary_industry",
             "tertiary_industry", "hospitals_per_100k", "beds_per_100k", "doctors_per_100k",
             "pension_insurance_per_100k", "medical_insurance_per_100k", "road_km", "water_pipe_km"]
    assert set(order) == set(reference.variable)
    for variable in order:
        indicator = indicator_names[variable]
        pair = wide[[indicator, "esl25_prevalence"]].dropna()
        rho, p = permutation_rho(pair[indicator].to_numpy(), pair.esl25_prevalence.to_numpy(), rng)
        expanded.append((variable, len(pair), rho, p))
    expanded = pd.DataFrame(expanded, columns=["variable", "n_cities", "rho", "permutation_p"])
    expanded["BH_q"] = bh_adjust(expanded.permutation_p.to_numpy())
    expanded.to_csv(OUT / "historical_spearman.tsv", sep="\t", index=False)
    if args.plot:
        draw(h, sh, lh, vh, sc, vc, table, stats)
    manifest = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
                "stats": stats, "permutations": 99999,
                "inputs_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in sorted(DATA.glob("*.tsv"))},
                "features": FEATURES, "prevalence_denominator": "ESL2.5 / all genomes",
                "contemporary_label": "Contemporary (Table S2 snapshot); source-vintage label, not newly audited annual data",
                "PC1_orientation": "Positive loading sum for GDPpc - primary industry + tertiary industry; outcome-independent",
                "panel_b_scale": "PC1 scores independently scaled to within-vintage SD=1",
                "panel_b_fit": "Equal-city-weight OLS with intercept; shaded pointwise 95% confidence interval for mean response, Student t with 16 df",
                "panel_c_outcome": "Fixed 2010-2017 ESL2.5 prevalence in the same 18 cities",
                "healthcare_definition_caution": "Historical licensed doctors versus contemporary HealthCarePro; units normalised, occupational definitions may differ",
                "q_value_family": "BH separately across seven predictors for each vintage; no tests of rho differences",
                "figure_generated": args.plot}
    (OUT / f"{PREFIX}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
