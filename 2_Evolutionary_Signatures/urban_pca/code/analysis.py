"""Outcome-blind SVD and urban PCA figure panels."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
PCA_VARIABLES = pd.read_csv(DATA / "S2b_Contemporary_PCA_loadings.tsv", sep="\t").Variable.tolist()

def outcome_blind_pca() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    data = pd.read_csv(DATA / "pca_input.tsv", sep="\t")
    complete = data.dropna(subset=PCA_VARIABLES).copy()
    if len(complete) != 60:
        raise ValueError(f"Expected 60 complete PCA-input cities, found {len(complete)}")
    x = complete[PCA_VARIABLES].to_numpy(float)
    means = x.mean(axis=0)
    sds = x.std(axis=0, ddof=1)
    z = (x - means) / sds
    u, singular_values, vt = np.linalg.svd(z, full_matrices=False)
    scores = u * singular_values
    loadings = vt.T
    eigenvalues = singular_values**2 / (len(complete) - 1)
    explained = eigenvalues / eigenvalues.sum()

    # Fixed sign conventions for reproducibility; signs do not alter the PCA.
    sign_pc1 = 1.0 if loadings[PCA_VARIABLES.index("Urbanization_rate"), 0] >= 0 else -1.0
    sign_pc2 = 1.0 if loadings[PCA_VARIABLES.index("HospitalBed_per_capita_log"), 1] >= 0 else -1.0
    scores[:, 0] *= sign_pc1
    loadings[:, 0] *= sign_pc1
    scores[:, 1] *= sign_pc2
    loadings[:, 1] *= sign_pc2

    score_table = complete[["City", "Province"]].copy()
    for i in range(scores.shape[1]):
        score_table[f"PC{i + 1}"] = scores[:, i]
        score_table[f"PC{i + 1}_variance_explained"] = explained[i]

    loading_table = pd.DataFrame(
        {
            "Variable": PCA_VARIABLES,
            "transformed_mean": means,
            "transformed_sample_sd": sds,
        }
    )
    for i in range(loadings.shape[1]):
        loading_table[f"PC{i + 1}_loading"] = loadings[:, i]
        loading_table[f"PC{i + 1}_correlation_loading"] = loadings[:, i] * np.sqrt(eigenvalues[i])
        loading_table[f"PC{i + 1}_variance_explained"] = explained[i]

    outcome = pd.read_csv(DATA / "S2a_Contemporary_PCA_scores.tsv", sep="\t")
    score_table = score_table.merge(outcome[["City", "Province", "ESL25_prevalence_Fig3"]],
                                    on=["City", "Province"], how="left", validate="one_to_one")
    return score_table, loading_table, explained



def draw(out, scores, explained):
    vectors = pd.read_csv(DATA / "S2c_Supplementary_PCA_vectors.tsv", sep="\t")
    for key, arrows in [("fig_3d", vectors.loc[vectors.Display_label.isin(["Population", "Total GDP", "Urbanisation rate"])]),
                        ("extended_data_fig_6", vectors)]:
        fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
        shown = scores.ESL25_prevalence_Fig3.gt(0)
        ax.scatter(scores.loc[~shown, "PC1"], scores.loc[~shown, "PC2"], color="0.7", s=18)
        sc = ax.scatter(scores.loc[shown, "PC1"], scores.loc[shown, "PC2"],
                        c=scores.loc[shown, "ESL25_prevalence_Fig3"], cmap="plasma", vmin=0, vmax=1, s=22)
        fig.colorbar(sc, ax=ax, label="ESL2.5 prevalence")
        for row in arrows.itertuples():
            x, y = 2 * row.PC1_correlation_loading, 2 * row.PC2_correlation_loading
            ax.annotate("", xy=(x, y), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#B62929"))
            ax.text(x, y, row.Display_label, fontsize=7)
        ax.axhline(0, color="0.85", zorder=0)
        ax.axvline(0, color="0.85", zorder=0)
        ax.set(xlabel=f"PC1 ({explained[0]:.1%})", ylabel=f"PC2 ({explained[1]:.1%})")
        ax.set_box_aspect(1)
        fig.savefig(out / f"{key}.svg")
        plt.close(fig)
    heat = vectors.loc[~vectors.Display_label.str.contains("Road|Highway", case=False)]
    matrix = heat[["PC1_correlation_loading", "PC2_correlation_loading"]].to_numpy()
    fig, ax = plt.subplots(figsize=(6, 8), layout="constrained")
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set(xticks=[0, 1], xticklabels=["PC1", "PC2"], yticks=range(len(heat)), yticklabels=heat.Display_label)
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:+.2f}", ha="center", va="center", color="white" if abs(value) > .65 else "black")
    fig.colorbar(im, ax=ax, label="Variable–PC correlation")
    fig.savefig(out / "fig_3e.svg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    scores, loadings, explained = outcome_blind_pca()
    scores.to_csv(args.out / "scores.tsv", sep="\t", index=False)
    loadings.to_csv(args.out / "loadings.tsv", sep="\t", index=False)
    x = pd.read_csv(DATA / "pca_input.tsv", sep="\t").dropna(subset=PCA_VARIABLES)
    x[PCA_VARIABLES] = (x[PCA_VARIABLES] - x[PCA_VARIABLES].mean()) / x[PCA_VARIABLES].std(ddof=1)
    x.to_csv(args.out / "standardized_matrix.tsv", sep="\t", index=False)
    if args.plot:
        draw(args.out, scores, explained)
