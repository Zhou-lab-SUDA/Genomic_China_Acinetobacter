"""City-level genomic sampling sensitivity for Supplementary Fig. 3a-c."""
import argparse
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

DATA = Path(__file__).resolve().parents[1] / "data"
SEED = 20261028
N_BOOT = 5000
GROUPS = [
    ("0 ≤ N < 20", lambda d: d.n_major_lineages.lt(20)),
    ("20 ≤ N < 50", lambda d: d.n_major_lineages.between(20, 49)),
    ("N ≥ 50", lambda d: d.n_major_lineages.ge(50)),
]
COLORS = {"ESL2.4": "#2878B5", "ESL2.5": "#D95F59"}
def bootstrap_spearman(x, y, n_boot=N_BOOT, seed=SEED):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    vals = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.unique(x[idx]).size > 1 and np.unique(y[idx]).size > 1:
            vals.append(stats.spearmanr(x[idx], y[idx]).statistic)
    return np.quantile(vals, [0.025, 0.975])



def bootstrap_mean(values, seed, n_boot=100000):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float)
    idx = rng.integers(0, len(values), (n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return np.quantile(means, [.025, .975])


def paired_signflip_p(differences, seed, monte_carlo=1000000):
    d = np.asarray(differences, float)
    observed = abs(d.mean())
    n = len(d)
    if n <= 20:
        exceed = 0
        total = 2 ** n
        for signs in itertools.product((-1, 1), repeat=n):
            exceed += abs(np.mean(d * np.asarray(signs))) >= observed - 1e-15
        return exceed / total, total, "exact sign-flip"
    rng = np.random.default_rng(seed)
    exceed = 0
    done = 0
    chunk = 10000
    while done < monte_carlo:
        size = min(chunk, monte_carlo - done)
        signs = rng.choice((-1, 1), size=(size, n))
        values = np.abs((signs * d).mean(axis=1))
        exceed += np.count_nonzero(values >= observed - 1e-15)
        done += size
    return (exceed + 1) / (monte_carlo + 1), monte_carlo, "Monte Carlo sign-flip"


def bh_adjust(p_values):
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0, 1)
    return out


def analyse(df):
    rows = []
    for index, (label, selector) in enumerate(GROUPS):
        sub = df.loc[selector(df)].copy()
        v24 = sub.esl24_prevalence_fig3.to_numpy(float)
        v25 = sub.esl25_prevalence_fig3.to_numpy(float)
        diff = v25 - v24
        ci24 = bootstrap_mean(v24, 20261026 + index * 10)
        ci25 = bootstrap_mean(v25, 20261027 + index * 10)
        cid = bootstrap_mean(diff, 20261028 + index * 10)
        p, permutations, method = paired_signflip_p(diff, 20261126 + index)
        rows.append({
            "sampling_group": label,
            "n_cities": len(sub),
            "mean_esl24_prevalence": v24.mean(),
            "esl24_mean_ci95_low_bootstrap": ci24[0],
            "esl24_mean_ci95_high_bootstrap": ci24[1],
            "mean_esl25_prevalence": v25.mean(),
            "esl25_mean_ci95_low_bootstrap": ci25[0],
            "esl25_mean_ci95_high_bootstrap": ci25[1],
            "mean_paired_difference_esl25_minus_esl24": diff.mean(),
            "paired_difference_ci95_low_bootstrap": cid[0],
            "paired_difference_ci95_high_bootstrap": cid[1],
            "paired_signflip_p": p,
            "permutation_method": method,
            "permutations": permutations,
        })
    results = pd.DataFrame(rows)
    results["paired_signflip_q_BH"] = bh_adjust(results.paired_signflip_p)
    return results



def summarize(data):
    known = data.dropna(subset=["degree"])
    rows = []
    subsets = [("All cities", known)] + [(name, known.loc[mask(known)]) for name, mask in GROUPS]
    pairs = [("Degree vs sampling depth", "All cities", known, "n_major_lineages", SEED)]
    pairs += [("Degree vs ESL2.5 prevalence", name, subset, "esl25_prevalence_fig3",
               SEED + (1 if name == "All cities" else len(subset))) for name, subset in subsets]
    for label, name, subset, col, seed in pairs:
        x, y = subset.degree, subset[col]
        result = stats.spearmanr(x, y)
        low, high = bootstrap_spearman(x, y, seed=seed)
        rows.append((label, name, len(subset), result.statistic, low, high, result.pvalue))
    return pd.DataFrame(rows, columns=["analysis", "sampling_group", "n_cities", "spearman_rho",
                                      "ci95_low_bootstrap", "ci95_high_bootstrap", "p_value"])


def star(q):
    if q < .001:
        return "***"
    if q < .01:
        return "**"
    if q < .05:
        return "*"
    return "ns"


def plot(df, results, out):
    plt.rcParams.update({"font.family": "Arial", "font.size": 9.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.8), sharey=True)
    rng = np.random.default_rng(20261026)
    for index, (ax, (label, selector)) in enumerate(zip(axes, GROUPS)):
        sub = df.loc[selector(df)].copy()
        row = results.iloc[index]
        jitter = rng.uniform(-.055, .055, len(sub))
        for j, (_, city) in enumerate(sub.iterrows()):
            x0, x1 = 0 + jitter[j], 1 + jitter[j]
            y0, y1 = city.esl24_prevalence_fig3, city.esl25_prevalence_fig3
            ax.plot([x0, x1], [y0, y1], color="#B8B8B8", lw=.65, alpha=.48, zorder=1)
            ax.scatter(x0, y0, s=20, color=COLORS["ESL2.4"], alpha=.62,
                       edgecolor="white", linewidth=.35, zorder=2)
            ax.scatter(x1, y1, s=20, color=COLORS["ESL2.5"], alpha=.62,
                       edgecolor="white", linewidth=.35, zorder=2)
        means = [row.mean_esl24_prevalence, row.mean_esl25_prevalence]
        lows = [row.esl24_mean_ci95_low_bootstrap, row.esl25_mean_ci95_low_bootstrap]
        highs = [row.esl24_mean_ci95_high_bootstrap, row.esl25_mean_ci95_high_bootstrap]
        for x, mean, low, high, lineage in zip([0, 1], means, lows, highs, ["ESL2.4", "ESL2.5"]):
            ax.errorbar(x, mean, yerr=[[mean - low], [high - mean]], fmt="D",
                        ms=7, color=COLORS[lineage], ecolor=COLORS[lineage],
                        elinewidth=2, capsize=4, markeredgecolor="white",
                        markeredgewidth=.7, zorder=4)
            ax.text(x, high + .025, f"{mean:.1%}", ha="center", va="bottom",
                    fontsize=8.5, weight="bold", color=COLORS[lineage])
        bracket_y = 1.09
        ax.plot([0, 0, 1, 1], [bracket_y - .02, bracket_y, bracket_y, bracket_y - .02],
                color="#333333", lw=1)
        q = row.paired_signflip_q_BH
        ax.text(.5, bracket_y + .012, f"{star(q)}  q={q:.3g}", ha="center", va="bottom",
                fontsize=9, weight="bold")
        ax.set_title(f"{label}\nn={int(row.n_cities)} cities", fontsize=10.5, weight="bold")
        ax.set_xticks([0, 1], ["ESL2.4", "ESL2.5"])
        ax.set_xlim(-.35, 1.35)
        ax.set_ylim(-.05, 1.18)
        ax.yaxis.set_major_formatter(lambda value, pos: f"{value:.0%}")
        ax.grid(axis="y", color="#E8E8E8", lw=.7, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-.13, 1.06, "abc"[index], transform=ax.transAxes,
                fontsize=15, weight="bold")
    axes[0].set_ylabel("City-level prevalence (Figure 3)")
    fig.suptitle("City-equal mean lineage prevalence by sampling depth",
                 fontsize=13, weight="bold", y=.99)
    fig.text(.5, .025,
             "Thin lines connect lineage prevalences within each city; diamonds show city-equal means with 95% bootstrap CIs. Significance uses paired sign-flip tests with BH correction.",
             ha="center", fontsize=8.2, color="#506078")
    fig.subplots_adjust(left=.08, right=.99, top=.78, bottom=.16, wspace=.18)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"mean_lineage_prevalence_three_sampling_groups.{ext}",
                    dpi=450 if ext == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)
def plot_spearman(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    known = data.dropna(subset=["degree"])
    axes[0].scatter(np.log10(known.n_major_lineages + 1), known.degree)
    axes[0].set(xlabel="log10(major-lineage genome count + 1)", ylabel="City network degree")
    for (name, mask), color in zip(GROUPS, ["#287F9A", "#DC8251", "#79569C"]):
        subset = known.loc[mask(known)]
        x, y = subset.degree, subset.esl25_prevalence_fig3
        rho = stats.spearmanr(x, y)
        axes[1].scatter(x, y, color=color, label=f"{name}: ρ={rho.statistic:.3f}")
        grid = np.linspace(x.min(), x.max())
        axes[1].plot(grid, np.polyval(np.polyfit(x, y, 1), grid), color=color)
    axes[1].set(xlabel="City network degree", ylabel="ESL2.5 prevalence")
    axes[1].legend(frameon=False)
    fig.savefig(output, dpi=400)
    plt.close(fig)


def compare_depth_groups(data):
    """Independent-city permutation tests with BH across both lineages."""
    records = []
    for lineage_index, lineage in enumerate(("ESL2.4", "ESL2.5")):
        col = "esl24_prevalence_fig3" if lineage_index == 0 else "esl25_prevalence_fig3"
        values = [data.loc[mask(data), col].to_numpy() for _, mask in GROUPS]
        for pair_index, (i, j) in enumerate(((0, 1), (1, 2), (0, 2))):
            first, second = values[i], values[j]
            observed = first.mean() - second.mean()
            rng = np.random.default_rng(20261226 + lineage_index * 10 + pair_index)
            pooled = np.concatenate([first, second])
            extreme, n_permutations = 0, 500000
            for _ in range(0, n_permutations, 10000):
                shuffled = np.stack([rng.permutation(pooled) for _ in range(10000)])
                delta = shuffled[:, :len(first)].mean(axis=1) - shuffled[:, len(first):].mean(axis=1)
                extreme += np.count_nonzero(np.abs(delta) >= abs(observed) - 1e-15)
            records.append((lineage, GROUPS[i][0], GROUPS[j][0], observed,
                            (extreme + 1) / (n_permutations + 1), n_permutations))
    table = pd.DataFrame(records, columns=["lineage", "group_1", "group_2",
                                           "mean_difference", "permutation_p", "permutations"])
    table["BH_q"] = bh_adjust(table.permutation_p)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cities = pd.read_csv(DATA / "cities.tsv", sep="\t")
    summarize(cities).to_csv(args.out / "sampling_correlations.tsv", sep="\t", index=False)
    means = analyse(cities)
    means.to_csv(args.out / "sampling_group_means.tsv", sep="\t", index=False)
    compare_depth_groups(cities).to_csv(args.out / "within_lineage_pairwise.tsv", sep="\t", index=False)
    if args.plot:
        plot_spearman(cities, args.out / "supp_fig_3ab.svg")
        plot(cities, means, args.out)
