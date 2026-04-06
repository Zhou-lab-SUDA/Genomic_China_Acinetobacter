#!/usr/bin/env python3

import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import mannwhitneyu
import ete3_extensions


def tip_pairs(tree, dists, clade_info):
    data = []
    missing = {"Unknown", "Pingtung", "Taipei", "Hsinchu"}

    for node in tree.traverse("postorder"):
        if node.is_leaf():
            node.d = [[node, node.annotations["date"]]]
            continue

        mrca_year = int(node.annotations["date"])

        for i, child1 in enumerate(node.children):
            for child0 in node.children[:i]:
                for tip0, year0 in child0.d:
                    for tip1, year1 in child1.d:
                        state0 = tip0.annotations["state"]
                        state1 = tip1.annotations["state"]

                        if state0 in missing or state1 in missing:
                            continue

                        key = tuple(sorted([state0, state1]))
                        geo_dist = dists.get(key, None) if state0 != state1 else 0.0
                        if geo_dist is None:
                            continue

                        clade0 = clade_info.get(tip0.name)
                        clade1 = clade_info.get(tip1.name)

                        if clade0 == "2.4" and clade1 == "2.4":
                            data.append([year0 + year1 - 2 * mrca_year, geo_dist, mrca_year, year0, year1, 0])
                        elif clade0 == "2.5" and clade1 == "2.5":
                            data.append([year0 + year1 - 2 * mrca_year, geo_dist, mrca_year, year0, year1, 1])

        node.d = [item for child in node.children for item in child.d]

    return np.array(data, dtype=float)


def estimate_speeds_smooth(observations, year_range, lambda_smooth=1.0, lambda_l2=0.1, n_bootstrap=100):
    year_to_idx = {year: i for i, year in enumerate(year_range)}
    n_years = len(year_range)
    n_obs = len(observations)

    involvement = np.zeros((n_obs, n_years), dtype=float)
    geo_dists = observations[:, 1].astype(float) + 1.0
    mrca = observations[:, 2].astype(int)
    year0 = observations[:, 3].astype(int)
    year1 = observations[:, 4].astype(int)

    weights = 1.0 / (observations[:, 0].astype(float) + 1.0)
    weights /= np.mean(weights)

    for i in range(n_obs):
        years_involved = list(range(mrca[i], year0[i])) + list(range(mrca[i], year1[i]))
        if not years_involved:
            years_involved = [mrca[i]]
        for y in years_involved:
            idx = year_to_idx.get(y)
            if idx is not None:
                involvement[i, idx] += 1.0

    def objective(speeds):
        predicted = involvement.dot(speeds)
        residuals = predicted - geo_dists
        fit_error = np.mean(weights * residuals ** 2)
        smooth_penalty = lambda_smooth * np.sum(np.diff(speeds) ** 2)
        l2_penalty = lambda_l2 * np.sum(speeds ** 2)
        return fit_error + smooth_penalty + l2_penalty

    result = minimize(
        objective,
        x0=np.ones(n_years) * 10.0,
        method="L-BFGS-B",
        bounds=[(0, None)] * n_years
    )
    speeds = result.x

    rng = np.random.default_rng(42)
    bootstrap_speeds = np.zeros((n_bootstrap, n_years))

    for b in range(n_bootstrap):
        idx = rng.integers(0, n_obs, size=n_obs)
        inv_b = involvement[idx]
        geo_b = geo_dists[idx]
        w_b = weights[idx]

        def objective_boot(s):
            predicted = inv_b.dot(s)
            residuals = predicted - geo_b
            fit_error = np.mean(w_b * residuals ** 2)
            smooth_penalty = lambda_smooth * np.sum(np.diff(s) ** 2)
            l2_penalty = lambda_l2 * np.sum(s ** 2)
            return fit_error + smooth_penalty + l2_penalty

        result_boot = minimize(
            objective_boot,
            x0=speeds,
            method="L-BFGS-B",
            bounds=[(0, None)] * n_years
        )
        bootstrap_speeds[b] = result_boot.x

    uncertainties = np.std(bootstrap_speeds, axis=0)
    counts = np.sum(involvement > 0, axis=0)

    return speeds, uncertainties, counts


def load_distance_matrix(path):
    dists = {}
    with open(path, "rt", encoding="utf-8") as fin:
        for line in fin:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            dists[tuple(sorted([parts[0], parts[1]]))] = float(parts[2])
    return dists


def load_clade_info(path):
    df = pd.read_csv(path, sep="\t")
    return {genome: clade for genome, clade in df[["genome", "clade"]].values}


def estimate_group(data, year_range, smooth, bootstrap):
    speeds, uncertainties, counts = estimate_speeds_smooth(
        data, year_range, lambda_smooth=smooth, n_bootstrap=bootstrap
    )
    ci_low = np.maximum(0, speeds - 1.96 * uncertainties)
    ci_high = speeds + 1.96 * uncertainties
    return {
        "years": np.array(year_range, dtype=int),
        "speeds": speeds,
        "unc": uncertainties,
        "counts": counts,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "cum": np.cumsum(speeds),
        "cum_low": np.cumsum(ci_low),
        "cum_high": np.cumsum(ci_high),
    }


def p_to_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def draw_bracket(ax, x, y0, y1, text, w=2.0, lw=0.8):
    ax.plot([x, x + w, x + w, x], [y0, y0, y1, y1], color="black", lw=lw, clip_on=False)
    ax.text(x + w + 1.2, (y0 + y1) / 2, text, va="center", ha="left", fontsize=8)


def plot_speed_summary(ax, res24, res25):
    periods = {
        "2010-2019": (2010, 2019),
        "2020-2023": (2020, 2023),
    }

    y_positions = {
        ("2.4", "2020-2023"): 3.0,
        ("2.4", "2010-2019"): 2.0,
        ("2.5", "2020-2023"): 1.0,
        ("2.5", "2010-2019"): 0.0,
    }

    colors = {"2.4": "#c96f6f", "2.5": "#eda83b"}

    def get_period_values(res, start, end):
        mask = (res["years"] >= start) & (res["years"] <= end)
        vals = res["speeds"][mask]
        years = res["years"][mask]
        return years, vals

    stats_map = {}

    for clade, res in [("2.4", res24), ("2.5", res25)]:
        for label, (start, end) in periods.items():
            years, vals = get_period_values(res, start, end)
            y = y_positions[(clade, label)]
            jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])

            ax.scatter(vals, np.full(len(vals), y) + jitter, s=14, color=colors[clade], alpha=0.7, edgecolor="none")
            mean_v = vals.mean()
            sem_v = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            ci_low = max(0.0, mean_v - 1.96 * sem_v)
            ci_high = mean_v + 1.96 * sem_v

            ax.plot([ci_low, ci_high], [y, y], color="black", lw=1.2)
            ax.scatter([mean_v], [y], s=70, color=colors[clade], edgecolor="black", linewidth=0.4, zorder=3)
            ax.text(ci_high + 6, y + 0.02, f"{mean_v:.1f}km/year", ha="left", va="center", fontsize=10)

            stats_map[(clade, label)] = vals

    ax.set_yticks([3.0, 2.0, 1.0, 0.0])
    ax.set_yticklabels(["2020-2023", "2010-2019", "2020-2023", "2010-2019"])
    ax.set_xlabel("Spread speed (km/year)", fontsize=12)
    ax.set_xlim(0, max(np.max(res24["speeds"]), np.max(res25["speeds"])) * 1.55)
    ax.set_ylim(-0.5, 3.5)

    ax.text(-0.16, 0.76, "ESL2.4", transform=ax.transAxes, fontsize=12, ha="right", va="center")
    ax.text(-0.16, 0.24, "ESL2.5", transform=ax.transAxes, fontsize=12, ha="right", va="center")
    ax.axhline(1.5, color="#dddddd", lw=0.8)

    p24 = mannwhitneyu(stats_map[("2.4", "2010-2019")], stats_map[("2.4", "2020-2023")], alternative="two-sided").pvalue
    p25 = mannwhitneyu(stats_map[("2.5", "2010-2019")], stats_map[("2.5", "2020-2023")], alternative="two-sided").pvalue
    p_2010 = mannwhitneyu(stats_map[("2.4", "2010-2019")], stats_map[("2.5", "2010-2019")], alternative="two-sided").pvalue
    p_2020 = mannwhitneyu(stats_map[("2.4", "2020-2023")], stats_map[("2.5", "2020-2023")], alternative="two-sided").pvalue

    x0 = ax.get_xlim()[1] * 0.82
    draw_bracket(ax, x0, 2.0, 3.0, p_to_stars(p24))
    draw_bracket(ax, x0 + 12, 0.0, 1.0, p_to_stars(p25))
    draw_bracket(ax, x0 + 24, 1.0, 3.0, p_to_stars(p_2020))
    draw_bracket(ax, x0 + 36, 0.0, 2.0, p_to_stars(p_2010))

    ax.text(-0.13, 1.03, "c", transform=ax.transAxes, fontsize=24, fontweight="bold", ha="left", va="top")


def plot_cumulative_distance(ax, res24, res25, res_all):
    colors = {
        "2.4": "#d08a8a",
        "2.5": "#eda83b",
        "all": "#bdbdbd",
    }

    x = res_all["years"] - res_all["years"].min()

    for res, color, label in [
        (res24, colors["2.4"], "ESL2.4"),
        (res25, colors["2.5"], "ESL2.5"),
        (res_all, colors["all"], "ESL All"),
    ]:
        ax.plot(x, res["cum"], lw=2.0, color=color, label=label)
        ax.fill_between(x, res["cum_low"], res["cum_high"], color=color, alpha=0.35, linewidth=0)

    ax.set_xlabel("Evolutionary time (year)", fontsize=12)
    ax.set_ylabel("Distance (km)", fontsize=12)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, max(res24["cum_high"].max(), res25["cum_high"].max(), res_all["cum_high"].max()) * 1.02)
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    ax.text(0.00, 1.07, "China", transform=ax.transAxes, fontsize=12, ha="left", va="bottom")
    ax.text(-0.13, 1.03, "d", transform=ax.transAxes, fontsize=24, fontweight="bold", ha="left", va="top")


@click.command()
@click.option("-n", "--nexus", required=True, help="Annotated Nexus tree.")
@click.option("-d", "--distance", required=True, help="Geographic distance matrix.")
@click.option("-c", "--clade", required=True, help="Genome-to-clade table.")
@click.option("-s", "--smooth", default=1.0, type=float, show_default=True, help="Smoothness penalty.")
@click.option("-b", "--bootstrap", default=100, type=int, show_default=True, help="Bootstrap replicates.")
@click.option("-o", "--output", default="spread_speed_panels_cd.png", show_default=True, help="Output figure path.")
def main(nexus, distance, clade, smooth, bootstrap, output):
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.linewidth"] = 1.0

    dists = load_distance_matrix(distance)
    clade_info = load_clade_info(clade)
    tree = ete3_extensions.read_nexus(nexus)[0]

    data = tip_pairs(tree, dists, clade_info)
    data = data[(data[:, 2] >= 2010) & (data[:, 0] <= 10)]  # keep recent observations and short evolutionary spans

    year_range = list(range(2010, 2024))

    data_24 = data[data[:, 5] == 0, :5]
    data_25 = data[data[:, 5] == 1, :5]
    data_all = data[:, :5]

    res24 = estimate_group(data_24, year_range, smooth, bootstrap)
    res25 = estimate_group(data_25, year_range, smooth, bootstrap)
    res_all = estimate_group(data_all, year_range, smooth, bootstrap)

    fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.2), dpi=300)
    plot_speed_summary(axes[0], res24, res25)
    plot_cumulative_distance(axes[1], res24, res25, res_all)

    plt.subplots_adjust(left=0.18, right=0.97, top=0.97, bottom=0.10, hspace=0.52)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
