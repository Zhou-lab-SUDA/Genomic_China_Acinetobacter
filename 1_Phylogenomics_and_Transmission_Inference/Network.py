#!/usr/bin/env python3

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import community as community_louvain


INPUT_FILE = "China_Transmission_Normalization.tsv"
OUTPUT_FIG = "Figure_module_scan_connection_and_city_level.png"
OUTPUT_SCAN = "Louvain_resolution_scan.tsv"
OUTPUT_ASSIGNMENT = "City_module_assignment_resolution_1.0.tsv"

RESOLUTION_VALUES = np.arange(0.0, 1.01, 0.05)
FINAL_RESOLUTION = 1.0
RANDOM_STATE = 42

MEGA_CITIES = {
    "Beijing", "Shanghai", "Guangzhou", "Shenzhen",
}

PROVINCIAL_CAPITALS = {
    "Shijiazhuang", "Taiyuan", "Hohhot", "Shenyang", "Changchun", "Harbin", "Chongqing", "Tianjin",
    "Nanjing", "Hangzhou", "Hefei", "Fuzhou", "Nanchang", "Jinan",
    "Zhengzhou", "Wuhan", "Changsha", "Nanning", "Haikou", "Chengdu",
    "Guiyang", "Kunming", "Lhasa", "Xi'an", "Lanzhou", "Xining",
    "Yinchuan", "Urumqi", "Suzhou"
}


def configure_plot_style():
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["font.size"] = 8


def load_transmission_network(path: str) -> tuple[pd.DataFrame, nx.Graph]:
    df = pd.read_csv(path, sep="\t")
    df = df[["Source", "Target", "Weight"]].dropna()
    df["Weight"] = df["Weight"].astype(float)

    graph = nx.Graph()
    graph.add_weighted_edges_from(df[["Source", "Target", "Weight"]].values.tolist())
    return df, graph


def city_level(city: str) -> str:
    if city in MEGA_CITIES:
        return "Mega city"
    if city in PROVINCIAL_CAPITALS:
        return "Provincial"
    return "Non-provincial"


def scan_louvain_resolution(
    graph: nx.Graph,
    resolution_values: np.ndarray,
    weight_key: str = "weight",
    random_state: int = 42
) -> pd.DataFrame:
    results = []

    for resolution in resolution_values:
        partition = community_louvain.best_partition(
            graph,
            resolution=float(resolution),
            weight=weight_key,
            random_state=random_state
        )

        modularity_q = community_louvain.modularity(
            partition,
            graph,
            weight=weight_key
        )

        n_clusters = len(set(partition.values()))
        results.append((float(resolution), n_clusters, modularity_q))

    return pd.DataFrame(results, columns=["Resolution", "Clusters", "Modularity"])


def get_partition(
    graph: nx.Graph,
    resolution: float,
    weight_key: str = "weight",
    random_state: int = 42
) -> dict:
    return community_louvain.best_partition(
        graph,
        resolution=resolution,
        weight=weight_key,
        random_state=random_state
    )


def summarize_connection_types(
    df: pd.DataFrame,
    partition: dict
) -> tuple[np.ndarray, np.ndarray, float]:
    data = df.copy()
    data["Source_module"] = data["Source"].map(partition)
    data["Target_module"] = data["Target"].map(partition)
    data = data.dropna(subset=["Source_module", "Target_module"])

    data["Connection"] = np.where(
        data["Source_module"] == data["Target_module"],
        "Within",
        "Cross module"
    )

    within_vals = data.loc[data["Connection"] == "Within", "Weight"].values
    cross_vals = data.loc[data["Connection"] == "Cross module", "Weight"].values

    _, p_value = mannwhitneyu(within_vals, cross_vals, alternative="two-sided")
    return within_vals, cross_vals, p_value


def summarize_cross_within_level(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["Source_level"] = data["Source"].map(city_level)
    data["Target_level"] = data["Target"].map(city_level)

    data["Level_relation"] = np.where(
        data["Source_level"] == data["Target_level"],
        "Within-level",
        "Cross-level"
    )

    total_weight = data["Weight"].sum()
    summary = (
        data.groupby("Level_relation", as_index=False)["Weight"]
        .sum()
        .rename(columns={"Weight": "Total_weight"})
    )
    summary["Percent"] = summary["Total_weight"] / total_weight * 100.0

    order = ["Cross-level", "Within-level"]
    summary["Level_relation"] = pd.Categorical(summary["Level_relation"], categories=order, ordered=True)
    summary = summary.sort_values("Level_relation").reset_index(drop=True)
    return summary


def summarize_weight_per_city(graph: nx.Graph) -> pd.DataFrame:
    weighted_degree = dict(graph.degree(weight="weight"))

    city_df = pd.DataFrame({
        "City": list(weighted_degree.keys()),
        "Transmission_weight": list(weighted_degree.values())
    })
    city_df["City_level"] = city_df["City"].map(city_level)

    summary = (
        city_df.groupby("City_level", as_index=False)["Transmission_weight"]
        .mean()
        .rename(columns={"Transmission_weight": "Mean_weight_per_city"})
    )

    order = ["Mega city", "Provincial", "Non-provincial"]
    summary["City_level"] = pd.Categorical(summary["City_level"], categories=order, ordered=True)
    summary = summary.sort_values("City_level").reset_index(drop=True)
    return summary


def p_to_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def plot_panel_b(ax, scan_df: pd.DataFrame):
    ax.set_box_aspect(1)

    line1 = ax.plot(
        scan_df["Resolution"],
        scan_df["Clusters"],
        "o-",
        color="#2c3969",
        linewidth=1.5,
        markersize=3,
        label="Num. of clusters"
    )[0]

    ax.set_xlabel("Resolution (γ)", fontsize=10)
    ax.set_ylabel("Num. of clusters", fontsize=10)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 25)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2, linestyle="-")

    ax_right = ax.twinx()
    ax_right.set_box_aspect(1)

    line2 = ax_right.plot(
        scan_df["Resolution"],
        scan_df["Modularity"],
        "o-",
        color="#00bfff",
        linewidth=1.5,
        markersize=3,
        label="Modularity"
    )[0]

    ax_right.set_ylabel("Modularity (Q)", fontsize=10)
    ax_right.set_ylim(0, 0.7)
    ax_right.tick_params(labelsize=8)

    ax.legend(
        [line1, line2],
        ["Num. of clusters", "Modularity"],
        loc="lower left",
        fontsize=8,
        frameon=False,
        ncol=1
    )



def plot_panel_c(ax, within_vals: np.ndarray, cross_vals: np.ndarray, p_value: float):
    ax.set_box_aspect(1)

    violin = ax.violinplot(
        [within_vals, cross_vals],
        positions=[1, 2],
        widths=0.23,
        showextrema=False
    )

    for body in violin["bodies"]:
        body.set_facecolor("none")
        body.set_edgecolor("black")
        body.set_linewidth(1.0)

    rng = np.random.default_rng(RANDOM_STATE)
    ax.scatter(rng.normal(1, 0.05, len(within_vals)), within_vals, s=3, color="black", alpha=0.5)
    ax.scatter(rng.normal(2, 0.05, len(cross_vals)), cross_vals, s=3, color="black", alpha=0.5)

    ax.set_xlabel("Connections", fontsize=10)
    ax.set_ylabel("Transmission weight", fontsize=10)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Within", "Cross module"])
    ax.set_ylim(0, 60)
    ax.tick_params(labelsize=8)

    sig_label = p_to_stars(p_value)
    ax.plot([1, 1, 2, 2], [59, 60, 60, 59], "k-", linewidth=0.6)
    ax.text(1.5, 62, sig_label, ha="center", fontweight="bold", fontsize=10)



def plot_panel_d(ax, summary_d: pd.DataFrame):
    ax.set_box_aspect(1)

    colors = ["#6f99c1", "#f29b43"]
    x = np.arange(len(summary_d))

    ax.bar(
        x,
        summary_d["Percent"].values,
        color=colors,
        width=0.38
    )

    ax.set_ylabel("Transmission events (%)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_d["Level_relation"].tolist(), rotation=45, ha="right")
    ax.set_ylim(0, max(85, summary_d["Percent"].max() + 5))
    ax.tick_params(labelsize=8)



def plot_panel_e(ax, summary_e: pd.DataFrame):
    ax.set_box_aspect(1)

    x = np.arange(len(summary_e))
    ax.bar(
        x,
        summary_e["Mean_weight_per_city"].values,
        color="#9ea0a3",
        width=0.42
    )

    ax.set_ylabel("Transmission weight per city", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_e["City_level"].tolist(), rotation=45, ha="right")
    ax.set_ylim(0, max(80, summary_e["Mean_weight_per_city"].max() + 5))
    ax.tick_params(labelsize=8)



def main():
    configure_plot_style()

    edge_df, graph = load_transmission_network(INPUT_FILE)

    scan_df = scan_louvain_resolution(
        graph=graph,
        resolution_values=RESOLUTION_VALUES,
        random_state=RANDOM_STATE
    )
    scan_df.to_csv(OUTPUT_SCAN, sep="\t", index=False)

    final_partition = get_partition(
        graph=graph,
        resolution=FINAL_RESOLUTION,
        random_state=RANDOM_STATE
    )

    assignment_df = pd.DataFrame(
        sorted(final_partition.items(), key=lambda x: (x[1], x[0])),
        columns=["City", "Module"]
    )
    assignment_df.to_csv(OUTPUT_ASSIGNMENT, sep="\t", index=False)

    within_vals, cross_vals, p_value = summarize_connection_types(
        df=edge_df,
        partition=final_partition
    )

    summary_d = summarize_cross_within_level(edge_df)
    summary_e = summarize_weight_per_city(graph)

    fig, axes = plt.subplots(2, 2, figsize=(6, 6), dpi=300)

    plot_panel_b(axes[0, 0], scan_df)
    plot_panel_c(axes[0, 1], within_vals, cross_vals, p_value)
    plot_panel_d(axes[1, 0], summary_d)
    plot_panel_e(axes[1, 1], summary_e)

    plt.subplots_adjust(left=0.12, right=0.97, top=0.96, bottom=0.12, hspace=0.42, wspace=0.42)
    plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
