#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic Simulation Visualization Module

Extracted visualization functions from model_main_d.py
Generate plots:
1. fig1 - genotype_trajectories
2. fig2 - adaptation
3. fig3 - hospital_network
4. fig4 - genotypes_comparison
5. fig5 - adaptation_interactive
6. fig6 - hospital_network_state
7. fig7 - genotypes_radar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import json



# ========== calculate_genotypes_radar_metrics ==========

def calculate_genotypes_radar_metrics(genotype_freqs, hospital_types, TAU_VALUES, BETA_VALUES,
                                 time_series, transfer_matrix, fst_per_genotype, G):
    """
    Calculate radar chart metrics for metro-specialist and peripheral-specialist genotypes.
    Parameters:
        genotype_freqs: shape (n_hospitals, n_genotypes)
        hospital_types: List of hospital types
        TAU_VALUES, BETA_VALUES: Genotype parameters
        time_series: Time series data, containing 'overall_freqs', 'metro_freqs', 'periph_freqs'
        transfer_matrix: Transfer probability matrix
        fst_per_genotype: FST value for each genotype
        G: NetworkX directed graph (already created)

    Returns:
        dict: {
            'metro_genotypes': {dimension_name: value},
            'periph_genotypes': {dimension_name: value}
        }
    """
    metro_genotypes = [g for g in range(len(TAU_VALUES)) if TAU_VALUES[g] > 0.5]
    periph_genotypes = [g for g in range(len(TAU_VALUES)) if TAU_VALUES[g] <= 0.5]

    n_hospitals = len(hospital_types)
    metro_indices = [i for i, t in enumerate(hospital_types) if t == 'metro']
    periph_indices = [i for i, t in enumerate(hospital_types) if t == 'peripheral']

    # ========== 1. Metro Preference ==========
    def calc_metro_preference(genotype_list):
        if len(genotype_list) == 0:
            return 0.5 # neutral preference

        biases = []
        for g in genotype_list:
            metro_freq = np.mean(genotype_freqs[metro_indices, g])
            periph_freq = np.mean(genotype_freqs[periph_indices, g])
            total = metro_freq + periph_freq

            if total > 0.001:
                bias = (metro_freq - periph_freq) / total  # [-1, 1]
                biases.append((bias + 1) / 2)  # [0, 1]
            else:
                biases.append(0.5)

        return np.mean(biases)

    metro_genotypes_metro_pref = calc_metro_preference(metro_genotypes)
    periph_genotypes_metro_pref = calc_metro_preference(periph_genotypes)

    # ========== 2. Spread Speed ==========
    def calc_spread_speed_saturation(genotype_list):
        """
        Calculate spread speed based on saturation point in frequency trajectory.
        """
        if len(genotype_list) == 0:
            return 0

        speeds = []
        overall_freqs = time_series['overall_freqs']  # shape: (n_gen, n_genotypes)

        for g in genotype_list:
            freq_history = overall_freqs[:, g]

            if len(freq_history) > 10:
                initial_freq = freq_history[0]
                growth_rates = np.diff(freq_history)
                window_size = min(5, len(growth_rates) // 10)
                if window_size >= 3:
                    smoothed_rates = np.convolve(growth_rates, np.ones(window_size)/window_size, mode='valid')
                else:
                    smoothed_rates = growth_rates
                early_period = max(1, len(smoothed_rates) // 5)
                if early_period > 0:
                    initial_growth_rate = np.mean(smoothed_rates[:early_period])
                else:
                    initial_growth_rate = smoothed_rates[0] if len(smoothed_rates) > 0 else 0

                # find saturation point where growth rate drops below threshold
                saturation_threshold = max(0.0001, abs(initial_growth_rate) * 0.1)
                saturation_gen = None

                for t in range(len(smoothed_rates)):
                    if abs(smoothed_rates[t]) < saturation_threshold:
                        saturation_gen = t
                        break

                # If no saturation point is found, use the maximum point
                if saturation_gen is None or saturation_gen == 0:
                    saturation_gen = np.argmax(freq_history)

                # Calculate true speed (slope)
                saturation_freq = freq_history[saturation_gen]
                if saturation_gen > 0 and (saturation_freq - initial_freq) > 0.001:
                    speed = (saturation_freq - initial_freq) / saturation_gen
                else:
                    speed = 0
            else:
                speed = 0

            speeds.append(speed)

        return np.mean(speeds)

    # Calculate raw spread speed (Δfreq/generation)
    metro_genotypes_speed_raw = calc_spread_speed_saturation(metro_genotypes)
    periph_genotypes_speed_raw = calc_spread_speed_saturation(periph_genotypes)

    # Normalize speed to [0,1] range for radar chart display
    # Assume reasonable speed range is 0-0.02 (i.e., 0-2% growth per generation)
    # Values exceeding 0.02 will be capped at 1.0
    max_speed_for_scaling = 0.02
    metro_genotypes_speed = min(1.0, metro_genotypes_speed_raw / max_speed_for_scaling)
    periph_genotypes_speed = min(1.0, periph_genotypes_speed_raw / max_speed_for_scaling)

    # ========== 3. Spatial Differentiation ==========
    def calc_spatial_diff(genotype_list):
        if len(genotype_list) == 0:
            return 0
        fst_values = [fst_per_genotype[g] for g in genotype_list]
        return np.mean(fst_values)

    metro_genotypes_fst = calc_spatial_diff(metro_genotypes)
    periph_genotypes_fst = calc_spatial_diff(periph_genotypes)

    # ========== 4. Colonization Success ==========
    def calc_colonization_success(genotype_list, presence_threshold=0.05):
        if len(genotype_list) == 0:
            return 0

        success_rates = []
        for g in genotype_list:
            n_colonized = np.sum(genotype_freqs[:, g] > presence_threshold)
            success_rate = n_colonized / n_hospitals
            success_rates.append(success_rate)

        return np.mean(success_rates)

    metro_genotypes_colonization = calc_colonization_success(metro_genotypes)
    periph_genotypes_colonization = calc_colonization_success(periph_genotypes)

    # ========== 5. Network Centrality ==========
    def calc_network_centrality(genotype_list):
        if len(genotype_list) == 0:
            return 0
        pagerank_scores = nx.pagerank(G, weight='weight')
        centrality = 0
        for g in genotype_list:
            for hospital_id in range(n_hospitals):
                centrality += genotype_freqs[hospital_id, g] * pagerank_scores[hospital_id]
        centrality /= len(genotype_list)

        return centrality

    metro_genotypes_centrality = calc_network_centrality(metro_genotypes)
    periph_genotypes_centrality = calc_network_centrality(periph_genotypes)

    # ========== 6. Find Top Genotype and Metrics ==========
    def find_top_genotype_and_metrics(genotype_list):
        """Find the top genotype in the camp and calculate its metrics"""
        if len(genotype_list) == 0:
            return None, {}

        # Find the genotype with the highest overall frequency
        overall_freqs_final = time_series['overall_freqs'][-1, :]  # Last generation frequencies
        top_g = max(genotype_list, key=lambda g: overall_freqs_final[g])

        # Calculate metrics for this genotype
        metro_freq = np.mean(genotype_freqs[metro_indices, top_g])
        periph_freq = np.mean(genotype_freqs[periph_indices, top_g])
        total = metro_freq + periph_freq
        metro_pref = ((metro_freq - periph_freq) / total + 1) / 2 if total > 0.001 else 0.5

        # Spread speed (saturation point slope method, consistent with camp average)
        freq_history = time_series['overall_freqs'][:, top_g]
        if len(freq_history) > 10:
            initial_freq = freq_history[0]
            growth_rates = np.diff(freq_history)
            window_size = min(5, len(growth_rates) // 10)
            if window_size >= 3:
                smoothed_rates = np.convolve(growth_rates, np.ones(window_size)/window_size, mode='valid')
            else:
                smoothed_rates = growth_rates

            early_period = max(1, len(smoothed_rates) // 5)
            initial_growth_rate = np.mean(smoothed_rates[:early_period]) if early_period > 0 else smoothed_rates[0]
            saturation_threshold = max(0.0001, abs(initial_growth_rate) * 0.1)
            saturation_gen = None

            for t in range(len(smoothed_rates)):
                if abs(smoothed_rates[t]) < saturation_threshold:
                    saturation_gen = t
                    break

            if saturation_gen is None or saturation_gen == 0:
                saturation_gen = np.argmax(freq_history)

            saturation_freq = freq_history[saturation_gen]
            if saturation_gen > 0 and (saturation_freq - initial_freq) > 0.001:
                speed = (saturation_freq - initial_freq) / saturation_gen
            else:
                speed = 0
        else:
            speed = 0

        spatial_diff = fst_per_genotype[top_g]
        colonization = np.sum(genotype_freqs[:, top_g] > 0.05) / n_hospitals
        pagerank_scores = nx.pagerank(G, weight='weight')
        centrality = sum(genotype_freqs[h, top_g] * pagerank_scores[h] for h in range(n_hospitals))
        max_speed_for_scaling = 0.02
        speed_normalized = min(1.0, speed / max_speed_for_scaling)
        metrics = {
            'Metro Preference': metro_pref,
            'Spread Speed': speed_normalized,  # Normalized value
            'Spatial Differentiation': spatial_diff,
            'Colonization Success': colonization,
            'Network Centrality': centrality
        }
        return top_g, metrics, speed

    M_top_g, M_top_metrics, M_top_speed_raw = find_top_genotype_and_metrics(metro_genotypes)
    C_top_g, C_top_metrics, C_top_speed_raw = find_top_genotype_and_metrics(periph_genotypes)

    print(f"\nRadar chart metrics calculation completed:")
    print(f"  Speed calculation method: Saturation point slope method (Δfreq/generation)")
    print(f"  Speed normalization: Original speed/0.02 → [0,1] range for radar chart display")
    print(f"\n  Metro-oriented genotypes average (n={len(metro_genotypes)}):")
    print(f"    Metro Pref={metro_genotypes_metro_pref:.3f}, Speed={metro_genotypes_speed_raw:.6f}/gen ({metro_genotypes_speed_raw*100:.2f}%/gen, normalized={metro_genotypes_speed:.3f}), FST={metro_genotypes_fst:.3f}, Colonization={metro_genotypes_colonization:.3f}, Centrality={metro_genotypes_centrality:.4f}")
    if M_top_g is not None:
        print(f"  Metro-oriented genotypes top (G{M_top_g}):")
        print(f"    Metro Pref={M_top_metrics['Metro Preference']:.3f}, Speed={M_top_speed_raw:.6f}/gen ({M_top_speed_raw*100:.2f}%/gen, normalized={M_top_metrics['Spread Speed']:.3f}), FST={M_top_metrics['Spatial Differentiation']:.3f}, Colonization={M_top_metrics['Colonization Success']:.3f}, Centrality={M_top_metrics['Network Centrality']:.4f}")

    print(f"\n  Periph-oriented genotypes average (n={len(periph_genotypes)}):")
    print(f"    Metro Pref={periph_genotypes_metro_pref:.3f}, Speed={periph_genotypes_speed_raw:.6f}/gen ({periph_genotypes_speed_raw*100:.2f}%/gen, normalized={periph_genotypes_speed:.3f}), FST={periph_genotypes_fst:.3f}, Colonization={periph_genotypes_colonization:.3f}, Centrality={periph_genotypes_centrality:.4f}")
    if C_top_g is not None:
        print(f"  Periph-oriented genotypes top (G{C_top_g}):")
        print(f"    Metro Pref={C_top_metrics['Metro Preference']:.3f}, Speed={C_top_speed_raw:.6f}/gen ({C_top_speed_raw*100:.2f}%/gen, normalized={C_top_metrics['Spread Speed']:.3f}), FST={C_top_metrics['Spatial Differentiation']:.3f}, Colonization={C_top_metrics['Colonization Success']:.3f}, Centrality={C_top_metrics['Network Centrality']:.4f}")

    return {
        'metro_genotypes_avg': {
            'Metro Preference': metro_genotypes_metro_pref,
            'Spread Speed': metro_genotypes_speed,
            'Spatial Differentiation': metro_genotypes_fst,
            'Colonization Success': metro_genotypes_colonization,
            'Network Centrality': metro_genotypes_centrality
        },
        'metro_genotypes_top': {
            'genotype_id': M_top_g,
            'metrics': M_top_metrics
        },
        'periph_genotypes_avg': {
            'Metro Preference': periph_genotypes_metro_pref,
            'Spread Speed': periph_genotypes_speed,
            'Spatial Differentiation': periph_genotypes_fst,
            'Colonization Success': periph_genotypes_colonization,
            'Network Centrality': periph_genotypes_centrality
        },
        'periph_genotypes_top': {
            'genotype_id': C_top_g,
            'metrics': C_top_metrics
        }
    }


# ========== plot_genotypes_radar_chart ==========

def plot_genotypes_radar_chart(radar_metrics, output_path='genotypes_radar_chart.png'):
    """
    Plot radar chart comparing metro_genotypes vs periph_genotypes
    Includes genotype group averages + top genotype for each group

    Parameters:
        radar_metrics: dict, from calculate_genotypes_radar_metrics()
        output_path: output path for the radar chart image
    """
    categories = list(radar_metrics['metro_genotypes_avg'].keys())
    metro_genotypes_avg_values = list(radar_metrics['metro_genotypes_avg'].values())
    periph_genotypes_avg_values = list(radar_metrics['periph_genotypes_avg'].values())

    metro_genotypes_top_values = list(radar_metrics['metro_genotypes_top']['metrics'].values()) if radar_metrics['metro_genotypes_top']['metrics'] else metro_genotypes_avg_values
    periph_genotypes_top_values = list(radar_metrics['periph_genotypes_top']['metrics'].values()) if radar_metrics['periph_genotypes_top']['metrics'] else periph_genotypes_avg_values

    M_top_id = radar_metrics['metro_genotypes_top']['genotype_id']
    C_top_id = radar_metrics['periph_genotypes_top']['genotype_id']

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

    metro_genotypes_avg_values += metro_genotypes_avg_values[:1]
    periph_genotypes_avg_values += periph_genotypes_avg_values[:1]
    metro_genotypes_top_values += metro_genotypes_top_values[:1]
    periph_genotypes_top_values += periph_genotypes_top_values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))

    # Plot Metro-oriented genotypes average base line (blue solid)
    ax.plot(angles, metro_genotypes_avg_values, 'o-', linewidth=3.0, label='Metro-oriented Average',
            color='#3182ce', markersize=10, alpha=0.5)
    ax.fill(angles, metro_genotypes_avg_values, alpha=0.15, color='#3182ce')

    # Plot Metro-oriented genotypes top genotype (blue dashed)
    if M_top_id is not None:
        ax.plot(angles, metro_genotypes_top_values, 'o--', linewidth=2.0, label=f'Metro-oriented Top G{M_top_id}',
                color='#2c5aa0', markersize=6, alpha=0.7)

    # Plot Periph-oriented genotypes average (red solid)
    ax.plot(angles, periph_genotypes_avg_values, 's-', linewidth=3.0, label='Periph-oriented Average',
            color='#e53e3e', markersize=10, alpha=0.5)
    ax.fill(angles, periph_genotypes_avg_values, alpha=0.15, color='#e53e3e')

    # Plot Periph-oriented genotypes top genotype (red dashed)
    if C_top_id is not None:
        ax.plot(angles, periph_genotypes_top_values, 's--', linewidth=2.0, label=f'Periph-oriented Top G{C_top_id}',
                color='#c53030', markersize=6, alpha=0.7)

    # Set tick labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=7, fontweight='bold')

    # Set Y-axis limits and labels
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=7)
    ax.set_rlabel_position(0)

    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=7, frameon=True)
    plt.title('Ecological Strategy Profile\n(Genotype Group Average + Top Genotype)\nNote: Spread Speed normalized to [0,1], raw value = normalized × 0.02 /gen',
             fontsize=7, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Radar chart saved: {output_path}")


# ========== create_network_from_transfer_matrix ==========

def create_network_from_transfer_matrix(transfer_matrix, hospital_types,
                                       edge_threshold=0.01, top_k_per_node=None):
    """
    Create a network graph based on the real transfer probability matrix (using the actual transmission network in the simulation)

    Parameters:
        transfer_matrix: Transfer probability matrix (n_hospitals × n_hospitals), from simulation
        hospital_types: List of hospital types
        edge_threshold: Edge weight threshold, only show edges with weights greater than this value (default 0.01)
        top_k_per_node: If provided, only show top K strongest outgoing edges per node (reduces clutter in dense networks)

    Returns:
        G: NetworkX directed graph (since transfers are directional)
    """
    n_hospitals = len(hospital_types)
    G = nx.DiGraph()

    for i in range(n_hospitals):
        G.add_node(i, type=hospital_types[i])

    # Add edges based on threshold and/or top-k filter
    for i in range(n_hospitals):
        # Collect all potential edges from node i
        edges_from_i = []
        for j in range(n_hospitals):
            if i != j and transfer_matrix[i, j] > edge_threshold:
                edges_from_i.append((j, transfer_matrix[i, j]))

        # If top_k is specified, only keep top K strongest edges
        if top_k_per_node is not None and len(edges_from_i) > top_k_per_node:
            edges_from_i.sort(key=lambda x: x[1], reverse=True)
            edges_from_i = edges_from_i[:top_k_per_node]

        # Add selected edges to graph
        for j, weight in edges_from_i:
            G.add_edge(i, j, weight=weight)

    print(f"\nNetwork statistics:")
    print(f"  Total nodes: {G.number_of_nodes()}")
    print(f"  Total edges: {G.number_of_edges()}")
    print(f"  Average out-degree: {G.number_of_edges() / G.number_of_nodes():.2f}")

    return G


def classify_genotypes_by_strategy(TAU_VALUES, exclude_neutral=True):
    """
    Define M-genotypes and C-genotypes based on M values

    Parameters:
        TAU_VALUES: ndarray, M values for each genotype
        exclude_neutral: Whether to exclude neutral genotypes (background genotypes with M≈0.5)

    Returns:
        M_genotypes: M-genotypes list (2.5-like)
        C_genotypes: C-genotypes list (2.4-like)
    """
    if exclude_neutral:
        M_genotypes = [g for g in range(len(TAU_VALUES)) if TAU_VALUES[g] > 0.6]
        C_genotypes = [g for g in range(len(TAU_VALUES)) if TAU_VALUES[g] < 0.4]
    else:
        M_threshold = 0.5
        M_genotypes = [g for g in range(len(TAU_VALUES)) if TAU_VALUES[g] > M_threshold]
        C_genotypes = [g for g in range(len(TAU_VALUES)) if TAU_VALUES[g] <= M_threshold]

    return M_genotypes, C_genotypes


# ========== plot_hospital_network ==========

def plot_hospital_network(G, hospital_types, genotype_freqs, TAU_VALUES, BETA_VALUES,
                         output_path='hospital_network.png'):
    """
    Plot the hospital network graph. Nodes are colored based on dominant genotype camp:
    - Blue: metro_genotypes dominant
    - Red: periph_genotypes dominant
    - Purple: Mixed (near equal)
    - Gray: No dominant genotype

    Parameters:
        G: NetworkX
        hospital_types: List of hospital types
        genotype_freqs: shape (n_hospitals, n_genotypes)
        TAU_VALUES: Genotype M values
        BETA_VALUES: Genotype C values
        output_path: Output path for the network image
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    metro_genotypes, periph_genotypes = classify_genotypes_by_strategy(TAU_VALUES)
    print(f"\nGenotype classification:")
    print(f"  Metro-oriented genotypes (M>0.5): {metro_genotypes}")
    print(f"  Periph-oriented genotypes (M≤0.5): {periph_genotypes}")

    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
    node_colors = []
    node_sizes = []

    for i in range(len(hospital_types)):
        # Calculate genotype frequencies for this hospital
        metro_genotypes_freq = np.sum(genotype_freqs[i, metro_genotypes]) if len(metro_genotypes) > 0 else 0
        periph_genotypes_freq = np.sum(genotype_freqs[i, periph_genotypes]) if len(periph_genotypes) > 0 else 0

        if metro_genotypes_freq + periph_genotypes_freq < 0.1:
            color = '#cbd5e0'
        elif metro_genotypes_freq > periph_genotypes_freq * 2:
            color = '#3182ce'
        elif periph_genotypes_freq > metro_genotypes_freq * 2:
            color = '#e53e3e'
        else:
            color = "#6b4ab2"

        node_colors.append(color)
        node_sizes.append(800 if hospital_types[i] == 'metro' else 300)

    if G.is_directed():
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1.0
        edge_widths = [w / max_weight * 2.0 for w in weights]  # 归一化到0-2

        nx.draw_networkx_edges(G, pos, alpha=0.15, width=edge_widths,
                              arrows=True, arrowsize=10, arrowstyle='->',
                              edge_color='gray', ax=ax)
    else:
        nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5, ax=ax)

    metro_nodes = [i for i, t in enumerate(hospital_types) if t == 'metro']
    periph_nodes = [i for i, t in enumerate(hospital_types) if t == 'peripheral']

    nx.draw_networkx_nodes(G, pos, nodelist=metro_nodes,
                          node_color=[node_colors[i] for i in metro_nodes],
                          node_size=[node_sizes[i] for i in metro_nodes],
                          node_shape='s', edgecolors='black', linewidths=1.5, ax=ax)

    nx.draw_networkx_nodes(G, pos, nodelist=periph_nodes,
                          node_color=[node_colors[i] for i in periph_nodes],
                          node_size=[node_sizes[i] for i in periph_nodes],
                          node_shape='o', edgecolors='black', linewidths=1, ax=ax)
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#cbd5e0',
                  markersize=10, label='No Dominant Genotype'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3182ce',
                  markersize=10, label='metro_genotypes Dominant (2.5-like)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#e53e3e',
                  markersize=10, label='periph_genotypes Dominant (2.4-like)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#805ad5',
                  markersize=10, label='Mixed'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                  markersize=12, label='Metro Hospital', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                  markersize=8, label='Peripheral Hospital', markeredgecolor='black')
    ]

    ax.legend(handles=legend_elements, loc='upper left', fontsize=7)

    if G.is_directed():
        title = 'Hospital Transfer Network (Based on Simulation)\nArrows=Transfer Direction, Width=Probability\nSquares=Metro, Circles=Peripheral, Color=Genotype Dominance'
    else:
        title = 'Hospital Network\nSquares=Metro, Circles=Peripheral\nColor=Genotype Dominance'

    ax.set_title(title, fontsize=7, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Network graph saved: {output_path}")
    plt.close()


# ========== plot_clade_comparison_time_series ==========

def plot_clade_comparison_time_series(time_series, hospital_types, TAU_VALUES,
                                     output_path='clade_comparison.png'):
    """
    Plot time series comparing metro_genotypes vs periph_genotypes prevalence
    for large hospitals, small hospitals, and system-wide   
    Parameters:
        time_series: dict, 'metro_freqs', 'periph_freqs', 'overall_freqs'
    """
    metro_genotypes, periph_genotypes = classify_genotypes_by_strategy(TAU_VALUES)

    metro_freqs = time_series['metro_freqs']
    periph_freqs = time_series['periph_freqs']
    overall_freqs = time_series['overall_freqs']

    timesteps = np.arange(len(overall_freqs))

    # Each category's total frequency for the two genotype camps
    metro_metro_genotypes_freq = np.sum(metro_freqs[:, metro_genotypes], axis=1) if len(metro_genotypes) > 0 else np.zeros(len(timesteps))
    metro_periph_genotypes_freq = np.sum(metro_freqs[:, periph_genotypes], axis=1) if len(periph_genotypes) > 0 else np.zeros(len(timesteps))

    periph_metro_genotypes_freq = np.sum(periph_freqs[:, metro_genotypes], axis=1) if len(metro_genotypes) > 0 else np.zeros(len(timesteps))
    periph_periph_genotypes_freq = np.sum(periph_freqs[:, periph_genotypes], axis=1) if len(periph_genotypes) > 0 else np.zeros(len(timesteps))

    overall_metro_genotypes_freq = np.sum(overall_freqs[:, metro_genotypes], axis=1) if len(metro_genotypes) > 0 else np.zeros(len(timesteps))
    overall_periph_genotypes_freq = np.sum(overall_freqs[:, periph_genotypes], axis=1) if len(periph_genotypes) > 0 else np.zeros(len(timesteps))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ========== Subplot 1: Large Hospitals ==========
    ax = axes[0]
    ax.plot(timesteps, metro_metro_genotypes_freq, color='blue',
           linewidth=2, label='Metro-oriented genotypes')
    ax.plot(timesteps, metro_periph_genotypes_freq, color='red',
           linewidth=2, label='Periph-oriented genotypes')

    ax.set_xlabel('Time Steps', fontsize=7)
    ax.set_ylabel('Prevalence', fontsize=7)
    ax.set_title('Large Hospitals', fontsize=7, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

    # ========== Subplot 2: Small Hospitals ==========
    ax = axes[1]
    ax.plot(timesteps, periph_metro_genotypes_freq, color='blue',
           linewidth=2, label='Metro-oriented genotypes')
    ax.plot(timesteps, periph_periph_genotypes_freq, color='red',
           linewidth=2, label='Periph-oriented genotypes')

    ax.set_xlabel('Time Steps', fontsize=7)
    ax.set_ylabel('Prevalence', fontsize=7)
    ax.set_title('Small Hospitals', fontsize=7, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

    # ========== Subplot 3: System-Wide ==========
    ax = axes[2]
    ax.plot(timesteps, overall_metro_genotypes_freq, color='blue',
           linewidth=2, label='Metro-oriented genotypes')
    ax.plot(timesteps, overall_periph_genotypes_freq, color='red',
           linewidth=2, label='Periph-oriented genotypes')

    ax.set_xlabel('Time Steps', fontsize=7)
    ax.set_ylabel('Prevalence', fontsize=7)
    ax.set_title('System-Wide', fontsize=7, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Lineage comparison time series plot saved: {output_path}")


# ========== export_hospital_network_state ==========

def export_hospital_network_state(hospital_types, genotype_freqs, coordinates,
                                  TAU_VALUES, BETA_VALUES, output_path='hospital_network_state.csv'):
    """
    Export hospital network state to CSV for static network visualization.

    Generates a table with columns:
    - id: Hospital ID
    - size: Hospital size (randomly assigned based on type)
    - type: metro or peripheral
    - x, y: Spatial coordinates
    - outpatient_capacity, inpatient_capacity: Capacities (derived from size)
    - Clade_2_4_final_load: Final prevalence of C-genotypes (M<0.4)
    - Clade_2_5_final_load: Final prevalence of M-genotypes (M>0.6)

    Parameters:
        hospital_types: List of hospital types
        genotype_freqs: Final genotype frequencies (n_hospitals × n_genotypes)
        coordinates: Tuple of (x_coords, y_coords), or None if not available
        TAU_VALUES: M values for each genotype
        BETA_VALUES: C values for each genotype
        output_path: Path to save CSV file

    Returns:
        DataFrame with hospital network state
    """
    n_hospitals = len(hospital_types)

    # Classify genotypes
    M_genotypes = [g for g in range(len(TAU_VALUES)) if TAU_VALUES[g] > 0.6]
    C_genotypes = [g for g in range(len(TAU_VALUES)) if TAU_VALUES[g] < 0.4]

    # Calculate clade loads for each hospital
    clade_24_loads = []
    clade_25_loads = []

    for h in range(n_hospitals):
        # Clade 2.4: sum of C-genotypes (peripheral specialists)
        clade_24 = sum(genotype_freqs[h, g] for g in C_genotypes)
        clade_24_loads.append(clade_24)

        # Clade 2.5: sum of M-genotypes (metro specialists)
        clade_25 = sum(genotype_freqs[h, g] for g in M_genotypes)
        clade_25_loads.append(clade_25)

    # Generate hospital sizes (consistent with type)
    np.random.seed(42)  # For reproducibility
    sizes = []
    for h_type in hospital_types:
        if h_type == 'metro':
            size = np.random.randint(800, 2000)
        else:
            size = np.random.randint(50, 500)
        sizes.append(size)

    # Calculate capacities (as fractions of size)
    outpatient_capacities = [s * np.random.uniform(0.6, 0.9) for s in sizes]
    inpatient_capacities = [s * np.random.uniform(0.1, 0.3) for s in sizes]

    # Build dataframe
    data = {
        'id': list(range(n_hospitals)),
        'size': sizes,
        'type': hospital_types,
        'outpatient_capacity': outpatient_capacities,
        'inpatient_capacity': inpatient_capacities,
        'Clade_2_4_final_load': clade_24_loads,
        'Clade_2_5_final_load': clade_25_loads
    }

    # Add coordinates if available
    if coordinates is not None:
        x_coords, y_coords = coordinates
        data['x'] = x_coords
        data['y'] = y_coords
        # Reorder columns to match user's example
        column_order = ['id', 'size', 'type', 'x', 'y', 'outpatient_capacity',
                       'inpatient_capacity', 'Clade_2_4_final_load', 'Clade_2_5_final_load']
    else:
        column_order = ['id', 'size', 'type', 'outpatient_capacity',
                       'inpatient_capacity', 'Clade_2_4_final_load', 'Clade_2_5_final_load']

    df = pd.DataFrame(data)
    df = df[column_order]

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Hospital network state saved: {output_path}")

    return df


# ========== plot_hospital_network_state ==========

def plot_hospital_network_state(csv_path=None, hospital_types=None, genotype_freqs=None,
                                coordinates=None, TAU_VALUES=None, BETA_VALUES=None,
                                output_path='hospital_network_state.png'):
    """
    Generate static visualization of hospital network state.

    Creates a scatter plot showing:
    - Hospital positions (x, y coordinates)
    - Hospital types (square=metro with dark blue border, circle=peripheral with light blue border)
    - Hospital sizes (node size proportional to hospital size)
    - Clade prevalence (overlaid circles: pink=Clade 2.4, yellow=Clade 2.5)

    Can be called either:
    1. From CSV file: plot_hospital_network_state(csv_path='file4.hospital_network_state.csv')
    2. From data: plot_hospital_network_state(hospital_types=..., genotype_freqs=..., coordinates=...)

    Parameters:
        csv_path: Path to exported CSV file
        hospital_types: List of hospital types (if not using CSV)
        genotype_freqs: Genotype frequencies (if not using CSV)
        coordinates: Tuple of (x_coords, y_coords) (if not using CSV)
        TAU_VALUES: M values for genotypes (if not using CSV)
        BETA_VALUES: C values for genotypes (if not using CSV)
        output_path: Path to save the plot
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Circle
    import pandas as pd

    # Load data from CSV or use provided data
    if csv_path is not None:
        df = pd.read_csv(csv_path)
        x_coords = df['x'].values
        y_coords = df['y'].values
        hospital_types_list = df['type'].values
        sizes = df['size'].values
        outpatient_caps = df['outpatient_capacity'].values
        inpatient_caps = df['inpatient_capacity'].values
        clade_24_loads = df['Clade_2_4_final_load'].values
        clade_25_loads = df['Clade_2_5_final_load'].values
    else:
        if coordinates is None or hospital_types is None or genotype_freqs is None:
            raise ValueError("Must provide either csv_path or (hospital_types, genotype_freqs, coordinates)")

        x_coords, y_coords = coordinates
        hospital_types_list = hospital_types
        n_hospitals = len(hospital_types)

        # Classify genotypes
        M_genotypes = [g for g in range(len(TAU_VALUES)) if TAU_VALUES[g] > 0.6]
        C_genotypes = [g for g in range(len(TAU_VALUES)) if TAU_VALUES[g] < 0.4]

        # Calculate clade loads
        clade_24_loads = np.array([sum(genotype_freqs[h, g] for g in C_genotypes) for h in range(n_hospitals)])
        clade_25_loads = np.array([sum(genotype_freqs[h, g] for g in M_genotypes) for h in range(n_hospitals)])

        # Generate sizes
        np.random.seed(42)
        sizes = np.array([np.random.randint(800, 2000) if ht == 'metro' else np.random.randint(50, 500)
                         for ht in hospital_types_list])

        # Calculate capacities
        outpatient_caps = np.array([s * np.random.uniform(0.6, 0.9) for s in sizes])
        inpatient_caps = np.array([s * np.random.uniform(0.1, 0.3) for s in sizes])

    # Calculate total hospital capacity (outpatient + inpatient)
    total_capacities = outpatient_caps + inpatient_caps

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))

    # Normalize node sizes based on total hospital capacity
    # Using square root to reduce the visual size difference between large and small hospitals
    min_capacity = total_capacities.min()
    max_capacity = total_capacities.max()

    # Normalize to 0-1 range, then apply square root for better visual scaling
    normalized_caps = (total_capacities - min_capacity) / (max_capacity - min_capacity + 1e-9)
    sqrt_normalized = np.sqrt(normalized_caps)

    # Scale to visualization size range (min 100, max 1000 for scatter plot)
    min_node_size = 50
    max_node_size = 1000
    node_sizes = min_node_size + sqrt_normalized * (max_node_size - min_node_size)

    # Normalize clade loads for circle radius
    max_load = max(clade_24_loads.max(), clade_25_loads.max())
    if max_load > 0:
        clade_24_radius = (clade_24_loads / max_load) * 3.0  # Max radius 3 units
        clade_25_radius = (clade_25_loads / max_load) * 2.0  # Max radius 2 units (inner circle)
    else:
        clade_24_radius = np.zeros_like(clade_24_loads)
        clade_25_radius = np.zeros_like(clade_25_loads)

    # Plot each hospital
    for i in range(len(x_coords)):
        x, y = x_coords[i], y_coords[i]
        h_type = hospital_types_list[i]

        # Determine marker shape based on hospital type (no blue colors)
        # Node size is now directly based on normalized hospital capacity
        if h_type == 'metro':
            marker = 's'  # square
        else:
            marker = 'o'  # circle

        # Use consistent styling for all hospitals
        edge_color = "#4F52A4"  # dark gray/black
        face_color = '#4F52A4'  # white fill

        # Plot base hospital node (size already normalized by capacity)
        ax.scatter(x, y, s=node_sizes[i], marker=marker,
                  edgecolors=edge_color, facecolors=face_color,
                  linewidths=2, alpha=0.6, zorder=1)

        # Overlay Clade 2.4 prevalence (pink circle, outer)
        if clade_24_loads[i] > 0.001:
            circle_24 = Circle((x, y), clade_24_radius[i],
                              color='#ff9999', alpha=0.6, zorder=2)
            ax.add_patch(circle_24)

            # Add inner ring
            circle_24_inner = Circle((x, y), clade_24_radius[i],
                                    fill=False, edgecolor='#ff6666',
                                    linewidth=2, alpha=0.8, zorder=3)
            ax.add_patch(circle_24_inner)

        # Overlay Clade 2.5 prevalence (yellow circle, inner)
        if clade_25_loads[i] > 0.001:
            circle_25 = Circle((x, y), clade_25_radius[i],
                              color='#ffeb3b', alpha=0.6, zorder=4)
            ax.add_patch(circle_25)

            # Add inner ring
            circle_25_inner = Circle((x, y), clade_25_radius[i],
                                    fill=False, edgecolor='#fdd835',
                                    linewidth=2, alpha=0.8, zorder=5)
            ax.add_patch(circle_25_inner)

    # Set axis properties
    ax.set_xlabel('X Coordinate', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y Coordinate', fontsize=14, fontweight='bold')
    ax.set_title('Final Hospital Network State', fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')

    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor='#4F52A4', edgecolor='#4F52A4', label='Peripheral'),
        mpatches.Patch(facecolor='#4F52A4', edgecolor='#4F52A4', label='Metro'),
        mpatches.Patch(facecolor='#ff9999', edgecolor='#ff6666', label='Clade_2.4 prevalence'),
        mpatches.Patch(facecolor='#ffeb3b', edgecolor='#fdd835', label='Clade_2.5 prevalence')
    ]
    first_legend = ax.legend(handles=legend_elements, loc='lower right', fontsize=12,
                            framealpha=0.9, edgecolor='black')

    # Add size legend for hospital capacity
    # Create representative capacity values
    size_capacities = [min_capacity, (min_capacity + max_capacity) / 2, max_capacity]
    size_labels = [f'{int(cap)}' for cap in size_capacities]

    # Calculate corresponding node sizes for legend
    size_legend_sizes = []
    for cap in size_capacities:
        norm_cap = (cap - min_capacity) / (max_capacity - min_capacity + 1e-9)
        sqrt_norm = np.sqrt(norm_cap)
        size = min_node_size + sqrt_norm * (max_node_size - min_node_size)
        size_legend_sizes.append(size)

    # Create dummy scatter plots for size legend
    size_legend_handles = []
    for i, (size, label) in enumerate(zip(size_legend_sizes, size_labels)):
        handle = ax.scatter([], [], s=size, marker='o',
                           edgecolors='#4F52A4', facecolors='#4F52A4',
                           linewidths=2, alpha=0.6,
                           label=f'Capacity: {label}')
        size_legend_handles.append(handle)

    # Add second legend for sizes
    size_legend = ax.legend(handles=size_legend_handles, loc='upper right',
                           fontsize=11, title='Hospital Capacity',
                           title_fontsize=12, framealpha=0.9,
                           edgecolor='black', scatterpoints=1)

    # Add first legend back to the plot (matplotlib removes it when adding second legend)
    ax.add_artist(first_legend)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Hospital network state visualization saved: {output_path}")


# ========== plot_three_strategies ==========

def plot_three_strategies(final_freqs, hospital_types, TAU_VALUES, BETA_VALUES,
                         time_series=None, output_path='three_strategies.png'):
    """
    Plot three strategy comparison figures:
    1. Case A: Average Frequencies (grouped bar chart)
    2. Case C: Metro Bias Index (horizontal bar chart)
    3. Case G: Strategy Quadrant (scatter plot)
    """

    N_GENOTYPES = len(TAU_VALUES)
    high_M_geno = np.argmax(TAU_VALUES)
    high_C_geno = np.argmax(BETA_VALUES)

    metro_indices = [i for i, t in enumerate(hospital_types) if t == 'metro']
    periph_indices = [i for i, t in enumerate(hospital_types) if t == 'peripheral']

    metro_avg_freqs = []
    periph_avg_freqs = []
    metro_bias_indices = []
    spread_speeds = []

    # Check if time_series data is available for speed calculation
    use_real_speed = time_series is not None and 'overall_freqs' in time_series
    if use_real_speed:
        print(f"  Using time series data to calculate real spread speed")
    else:
        print(f"  Warning: No time series data available, using proxy metrics for speed")

    for g in range(N_GENOTYPES):
        # Case A: Average Frequencies
        metro_avg = np.mean(final_freqs[metro_indices, g])
        periph_avg = np.mean(final_freqs[periph_indices, g])
        metro_avg_freqs.append(metro_avg)
        periph_avg_freqs.append(periph_avg)

        # Case C: Metro Bias Index
        total = metro_avg + periph_avg
        if total > 0.001:
            bias = (metro_avg - periph_avg) / total
        else:
            bias = 0
        metro_bias_indices.append(bias)

        # Case G: Spread Speed (for Y-axis)
        if time_series is not None and 'overall_freqs' in time_series:
            # Calculate growth rate from time series
            # overall_freqs shape: (n_generations, n_genotypes)
            freq_history = time_series['overall_freqs'][:, g]  # (n_generations,)

            # Measure speed based on saturation point method
            if len(freq_history) > 10:
                initial_freq = freq_history[0]

                # Calculate growth rate for each generation (first derivative)
                growth_rates = np.diff(freq_history)  # freq[t+1] - freq[t]

                # Find saturation point: growth rate drops below 10% of initial growth rate
                # Use sliding window to smooth growth rates and avoid noise
                window_size = min(5, len(growth_rates) // 10)
                if window_size >= 3:
                    smoothed_rates = np.convolve(growth_rates, np.ones(window_size)/window_size, mode='valid')
                else:
                    smoothed_rates = growth_rates

                # Find early period (first 20%) average growth rate
                early_period = max(1, len(smoothed_rates) // 5)
                if early_period > 0:
                    initial_growth_rate = np.mean(smoothed_rates[:early_period])
                else:
                    initial_growth_rate = smoothed_rates[0] if len(smoothed_rates) > 0 else 0

                # Find saturation point: growth rate drops below 10% of initial growth rate
                saturation_threshold = max(0.0001, abs(initial_growth_rate) * 0.1)
                saturation_gen = None

                for t in range(len(smoothed_rates)):
                    if abs(smoothed_rates[t]) < saturation_threshold:
                        saturation_gen = t
                        break

                # Use max frequency generation if saturation not found
                if saturation_gen is None or saturation_gen == 0:
                    saturation_gen = np.argmax(freq_history)

                # Calculate average speed from initial to saturation point
                saturation_freq = freq_history[saturation_gen]
                if saturation_gen > 0 and (saturation_freq - initial_freq) > 0.001:
                    speed = (saturation_freq - initial_freq) / saturation_gen
                else:
                    speed = 0
            else:
                speed = 0
        else:
            # If no time series, use proxy: average frequency × presence
            overall_avg = (metro_avg + periph_avg) / 2
            threshold = 0.05
            metro_presence = np.sum(final_freqs[metro_indices, g] > threshold) / len(metro_indices)
            periph_presence = np.sum(final_freqs[periph_indices, g] > threshold) / len(periph_indices)
            overall_presence = (metro_presence + periph_presence) / 2
            speed = overall_avg * overall_presence

        spread_speeds.append(speed)

    if use_real_speed:
        freq_hist = time_series['overall_freqs'][:, high_M_geno]

        # Report detailed spread speed calculation for the key genotype
        growth_rates = np.diff(freq_hist)
        window_size = min(5, len(growth_rates) // 10)
        if window_size >= 3:
            smoothed_rates = np.convolve(growth_rates, np.ones(window_size)/window_size, mode='valid')
        else:
            smoothed_rates = growth_rates

        early_period = max(1, len(smoothed_rates) // 5)
        initial_growth_rate = np.mean(smoothed_rates[:early_period]) if early_period > 0 else smoothed_rates[0]
        saturation_threshold = max(0.0001, abs(initial_growth_rate) * 0.1)

        saturation_gen = None
        for t in range(len(smoothed_rates)):
            if abs(smoothed_rates[t]) < saturation_threshold:
                saturation_gen = t
                break

        if saturation_gen is None or saturation_gen == 0:
            saturation_gen = np.argmax(freq_hist)

        print(f"  Initial frequency: {freq_hist[0]:.4f} (Generation 0)")
        print(f"  Initial growth rate: {initial_growth_rate:.6f} per generation")
        print(f"  Saturation frequency: {freq_hist[saturation_gen]:.4f} (Generation {saturation_gen})")
        print(f"  Maximum frequency: {np.max(freq_hist):.4f} (Generation {np.argmax(freq_hist)})")
        print(f"  Calculation: Speed = ({freq_hist[saturation_gen]:.4f} - {freq_hist[0]:.4f}) / {saturation_gen} = {spread_speeds[high_M_geno]:.6f} per generation")
        print(f"  Implication: From initial to saturation, average growth per generation {spread_speeds[high_M_geno]*100:.2f}%")

    print(f"\nKey Genotype Data:")
    print(f"G{high_M_geno} (Clade 2.5-like):")
    print(f"  Metro Average Frequency: {metro_avg_freqs[high_M_geno]:.3f}")
    print(f"  Periph Average Frequency: {periph_avg_freqs[high_M_geno]:.3f}")
    print(f"  Metro Bias: {metro_bias_indices[high_M_geno]:+.3f}")
    print(f"  Spread Speed: {spread_speeds[high_M_geno]:.4f} (Growth per generation {spread_speeds[high_M_geno]*100:.2f}%)")

    print(f"\nG{high_C_geno} (Clade 2.4-like):")
    print(f"  Metro Average Frequency: {metro_avg_freqs[high_C_geno]:.3f}")
    print(f"  Periph Average Frequency: {periph_avg_freqs[high_C_geno]:.3f}")
    print(f"  Metro Bias: {metro_bias_indices[high_C_geno]:+.3f}")
    print(f"  Spread Speed: {spread_speeds[high_C_geno]:.4f} (Growth per generation {spread_speeds[high_C_geno]*100:.2f}%)")

    fig = plt.figure(figsize=(20, 7))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)

    # ========== Plot 1: Scheme A - Average Frequency Comparison ==========
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(N_GENOTYPES)
    width = 0.35
    ax1.bar(x - width/2, metro_avg_freqs, width,
            label='Metropolitan', color='steelblue', alpha=0.8,
            edgecolor='black', linewidth=0.5)
    ax1.bar(x + width/2, periph_avg_freqs, width,
            label='Peripheral', color='coral', alpha=0.8,
            edgecolor='black', linewidth=0.5)
    for i, (m_freq, p_freq) in enumerate(zip(metro_avg_freqs, periph_avg_freqs)):
        if m_freq > 0.01:
            ax1.text(i - width/2, m_freq, f'{m_freq:.2f}',
                    ha='center', va='bottom', fontsize=7)
        if p_freq > 0.01:
            ax1.text(i + width/2, p_freq, f'{p_freq:.2f}',
                    ha='center', va='bottom', fontsize=7)

    ax1.set_xlabel('Genotype', fontsize=7, fontweight='bold')
    ax1.set_ylabel('Average Frequency', fontsize=7, fontweight='bold')
    ax1.set_title('A. Average Frequency Comparison',
                fontsize=7, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'G{g}\nτ={TAU_VALUES[g]:.2f}' for g in range(N_GENOTYPES)],
                        fontsize=7)
    ax1.legend(fontsize=7, loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.set_ylim(0, max(max(metro_avg_freqs), max(periph_avg_freqs)) * 1.2)
    ax1.tick_params(axis='x', labelsize=7)
    ax1.tick_params(axis='y', labelsize=7)

    # ========== Plot 2: Scheme C - Metro Bias Index ==========
    ax2 = fig.add_subplot(gs[0, 1])
    colors = ['steelblue' if bias > 0 else 'coral' for bias in metro_bias_indices]
    ax2.barh(range(N_GENOTYPES), metro_bias_indices,
             color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
    for i, bias in enumerate(metro_bias_indices):
        if abs(bias) > 0.01:
            x_pos = bias + (0.05 if bias > 0 else -0.05)
            ha = 'left' if bias > 0 else 'right'
            fontweight = 'bold' if i in [high_M_geno, high_C_geno] else 'normal'
            ax2.text(x_pos, i, f'{bias:+.2f}',
                    ha=ha, va='center', fontsize=7)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.8)
    ax2.axvspan(-1, 0, alpha=0.1, color='coral', label='Peripheral bias zone')
    ax2.axvspan(0, 1, alpha=0.1, color='steelblue', label='Metro bias zone')

    ax2.set_xlabel('Metro Bias Index\n← Peripheral Bias  |  Metro Bias →',
                   fontsize=7, fontweight='bold')
    ax2.set_ylabel('Genotype', fontsize=7, fontweight='bold')
    ax2.set_title('C. Metro Bias Index\n(Quantifies Strategy Differences)',
                  fontsize=7, fontweight='bold', pad=15)
    ax2.set_yticks(range(N_GENOTYPES))
    ax2.set_yticklabels([f'G{g} (τ={TAU_VALUES[g]:.2f})' for g in range(N_GENOTYPES)],
                        fontsize=7)
    ax2.set_xlim(-1.0, 1.0)
    ax2.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax2.legend(fontsize=7, loc='lower right')
    ax2.tick_params(axis='both', labelsize=7)

    # ========== Plot 3: Scheme G - Strategy Quadrant ==========
    ax3 = fig.add_subplot(gs[0, 2])
    scatter_colors = []
    scatter_sizes = []
    scatter_alphas = []
    for g in range(N_GENOTYPES):
        if g == high_M_geno:
            scatter_colors.append('steelblue')
            scatter_sizes.append(150)
            scatter_alphas.append(0.5)
        elif g == high_C_geno:
            scatter_colors.append('coral')
            scatter_sizes.append(150)
            scatter_alphas.append(0.5)
        else:
            scatter_colors.append('gray')
            scatter_sizes.append(150)
            scatter_alphas.append(0.5)
    for g in range(N_GENOTYPES):
        ax3.scatter(metro_bias_indices[g], spread_speeds[g],
                   c=scatter_colors[g], s=scatter_sizes[g],
                   alpha=scatter_alphas[g],
                   edgecolors='black', linewidths=2 if g in [high_M_geno, high_C_geno] else 1,
                   zorder=3 if g in [high_M_geno, high_C_geno] else 2)
    for g in range(N_GENOTYPES):
        if g in [high_M_geno, high_C_geno] or spread_speeds[g] > 0.05:
            fontweight = 'bold' if g in [high_M_geno, high_C_geno] else 'normal'
            fontsize = 7 if g in [high_M_geno, high_C_geno] else 7
            ax3.annotate(f'G{g}',
                        (metro_bias_indices[g], spread_speeds[g]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=fontsize)
    mean_speed = np.mean([s for s in spread_speeds if s > 0.01])
    ax3.axhline(y=mean_speed, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    max_speed = max(spread_speeds)
    max_bias = max(abs(min(metro_bias_indices)), max(metro_bias_indices))

    ax3.set_xlabel('Metro Bias Index', fontsize=7, fontweight='bold')
    if time_series is not None and 'overall_freqs' in time_series:
        ax3.set_ylabel('Spread Speed (Δfreq/generation)', fontsize=7, fontweight='bold')
    else:
        ax3.set_ylabel('Spread Speed Proxy', fontsize=7, fontweight='bold')
    ax3.set_title('G. Strategy Quadrant\n(Validating 2.5 vs 2.4 Hypotheses)',
                  fontsize=7, fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(-max_bias * 1.1, max_bias * 1.1)
    ax3.set_ylim(-max_speed * 0.05, max_speed * 1.1)
    ax3.tick_params(axis='both', labelsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nImage saved: {output_path}")
    plt.close()
    return {
        'metro_avg_freqs': metro_avg_freqs,
        'periph_avg_freqs': periph_avg_freqs,
        'metro_bias_indices': metro_bias_indices,
        'spread_speeds': spread_speeds
    }


# ========== plot_three_strategies_interactive ==========

def plot_three_strategies_interactive(final_freqs, hospital_types, TAU_VALUES, BETA_VALUES,
                                     time_series=None, output_path='three_strategies_interactive.html'):
    N_GENOTYPES = len(TAU_VALUES)
    metro_indices = [i for i, t in enumerate(hospital_types) if t == 'metro']
    periph_indices = [i for i, t in enumerate(hospital_types) if t == 'peripheral']

    metro_avg_freqs = []
    periph_avg_freqs = []
    metro_bias_indices = []
    spread_speeds = []
    use_real_speed = time_series is not None and 'overall_freqs' in time_series
    for g in range(N_GENOTYPES):
        metro_avg = np.mean(final_freqs[metro_indices, g])
        periph_avg = np.mean(final_freqs[periph_indices, g])
        metro_avg_freqs.append(metro_avg)
        periph_avg_freqs.append(periph_avg)

        total = metro_avg + periph_avg
        bias = (metro_avg - periph_avg) / total if total > 0.001 else 0
        metro_bias_indices.append(bias)

        if use_real_speed:
            freq_history = time_series['overall_freqs'][:, g]
            if len(freq_history) > 10:
                initial_freq = freq_history[0]
                growth_rates = np.diff(freq_history)

                window_size = min(5, len(growth_rates) // 10)
                if window_size >= 3:
                    smoothed_rates = np.convolve(growth_rates, np.ones(window_size)/window_size, mode='valid')
                else:
                    smoothed_rates = growth_rates

                early_period = max(1, len(smoothed_rates) // 5)
                initial_growth_rate = np.mean(smoothed_rates[:early_period]) if early_period > 0 else smoothed_rates[0]
                saturation_threshold = max(0.0001, abs(initial_growth_rate) * 0.1)
                saturation_gen = None

                for t in range(len(smoothed_rates)):
                    if abs(smoothed_rates[t]) < saturation_threshold:
                        saturation_gen = t
                        break

                if saturation_gen is None or saturation_gen == 0:
                    saturation_gen = np.argmax(freq_history)

                saturation_freq = freq_history[saturation_gen]
                speed = (saturation_freq - initial_freq) / saturation_gen if saturation_gen > 0 and (saturation_freq - initial_freq) > 0.001 else 0
            else:
                speed = 0
        else:
            overall_avg = (metro_avg + periph_avg) / 2
            threshold = 0.05
            metro_presence = np.sum(final_freqs[metro_indices, g] > threshold) / len(metro_indices)
            periph_presence = np.sum(final_freqs[periph_indices, g] > threshold) / len(periph_indices)
            overall_presence = (metro_presence + periph_presence) / 2
            speed = overall_avg * overall_presence

        spread_speeds.append(speed)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            '<b>A. Frequency Comparison</b><br><sub>Metro vs Peripheral</sub>',
            '<b>B. Metro Bias Index</b><br><sub>← Peripheral | Metro →</sub>',
            '<b>C. Strategy Quadrant</b><br><sub>Validating 2.5 vs 2.4</sub>'
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "scatter"}]],
        horizontal_spacing=0.12
    )

    genotype_labels = [f'G{g}' for g in range(N_GENOTYPES)]

    fig.add_trace(
        go.Bar(
            name='Metropolitan',
            x=genotype_labels,
            y=metro_avg_freqs,
            marker_color='steelblue',
            hovertemplate='<b>%{x}</b><br>Metro freq: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            name='Peripheral',
            x=genotype_labels,
            y=periph_avg_freqs,
            marker_color='coral',
            hovertemplate='<b>%{x}</b><br>Periph freq: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )

    colors_bias = ['steelblue' if b > 0 else 'coral' for b in metro_bias_indices]

    hover_text_bias = [
        f"<b>G{g} (τ={TAU_VALUES[g]:.2f})</b><br>"
        f"Metro Bias: {metro_bias_indices[g]:.3f}<br>"
        f"Metro freq: {metro_avg_freqs[g]:.4f}<br>"
        f"Periph freq: {periph_avg_freqs[g]:.4f}"
        for g in range(N_GENOTYPES)
    ]

    fig.add_trace(
        go.Bar(
            name='Metro Bias',
            x=metro_bias_indices,
            y=[f'G{g} (τ={TAU_VALUES[g]:.2f})' for g in range(N_GENOTYPES)],
            orientation='h',
            marker_color=colors_bias,
            hovertext=hover_text_bias,
            hoverinfo='text',
            showlegend=False
        ),
        row=1, col=2
    )

    metro_specialists = [g for g in range(N_GENOTYPES) if TAU_VALUES[g] > 0.6]
    periph_specialists = [g for g in range(N_GENOTYPES) if BETA_VALUES[g] > 0.6 and TAU_VALUES[g] <= 0.6]
    generalists = [g for g in range(N_GENOTYPES) if g not in metro_specialists and g not in periph_specialists]
    def create_hover_text(g):
        return (
            f"<b>Genotype {g}</b><br>"
            f"τ-value: {TAU_VALUES[g]:.3f}<br>"
            f"β-value: {BETA_VALUES[g]:.3f}<br>"
            f"<br>"
            f"Metro freq: {metro_avg_freqs[g]:.4f}<br>"
            f"Periph freq: {periph_avg_freqs[g]:.4f}<br>"
            f"Metro Bias: {metro_bias_indices[g]:.3f}<br>"
            f"Spread Speed: {spread_speeds[g]:.6f}"
        )
    if metro_specialists:
        fig.add_trace(
            go.Scatter(
                name='Metro specialist',
                x=[metro_bias_indices[g] for g in metro_specialists],
                y=[spread_speeds[g] for g in metro_specialists],
                mode='markers+text',
                text=[f'G{g}' for g in metro_specialists],
                textposition='top center',
                marker=dict(size=15, color='steelblue', line=dict(width=2, color='black')),
                hovertext=[create_hover_text(g) for g in metro_specialists],
                hoverinfo='text'
            ),
            row=1, col=3
        )
    if periph_specialists:
        fig.add_trace(
            go.Scatter(
                name='Peripheral specialist',
                x=[metro_bias_indices[g] for g in periph_specialists],
                y=[spread_speeds[g] for g in periph_specialists],
                mode='markers+text',
                text=[f'G{g}' for g in periph_specialists],
                textposition='top center',
                marker=dict(size=15, color='coral', line=dict(width=2, color='black')),
                hovertext=[create_hover_text(g) for g in periph_specialists],
                hoverinfo='text'
            ),
            row=1, col=3
        )
    if generalists:
        fig.add_trace(
            go.Scatter(
                name='Generalist',
                x=[metro_bias_indices[g] for g in generalists],
                y=[spread_speeds[g] for g in generalists],
                mode='markers+text',
                text=[f'G{g}' for g in generalists],
                textposition='top center',
                marker=dict(size=15, color='gray', line=dict(width=1, color='black'), opacity=0.6),
                hovertext=[create_hover_text(g) for g in generalists],
                hoverinfo='text'
            ),
            row=1, col=3
        )

    mean_speed = np.mean([s for s in spread_speeds if s > 0.01])
    max_bias = max(abs(min(metro_bias_indices)), max(metro_bias_indices))
    max_speed = max(spread_speeds)
    fig.add_shape(
        type='line',
        x0=0, x1=0,
        y0=0, y1=max_speed * 1.1,
        line=dict(color='gray', dash='dash', width=1),
        row=1, col=3
    )
    fig.add_shape(
        type='line',
        x0=-max_bias * 1.1, x1=max_bias * 1.1,
        y0=mean_speed, y1=mean_speed,
        line=dict(color='gray', dash='dash', width=1),
        row=1, col=3
    )

    speed_label = 'Spread Speed (Δfreq/gen)' if use_real_speed else 'Spread Speed Proxy'

    fig.update_xaxes(title_text='Genotype', row=1, col=1)
    fig.update_yaxes(title_text='Average Frequency', row=1, col=1)

    fig.update_xaxes(title_text='Metro Bias Index', row=1, col=2, range=[-1.1, 1.1])
    fig.update_yaxes(title_text='Genotype (M-value)', row=1, col=2)

    fig.update_xaxes(title_text='Metro Bias Index', row=1, col=3, range=[-max_bias * 1.2, max_bias * 1.2])
    fig.update_yaxes(title_text=speed_label, row=1, col=3, range=[-max_speed * 0.05, max_speed * 1.2])

    fig.update_layout(
        title_text='<b>Interactive Three-Strategy Analysis</b><br><sub>Hover over points for detailed information</sub>',
        title_x=0.5,
        title_font_size=16,
        showlegend=True,
        legend=dict(x=1.02, y=0.5),
        height=500,
        width=1600,
        hovermode='closest',
        plot_bgcolor='white'
    )

    fig.write_html(output_path)
    print(f"Plotly interactive chart saved: {output_path}")


# ========== plot_genotype_trajectories ==========

def plot_spatial_spread_over_time(spatial_spread_data, output_path='spatial_spread_over_time.png',
                                  intervention_start=None, recovery_start=None):
    """
    Plot spatial spread (geographic dissemination) over time for M and C genotype groups.

    Similar to the reference image showing "Spatial Dissemination Over Time" with
    Clade 2.4 and Clade 2.5 trajectories.

    Parameters:
        spatial_spread_data: dict with:
            - 'timesteps': array of generation numbers
            - 'm_spread': array of M-genotype spatial spread (km)
            - 'c_spread': array of C-genotype spatial spread (km)
        output_path: path to save the figure
        intervention_start: optional, generation when intervention starts (for vertical line)
        recovery_start: optional, generation when recovery starts (for vertical line)

    Returns:
        None (saves figure to output_path)
    """
    timesteps = spatial_spread_data['timesteps']
    m_spread = spatial_spread_data['m_spread']
    c_spread = spatial_spread_data['c_spread']

    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot M-genotype spread (blue line, similar to Clade 2.5)
    ax.plot(timesteps, m_spread, linewidth=2.5, color='#2196F3',
           label='M-genotype (Metro-specialist, Clade 2.5-like)', alpha=0.8)

    # Plot C-genotype spread (orange line, similar to Clade 2.4)
    ax.plot(timesteps, c_spread, linewidth=2.5, color='#FF9800',
           label='C-genotype (Peripheral-specialist, Clade 2.4-like)', alpha=0.8)

    # Add intervention phases if provided
    if intervention_start is not None:
        ax.axvline(x=intervention_start, color='red', linestyle='--',
                  linewidth=2, alpha=0.6, label=f'Intervention Start (Gen {intervention_start})')

        # Shade the intervention period if recovery_start is also provided
        if recovery_start is not None:
            ax.axvspan(intervention_start, recovery_start, alpha=0.15, color='red',
                      label='COVID Intervention Period')
            ax.axvline(x=recovery_start, color='green', linestyle='--',
                      linewidth=2, alpha=0.6, label=f'Recovery Start (Gen {recovery_start})')

    # Formatting
    ax.set_xlabel('Timestep (Generation)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Geographic Spread (distance units)', fontsize=14, fontweight='bold')
    ax.set_title('Spatial Dissemination Over Time', fontsize=18, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', labelsize=12)

    # Set reasonable y-axis limits
    all_spreads = np.concatenate([m_spread, c_spread])
    valid_spreads = all_spreads[all_spreads > 0]
    if len(valid_spreads) > 0:
        y_min = max(0, np.min(valid_spreads) * 0.9)
        y_max = np.max(valid_spreads) * 1.1
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSpatial spread time series plot saved: {output_path}")
    plt.close()

    # Print summary statistics
    print(f"\n{'='*70}")
    print(f"Spatial Spread Summary Statistics")
    print(f"{'='*70}")

    # Filter out zero values for meaningful statistics
    m_nonzero = m_spread[m_spread > 0]
    c_nonzero = c_spread[c_spread > 0]

    if len(m_nonzero) > 0:
        print(f"\nM-genotype (Metro-specialist) spatial spread:")
        print(f"  Mean: {np.mean(m_nonzero):.2f} km")
        print(f"  Min:  {np.min(m_nonzero):.2f} km")
        print(f"  Max:  {np.max(m_nonzero):.2f} km")
        print(f"  Final: {m_spread[-1]:.2f} km")

    if len(c_nonzero) > 0:
        print(f"\nC-genotype (Peripheral-specialist) spatial spread:")
        print(f"  Mean: {np.mean(c_nonzero):.2f} km")
        print(f"  Min:  {np.min(c_nonzero):.2f} km")
        print(f"  Max:  {np.max(c_nonzero):.2f} km")
        print(f"  Final: {c_spread[-1]:.2f} km")

    if len(m_nonzero) > 0 and len(c_nonzero) > 0:
        print(f"\nComparison:")
        ratio = np.mean(m_nonzero) / np.mean(c_nonzero)
        print(f"  M/C spread ratio: {ratio:.2f}x")
        if ratio > 1.2:
            print(f"  → M-genotype shows higher geographic dispersal (metro-specialist strategy)")
        elif ratio < 0.8:
            print(f"  → C-genotype shows higher geographic dispersal (unexpected!)")
        else:
            print(f"  → Both genotypes show similar geographic dispersal")

    print(f"{'='*70}\n")


def plot_genotype_trajectories(time_series, final_freqs, hospital_types, TAU_VALUES, BETA_VALUES,
                               output_path='genotype_trajectories.png'):
    """
    Plot the trajectory of genotype frequencies over time and the final hospital distribution.

    Parameters:
        time_series: dict, contains 'overall_freqs', 'metro_freqs', 'periph_freqs'
        final_freqs: ndarray, shape (n_hospitals, n_genotypes)
        hospital_types: list of str
        TAU_VALUES: ndarray
        BETA_VALUES: ndarray
        output_path: str
    """
    N_GENOTYPES = len(TAU_VALUES)
    high_M_geno = np.argmax(TAU_VALUES)
    high_C_geno = np.argmax(BETA_VALUES)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # ============ Genotype Frequency Time Series (Excluding Neutral Background) ============
    ax = axes[0]

    overall_freqs = time_series['overall_freqs']
    timesteps = np.arange(len(overall_freqs))

    colors = plt.cm.tab10(np.linspace(0, 1, N_GENOTYPES))

    # Identify and exclude background genotypes (neutral genotypes, 0.4 <= M <= 0.6)
    for g in range(N_GENOTYPES):
        # Skip background genotypes
        if 0.4 <= TAU_VALUES[g] <= 0.6:
            continue

        label = f'G{g} τ={TAU_VALUES[g]:.2f}'
        if g == high_M_geno:
            ax.plot(timesteps, overall_freqs[:, g], linewidth=2, label=label, color='blue')
        elif g == high_C_geno:
            ax.plot(timesteps, overall_freqs[:, g], linewidth=2, label=label, color='orange')
        else:
            ax.plot(timesteps, overall_freqs[:, g], linewidth=1.25, alpha=0.7, label=label, color=colors[g])

    ax.set_xlabel('Timestep (Generations)', fontsize=7)
    ax.set_ylabel('Relative Frequency', fontsize=7)
    ax.set_title('Infection Genotype Trajectories Over Time\n(Excluding Neutral Background)', fontsize=7, fontweight='bold')
    ax.legend(loc='center left', fontsize=7, bbox_to_anchor=(1.02, 0.5))
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(timesteps))
    ax.set_ylim(0, 1.0)

    # ============ Genotype Distribution in Hospitals (Excluding Background Genotypes) ============
    ax = axes[1]

    metro_indices = [i for i, t in enumerate(hospital_types) if t == 'metro']
    periph_indices = [i for i, t in enumerate(hospital_types) if t == 'peripheral']

    # Calculate the colonization count for each genotype in metropolitan and peripheral hospitals
    metro_colonized = {}
    periph_colonized = {}

    threshold = 0.05  # Frequency > 5% is considered successful colonization

    # Only count infecting genotypes (excluding background)
    for g in range(N_GENOTYPES):
        # Skip background genotypes
        if 0.4 <= TAU_VALUES[g] <= 0.6:
            continue

        metro_count = np.sum(final_freqs[metro_indices, g] > threshold)
        periph_count = np.sum(final_freqs[periph_indices, g] > threshold)

        metro_colonized[g] = metro_count
        periph_colonized[g] = periph_count

    # Sort by M value (only include infecting genotypes)
    infection_genotypes = list(metro_colonized.keys())
    genotype_order = sorted(infection_genotypes, key=lambda g: TAU_VALUES[g], reverse=True)

    x_pos = np.arange(len(genotype_order))
    width = 0.35

    # Plot the bar chart for infecting genotypes
    for idx, g in enumerate(genotype_order):
        if g == high_M_geno:
            color_metro = 'steelblue'
            color_periph = 'lightblue'
        elif g == high_C_geno:
            color_metro = 'coral'
            color_periph = 'lightsalmon'
        else:
            color_metro = 'gray'
            color_periph = 'lightgray'

        ax.bar(idx - width/2, metro_colonized[g], width,
               label='Metropolitan' if idx == 0 else '',
               color=color_metro)
        ax.bar(idx + width/2, periph_colonized[g], width,
               label='Peripheral' if idx == 0 else '',
               color=color_periph)

        # Add value labels
        if metro_colonized[g] > 0:
            ax.text(idx - width/2, metro_colonized[g], str(metro_colonized[g]),
                   ha='center', va='bottom', fontsize=7, fontweight='bold')
        if periph_colonized[g] > 0:
            ax.text(idx + width/2, periph_colonized[g], str(periph_colonized[g]),
                   ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_ylabel('Final Colonized Count', fontsize=7)
    ax.set_xlabel('Genotype', fontsize=7)
    ax.set_title('Hospital Type Distribution (Infection Genotypes Only)', fontsize=7, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'G{g}\nτ={TAU_VALUES[g]:.2f}' for g in genotype_order], fontsize=7)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


