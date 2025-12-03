#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Sensitivity Analysis Results

Usage:
    python plot_sensitivity.py sensitivity_oneway_20251118_020000.csv
    python plot_sensitivity.py sensitivity_oneway_*.csv --output figure.pdf
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12


def plot_parameter_sensitivity(df, output_path='sensitivity_analysis.pdf'):
    """
    Create main sensitivity figure showing frequencies across parameters.

    Parameters:
        df: DataFrame with sensitivity results
        output_path: Output file path
    """
    # Get unique parameters (limit to max 6 for layout)
    params = df['parameter'].unique()[:6]
    n_params = len(params)

    # Calculate grid layout
    nrows = (n_params + 1) // 2
    ncols = 2

    # Create figure
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5*nrows))
    if n_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Parameter display names
    param_labels = {
        'selection': 'Selection Strength (s)',
        'metro_migration': 'Metro Migration Rate',
        'peripheral_migration': 'Peripheral Migration Rate',
        'decay_length': 'Distance Decay Length (km)',
        'metros': 'Number of Metro Cities',
    }

    colors = {'metro': "#1D4DBD", 'periph': "#C55137"}

    for idx, param in enumerate(params):
        ax = axes[idx]

        # Filter data
        subset = df[df['parameter'] == param].copy()

        # Build aggregation dict based on available columns
        agg_dict = {
            'metro_m_freq': ['mean', 'std', 'count'],
            'periph_m_freq': ['mean', 'std'],
        }

        # Use new or old coexistence column
        if 'acceptable_coexistence' in subset.columns:
            agg_dict['acceptable_coexistence'] = 'mean'
        elif 'coexistence' in subset.columns:
            agg_dict['coexistence'] = 'mean'

        # Calculate mean and std for each value
        summary = subset.groupby('value').agg(agg_dict).reset_index()

        values = summary['value'].values
        metro_mean = summary[('metro_m_freq', 'mean')].values
        metro_std = summary[('metro_m_freq', 'std')].values
        periph_mean = summary[('periph_m_freq', 'mean')].values
        periph_std = summary[('periph_m_freq', 'std')].values

        # Handle coexistence - extract from the right column
        coexist_rate = np.ones(len(values))  # default
        for col_name in ['acceptable_coexistence', 'coexistence']:
            if col_name in summary.columns.get_level_values(0):
                coexist_rate = summary[(col_name, 'mean')].values
                break

        # Plot lines with error bars
        ax.errorbar(values, metro_mean, yerr=metro_std,
                   label='Metro hospitals', marker='o', markersize=8,
                   linewidth=2, capsize=5, color=colors['metro'], zorder=3)

        ax.errorbar(values, periph_mean, yerr=periph_std,
                   label='Peripheral hospitals', marker='s', markersize=8,
                   linewidth=2, capsize=5, color=colors['periph'], zorder=3)

        # # Add coexistence indicator (background shading)
        # low_coexist_plotted = False
        # for i, (v, rate) in enumerate(zip(values, coexist_rate)):
        #     if rate < 0.5:  # Low coexistence
        #         label = 'Low coexistence\n(<50%)' if not low_coexist_plotted else None
        #         ax.axvspan(v - (values[1]-values[0])*0.4 if i > 0 else v - 1,
        #                   v + (values[1]-values[0])*0.4 if i < len(values)-1 else v + 1,
        #                   alpha=0.2, color='red', zorder=0, label=label)
        #         low_coexist_plotted = True

        # Formatting
        ax.set_xlabel(param_labels.get(param, param), fontsize=11, fontweight='bold')
        ax.set_ylabel('Clade 2.5 Frequency', fontsize=11, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3, linestyle='--', zorder=1)
        ax.legend(loc='best', frameon=True, fontsize=9)

        # Add panel label
        ax.text(0.02, 0.98, f"({chr(65+idx)})", transform=ax.transAxes,
               fontsize=14, fontweight='bold', va='top', ha='left')

        # Add baseline vertical line if it's in the range
        baseline_values = {
            'selection': 0.1,
            'metro_migration': 0.07,
            'peripheral_migration': 0.02,
            'decay_length': 30,
        }
        if param in baseline_values:
            ax.axvline(baseline_values[param], color='gray',
                      linestyle='--', linewidth=1.5, alpha=0.7,
                      label='Baseline')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {output_path}")

    return fig


def plot_fst_diversity(df, output_path='fst_diversity_analysis.pdf'):
    """
    Plot FST and Shannon diversity across parameter ranges.

    Parameters:
        df: DataFrame with sensitivity results
        output_path: Output file path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    params = df['parameter'].unique()

    param_labels = {
        'selection': 'Selection',
        'metro_migration': 'Metro Mig.',
        'peripheral_migration': 'Periph Mig.',
        'decay_length': 'Decay',
        'metros': 'N_metros',
    }

    colors = plt.cm.Set3(np.linspace(0, 1, len(params)))

    # Plot 1: FST across parameters
    x_offset = 0
    bar_width = 0.15

    for pidx, param in enumerate(params):
        param_data = df[df['parameter'] == param]
        summary = param_data.groupby('value').agg({
            'fst_weighted': ['mean', 'std']
        }).reset_index()

        values = summary['value'].values
        fst_mean = summary[('fst_weighted', 'mean')].values
        fst_std = summary[('fst_weighted', 'std')].values

        positions = np.arange(len(values)) * 0.2 + pidx * len(values) * 0.2 + pidx * 0.1
        ax1.bar(positions, fst_mean, bar_width, yerr=fst_std,
               label=param_labels.get(param, param),
               color=colors[pidx], alpha=0.8, capsize=3)

    ax1.set_ylabel('FST (weighted)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Parameter Values', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', frameon=True)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_title('(A) Genetic Differentiation Across Parameters', fontweight='bold')

    # Plot 2: Shannon diversity across parameters
    x_offset = 0

    for pidx, param in enumerate(params):
        param_data = df[df['parameter'] == param]
        summary = param_data.groupby('value').agg({
            'shannon_diversity': ['mean', 'std']
        }).reset_index()

        values = summary['value'].values
        div_mean = summary[('shannon_diversity', 'mean')].values
        div_std = summary[('shannon_diversity', 'std')].values

        positions = np.arange(len(values)) * 0.2 + pidx * len(values) * 0.2 + pidx * 0.1
        ax2.bar(positions, div_mean, bar_width, yerr=div_std,
               label=param_labels.get(param, param),
               color=colors[pidx], alpha=0.8, capsize=3)

    ax2.set_ylabel('Shannon Diversity', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Parameter Values', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', frameon=True)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_title('(B) Genotypic Diversity Across Parameters', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"FST & Diversity plot saved: {output_path}")

    return fig


def plot_coexistence_stability(df, output_path='coexistence_stability.pdf'):
    """
    Plot coexistence stability across parameter ranges - CLEAR VERSION.
    Each parameter gets its own grouped section.

    Parameters:
        df: DataFrame with sensitivity results
        output_path: Output file path
    """
    params = df['parameter'].unique()
    n_params = len(params)

    # Create subplots - one for each parameter
    fig, axes = plt.subplots(1, n_params, figsize=(4*n_params, 5), sharey=True)
    if n_params == 1:
        axes = [axes]

    param_labels = {
        'selection': 'Selection\nStrength (s)',
        'metro_migration': 'Metro\nMigration',
        'peripheral_migration': 'Peripheral\nMigration',
        'decay_length': 'Decay Length\n(km)',
        'metros': 'N Metro\nHospitals',
    }

    # Determine which coexistence metric to use
    coexist_col = 'acceptable_coexistence' if 'acceptable_coexistence' in df.columns else 'coexistence'

    for idx, param in enumerate(params):
        ax = axes[idx]
        param_data = df[df['parameter'] == param]

        # Calculate summary
        summary = param_data.groupby('value').agg({
            coexist_col: ['mean', 'std', 'count']
        }).reset_index()

        values = summary['value'].values
        coexist_mean = summary[(coexist_col, 'mean')].values
        coexist_std = summary[(coexist_col, 'std')].values
        n_samples = summary[(coexist_col, 'count')].values

        # Color bars based on coexistence rate
        colors = ['#2ecc71' if rate >= 0.8 else '#f39c12' if rate >= 0.5 else '#e74c3c'
                 for rate in coexist_mean]

        # Plot bars
        x_pos = np.arange(len(values))
        bars = ax.bar(x_pos, coexist_mean, color=colors, alpha=0.8,
                     edgecolor='black', linewidth=1.5)

        # Add error bars
        ax.errorbar(x_pos, coexist_mean, yerr=coexist_std,
                   fmt='none', ecolor='black', capsize=5, capthick=2)

        # Add value labels on bars
        for i, (bar, val, n) in enumerate(zip(bars, coexist_mean, n_samples)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{val*100:.0f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Formatting
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{v:.3g}' for v in values], rotation=45, ha='right')
        ax.set_xlabel(param_labels.get(param, param), fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.axhline(0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5, zorder=0)
        ax.axhline(0.8, color='green', linestyle=':', linewidth=1.5, alpha=0.5, zorder=0)
        ax.grid(axis='y', alpha=0.2, linestyle='--')

        # Panel label
        ax.text(0.02, 0.98, f"({chr(65+idx)})", transform=ax.transAxes,
               fontsize=14, fontweight='bold', va='top', ha='left')

    # Y-label only on first panel
    axes[0].set_ylabel('Coexistence Rate', fontsize=12, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='High (≥80%)', alpha=0.8),
        Patch(facecolor='#f39c12', label='Medium (50-80%)', alpha=0.8),
        Patch(facecolor='#e74c3c', label='Low (<50%)', alpha=0.8),
    ]
    axes[-1].legend(handles=legend_elements, loc='upper right',
                   frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Coexistence stability plot saved: {output_path}")

    return fig


def print_summary_table(df):
    """
    Print a summary table of sensitivity analysis results.

    Parameters:
        df: DataFrame with sensitivity results
    """
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("="*80)

    params = df['parameter'].unique()

    for param in params:
        print(f"\n{param.upper()}")
        print("-"*80)

        subset = df[df['parameter'] == param]

        # Build aggregation dict based on available columns
        agg_dict = {
            'metro_m_freq': ['mean', 'std'],
            'periph_m_freq': ['mean', 'std'],
            'overall_m_freq': ['mean', 'std'],
        }

        # Use new column names if available, fallback to old
        if 'ideal_coexistence' in subset.columns:
            agg_dict['ideal_coexistence'] = 'mean'
            agg_dict['acceptable_coexistence'] = 'mean'
            agg_dict['segregation_strength'] = 'mean'
        elif 'coexistence' in subset.columns:
            agg_dict['coexistence'] = 'mean'
            agg_dict['segregation'] = 'mean'

        # Add FST and diversity if available
        if 'fst_weighted' in subset.columns:
            agg_dict['fst_weighted'] = ['mean', 'std']
        if 'shannon_diversity' in subset.columns:
            agg_dict['shannon_diversity'] = ['mean', 'std']

        summary = subset.groupby('value').agg(agg_dict).round(3)

        # Flatten column names for display
        new_cols = []
        for col in summary.columns:
            if isinstance(col, tuple):
                new_cols.append(f"{col[0]}_{col[1]}")
            else:
                new_cols.append(str(col))
        summary.columns = new_cols

        print(summary.to_string())

        # Identify stable range based on available metrics
        stable_values = []
        if 'acceptable_coexistence_mean' in summary.columns:
            stable_values = summary[summary['acceptable_coexistence_mean'] >= 0.8].index.tolist()
        elif 'coexistence_mean' in summary.columns:
            stable_values = summary[summary['coexistence_mean'] >= 0.8].index.tolist()

        if stable_values:
            print(f"\n  ✓ Stable coexistence range: {stable_values}")
        else:
            print(f"\n  ⚠ WARNING: No values with ≥80% coexistence rate")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Plot sensitivity analysis results'
    )
    parser.add_argument('input', type=str,
                       help='Input CSV file from sensitivity_oneway.py')
    parser.add_argument('--output', type=str, default='sensitivity_analysis.pdf',
                       help='Output figure filename')
    parser.add_argument('--output-coexist', type=str, default='coexistence_stability.pdf',
                       help='Output coexistence stability figure')

    args = parser.parse_args()

    # Load data
    print(f"Loading data from: {args.input}")
    df = pd.read_csv(args.input)

    print(f"Loaded {len(df)} simulation results")
    print(f"Parameters tested: {df['parameter'].unique().tolist()}")

    # Print summary table
    print_summary_table(df)

    # Generate plots
    print(f"\nGenerating main sensitivity plot...")
    plot_parameter_sensitivity(df, args.output)

    # Only plot FST if data exists
    if 'fst_weighted' in df.columns:
        print(f"\nGenerating FST & diversity plot...")
        plot_fst_diversity(df, 'fst_diversity_analysis.pdf')
    else:
        print(f"\nSkipping FST plot (data not available in old results)")

    print(f"\nGenerating coexistence stability plot...")
    plot_coexistence_stability(df, args.output_coexist)

    print("\nDone!")


if __name__ == '__main__':
    main()
