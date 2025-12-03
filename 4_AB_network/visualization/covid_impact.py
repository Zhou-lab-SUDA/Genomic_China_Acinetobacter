#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COVID Impact Visualization
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calculate_dissemination_speed(timeline_df, phase_col='phase',
                                  intervention_start=300, recovery_start=800):
    """
    Calculate dissemination speed (average rate of frequency change) for Clade 2.4 and Clade 2.5
    during baseline and COVID intervention phases.

    Parameters:
        timeline_df: DataFrame with timeline data
        phase_col: column name for phase
        intervention_start: generation when intervention starts
        recovery_start: generation when recovery starts

    Returns:
        dict: {
            'clade_24_pre': speed (signed),
            'clade_24_covid': speed (signed),
            'clade_25_pre': speed (signed),
            'clade_25_covid': speed (signed)
        }
    """
    # Extract data for different phases
    baseline_data = timeline_df[timeline_df['generation'] < intervention_start]
    covid_data = timeline_df[(timeline_df['generation'] >= intervention_start) &
                            (timeline_df['generation'] < recovery_start)]

    # Calculate dissemination speed (average rate of frequency change) for Clade 2.5
    if len(baseline_data) > 1:
        # Baseline：Calculate average net change rate of overall_m_freq (keep sign)
        baseline_m_speed = np.diff(baseline_data['overall_m_freq'].values).mean()
    else:
        baseline_m_speed = 0

    if len(covid_data) > 1:
        # COVID：Calculate average net change rate of overall_m_freq (keep sign)
        covid_m_speed = np.diff(covid_data['overall_m_freq'].values).mean()
    else:
        covid_m_speed = 0

    # Calculate dissemination speed (average rate of frequency change) for Clade 2.4
    if len(baseline_data) > 1:
        baseline_c_speed = np.diff(baseline_data['overall_c_freq'].values).mean()
    else:
        baseline_c_speed = 0

    if len(covid_data) > 1:
        covid_c_speed = np.diff(covid_data['overall_c_freq'].values).mean()
    else:
        covid_c_speed = 0

    return {
        'clade_24_pre': baseline_c_speed,
        'clade_24_covid': covid_c_speed,
        'clade_25_pre': baseline_m_speed,
        'clade_25_covid': covid_m_speed
    }


def calculate_resilience_score(timeline_df, intervention_start=300, recovery_start=800):
    """
    Calculate resilience score (frequency retention ratio) for Clade 2.4 and Clade 2.5.

    Uses steady-state baseline (last 100 generations or half of baseline) as reference
    to avoid contamination from initial invasion growth phase.

    Parameters:
        timeline_df: DataFrame with timeline data
        intervention_start: generation when intervention starts
        recovery_start: generation when recovery starts

    Returns:
        dict: {
            'resilience_24': ratio (COVID freq / Pre-COVID steady state freq),
            'resilience_25': ratio
        }
    """
    # Extract data
    baseline_data = timeline_df[timeline_df['generation'] < intervention_start]
    covid_data = timeline_df[(timeline_df['generation'] >= intervention_start) &
                            (timeline_df['generation'] < recovery_start)]

    # Use last 100 generations (or half) of baseline as steady state
    baseline_window = min(100, len(baseline_data) // 2)
    baseline_steady = baseline_data.iloc[-baseline_window:]

    # Calculate resilience for Clade 2.4
    if len(baseline_steady) > 0 and len(covid_data) > 0:
        pre_freq_24 = baseline_steady['overall_c_freq'].mean()
        covid_freq_24 = covid_data['overall_c_freq'].mean()
        resilience_24 = covid_freq_24 / pre_freq_24 if pre_freq_24 > 0 else 1.0
    else:
        resilience_24 = 1.0

    # Calculate resilience for Clade 2.5
    if len(baseline_steady) > 0 and len(covid_data) > 0:
        pre_freq_25 = baseline_steady['overall_m_freq'].mean()
        covid_freq_25 = covid_data['overall_m_freq'].mean()
        resilience_25 = covid_freq_25 / pre_freq_25 if pre_freq_25 > 0 else 1.0
    else:
        resilience_25 = 1.0

    return {
        'resilience_24': resilience_24,
        'resilience_25': resilience_25
    }


def calculate_three_phase_metrics(timeline_df, intervention_start=300, recovery_start=800):
    """
    Calculate Growth Rate and Resilience for all three phases: Baseline, COVID, Recovery.

    Parameters:
        timeline_df: DataFrame with timeline data
        intervention_start: generation when intervention starts
        recovery_start: generation when recovery starts

    Returns:
        dict with growth rates and resilience scores for all three phases
    """
    # Extract data for three phases
    baseline_data = timeline_df[timeline_df['generation'] < intervention_start]
    covid_data = timeline_df[(timeline_df['generation'] >= intervention_start) &
                            (timeline_df['generation'] < recovery_start)]
    recovery_data = timeline_df[timeline_df['generation'] >= recovery_start]

    # Steady state baseline (for resilience calculation)
    baseline_window = min(100, len(baseline_data) // 2)
    baseline_steady = baseline_data.iloc[-baseline_window:]

    results = {}

    # ========== Growth Rate for all three phases ==========
    # Baseline
    if len(baseline_data) > 1:
        baseline_m_speed = np.diff(baseline_data['overall_m_freq'].values).mean()
        baseline_c_speed = np.diff(baseline_data['overall_c_freq'].values).mean()
    else:
        baseline_m_speed = baseline_c_speed = 0

    # COVID
    if len(covid_data) > 1:
        covid_m_speed = np.diff(covid_data['overall_m_freq'].values).mean()
        covid_c_speed = np.diff(covid_data['overall_c_freq'].values).mean()
    else:
        covid_m_speed = covid_c_speed = 0

    # Recovery
    if len(recovery_data) > 1:
        recovery_m_speed = np.diff(recovery_data['overall_m_freq'].values).mean()
        recovery_c_speed = np.diff(recovery_data['overall_c_freq'].values).mean()
    else:
        recovery_m_speed = recovery_c_speed = 0

    results['growth'] = {
        'clade_24_baseline': baseline_c_speed,
        'clade_24_covid': covid_c_speed,
        'clade_24_recovery': recovery_c_speed,
        'clade_25_baseline': baseline_m_speed,
        'clade_25_covid': covid_m_speed,
        'clade_25_recovery': recovery_m_speed,
    }

    # ========== Resilience Score for COVID and Recovery ==========
    # (relative to baseline steady state)
    pre_freq_24 = baseline_steady['overall_c_freq'].mean() if len(baseline_steady) > 0 else 1.0
    pre_freq_25 = baseline_steady['overall_m_freq'].mean() if len(baseline_steady) > 0 else 1.0

    covid_freq_24 = covid_data['overall_c_freq'].mean() if len(covid_data) > 0 else pre_freq_24
    covid_freq_25 = covid_data['overall_m_freq'].mean() if len(covid_data) > 0 else pre_freq_25

    recovery_freq_24 = recovery_data['overall_c_freq'].mean() if len(recovery_data) > 0 else covid_freq_24
    recovery_freq_25 = recovery_data['overall_m_freq'].mean() if len(recovery_data) > 0 else covid_freq_25

    results['resilience'] = {
        'clade_24_baseline': 1.0,  # By definition
        'clade_24_covid': covid_freq_24 / pre_freq_24 if pre_freq_24 > 0 else 1.0,
        'clade_24_recovery': recovery_freq_24 / pre_freq_24 if pre_freq_24 > 0 else 1.0,
        'clade_25_baseline': 1.0,  # By definition
        'clade_25_covid': covid_freq_25 / pre_freq_25 if pre_freq_25 > 0 else 1.0,
        'clade_25_recovery': recovery_freq_25 / pre_freq_25 if pre_freq_25 > 0 else 1.0,
    }

    return results


def plot_covid_impact_combined(timeline_df, intervention_start=300, recovery_start=800,
                                output_path=None):
    """
    Plot combined COVID-19 impact with THREE phases: Baseline, COVID, Recovery.
    Shows Growth Rate + Resilience Score in one figure.

    Parameters:
        timeline_df: DataFrame with timeline data
        intervention_start: generation when intervention starts
        recovery_start: generation when recovery starts
        output_path: path to save the figure

    Returns:
        None (saves figure to output_path)
    """
    # Calculate three-phase metrics
    metrics = calculate_three_phase_metrics(timeline_df,
                                           intervention_start=intervention_start,
                                           recovery_start=recovery_start)
    growth = metrics['growth']
    resilience = metrics['resilience']

    # Prepare data
    clades = ['Clade 2.4', 'Clade 2.5']
    x = np.arange(len(clades))
    width = 0.25  # Narrower bars to fit 3 phases

    # Unified color scheme for three phases
    color_baseline = '#5B8DBE'    # Muted blue
    color_covid = '#E07B5F'       # Muted coral
    color_recovery = '#7FBA7A'    # Muted green
    edge_baseline = '#2E5C8A'     # Dark blue
    edge_covid = '#B84E3A'        # Dark red
    edge_recovery = '#3D7A3A'     # Dark green

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # ==================== LEFT PANEL: Growth Rate (THREE PHASES) ====================
    baseline_speed = [growth['clade_24_baseline'], growth['clade_25_baseline']]
    covid_speed = [growth['clade_24_covid'], growth['clade_25_covid']]
    recovery_speed = [growth['clade_24_recovery'], growth['clade_25_recovery']]

    # Color bars based on sign (positive=blue, negative=coral)
    def get_color_edge(value):
        if value >= 0:
            return color_baseline, edge_baseline
        else:
            return color_covid, edge_covid

    colors_baseline = [get_color_edge(v)[0] for v in baseline_speed]
    edges_baseline = [get_color_edge(v)[1] for v in baseline_speed]
    colors_covid_val = [get_color_edge(v)[0] for v in covid_speed]
    edges_covid_val = [get_color_edge(v)[1] for v in covid_speed]
    colors_recovery = [get_color_edge(v)[0] for v in recovery_speed]
    edges_recovery = [get_color_edge(v)[1] for v in recovery_speed]

    # Plot three bars for each clade
    bars1 = ax1.bar(x - width, baseline_speed, width, label='Baseline',
                    color=colors_baseline, alpha=0.85, edgecolor=edges_baseline, linewidth=1.5)
    bars2 = ax1.bar(x, covid_speed, width, label='COVID',
                    color=colors_covid_val, alpha=0.85, edgecolor=edges_covid_val, linewidth=1.5)
    bars3 = ax1.bar(x + width, recovery_speed, width, label='Recovery',
                    color=colors_recovery, alpha=0.85, edgecolor=edges_recovery, linewidth=1.5)

    ax1.set_xlabel('Clade', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Growth Rate (Δfrequency/generation)', fontsize=12)
    ax1.set_title('Growth Rate Across Three Phases', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(clades, fontsize=12)
    ax1.legend(fontsize=11, loc='upper right', ncol=3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    all_speed_values = baseline_speed + covid_speed + recovery_speed
    y_min = min(all_speed_values + [0]) * 1.25
    y_max = max(all_speed_values + [0]) * 1.25
    ax1.set_ylim(y_min, y_max)

    # ==================== RIGHT PANEL: Resilience Score (THREE PHASES) ====================
    resilience_baseline = [resilience['clade_24_baseline'], resilience['clade_25_baseline']]
    resilience_covid = [resilience['clade_24_covid'], resilience['clade_25_covid']]
    resilience_recovery = [resilience['clade_24_recovery'], resilience['clade_25_recovery']]

    # Color based on whether >1 (gain) or <1 (loss)
    def get_resilience_color_edge(value):
        if value >= 1:
            return color_baseline, edge_baseline  # Blue for gain
        else:
            return color_covid, edge_covid  # Coral for loss

    colors_res_baseline = [get_resilience_color_edge(r)[0] for r in resilience_baseline]
    edges_res_baseline = [get_resilience_color_edge(r)[1] for r in resilience_baseline]
    colors_res_covid = [get_resilience_color_edge(r)[0] for r in resilience_covid]
    edges_res_covid = [get_resilience_color_edge(r)[1] for r in resilience_covid]
    colors_res_recovery = [get_resilience_color_edge(r)[0] for r in resilience_recovery]
    edges_res_recovery = [get_resilience_color_edge(r)[1] for r in resilience_recovery]

    # Plot three bars
    bars4 = ax2.bar(x - width, resilience_baseline, width, label='Baseline',
                    color=colors_res_baseline, alpha=0.85, edgecolor=edges_res_baseline, linewidth=1.5)
    bars5 = ax2.bar(x, resilience_covid, width, label='COVID',
                    color=colors_res_covid, alpha=0.85, edgecolor=edges_res_covid, linewidth=1.5)
    bars6 = ax2.bar(x + width, resilience_recovery, width, label='Recovery',
                    color=colors_res_recovery, alpha=0.85, edgecolor=edges_res_recovery, linewidth=1.5)

    ax2.axhline(1.0, color='black', linewidth=2, linestyle='--', alpha=0.7)
    ax2.set_xlabel('Clade', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Resilience Score (freq / Pre-COVID steady)', fontsize=11)
    ax2.set_title('Resilience Score Across Three Phases', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(clades, fontsize=12)
    ax2.legend(fontsize=11, loc='upper right', ncol=3)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    all_resilience = resilience_baseline + resilience_covid + resilience_recovery
    ax2.set_ylim(0, max(all_resilience) * 1.25)

    # Add value labels on bars
    for bars_set in [bars4, bars5, bars6]:
        for bar in bars_set:
            height = bar.get_height()
            if height > 0.1:  # Only label if visible
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Overall title
    fig.suptitle('COVID-19 Three-Phase Impact Analysis',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Three-phase COVID impact plot saved: {output_path}")

    plt.close()

    # Print summary
    print(f"\n{'='*80}")
    print("THREE-PHASE COVID-19 IMPACT SUMMARY")
    print(f"{'='*80}")

    print(f"\n【Growth Rate (Process)】")
    print(f"  Clade 2.4:")
    print(f"    Baseline:   {growth['clade_24_baseline']:+.6f} Δfreq/gen")
    print(f"    COVID:      {growth['clade_24_covid']:+.6f} Δfreq/gen")
    print(f"    Recovery:   {growth['clade_24_recovery']:+.6f} Δfreq/gen")

    print(f"\n  Clade 2.5:")
    print(f"    Baseline:   {growth['clade_25_baseline']:+.6f} Δfreq/gen")
    print(f"    COVID:      {growth['clade_25_covid']:+.6f} Δfreq/gen")
    print(f"    Recovery:   {growth['clade_25_recovery']:+.6f} Δfreq/gen")

    print(f"\n{'='*80}")
    print(f"【Resilience Score (Outcome, relative to Baseline)】")
    print(f"  Clade 2.4:")
    print(f"    Baseline:   {resilience['clade_24_baseline']:.3f} (reference)")
    print(f"    COVID:      {resilience['clade_24_covid']:.3f} ({(resilience['clade_24_covid']-1)*100:+.0f}%)")
    print(f"    Recovery:   {resilience['clade_24_recovery']:.3f} ({(resilience['clade_24_recovery']-1)*100:+.0f}%)")

    print(f"\n  Clade 2.5:")
    print(f"    Baseline:   {resilience['clade_25_baseline']:.3f} (reference)")
    print(f"    COVID:      {resilience['clade_25_covid']:.3f} ({(resilience['clade_25_covid']-1)*100:+.0f}%)")
    print(f"    Recovery:   {resilience['clade_25_recovery']:.3f} ({(resilience['clade_25_recovery']-1)*100:+.0f}%)")

    print(f"\n{'='*80}")
    print("KEY FINDINGS:")
    print(f"{'='*80}")
    print(f"\n1. COVID Impact:")
    print(f"   • Clade 2.4: Growth rate slowed by {((growth['clade_24_covid']-growth['clade_24_baseline'])/abs(growth['clade_24_baseline'])*100):+.0f}%, but gained {(resilience['clade_24_covid']-1)*100:+.0f}% frequency")
    print(f"   • Clade 2.5: Growth rate slowed by {((growth['clade_25_covid']-growth['clade_25_baseline'])/abs(growth['clade_25_baseline'])*100):+.0f}%, and lost {abs(resilience['clade_25_covid']-1)*100:.0f}% frequency")

    print(f"\n2. Recovery Phase:")
    if growth['clade_24_recovery'] > growth['clade_24_covid']:
        print(f"   • Clade 2.4: Recovering (growth rate increased)")
    else:
        print(f"   • Clade 2.4: Still suppressed (growth rate declined further)")

    if growth['clade_25_recovery'] > growth['clade_25_covid']:
        print(f"   • Clade 2.5: Recovering (growth rate increased from {growth['clade_25_covid']:+.6f} to {growth['clade_25_recovery']:+.6f})")
    else:
        print(f"   • Clade 2.5: Still declining (growth rate: {growth['clade_25_recovery']:+.6f})")

    print(f"\n3. Final Outcome (Recovery vs Baseline):")
    print(f"   • Clade 2.4: {(resilience['clade_24_recovery']-1)*100:+.0f}% change from baseline")
    print(f"   • Clade 2.5: {(resilience['clade_25_recovery']-1)*100:+.0f}% change from baseline")
    print(f"{'='*80}\n")


def plot_covid_impact_barplot(timeline_df, intervention_start=300, recovery_start=800,
                              output_path=None):
    """
    Plot COVID-19 impact on dissemination speed as a barplot.

    NOTE: This is the ORIGINAL function. For combined Growth Rate + Resilience,
    use plot_covid_impact_combined() instead.

    Parameters:
        timeline_df: DataFrame with timeline data
        intervention_start: generation when intervention starts
        recovery_start: generation when recovery starts
        output_path: path to save the figure

    Returns:
        None (saves figure to output_path)
    """
    # Calculate dissemination speed
    speeds = calculate_dissemination_speed(timeline_df,
                                          intervention_start=intervention_start,
                                          recovery_start=recovery_start)

    # Prepare data
    clades = ['Clade 2.4', 'Clade 2.5']
    pre_covid = [speeds['clade_24_pre'], speeds['clade_25_pre']]
    covid_era = [speeds['clade_24_covid'], speeds['clade_25_covid']]

    # Calculate speed change percentage (considering sign)
    def calculate_speed_change(pre, covid):
        """Calculate speed change percentage, handling positive and negative values"""
        if abs(pre) > 1e-10:  # Avoid division by zero
            return (covid - pre) / abs(pre) * 100
        else:
            return 0

    change_24 = calculate_speed_change(speeds['clade_24_pre'], speeds['clade_24_covid'])
    change_25 = calculate_speed_change(speeds['clade_25_pre'], speeds['clade_25_covid'])

    # Set bar plot parameters
    x = np.arange(len(clades))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_pre = ['steelblue' if v >= 0 else 'lightcoral' for v in pre_covid]
    colors_covid = ['steelblue' if v >= 0 else 'lightcoral' for v in covid_era]

    bars1 = ax.bar(x - width/2, pre_covid, width, label='Pre-COVID',
                   color=colors_pre, alpha=0.8, edgecolor='navy', linewidth=1.5)
    bars2 = ax.bar(x + width/2, covid_era, width, label='COVID Era',
                   color=colors_covid, alpha=0.8, edgecolor='darkred', linewidth=1.5)
    ax.set_xlabel('Clade', fontsize=13, fontweight='bold')
    ax.set_ylabel('Dissemination Speed (Δfrequency/generation)', fontsize=13)
    ax.set_title('COVID-19 Impact on Dissemination Speed', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(clades, fontsize=12)
    ax.legend(fontsize=11)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    for i, (change, clade_pre, clade_covid) in enumerate([(change_24, pre_covid[0], covid_era[0]),
                                                            (change_25, pre_covid[1], covid_era[1])]):
        max_val = max(abs(clade_pre), abs(clade_covid))
        y_pos = max_val * 1.1 if max_val > 0 else 0.0001
        if change > 0:
            label = f'+{change:.0f}%'
            color = 'darkgreen'
        elif change < 0:
            label = f'{change:.0f}%'
            color = 'darkred'
        else:
            label = '0%'
            color = 'gray'

        ax.text(x[i], y_pos, label,
                ha='center', va='bottom', fontsize=12, fontweight='bold', color=color)
    all_values = pre_covid + covid_era
    y_min = min(all_values + [0]) * 1.15
    y_max = max(all_values + [0]) * 1.15
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"COVID impact barplot saved: {output_path}")

    plt.close()

    # Print summary statistics
    print(f"\n")
    print(f"COVID-19 Impact on Dissemination Speed")
    print(f"Clade 2.4:")
    print(f"  Pre-COVID:  {speeds['clade_24_pre']:+.6f} Δfreq/generation")
    print(f"  COVID Era:  {speeds['clade_24_covid']:+.6f} Δfreq/generation")
    print(f"  Change:     {change_24:+.1f}%")
    print(f"\nClade 2.5:")
    print(f"  Pre-COVID:  {speeds['clade_25_pre']:+.6f} Δfreq/generation")
    print(f"  COVID Era:  {speeds['clade_25_covid']:+.6f} Δfreq/generation")
    print(f"  Change:     {change_25:+.1f}%")
    print(f"Key Finding:")

    # Analyze and print key findings
    if change_24 < 0 and change_25 < 0:
        # If both slowed down
        if abs(change_25) > abs(change_24):
            diff = abs(change_25) - abs(change_24)
            print(f"  ✓ Both clades slowed down during COVID-19 intervention")
            print(f"  ✓ Clade 2.5 (metro-dependent) was hit {diff:.1f}% harder than Clade 2.4")
            print(f"    This confirms that high-mobility strategy suffers more from")
            print(f"    hospital transfer restrictions during COVID-19 interventions.")
        else:
            diff = abs(change_24) - abs(change_25)
            print(f"  ✓ Both clades slowed down during COVID-19 intervention")
            print(f"  ✓ Clade 2.4 was hit {diff:.1f}% harder (unexpected)")
    elif change_24 > 0 and change_25 > 0:
        # If both sped up
        print(f"  ! Both clades accelerated during COVID-19 (unexpected)")
    else:
        # One sped up, one slowed down
        print(f"  Mixed impact: one clade accelerated while the other slowed down")



def plot_covid_impact_migration_velocity(timeline_df, intervention_start=300,
                                         recovery_start=800, output_path=None,
                                         baseline_metro_migration=0.09,
                                         baseline_periph_migration=0.02,
                                         covid_metro_migration=0.015,
                                         covid_periph_migration=0.01):
    """
    Plot COVID-19 impact on migration velocity as a barplot.

    Parameters:
        timeline_df: DataFrame with timeline data
        intervention_start, recovery_start: phase boundaries
        output_path: path to save figure
        baseline_metro_migration, baseline_periph_migration: baseline migration rates
        covid_metro_migration, covid_periph_migration: COVID-era migration rates

    Returns:
        None (saves figure to output_path)
    """
    # Extract clade frequencies in different hospital types from timeline
    baseline_data = timeline_df[timeline_df['generation'] < intervention_start]
    covid_data = timeline_df[(timeline_df['generation'] >= intervention_start) &
                            (timeline_df['generation'] < recovery_start)]

    # Calculate average frequencies
    if len(baseline_data) > 0:
        baseline_m_metro = baseline_data['metro_m_freq'].mean()
        baseline_m_periph = baseline_data['periph_m_freq'].mean()
        baseline_c_metro = baseline_data['metro_c_freq'].mean()
        baseline_c_periph = baseline_data['periph_c_freq'].mean()
    else:
        baseline_m_metro = baseline_m_periph = 0.5
        baseline_c_metro = baseline_c_periph = 0.5

    # Calculate weighted migration velocities (considering clade distribution across hospital types)
    # Clade 2.5 is mainly in metro, so it is more affected by metro migration rates
    clade_25_pre_velocity = (baseline_m_metro * baseline_metro_migration +
                             baseline_m_periph * baseline_periph_migration)
    clade_25_covid_velocity = (baseline_m_metro * covid_metro_migration +
                               baseline_m_periph * covid_periph_migration)

    # Clade 2.4 is mainly in peripheral, so it is more affected by peripheral migration rates
    clade_24_pre_velocity = (baseline_c_metro * baseline_metro_migration +
                            baseline_c_periph * baseline_periph_migration)
    clade_24_covid_velocity = (baseline_c_metro * covid_metro_migration +
                              baseline_c_periph * covid_periph_migration)

    # Calculate decline percentages
    decline_24 = (clade_24_pre_velocity - clade_24_covid_velocity) / clade_24_pre_velocity * 100
    decline_25 = (clade_25_pre_velocity - clade_25_covid_velocity) / clade_25_pre_velocity * 100

    # Plotting
    clades = ['Clade 2.4', 'Clade 2.5']
    pre_covid = [clade_24_pre_velocity, clade_25_pre_velocity]
    covid_era = [clade_24_covid_velocity, clade_25_covid_velocity]

    x = np.arange(len(clades))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, pre_covid, width, label='Pre-COVID',
                   color='steelblue', alpha=0.8, edgecolor='navy', linewidth=1.5)
    bars2 = ax.bar(x + width/2, covid_era, width, label='COVID Era',
                   color='coral', alpha=0.8, edgecolor='darkred', linewidth=1.5)

    ax.set_xlabel('Clade', fontsize=13, fontweight='bold')
    ax.set_ylabel('Migration Velocity (units/timestep)', fontsize=13)
    ax.set_title('COVID-19 Impact on Dissemination Speed', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(clades, fontsize=12)
    ax.legend(fontsize=11, loc='upper right')

    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add decline percentage annotations (above the bars)
    ax.text(x[0], max(pre_covid[0], covid_era[0]) * 1.05,
            f'–{decline_24:.0f}%',
            ha='center', va='bottom', fontsize=13, fontweight='bold', color='darkred')

    ax.text(x[1], max(pre_covid[1], covid_era[1]) * 1.05,
            f'–{decline_25:.0f}%',
            ha='center', va='bottom', fontsize=13, fontweight='bold', color='darkred')

    ax.set_ylim(0, max(pre_covid + covid_era) * 1.2)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"COVID impact migration velocity plot saved: {output_path}")

    plt.close()

    # Print statistics
    print(f"\n")
    print(f"COVID-19 Impact on Migration Velocity")
    print(f"Clade 2.4 (low-mobility strategy):")
    print(f"  Pre-COVID:  {clade_24_pre_velocity:.4f} units/timestep")
    print(f"  COVID Era:  {clade_24_covid_velocity:.4f} units/timestep")
    print(f"  Reduction:  {decline_24:.1f}%")
    print(f"\nClade 2.5 (high-mobility strategy):")
    print(f"  Pre-COVID:  {clade_25_pre_velocity:.4f} units/timestep")
    print(f"  COVID Era:  {clade_25_covid_velocity:.4f} units/timestep")
    print(f"  Reduction:  {decline_25:.1f}%")


def calculate_spatial_dispersion(hospital_populations, hospital_types, coordinates,
                                M_genotypes, C_genotypes):
    """
    Calculate spatial dispersion (Method 1)

    Formula: Effective Distance = Σ(distance_ij × prevalence_i × prevalence_j) / Σ(prevalence_i × prevalence_j)

    Physical Meaning: The "average distance" between hospitals where infections are concentrated
    - Reflects spatial distribution patterns of genotypes
    - Screenshop between highly clustered vs. widely dispersed patterns

    Parameters:
        hospital_populations: List of population arrays for each hospital
        hospital_types: List of hospital types
        coordinates: Tuple of (x_coords, y_coords)
        M_genotypes: List of M-genotype indices (Clade 2.5)
        C_genotypes: List of C-genotype indices (Clade 2.4)

    Returns:
        dict: {
            'dispersion_24': float,  # Clade 2.4空间离散度 (km)
            'dispersion_25': float   # Clade 2.5空间离散度 (km)
        }
    """
    x_coords, y_coords = coordinates
    n_hospitals = len(hospital_populations)

    # Calculate prevalence of each genotype in each hospital
    prevalence_24 = np.zeros(n_hospitals)
    prevalence_25 = np.zeros(n_hospitals)

    for h in range(n_hospitals):
        pop = hospital_populations[h]
        total = len(pop)
        if total > 0:
            n24 = sum(1 for g in pop if g in C_genotypes)
            n25 = sum(1 for g in pop if g in M_genotypes)
            prevalence_24[h] = n24 / total
            prevalence_25[h] = n25 / total

    # Calculate distance matrix between all hospitals
    distance_matrix = np.zeros((n_hospitals, n_hospitals))
    for i in range(n_hospitals):
        for j in range(n_hospitals):
            if i != j:
                dx = x_coords[i] - x_coords[j]
                dy = y_coords[i] - y_coords[j]
                distance_matrix[i, j] = np.sqrt(dx**2 + dy**2)

    # Calculate Clade 2.4 spatial dispersion
    numerator_24 = 0
    denominator_24 = 0
    for i in range(n_hospitals):
        for j in range(i+1, n_hospitals):  # Only calculate upper triangle to avoid duplication
            weight = prevalence_24[i] * prevalence_24[j]
            numerator_24 += distance_matrix[i, j] * weight
            denominator_24 += weight

    dispersion_24 = numerator_24 / denominator_24 if denominator_24 > 0 else 0

    # Calculate Clade 2.5 spatial dispersion
    numerator_25 = 0
    denominator_25 = 0
    for i in range(n_hospitals):
        for j in range(i+1, n_hospitals):
            weight = prevalence_25[i] * prevalence_25[j]
            numerator_25 += distance_matrix[i, j] * weight
            denominator_25 += weight

    dispersion_25 = numerator_25 / denominator_25 if denominator_25 > 0 else 0

    return {
        'dispersion_24': dispersion_24,
        'dispersion_25': dispersion_25
    }


def calculate_effective_transmission_distance(hospital_populations, hospital_types,
                                              coordinates, transfer_matrix,
                                              M_genotypes, C_genotypes):
    """
    Calculate effective transmission distance (Method 2)

    F = Σ(distance_ij × transfer_ij × n_i) / Σ(transfer_ij × n_i)

    Physical Meaning: The "average jump distance" of pathogens transmitted through the transfer network
    - Considers actual transfer directions
    - Reflects transmission dynamics
    - Only calculates pairs of hospitals with actual transfers

    Parameters:
        hospital_populations: List of population arrays
        hospital_types: List of hospital types
        coordinates: Tuple of (x_coords, y_coords)
        transfer_matrix: Transfer probability matrix [n_hospitals, n_hospitals]
        M_genotypes: List of M-genotype indices (Clade 2.5)
        C_genotypes: List of C-genotype indices (Clade 2.4)

    Returns:
        dict: {
            'transmission_dist_24': float,  # Clade 2.4传播距离 (km)
            'transmission_dist_25': float   # Clade 2.5传播距离 (km)
        }
    """
    x_coords, y_coords = coordinates
    n_hospitals = len(hospital_populations)

    # Calculate absolute numbers for each hospital
    n_24 = np.zeros(n_hospitals)
    n_25 = np.zeros(n_hospitals)

    for h in range(n_hospitals):
        pop = hospital_populations[h]
        n_24[h] = sum(1 for g in pop if g in C_genotypes)
        n_25[h] = sum(1 for g in pop if g in M_genotypes)

    # Calculate distance matrix
    distance_matrix = np.zeros((n_hospitals, n_hospitals))
    for i in range(n_hospitals):
        for j in range(n_hospitals):
            if i != j:
                dx = x_coords[i] - x_coords[j]
                dy = y_coords[i] - y_coords[j]
                distance_matrix[i, j] = np.sqrt(dx**2 + dy**2)

    # Calculate Clade 2.4 effective transmission distance
    numerator_24 = 0
    denominator_24 = 0
    for i in range(n_hospitals):
        for j in range(n_hospitals):
            if i != j and transfer_matrix[i, j] > 0:
                weight = transfer_matrix[i, j] * n_24[i]
                numerator_24 += distance_matrix[i, j] * weight
                denominator_24 += weight

    transmission_dist_24 = numerator_24 / denominator_24 if denominator_24 > 0 else 0

    # Calculate Clade 2.5 effective transmission distance
    numerator_25 = 0
    denominator_25 = 0
    for i in range(n_hospitals):
        for j in range(n_hospitals):
            if i != j and transfer_matrix[i, j] > 0:
                weight = transfer_matrix[i, j] * n_25[i]
                numerator_25 += distance_matrix[i, j] * weight
                denominator_25 += weight

    transmission_dist_25 = numerator_25 / denominator_25 if denominator_25 > 0 else 0

    return {
        'transmission_dist_24': transmission_dist_24,
        'transmission_dist_25': transmission_dist_25
    }


def plot_distance_metrics_comparison(metrics_dict, output_path=None):
    """
    Plot comparison of two distance metrics (2 rows, 3 columns)

    Parameters:
        metrics_dict: dict with structure:
            {
                'phase1': {'dispersion_24': float, 'dispersion_25': float,
                          'transmission_dist_24': float, 'transmission_dist_25': float},
                'phase2': {...},
                'phase3': {...}
            }
        output_path: path to save figure

    Returns:
        None (saves figure to output_path)
    """
    phases = ['Phase 1\n(Baseline)', 'Phase 2\n(COVID)', 'Phase 3\n(Recovery)']
    phase_keys = ['phase1', 'phase2', 'phase3']

    # Extract data
    dispersion_24 = [metrics_dict[pk]['dispersion_24'] for pk in phase_keys]
    dispersion_25 = [metrics_dict[pk]['dispersion_25'] for pk in phase_keys]
    transmission_24 = [metrics_dict[pk]['transmission_dist_24'] for pk in phase_keys]
    transmission_25 = [metrics_dict[pk]['transmission_dist_25'] for pk in phase_keys]

    # Create subplots (2 rows, 1 column)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    x = np.arange(len(phases))
    width = 0.35

    # ==================== Panel 1: Spatial Dispersion ====================
    bars1_24 = ax1.bar(x - width/2, dispersion_24, width,
                       label='Clade 2.4', color='indianred', alpha=0.8, edgecolor='darkred', linewidth=1.5)
    bars1_25 = ax1.bar(x + width/2, dispersion_25, width,
                       label='Clade 2.5', color='steelblue', alpha=0.8, edgecolor='navy', linewidth=1.5)

    ax1.set_ylabel('Spatial Dispersion (km)', fontsize=13, fontweight='bold')
    ax1.set_title('Method1: Spatial Dispersion\n(Effective Distance)',
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(phases, fontsize=11)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # Add value labels
    for bars in [bars1_24, bars1_25]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=10)

    # ==================== Panel 2: Effective Transmission Distance ====================
    bars2_24 = ax2.bar(x - width/2, transmission_24, width,
                       label='Clade 2.4', color='indianred', alpha=0.8, edgecolor='darkred', linewidth=1.5)
    bars2_25 = ax2.bar(x + width/2, transmission_25, width,
                       label='Clade 2.5', color='steelblue', alpha=0.8, edgecolor='navy', linewidth=1.5)

    ax2.set_ylabel('Effective Transmission Distance (km)', fontsize=13, fontweight='bold')
    ax2.set_title('Method2: Effective Transmission Distance\nAverage Jump Distance via Transfer Network',
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(phases, fontsize=11)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    # Add value labels
    for bars in [bars2_24, bars2_25]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n")
        print(f"Distance metrics comparison plot saved: {output_path}")

    plt.close()

    # Print detailed statistics
    print(f"\n")
    print(f"Distance Metrics Analysis")

    print(f"\nMethod1: Spatial Dispersion:")
    # print(f"  物理意义: 感染医院之间的加权平均距离")
    print(f"  Clade 2.4:")
    print(f"    Phase 1: {dispersion_24[0]:.2f} km")
    print(f"    Phase 2: {dispersion_24[1]:.2f} km  (Δ = {dispersion_24[1]-dispersion_24[0]:+.2f} km)")
    print(f"    Phase 3: {dispersion_24[2]:.2f} km  (Δ = {dispersion_24[2]-dispersion_24[1]:+.2f} km)")
    print(f"  Clade 2.5:")
    print(f"    Phase 1: {dispersion_25[0]:.2f} km")
    print(f"    Phase 2: {dispersion_25[1]:.2f} km  (Δ = {dispersion_25[1]-dispersion_25[0]:+.2f} km)")
    print(f"    Phase 3: {dispersion_25[2]:.2f} km  (Δ = {dispersion_25[2]-dispersion_25[1]:+.2f} km)")

    print(f"\nMethod2: Effective Transmission Distance - Recommended:")
    # print(f"  物理意义: 通过转院网络传播的平均跳跃距离")
    print(f"  Clade 2.4:")
    print(f"    Phase 1: {transmission_24[0]:.2f} km")
    print(f"    Phase 2: {transmission_24[1]:.2f} km  (Δ = {transmission_24[1]-transmission_24[0]:+.2f} km)")
    print(f"    Phase 3: {transmission_24[2]:.2f} km  (Δ = {transmission_24[2]-transmission_24[1]:+.2f} km)")
    print(f"  Clade 2.5:")
    print(f"    Phase 1: {transmission_25[0]:.2f} km")
    print(f"    Phase 2: {transmission_25[1]:.2f} km  (Δ = {transmission_25[1]-transmission_25[0]:+.2f} km)")
    print(f"    Phase 3: {transmission_25[2]:.2f} km  (Δ = {transmission_25[2]-transmission_25[1]:+.2f} km)")

    # Calculate percentage changes
    if transmission_25[0] > 0:
        pct_change_25_p2 = (transmission_25[1] - transmission_25[0]) / transmission_25[0] * 100
        pct_change_25_p3 = (transmission_25[2] - transmission_25[1]) / transmission_25[1] * 100
    else:
        pct_change_25_p2 = pct_change_25_p3 = 0

    if transmission_24[0] > 0:
        pct_change_24_p2 = (transmission_24[1] - transmission_24[0]) / transmission_24[0] * 100
        pct_change_24_p3 = (transmission_24[2] - transmission_24[1]) / transmission_24[1] * 100
    else:
        pct_change_24_p2 = pct_change_24_p3 = 0

    print(f"\n【关键发现】:")
    print(f"  COVID干预期 (Phase 1→2):")
    print(f"    Clade 2.4传播距离变化: {pct_change_24_p2:+.1f}%")
    print(f"    Clade 2.5传播距离变化: {pct_change_25_p2:+.1f}%")
    if abs(pct_change_25_p2) > abs(pct_change_24_p2):
        print(f"    ✓ Clade 2.5受COVID干预影响更大（依赖长距离转院）")

    print(f"\n  恢复期 (Phase 2→3):")
    print(f"    Clade 2.4传播距离变化: {pct_change_24_p3:+.1f}%")
    print(f"    Clade 2.5传播距离变化: {pct_change_25_p3:+.1f}%")
    if abs(pct_change_25_p3) > abs(pct_change_24_p3):
        print(f"    ✓ Clade 2.5恢复更显著（依赖长距离转院）")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else 'covid_impact_barplot.png'

        df = pd.read_csv(csv_path)
        plot_covid_impact_barplot(df, output_path=output_path)
    else:
        print("Usage: python covid_impact.py <timeline_csv> [output_path]")
