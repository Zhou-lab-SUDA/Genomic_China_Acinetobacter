#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Invasion Visualization Module

Includes functions to plot the invasion dynamics of 
two competing pathogen clades (2.4 and 2.5) in a hospital network simulation:
- plot_invasion_curve_combined()
- plot_invasion_curve_combined_plotly()
- plot_invasion_curve()


invasion_timeline DataFrame:
    - generation
    - metro_m_freq
    - periph_m_freq
    - overall_m_freq
    - metro_c_freq
    - periph_c_freq
    - overall_c_freq
"""

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import pandas as pd


def smooth_data(data, window_size=15):
    """
    Apply moving average smoothing to the data

    Parameters:
        data
        window_size: 15

    Returns:
        Smoothed data array
    """
    if isinstance(data, pd.Series):
        return data.rolling(window=window_size, min_periods=1, center=True).mean()
    else:
        df = pd.Series(data)
        return df.rolling(window=window_size, min_periods=1, center=True).mean().values


def plot_invasion_curve_combined(invasion_timeline, introduction_time, output_path, smooth=True, smooth_window=20, second_phase_time=None):
    """
    Plot both clades on the same graph to show competition dynamics (static version)

    Parameters:
        invasion_timeline: DataFrame with [generation, metro_m_freq, metro_c_freq, ...]
        introduction_time: Generation when 2.5 was introduced OR when phase 1 ends
        output_path: Output file path
        smooth: If True, apply moving average smoothing (default: True)
        smooth_window: Window size for smoothing (default: 20)
        second_phase_time: Optional. If provided, marks end of phase 2 (for 3-phase scenarios)

    Returns:
        None (saves figure to output_path)
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    generations = invasion_timeline['generation']

    # Smooth data (if needed)
    if smooth:
        overall_c_freq = smooth_data(invasion_timeline['overall_c_freq'], smooth_window)
        metro_c_freq = smooth_data(invasion_timeline['metro_c_freq'], smooth_window)
        periph_c_freq = smooth_data(invasion_timeline['periph_c_freq'], smooth_window)
        overall_m_freq = smooth_data(invasion_timeline['overall_m_freq'], smooth_window)
        metro_m_freq = smooth_data(invasion_timeline['metro_m_freq'], smooth_window)
        periph_m_freq = smooth_data(invasion_timeline['periph_m_freq'], smooth_window)
    else:
        overall_c_freq = invasion_timeline['overall_c_freq']
        metro_c_freq = invasion_timeline['metro_c_freq']
        periph_c_freq = invasion_timeline['periph_c_freq']
        overall_m_freq = invasion_timeline['overall_m_freq']
        metro_m_freq = invasion_timeline['metro_m_freq']
        periph_m_freq = invasion_timeline['periph_m_freq']

    # ============ Clade 2.4 (RED) ============
    ax.plot(generations, overall_c_freq,
            'darkred', linewidth=3, label='Clade 2.4 (overall)', linestyle='-', alpha=0.8)
    ax.plot(generations, metro_c_freq,
            'red', linewidth=2, label='Clade 2.4 (metro)', linestyle='--', alpha=0.6)
    ax.plot(generations, periph_c_freq,
            'indianred', linewidth=2, label='Clade 2.4 (peripheral)', linestyle=':', alpha=0.6)

    # ============ Clade 2.5 (BLUE) ============
    ax.plot(generations, overall_m_freq,
            'darkblue', linewidth=3, label='Clade 2.5 (overall)', linestyle='-', alpha=0.8)
    ax.plot(generations, metro_m_freq,
            'blue', linewidth=2, label='Clade 2.5 (metro)', linestyle='--', alpha=0.6)
    ax.plot(generations, periph_m_freq,
            'cornflowerblue', linewidth=2, label='Clade 2.5 (peripheral)', linestyle=':', alpha=0.6)

    # Mark phase transitions
    if second_phase_time is not None:
        # Three-phase scenario (e.g., COVID intervention)
        ax.axvline(x=introduction_time, color='orange', linestyle='--',
                  linewidth=2.5, alpha=0.7, label=f'Intervention Start (Gen {introduction_time})')
        ax.axvline(x=second_phase_time, color='green', linestyle='--',
                  linewidth=2.5, alpha=0.7, label=f'Recovery Start (Gen {second_phase_time})')

        # Add phase labels
        mid_phase1 = introduction_time / 2
        mid_phase2 = (introduction_time + second_phase_time) / 2
        mid_phase3 = (second_phase_time + generations.max()) / 2

        ax.text(mid_phase1, 1.02, 'Phase 1: Baseline', ha='center', va='bottom',
                fontsize=10, style='italic', alpha=0.6)
        ax.text(mid_phase2, 1.02, 'Phase 2: Intervention', ha='center', va='bottom',
                fontsize=10, style='italic', alpha=0.6, color='orange')
        ax.text(mid_phase3, 1.02, 'Phase 3: Recovery', ha='center', va='bottom',
                fontsize=10, style='italic', alpha=0.6, color='green')

        title = 'COVID-19 Intervention Impact on Clade Competition'
    else:
        # Two-phase scenario (invasion)
        ax.axvline(x=introduction_time, color='green', linestyle='--',
                  linewidth=2.5, alpha=0.7, label=f'Clade 2.5 Introduction (Gen {introduction_time})')
        title = 'Clade Competition Dynamics: 2.4 (Red) vs 2.5 (Blue)'

    ax.set_xlabel('Generation', fontsize=13)
    ax.set_ylabel('Clade Frequency', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.legend(loc='center right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(generations.min(), generations.max())

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined invasion curve saved: {output_path}")
    plt.close()


def plot_invasion_curve_combined_plotly(invasion_timeline, introduction_time, output_path):
    """
    Plot both clades on the same graph using Plotly (interactive version)

    Parameters:
        invasion_timeline: DataFrame with [generation, metro_m_freq, metro_c_freq, ...]
        introduction_time: Generation when 2.5 was introduced
        output_path: Output file path (should end with .html)

    Returns:
        fig (plotly.graph_objects.Figure): Plotly figure object
    """
    fig = go.Figure()

    generations = invasion_timeline['generation']

    # ============ Clade 2.4 (RED) - Overall ============
    fig.add_trace(go.Scatter(
        x=generations,
        y=invasion_timeline['overall_c_freq'],
        mode='lines',
        name='Clade 2.4 (overall)',
        line=dict(color='darkred', width=3),
        legendgroup='clade24',
        hovertemplate='<b>Clade 2.4 Overall</b><br>Gen: %{x}<br>Freq: %{y:.3f}<extra></extra>'
    ))

    # Clade 2.4 - Metro
    fig.add_trace(go.Scatter(
        x=generations,
        y=invasion_timeline['metro_c_freq'],
        mode='lines',
        name='Clade 2.4 (metro)',
        line=dict(color='red', width=2, dash='dash'),
        legendgroup='clade24',
        hovertemplate='<b>Clade 2.4 Metro</b><br>Gen: %{x}<br>Freq: %{y:.3f}<extra></extra>'
    ))

    # Clade 2.4 - Peripheral
    fig.add_trace(go.Scatter(
        x=generations,
        y=invasion_timeline['periph_c_freq'],
        mode='lines',
        name='Clade 2.4 (peripheral)',
        line=dict(color='indianred', width=2, dash='dot'),
        legendgroup='clade24',
        hovertemplate='<b>Clade 2.4 Peripheral</b><br>Gen: %{x}<br>Freq: %{y:.3f}<extra></extra>'
    ))

    # ============ Clade 2.5 (BLUE) - Overall ============
    fig.add_trace(go.Scatter(
        x=generations,
        y=invasion_timeline['overall_m_freq'],
        mode='lines',
        name='Clade 2.5 (overall)',
        line=dict(color='darkblue', width=3),
        legendgroup='clade25',
        hovertemplate='<b>Clade 2.5 Overall</b><br>Gen: %{x}<br>Freq: %{y:.3f}<extra></extra>'
    ))

    # Clade 2.5 - Metro
    fig.add_trace(go.Scatter(
        x=generations,
        y=invasion_timeline['metro_m_freq'],
        mode='lines',
        name='Clade 2.5 (metro)',
        line=dict(color='blue', width=2, dash='dash'),
        legendgroup='clade25',
        hovertemplate='<b>Clade 2.5 Metro</b><br>Gen: %{x}<br>Freq: %{y:.3f}<extra></extra>'
    ))

    # Clade 2.5 - Peripheral
    fig.add_trace(go.Scatter(
        x=generations,
        y=invasion_timeline['periph_m_freq'],
        mode='lines',
        name='Clade 2.5 (peripheral)',
        line=dict(color='cornflowerblue', width=2, dash='dot'),
        legendgroup='clade25',
        hovertemplate='<b>Clade 2.5 Peripheral</b><br>Gen: %{x}<br>Freq: %{y:.3f}<extra></extra>'
    ))

    # Add vertical line for introduction time
    fig.add_vline(
        x=introduction_time,
        line_width=2.5,
        line_dash='dash',
        line_color='green',
        annotation_text=f'Clade 2.5 Introduction (Gen {introduction_time})',
        annotation_position='top'
    )

    # Layout
    fig.update_layout(
        title='Clade Competition Dynamics: 2.4 (Red) vs 2.5 (Blue)',
        xaxis_title='Generation',
        yaxis_title='Clade Frequency',
        hovermode='x unified',
        template='plotly_white',
        font=dict(size=12),
        legend=dict(x=1.02, y=0.5),
        width=1200,
        height=700
    )

    fig.update_yaxes(range=[-0.05, 1.05])

    # Save
    fig.write_html(output_path)
    print(f"Interactive invasion curve saved: {output_path}")

    return fig


def plot_invasion_curve(invasion_timeline, introduction_time, output_path):
    """
    Plot invasion success trajectories (both clades, separate panels)

    Parameters:
        invasion_timeline: DataFrame with [generation, metro_m_freq, metro_c_freq, ...]
        introduction_time: Generation when 2.5 was introduced
        output_path: Output file path

    Returns:
        None (saves figure to output_path)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    generations = invasion_timeline['generation']

    # ============ Panel 1: Clade 2.4 (RED) ============
    ax1.plot(generations, invasion_timeline['metro_c_freq'],
            'darkred', linewidth=2.5, label='Metro hospitals', linestyle='-')
    ax1.plot(generations, invasion_timeline['periph_c_freq'],
            'indianred', linewidth=2.5, label='Peripheral hospitals', linestyle='-')
    ax1.plot(generations, invasion_timeline['overall_c_freq'],
            'gray', linewidth=2, linestyle='--', alpha=0.7, label='System-wide')

    # Mark introduction time
    ax1.axvline(x=introduction_time, color='green', linestyle='--',
               linewidth=2, alpha=0.7, label=f'Clade 2.5 Intro (Gen {introduction_time})')

    ax1.set_ylabel('Clade 2.4 Frequency', fontsize=12)
    ax1.set_title('Clade 2.4 Dynamics (C-genotypes, Periph-oriented)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    # ============ Panel 2: Clade 2.5 (BLUE) ============
    ax2.plot(generations, invasion_timeline['metro_m_freq'],
            'darkblue', linewidth=2.5, label='Metro hospitals', linestyle='-')
    ax2.plot(generations, invasion_timeline['periph_m_freq'],
            'cornflowerblue', linewidth=2.5, label='Peripheral hospitals', linestyle='-')
    ax2.plot(generations, invasion_timeline['overall_m_freq'],
            'gray', linewidth=2, linestyle='--', alpha=0.7, label='System-wide')

    # Mark introduction time
    ax2.axvline(x=introduction_time, color='green', linestyle='--',
               linewidth=2, alpha=0.7, label=f'Clade 2.5 Intro (Gen {introduction_time})')

    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Clade 2.5 Frequency', fontsize=12)
    ax2.set_title('Clade 2.5 Dynamics (M-genotypes, Metro-oriented)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Invasion curve (separated panels) saved: {output_path}")
    plt.close()


def plot_prevalence_dynamics(invasion_timeline, introduction_time, output_path, smooth=True, smooth_window=20, second_phase_time=None):
    """
    Plot absolute prevalence (n/N_total) for both clades over time
    Parameters:
        invasion_timeline: DataFrame with [generation, prevalence24, prevalence25, ...]
        introduction_time: Generation when 2.5 was introduced OR when phase 1 ends
        output_path: Output file path
        smooth: If True, apply moving average smoothing (default: True)
        smooth_window: Window size for smoothing (default: 20)
        second_phase_time: Optional. If provided, marks end of phase 2 (for 3-phase scenarios)

    Returns:
        None (saves figure to output_path)
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    generations = invasion_timeline['generation']

    if smooth:
        prevalence24 = smooth_data(invasion_timeline['prevalence24'], smooth_window)
        prevalence25 = smooth_data(invasion_timeline['prevalence25'], smooth_window)
    else:
        prevalence24 = invasion_timeline['prevalence24']
        prevalence25 = invasion_timeline['prevalence25']

    # ============ Plot prevalence lines ============
    ax.plot(generations, prevalence24,
            'darkred', linewidth=3, label='Clade 2.4 prevalence', linestyle='-', alpha=0.8)
    ax.plot(generations, prevalence25,
            'darkblue', linewidth=3, label='Clade 2.5 prevalence', linestyle='-', alpha=0.8)

    # Mark phase transitions
    if second_phase_time is not None:
        # Three-phase scenario (e.g., COVID intervention)
        ax.axvline(x=introduction_time, color='orange', linestyle='--',
                  linewidth=2.5, alpha=0.7, label=f'Intervention Start (Gen {introduction_time})')
        ax.axvline(x=second_phase_time, color='green', linestyle='--',
                  linewidth=2.5, alpha=0.7, label=f'Recovery Start (Gen {second_phase_time})')

        # Add phase labels
        mid_phase1 = introduction_time / 2
        mid_phase2 = (introduction_time + second_phase_time) / 2
        mid_phase3 = (second_phase_time + generations.max()) / 2

        y_pos = ax.get_ylim()[1] * 0.95
        ax.text(mid_phase1, y_pos, 'Phase 1: Baseline', ha='center', va='top',
                fontsize=10, style='italic', alpha=0.6)
        ax.text(mid_phase2, y_pos, 'Phase 2: Intervention', ha='center', va='top',
                fontsize=10, style='italic', alpha=0.6, color='orange')
        ax.text(mid_phase3, y_pos, 'Phase 3: Recovery', ha='center', va='top',
                fontsize=10, style='italic', alpha=0.6, color='green')

        title = 'Absolute Prevalence Dynamics: COVID-19 Intervention Impact'
    else:
        # Two-phase scenario (invasion)
        ax.axvline(x=introduction_time, color='green', linestyle='--',
                  linewidth=2.5, alpha=0.7, label=f'Clade 2.5 Introduction (Gen {introduction_time})')
        title = 'Absolute Prevalence Dynamics: Clade 2.4 vs 2.5'

    ax.set_xlabel('Generation', fontsize=13)
    ax.set_ylabel('Prevalence (n/N_total)', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(generations.min(), generations.max())

    # Set y-axis to start from 0
    y_max = max(prevalence24.max(), prevalence25.max()) * 1.1
    ax.set_ylim(0, y_max)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Prevalence dynamics plot saved: {output_path}")
    plt.close()
