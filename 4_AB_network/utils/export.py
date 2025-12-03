#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export Module
From model_main_d.py ：
- file1: Genotype configuration CSV
- file2: Summary text file
- file3: Results CSV
- file4: Network analysis JSON
- file5: Transfer matrix CSV
- file6: Transfer summary 2x2 CSV
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime


def classify_genotypes_by_strategy(TAU_VALUES):
    """
    Classify genotypes into metro-oriented and peripheral-oriented strategies.

    Parameters:
        TAU_VALUES: ndarray, M values for each genotype

    Returns:
        tuple: (metro_genotypes, periph_genotypes) - lists of genotype indices
    """
    METRO_THRESHOLD = 0.6
    PERIPH_THRESHOLD = 0.4

    metro_genotypes = [g for g in range(len(TAU_VALUES)) if TAU_VALUES[g] > METRO_THRESHOLD]
    periph_genotypes = [g for g in range(len(TAU_VALUES)) if TAU_VALUES[g] < PERIPH_THRESHOLD]

    return metro_genotypes, periph_genotypes


def print_degree_statistics(degree_stats, hospital_types):
    """
    Print degree statistics summary to console

    Parameters:
        degree_stats: dict, degree statistics for each hospital
        hospital_types: list, hospital types
    """
    print("\n" + "="*70)
    print("Network Degree Statistics")
    print("="*70)

    # Extract degree values by hospital type
    metro_degrees = []
    periph_degrees = []
    all_degrees = []

    for hospital_id, info in degree_stats.items():
        degree = info['degree']
        all_degrees.append(degree)

        if info['type'] == 'metro':
            metro_degrees.append(degree)
        else:
            periph_degrees.append(degree)

    # Calculate statistics
    print(f"\nOverall (n={len(all_degrees)}):")
    print(f"  Mean degree:   {np.mean(all_degrees):.2f}")
    print(f"  Median degree: {np.median(all_degrees):.2f}")
    print(f"  Min degree:    {np.min(all_degrees)}")
    print(f"  Max degree:    {np.max(all_degrees)}")
    print(f"  Std degree:    {np.std(all_degrees):.2f}")

    if len(metro_degrees) > 0:
        print(f"\nLarge (Metro) hospitals (n={len(metro_degrees)}):")
        print(f"  Mean degree:   {np.mean(metro_degrees):.2f}")
        print(f"  Median degree: {np.median(metro_degrees):.2f}")
        print(f"  Min degree:    {np.min(metro_degrees)}")
        print(f"  Max degree:    {np.max(metro_degrees)}")
        print(f"  Std degree:    {np.std(metro_degrees):.2f}")

    if len(periph_degrees) > 0:
        print(f"\nSmall (Peripheral) hospitals (n={len(periph_degrees)}):")
        print(f"  Mean degree:   {np.mean(periph_degrees):.2f}")
        print(f"  Median degree: {np.median(periph_degrees):.2f}")
        print(f"  Min degree:    {np.min(periph_degrees)}")
        print(f"  Max degree:    {np.max(periph_degrees)}")
        print(f"  Std degree:    {np.std(periph_degrees):.2f}")

    # Analyze transfer patterns
    transfer_patterns = {
        'metro_to_metro': 0,
        'metro_to_periph': 0,
        'periph_to_metro': 0,
        'periph_to_periph': 0
    }

    total_connections = 0
    for hospital_id, info in degree_stats.items():
        source_type = info['type']
        connections = info.get('connections', [])

        for target_id in connections:
            target_info = degree_stats.get(str(target_id), {})
            target_type = target_info.get('type', 'unknown')
            total_connections += 1

            if source_type == 'metro' and target_type == 'metro':
                transfer_patterns['metro_to_metro'] += 1
            elif source_type == 'metro' and target_type == 'peripheral':
                transfer_patterns['metro_to_periph'] += 1
            elif source_type == 'peripheral' and target_type == 'metro':
                transfer_patterns['periph_to_metro'] += 1
            elif source_type == 'peripheral' and target_type == 'peripheral':
                transfer_patterns['periph_to_periph'] += 1

    print(f"\n" + "="*70)
    print("Transfer Pattern Distribution")
    print("="*70)
    print(f"\nTotal connections: {total_connections}")

    if total_connections > 0:
        print(f"\n{'Transfer Type':<25} {'Count':>10} {'Percentage':>12}")
        print("-"*50)

        pattern_labels = {
            'metro_to_metro': 'Large → Large',
            'metro_to_periph': 'Large → Small',
            'periph_to_metro': 'Small → Large',
            'periph_to_periph': 'Small → Small'
        }

        for pattern_name, count in transfer_patterns.items():
            percentage = (count / total_connections * 100)
            print(f"{pattern_labels[pattern_name]:<25} {count:>10} {percentage:>11.1f}%")

    print("="*70 + "\n")


# export_genotype_config() - REMOVED (replaced by run_config.json)


# export_summary_text() - REMOVED (data already in results.csv)


def export_results_csv(stats, metro_migration, periph_migration, output_path):
    """
    Export detailed results CSV (file3)
    
    Parameters:
        stats: dict, statistics from calculate_fst_statistics
        metro_migration: float
        periph_migration: float
        output_path: str
    """
    row = {
        'metro_migration': metro_migration,
        'peripheral_migration': periph_migration,
        'fst_mean_weighted': stats['fst_mean_weighted'],
        'metro_specialist_freq': stats['metro_specialist_freq'],
        'periph_specialist_freq': stats['periph_specialist_freq'],
        'dominant_genotype': stats['dominant_genotype'],
        'shannon_diversity': np.mean(stats['shannon_diversity']),
        'metro_within_fst_simple': stats['metro_within_fst_simple'],
        'metro_within_fst_weighted': stats['metro_within_fst_weighted'],
        'periph_within_fst_simple': stats['periph_within_fst_simple'],
        'periph_within_fst_weighted': stats['periph_within_fst_weighted'],
    }
    
    # Add per-genotype FST
    for g, fst in enumerate(stats['fst_per_genotype']):
        row[f'fst_g{g}'] = fst
    
    df = pd.DataFrame([row])
    df.to_csv(output_path, index=False)
    print(f"Results CSV saved: {output_path}")


def export_transfer_matrix(transfer_matrix, hospital_types, output_path):
    """
    Export transfer matrix CSV (file5)
    
    Parameters:
        transfer_matrix: ndarray, shape (n_hospitals, n_hospitals)
        hospital_types: list of str
        output_path: str
    """
    hospital_labels = [f"{htype}_{i}" for i, htype in enumerate(hospital_types)]
    
    df_matrix = pd.DataFrame(
        transfer_matrix,
        index=hospital_labels,
        columns=hospital_labels
    )
    df_matrix.to_csv(output_path)
    print(f"Transfer matrix saved: {output_path}")


# export_transfer_summary_2x2() - REMOVED (not a true summary, just sample values)


def export_run_config(args, TAU_VALUES, BETA_VALUES, scenario_type, output_path):
    """
    Export run configuration to JSON (replaces file1 and records all parameters)

    Parameters:
        args: argparse.Namespace, CLI arguments
        TAU_VALUES: ndarray, M values for genotypes
        BETA_VALUES: ndarray, C values for genotypes
        scenario_type: str, 'basic' / 'invasion' / 'covid'
        output_path: str, output file path
    """
    config_data = {
        'scenario': scenario_type,
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'hospitals': args.hospitals,
            'metros': args.metros,
            'population': args.population,
            'genotypes': args.genotypes,
            'generations': args.generations,
            'seed': args.seed,
            'metro_migration': args.metro_migration,
            'peripheral_migration': args.peripheral_migration,
            'selection_strength': args.selection,
            'spatial_scale': args.spatial_scale,
        },
        'genotype_values': {
            'TAU_VALUES': [float(m) for m in TAU_VALUES],
            'BETA_VALUES': [float(c) for c in BETA_VALUES],
        }
    }

    # Add decay_length only if it exists (basic and invasion scenarios)
    if hasattr(args, 'decay_length'):
        config_data['parameters']['decay_length'] = args.decay_length

    # Add scenario-specific parameters
    if scenario_type == 'invasion':
        config_data['parameters'].update({
            'introduction_time': args.introduction_time,
            'introduction_dose': args.introduction_dose,
            'introduction_sites': args.introduction_sites,
        })
    elif scenario_type == 'covid':
        config_data['parameters'].update({
            'intervention_start': args.intervention_start,
            'recovery_start': args.recovery_start,
            'baseline_decay': args.baseline_decay,
            'intervention_decay': args.intervention_decay,
            'recovery_decay': args.recovery_decay,
            'metro_closure': args.metro_closure,
            'periph_closure': args.periph_closure,
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)

    print(f"Run config saved: {output_path}")


def export_network_analysis_json(G, hospital_types, genotype_freqs, TAU_VALUES, BETA_VALUES,
                                 time_series, current_step, radar_metrics=None,
                                 output_path='network_analysis.json'):
    """
    Export network analysis JSON file (similar to test5 format)

    Parameters:
        G: NetworkX graph
        hospital_types: list of hospital types
        genotype_freqs: shape (n_hospitals, n_genotypes)
        TAU_VALUES, BETA_VALUES: genotype parameters
        time_series: time series data
        current_step: current step
        radar_metrics: dict, optional radar chart metrics (integrates file7)
        output_path: output path for the JSON file
    """
    metro_genotypes, periph_genotypes = classify_genotypes_by_strategy(TAU_VALUES)

    # Degree statistics
    degrees = dict(G.degree())
    degree_stats = {}

    for i in range(len(hospital_types)):
        # Calculate the frequency of the two major genotype groups
        metro_genotypes_total = np.sum(genotype_freqs[i, metro_genotypes]) if len(metro_genotypes) > 0 else 0
        periph_genotypes_total = np.sum(genotype_freqs[i, periph_genotypes]) if len(periph_genotypes) > 0 else 0

        # Determine the dominant genotype group
        if metro_genotypes_total > periph_genotypes_total * 2:
            dominant_genotype_group = 'metro_oriented'
        elif periph_genotypes_total > metro_genotypes_total * 2:
            dominant_genotype_group = 'periph_oriented'
        elif metro_genotypes_total + periph_genotypes_total < 0.1:
            dominant_genotype_group = 'none'
        else:
            dominant_genotype_group = 'mixed'

        degree_stats[str(i)] = {
            'hospital_id': str(i),
            'type': hospital_types[i],
            'degree': degrees.get(i, 0),
            'label': f"{hospital_types[i]}_{i}",
            'connections': [int(n) for n in G.neighbors(i)],
            'metro_genotypes_freq': float(metro_genotypes_total),
            'periph_genotypes_freq': float(periph_genotypes_total),
            'dominant_genotype_group': dominant_genotype_group,
            'genotype_freqs': [float(f) for f in genotype_freqs[i, :]]
        }

    # Hospital frequency summary
    hospital_frequencies = {}
    for i in range(len(hospital_types)):
        metro_genotypes_total = np.sum(genotype_freqs[i, metro_genotypes]) if len(metro_genotypes) > 0 else 0
        periph_genotypes_total = np.sum(genotype_freqs[i, periph_genotypes]) if len(periph_genotypes) > 0 else 0

        if metro_genotypes_total > periph_genotypes_total * 2:
            dominant_genotype_group = 'metro_oriented'
        elif periph_genotypes_total > metro_genotypes_total * 2:
            dominant_genotype_group = 'periph_oriented'
        elif metro_genotypes_total + periph_genotypes_total < 0.1:
            dominant_genotype_group = 'none'
        else:
            dominant_genotype_group = 'mixed'

        hospital_frequencies[str(i)] = {
            'type': hospital_types[i],
            'metro_genotypes_total': float(metro_genotypes_total),
            'periph_genotypes_total': float(periph_genotypes_total),
            'dominant_genotype_group': dominant_genotype_group,
            'genotype_freqs': [float(f) for f in genotype_freqs[i, :]]
        }

    analysis_data = {
        'timestamp': datetime.now().isoformat(),
        'current_step': current_step,
        'network_info': {
            'total_hospitals': len(hospital_types),
            'total_connections': G.number_of_edges(),
            'num_metro': len([t for t in hospital_types if t == 'metro']),
            'num_peripheral': len([t for t in hospital_types if t == 'peripheral'])
        },
        'genotype_classification': {
            'metro_oriented_genotypes': metro_genotypes,
            'periph_oriented_genotypes': periph_genotypes,
            'TAU_VALUES': [float(m) for m in TAU_VALUES],
            'BETA_VALUES': [float(c) for c in BETA_VALUES]
        },
        'degree_stats': degree_stats,
        'hospital_frequencies': hospital_frequencies
    }

    # Add radar_metrics if provided (integrates file7)
    if radar_metrics is not None:
        analysis_data['radar_metrics'] = radar_metrics

    if time_series is not None:
        # Convert time_series data to JSON-serializable format
        # Handle both numpy arrays and lists of numpy arrays
        def convert_to_list(data):
            if isinstance(data, list):
                # List of numpy arrays or nested lists
                return [arr.tolist() if hasattr(arr, 'tolist') else arr for arr in data]
            elif hasattr(data, 'tolist'):
                # Single numpy array
                return data.tolist()
            else:
                # Already a list
                return data

        analysis_data['time_series'] = {
            'overall_freqs': convert_to_list(time_series['overall_freqs']),
            'metro_freqs': convert_to_list(time_series['metro_freqs']),
            'periph_freqs': convert_to_list(time_series['periph_freqs'])
        }
        if 'hospital_freqs' in time_series:
            analysis_data['time_series']['hospital_freqs'] = convert_to_list(time_series['hospital_freqs'])

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, ensure_ascii=False, indent=2)

    print(f"Network analysis JSON saved: {output_path}")

    # Print degree statistics summary
    print_degree_statistics(degree_stats, hospital_types)

    return analysis_data



