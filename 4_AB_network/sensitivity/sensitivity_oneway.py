#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-at-a-time Sensitivity Analysis
Tests core parameters independently to assess model robustness.

Usage:
    python sensitivity_oneway.py --output-dir sensitivity_results
    python sensitivity_oneway.py --quick  # Fast test with fewer replicates
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DEFAULT_N_HOSPITALS, DEFAULT_N_METROS, DEFAULT_GENERATIONS,
    DEFAULT_POPULATION, DEFAULT_N_GENOTYPES, DEFAULT_SEED,
    DEFAULT_METRO_MIGRATION, DEFAULT_PERIPH_MIGRATION,
    DEFAULT_SELECTION_STRENGTH, DEFAULT_SPATIAL_SCALE, DEFAULT_DECAY_LENGTH
)

from core.engine import (
    calculate_fitness,
    wright_fisher_reproduction,
    migrate_between_hospitals_with_transfer,
    generate_hospital_coordinates,
    initialize_simulation,
)

from config import generate_random_genotype_values, classify_genotypes
from utils.statistics import calculate_fst_statistics


def run_single_simulation(params):
    """
    Run a single simulation and return key metrics.

    Parameters:
        params: dict with simulation parameters

    Returns:
        dict with results
    """
    np.random.seed(params['seed'])

    # Generate genotypes
    TAU_VALUES, BETA_VALUES = generate_random_genotype_values(
        n_genotypes=params['genotypes'],
        seed=params['seed'],
        strategy='complementary'
    )

    genotype_classes = classify_genotypes(TAU_VALUES, exclude_neutral=False)
    M_genotypes = genotype_classes['metro']
    C_genotypes = genotype_classes['periph']

    # Generate coordinates
    coordinates = generate_hospital_coordinates(
        params['hospitals'],
        params['spatial_scale'],
        seed=params['seed']
    )

    # Initialize populations
    populations, hospital_types = initialize_simulation(
        n_hospitals=params['hospitals'],
        n_metros=params['metros'],
        population_size=params['population'],
        n_genotypes=params['genotypes'],
        TAU_array=TAU_VALUES,
        initial_prevalence=0.05,
        seed_genotypes='balanced'
    )

    # Set migration rates
    migration_rates = [
        params['metro_migration'] if ht == 'metro' else params['peripheral_migration']
        for ht in hospital_types
    ]

    metro_indices = [i for i, t in enumerate(hospital_types) if t == 'metro']
    periph_indices = [i for i, t in enumerate(hospital_types) if t == 'peripheral']

    # Main simulation loop (simplified - no detailed recording)
    for gen in range(params['generations']):
        # Selection and reproduction
        for h in range(params['hospitals']):
            fitness = np.array([
                calculate_fitness(g, hospital_types[h], TAU_VALUES, BETA_VALUES, params['selection'])
                for g in populations[h]
            ])
            populations[h] = wright_fisher_reproduction(
                populations[h], fitness, params['population']
            )

        # Migration
        populations = migrate_between_hospitals_with_transfer(
            populations, migration_rates, hospital_types, params['population'],
            coordinates=coordinates,
            decay_length=params['decay_length'],
            return_matrix=False
        )

    # Calculate final metrics
    # 1. Basic frequencies
    genotype_freqs = np.zeros((params['hospitals'], params['genotypes']))
    m_freqs = np.zeros(params['hospitals'])
    c_freqs = np.zeros(params['hospitals'])

    for h in range(params['hospitals']):
        for g in range(params['genotypes']):
            genotype_freqs[h, g] = np.mean(populations[h] == g)
        m_freqs[h] = np.sum([genotype_freqs[h, g] for g in M_genotypes])
        c_freqs[h] = np.sum([genotype_freqs[h, g] for g in C_genotypes])

    metro_m_freq = np.mean(m_freqs[metro_indices]) if metro_indices else 0.0
    periph_m_freq = np.mean(m_freqs[periph_indices]) if periph_indices else 0.0
    overall_m_freq = np.mean(m_freqs)

    # 2. FST and diversity statistics
    fst_stats = calculate_fst_statistics(genotype_freqs, hospital_types, TAU_VALUES, BETA_VALUES)

    # 3. Coexistence criteria
    # Ideal: matches real data pattern (metro 70-90%, periph 25-40%, overall 40-60%)
    ideal_coexistence = (
        0.70 < metro_m_freq < 0.90 and
        0.25 < periph_m_freq < 0.40 and
        0.40 < overall_m_freq < 0.60 and
        (metro_m_freq - periph_m_freq) > 0.30  # Strong segregation
    )

    # Acceptable: relaxed criteria
    acceptable_coexistence = (
        metro_m_freq > periph_m_freq + 0.20 and
        0.30 < overall_m_freq < 0.70
    )

    # Spatial segregation strength
    segregation_strength = metro_m_freq - periph_m_freq

    return {
        # Frequencies
        'metro_m_freq': metro_m_freq,
        'periph_m_freq': periph_m_freq,
        'overall_m_freq': overall_m_freq,
        'metro_c_freq': 1.0 - metro_m_freq,
        'periph_c_freq': 1.0 - periph_m_freq,

        # Coexistence metrics
        'ideal_coexistence': ideal_coexistence,
        'acceptable_coexistence': acceptable_coexistence,
        'segregation_strength': segregation_strength,

        # FST and diversity
        'fst_weighted': fst_stats['fst_mean_weighted'],
        'shannon_diversity': np.mean(fst_stats['shannon_diversity']),
        'metro_within_fst': fst_stats['metro_within_fst_weighted'],
        'periph_within_fst': fst_stats['periph_within_fst_weighted'],
    }


def main():
    parser = argparse.ArgumentParser(
        description='One-at-a-time sensitivity analysis'
    )
    parser.add_argument('--output-dir', type=str, default='sensitivity_results',
                       help='Output directory')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: fewer replicates for testing')
    parser.add_argument('--n-replicates', type=int, default=10,
                       help='Number of replicates per parameter value')
    parser.add_argument('--param-filter', type=str, default=None,
                       help='Only test specific parameter(s). Options: selection, metro_migration, '
                            'peripheral_migration, decay_length, metros, mobility_ratio, or "all" for all parameters')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Baseline parameters
    baseline = {
        'hospitals': DEFAULT_N_HOSPITALS,
        'metros': DEFAULT_N_METROS,
        'population': DEFAULT_POPULATION,
        'genotypes': DEFAULT_N_GENOTYPES,
        'generations': 1000,  # Shorter than default for speed
        'selection': DEFAULT_SELECTION_STRENGTH,
        'metro_migration': 0.07,  # Updated baseline from real data calibration
        'peripheral_migration': DEFAULT_PERIPH_MIGRATION,
        'spatial_scale': DEFAULT_SPATIAL_SCALE,
        'decay_length': DEFAULT_DECAY_LENGTH,
        'seed': DEFAULT_SEED,
    }

    # Quick mode: reduce replicates and generations
    if args.quick:
        n_replicates = 3
        baseline['generations'] = 500
        print("\n*** QUICK MODE: 3 replicates, 500 generations ***\n")
    else:
        n_replicates = args.n_replicates

    # Parameter ranges to test
    param_ranges = {
        'selection': [0.05, 0.075, 0.1, 0.15, 0.2],
        'metro_migration': [0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
        'peripheral_migration': [0.01, 0.015, 0.02, 0.03, 0.04],
        'decay_length': [15, 20, 30, 50, 100],
        'metros': [25, 28, 35, 42, 45, 50],  # Network structure: 20%-50% metro ratio

        # NEW: Mobility ratio test
        # This directly tests the influence of O/I ratio disparity
        # Ratio = metro_migration / peripheral_migration
        # Reflects differential patient mobility between city types
        'mobility_ratio': [2.0, 3.0, 4.5, 6.0, 8.0],
    }

    # Apply parameter filter if specified
    if args.param_filter and args.param_filter != 'all':
        if args.param_filter in param_ranges:
            param_ranges = {args.param_filter: param_ranges[args.param_filter]}
            print(f"\n🎯 FILTER MODE: Testing only '{args.param_filter}'")
        else:
            print(f"\n❌ ERROR: Unknown parameter '{args.param_filter}'")
            print(f"   Available parameters: {', '.join(param_ranges.keys())}")
            print(f"\n   Usage: --param-filter <parameter_name>")
            return

    print("="*70)
    print("One-at-a-time Sensitivity Analysis")
    print("="*70)
    print(f"\nBaseline parameters:")
    for key, val in baseline.items():
        print(f"  {key}: {val}")

    print(f"\nTesting {len(param_ranges)} parameters")
    print(f"Replicates per value: {n_replicates}")

    total_runs = sum(len(values) * n_replicates for values in param_ranges.values())
    print(f"Total simulations: {total_runs}")
    print(f"Estimated time: {total_runs * 2 / 60:.1f} minutes\n")

    # Run sensitivity analysis
    results = []
    run_count = 0
    start_time = datetime.now()

    for param_name, values in param_ranges.items():
        print(f"\n{'='*70}")
        print(f"Testing parameter: {param_name}")
        print(f"Range: {values}")
        print(f"{'='*70}")

        for value in values:
            # Copy baseline and modify parameter
            params = baseline.copy()

            # Special handling for mobility_ratio
            if param_name == 'mobility_ratio':
                # Fix metro_migration at 0.07, vary peripheral_migration to achieve target ratio
                # This isolates the effect of mobility disparity (reflecting O/I ratio difference)
                params['metro_migration'] = 0.07
                params['peripheral_migration'] = 0.07 / value

                print(f"\n  {param_name} = {value:.1f}x")
                print(f"    → metro_migration = {params['metro_migration']:.4f}")
                print(f"    → peripheral_migration = {params['peripheral_migration']:.4f}")
            else:
                params[param_name] = value
                print(f"\n  {param_name} = {value}")

            # Run multiple replicates with different seeds
            for rep in range(n_replicates):
                params['seed'] = DEFAULT_SEED + rep

                try:
                    output = run_single_simulation(params)

                    # Store results
                    result_record = {
                        'parameter': param_name,
                        'value': value,
                        'replicate': rep,
                        'seed': params['seed'],
                        # Store actual migration values used (important for mobility_ratio analysis)
                        'actual_metro_migration': params['metro_migration'],
                        'actual_periph_migration': params['peripheral_migration'],
                        **output
                    }
                    results.append(result_record)

                    run_count += 1

                    # Progress indicator
                    if rep == 0:
                        coexist_symbol = '✓✓' if output['ideal_coexistence'] else '✓' if output['acceptable_coexistence'] else '✗'
                        print(f"    Rep {rep+1}/{n_replicates}: M-freq(metro)={output['metro_m_freq']:.2f}, "
                              f"M-freq(periph)={output['periph_m_freq']:.2f}, "
                              f"Coexist={coexist_symbol}")

                except Exception as e:
                    print(f"    ERROR in replicate {rep}: {e}")
                    continue

            # Print summary for this value
            value_results = [r for r in results if r['parameter'] == param_name and r['value'] == value]
            if value_results:
                mean_metro = np.mean([r['metro_m_freq'] for r in value_results])
                mean_periph = np.mean([r['periph_m_freq'] for r in value_results])
                ideal_rate = np.mean([r['ideal_coexistence'] for r in value_results])
                accept_rate = np.mean([r['acceptable_coexistence'] for r in value_results])
                print(f"    Summary: M-freq(metro)={mean_metro:.2f}±{np.std([r['metro_m_freq'] for r in value_results]):.2f}, "
                      f"M-freq(periph)={mean_periph:.2f}±{np.std([r['periph_m_freq'] for r in value_results]):.2f}, "
                      f"Ideal={ideal_rate*100:.0f}%, Accept={accept_rate*100:.0f}%")

    # Save results
    df = pd.DataFrame(results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"sensitivity_oneway_{timestamp}.csv"
    df.to_csv(output_file, index=False)

    elapsed = (datetime.now() - start_time).total_seconds()

    print("\n" + "="*70)
    print("Analysis Complete")
    print("="*70)
    print(f"Total runs: {run_count}/{total_runs}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Results saved: {output_file}")

    # Print summary statistics
    print("\n" + "="*70)
    print("Summary Statistics")
    print("="*70)

    for param_name in param_ranges.keys():
        param_data = df[df['parameter'] == param_name]

        print(f"\n{param_name}:")
        summary = param_data.groupby('value').agg({
            'metro_m_freq': ['mean', 'std'],
            'periph_m_freq': ['mean', 'std'],
            'segregation_strength': ['mean', 'std'],
            'ideal_coexistence': 'mean',
            'acceptable_coexistence': 'mean',
            'fst_weighted': ['mean', 'std'],
            'shannon_diversity': ['mean', 'std'],
        }).round(3)

        print(summary.to_string())

    # Overall statistics
    ideal_rate = df['ideal_coexistence'].mean()
    acceptable_rate = df['acceptable_coexistence'].mean()
    mean_fst = df['fst_weighted'].mean()
    std_fst = df['fst_weighted'].std()
    mean_diversity = df['shannon_diversity'].mean()
    std_diversity = df['shannon_diversity'].std()
    mean_seg = df['segregation_strength'].mean()
    std_seg = df['segregation_strength'].std()

    print(f"\n{'='*70}")
    print(f"Overall Statistics (across {len(df)} simulations)")
    print(f"{'='*70}")
    print(f"Ideal coexistence rate:      {ideal_rate*100:.1f}%  (matches real data)")
    print(f"Acceptable coexistence rate: {acceptable_rate*100:.1f}%  (has segregation)")
    print(f"Mean FST (weighted):         {mean_fst:.3f} ± {std_fst:.3f}")
    print(f"Mean Shannon diversity:      {mean_diversity:.3f} ± {std_diversity:.3f}")
    print(f"Mean segregation strength:   {mean_seg:.3f} ± {std_seg:.3f}")
    print(f"{'='*70}\n")

    return df


if __name__ == '__main__':
    main()
