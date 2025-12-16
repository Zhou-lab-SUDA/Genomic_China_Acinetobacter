#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Invasion Scenario: Clade 2.5 Invasion into Clade 2.4-dominated System

Scenario Description:
    Phase 1: Clade 2.4 (C-genotypes) establishes equilibrium (0 → introduction_time)
    Phase 2: Clade 2.5 (M-genotypes) is introduced and competes (introduction_time → end)

Usage Example:
    python scenarios/invasion.py --hospitals 100 --metros 30 --generations 2000 --introduction-time 500

    Optional Parameters:
        --introduction-dose 0.05    # Introduction dose (default 5%)
        --introduction-sites 3       # Number of metropolitan hospitals for introduction (default 3)
        --seed 42                    # Random seed
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration
from config import (
    TAU_VALUES, BETA_VALUES,
    DEFAULT_N_HOSPITALS, DEFAULT_N_METROS, DEFAULT_GENERATIONS,
    DEFAULT_POPULATION, DEFAULT_N_GENOTYPES,
    DEFAULT_METRO_MIGRATION, DEFAULT_PERIPH_MIGRATION,
    DEFAULT_SELECTION_STRENGTH, DEFAULT_SPATIAL_SCALE,
    INVASION_DEFAULT_INTRODUCTION_TIME, INVASION_DEFAULT_DOSE, INVASION_DEFAULT_SITES,
    METRO_THRESHOLD, PERIPH_THRESHOLD,
    classify_genotypes, get_timestamp, generate_random_genotype_values
)

# Import core engine functions
from core.engine import (
    calculate_fitness,
    wright_fisher_reproduction,
    migrate_between_hospitals_with_transfer,
    generate_hospital_coordinates,
    initialize_simulation,
    calculate_spatial_spread
)

# Import visualization
from visualization.invasion import (
    plot_invasion_curve_combined,
    plot_invasion_curve_combined_plotly,
    plot_invasion_curve,
    plot_prevalence_dynamics
)

from visualization.basic_plots import (
    export_hospital_network_state,
    plot_hospital_network_state,
    create_network_from_transfer_matrix,
    plot_spatial_spread_over_time
)

from utils.export import (
    export_network_analysis_json,
    export_transfer_matrix,
    export_results_csv,
    export_run_config,
    export_genotype_timeline_csv
)

from utils.statistics import calculate_fst_statistics


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Invasion Scenario: Clade 2.5 invades Clade 2.4 system',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-n', '--hospitals', type=int, default=DEFAULT_N_HOSPITALS,
                       help='Total number of hospitals')
    parser.add_argument('-m', '--metros', type=int, default=DEFAULT_N_METROS,
                       help='Number of metropolitan hospitals')
    parser.add_argument('-g', '--generations', type=int, default=2000,
                       help='Total number of generations')
    parser.add_argument('--population', type=int, default=DEFAULT_POPULATION,
                       help='Population size per hospital')
    parser.add_argument('--genotypes', type=int, default=DEFAULT_N_GENOTYPES,
                       help='Number of genotypes')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--fixed-genotypes', action='store_true',
                       help='Use fixed M/C genotype values from config (default: generate random values based on seed)')
    parser.add_argument('--metro-migration', type=float, default=DEFAULT_METRO_MIGRATION,
                       help='Metro hospital migration rate')
    parser.add_argument('--peripheral-migration', type=float, default=DEFAULT_PERIPH_MIGRATION,
                       help='Peripheral hospital migration rate')
    parser.add_argument('--selection', type=float, default=DEFAULT_SELECTION_STRENGTH,
                       help='Selection strength')
    # Intervention-specific parameters
    parser.add_argument('--initial-prevalence', type=float, default=0.05,
                       help='Initial Clade 2.4 prevalence before invasion (0.0-1.0, default: 0.05)')
    parser.add_argument('--seed-hospital-fraction', type=float, default=1.0,
                       help='Fraction of hospitals with initial infections (0.0-1.0, default: 1.0 for invasion)')
    parser.add_argument('--introduction-time', type=int, default=INVASION_DEFAULT_INTRODUCTION_TIME,
                       help='Generation to introduce Clade 2.5')
    parser.add_argument('--introduction-dose', type=float, default=INVASION_DEFAULT_DOSE,
                       help='Initial frequency of introduced genotypes (0.0-1.0)')
    parser.add_argument('--introduction-sites', type=int, default=INVASION_DEFAULT_SITES,
                       help='Number of metro hospitals to introduce Clade 2.5')
    parser.add_argument('--spatial-scale', type=float, default=DEFAULT_SPATIAL_SCALE,
                       help='Spatial scale (km)')
    parser.add_argument('--decay-length', type=float, default=None,
                       help='Distance decay length (km). If None, no spatial decay')
    parser.add_argument('-o', '--output-dir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--no-smooth', action='store_true',
                       help='Disable smoothing in plots (show raw data with genetic drift noise)')
    parser.add_argument('--smooth-window', type=int, default=15,
                       help='Smoothing window size (default: 15, larger = smoother)')

    return parser.parse_args()


def introduce_m_genotypes(hospital_populations, introduction_sites, metro_indices,
                         M_genotypes, introduction_dose, population_size):
    """
    Introduce M-genotypes (Clade 2.5) to specified hospitals

    Parameters:
        hospital_populations: List of hospital populations
        introduction_sites: List of indices of hospitals to introduce
        metro_indices: List of indices of metropolitan hospitals
        M_genotypes: List of M-genotype IDs
        introduction_dose: Introduction dose (0.0-1.0)
        population_size: Population size per hospital

    Returns:
        Modified hospital_populations
    """
    if len(M_genotypes) == 0:
        raise ValueError(f"No M-genotypes found (τ > {METRO_THRESHOLD})!")

    # Select hospitals for introduction (randomly chosen from metropolitan hospitals)
    n_sites = min(introduction_sites, len(metro_indices))
    selected_hospitals = np.random.choice(metro_indices, size=n_sites, replace=False)

    print(f"\n")
    print(f"Introducing Clade 2.5 (M-genotypes)")
    print(f"  Introduction sites: {n_sites} metro hospitals")
    print(f"  Hospital IDs: {sorted(selected_hospitals.tolist())}")
    print(f"  Introduction dose: {introduction_dose:.1%} of population")
    print(f"  M-genotypes to introduce: {M_genotypes}")

    for h in selected_hospitals:
        # Calculate number to introduce
        n_introduce = int(population_size * introduction_dose)

        if n_introduce > 0:
            # Randomly select individuals to replace
            replace_indices = np.random.choice(population_size, n_introduce, replace=False)

            # Replace with M-genotypes
            for idx in replace_indices:
                hospital_populations[h][idx] = np.random.choice(M_genotypes)

    # Calculate post-introduction frequency
    m_count_total = sum(np.sum([pop == g for g in M_genotypes]) for pop in hospital_populations)
    total_individuals = len(hospital_populations) * population_size
    print(f"  Post-introduction M-genotype frequency: {m_count_total/total_individuals:.1%}")

    return hospital_populations


def calculate_clade_frequencies(hospital_populations, hospital_types, M_genotypes, C_genotypes):
    """
    Calculate the frequency of each clade in different hospital types

    Returns:
        dict with keys: metro_m_freq, periph_m_freq, overall_m_freq,
                       metro_c_freq, periph_c_freq, overall_c_freq
    """
    n_hospitals = len(hospital_populations)
    m_freqs = np.zeros(n_hospitals)
    c_freqs = np.zeros(n_hospitals)

    for h in range(n_hospitals):
        pop = hospital_populations[h]
        # M-genotypes frequency
        m_freqs[h] = np.sum([np.mean(pop == g) for g in M_genotypes])
        # C-genotypes frequency
        c_freqs[h] = np.sum([np.mean(pop == g) for g in C_genotypes])

    metro_indices = [i for i, t in enumerate(hospital_types) if t == 'metro']
    periph_indices = [i for i, t in enumerate(hospital_types) if t == 'peripheral']

    return {
        'metro_m_freq': np.mean(m_freqs[metro_indices]),
        'periph_m_freq': np.mean(m_freqs[periph_indices]),
        'overall_m_freq': np.mean(m_freqs),
        'metro_c_freq': np.mean(c_freqs[metro_indices]),
        'periph_c_freq': np.mean(c_freqs[periph_indices]),
        'overall_c_freq': np.mean(c_freqs),
    }


def run_invasion_scenario(args):
    """Run invasion scenario main function"""

    # Set random seed
    np.random.seed(args.seed)

    # Validate parameters
    if args.introduction_time >= args.generations:
        raise ValueError(
            f"introduction_time ({args.introduction_time}) must be less than generations ({args.generations})"
        )
    if args.introduction_time < 1:
        raise ValueError(
            f"introduction_time ({args.introduction_time}) must be at least 1"
        )

    # Create independent run directory
    timestamp = get_timestamp()
    output_run_dir = f"{args.output_dir}/run_{timestamp}_invasion_h{args.hospitals}_m{args.metros}_g{args.generations}_s{args.seed}"
    os.makedirs(f"{output_run_dir}/data", exist_ok=True)
    os.makedirs(f"{output_run_dir}/figures", exist_ok=True)

    print("\n" + "="*70)
    print("Invasion Scenario Simulation")
    print("="*70)
    print(f"\nOutput directory: {output_run_dir}")
    print(f"\nParameters:")
    print(f"  Hospitals: {args.hospitals} ({args.metros} metro, {args.hospitals - args.metros} peripheral)")
    print(f"  Generations: {args.generations}")
    print(f"  Initial Clade 2.4 prevalence: {args.initial_prevalence:.1%}")
    print(f"  Introduction time: {args.introduction_time}")
    print(f"  Introduction dose: {args.introduction_dose:.1%}")
    print(f"  Introduction sites: {args.introduction_sites}")
    print(f"  Seed: {args.seed}")

    # Generate M/C values (random by default, fixed if specified)
    if args.fixed_genotypes:
        print("\nUsing fixed genotype values from config...")
        TAU_sim = TAU_VALUES
        BETA_sim = BETA_VALUES
    else:
        print(f"\nGenerating random genotype values based on seed {args.seed}...")
        TAU_sim, BETA_sim = generate_random_genotype_values(
            n_genotypes=args.genotypes,
            seed=args.seed,
            strategy='complementary'
        )
        print(f"Generated TAU_VALUES: {TAU_sim}")
        print(f"Generated BETA_VALUES: {BETA_sim}")

    # Classify genotypes
    genotype_classes = classify_genotypes(TAU_sim, exclude_neutral=False)
    M_genotypes = genotype_classes['metro']
    C_genotypes = genotype_classes['periph']

    print(f"\nGenotype classification:")
    print(f"  M-genotypes (Clade 2.5): {M_genotypes}")
    print(f"  C-genotypes (Clade 2.4): {C_genotypes}")

    # Generate spatial coordinates
    print(f"\nSpatial parameters:")
    print(f"  Spatial scale: {args.spatial_scale} km")
    coordinates = generate_hospital_coordinates(
        args.hospitals,
        args.spatial_scale,
        seed=args.seed
    )

    if args.decay_length is not None:
        print(f"  Decay length: {args.decay_length} km (distance decay enabled)")
    else:
        print(f"  Decay length: None (no distance decay, uniform transfer probability)")

    # ========== Phase 1: Establishing Clade 2.4 ==========
    print(f"\n")
    print(f"Phase 1: Establishing Clade 2.4 (0 → {args.introduction_time})")
    print(f"Initial Clade 2.4 prevalence: {args.initial_prevalence:.1%}")

    # Initialize: only C-genotypes
    hospital_populations, hospital_types = initialize_simulation(
        args.hospitals,
        args.metros,
        args.population,
        args.genotypes,
        TAU_sim,
        initial_prevalence=args.initial_prevalence,
        seed_genotypes='C_only',
        seed_hospital_fraction=args.seed_hospital_fraction
    )

    # Get hospital indices
    metro_indices = [i for i, t in enumerate(hospital_types) if t == 'metro']
    periph_indices = [i for i, t in enumerate(hospital_types) if t == 'peripheral']

    # Set migration rates
    hospital_migration_rates = [
        args.metro_migration if hospital_types[i] == 'metro' else args.peripheral_migration
        for i in range(args.hospitals)
    ]

    # Time series data
    invasion_timeline = []

    # Detailed time series data (for network_analysis JSON)
    time_series = {
        'overall_freqs': [],
        'metro_freqs': [],
        'periph_freqs': [],
        'hospital_freqs': [],  
        'spatial_spread': {'timesteps': [], 'm_spread': [], 'c_spread': []}  # NEW: spatial spread time series
    }

    transfer_matrix = None
    transfer_matrix_before = None  # Save before invasion

    # Phase 1 simulation loop
    for gen in range(args.introduction_time):
        # 1. Calculate fitness
        for h in range(args.hospitals):
            fitness_values = np.array([
                calculate_fitness(geno, hospital_types[h], TAU_sim, BETA_sim, args.selection)
                for geno in hospital_populations[h]
            ])

            # 2. Wright-Fisher reproduction
            hospital_populations[h] = wright_fisher_reproduction(
                hospital_populations[h],
                fitness_values,
                args.population
            )

        # 3. Migration (need to get transfer_matrix when calculating spatial spread)
        need_matrix = (gen % 10 == 0 or gen == 0 or gen == args.introduction_time - 1)
        if need_matrix:
            hospital_populations, current_transfer_matrix = migrate_between_hospitals_with_transfer(
                hospital_populations,
                hospital_migration_rates,
                hospital_types,
                args.population,
                coordinates=coordinates,
                decay_length=args.decay_length,
                return_matrix=True
            )
            # Save transfer matrix before invasion (last generation of Phase 1)
            if gen == args.introduction_time - 1:
                transfer_matrix_before = current_transfer_matrix.copy()
        else:
            hospital_populations = migrate_between_hospitals_with_transfer(
                hospital_populations,
                hospital_migration_rates,
                hospital_types,
                args.population,
                coordinates=coordinates,
                decay_length=args.decay_length
            )

        # 4. Record data
        metrics = calculate_clade_frequencies(hospital_populations, hospital_types,
                                             M_genotypes, C_genotypes)

        # Calculate prevalence (absolute numbers / total population)
        total_population = args.hospitals * args.population
        n24_total = sum(np.sum([pop == g for g in C_genotypes]) for pop in hospital_populations)
        n25_total = sum(np.sum([pop == g for g in M_genotypes]) for pop in hospital_populations)
        prevalence24 = n24_total / total_population
        prevalence25 = n25_total / total_population

        invasion_timeline.append({
            'generation': gen,
            **metrics,
            'prevalence24': prevalence24,
            'prevalence25': prevalence25
        })

        # Record detailed genotype frequencies (for network_analysis JSON)
        current_genotype_freqs = np.zeros((args.hospitals, args.genotypes))
        for h in range(args.hospitals):
            for g in range(args.genotypes):
                current_genotype_freqs[h, g] = np.mean([geno == g for geno in hospital_populations[h]])

        time_series['overall_freqs'].append(np.mean(current_genotype_freqs, axis=0))
        time_series['metro_freqs'].append(np.mean(current_genotype_freqs[metro_indices], axis=0))
        time_series['periph_freqs'].append(np.mean(current_genotype_freqs[periph_indices], axis=0))

        # Record frequencies for each hospital
        hospital_freqs_dict = {}
        for h in range(args.hospitals):
            hospital_freqs_dict[h] = current_genotype_freqs[h, :].tolist()
        time_series['hospital_freqs'].append(hospital_freqs_dict)

        # Calculate spatial spread (every 10 generations to reduce computation)
        if gen % 10 == 0 or gen == 0:
            m_spread, c_spread = calculate_spatial_spread(
                hospital_populations, coordinates, current_transfer_matrix, M_genotypes, C_genotypes
            )
            time_series['spatial_spread']['timesteps'].append(gen)
            time_series['spatial_spread']['m_spread'].append(m_spread)
            time_series['spatial_spread']['c_spread'].append(c_spread)

        # Progress display
        if (gen + 1) % 50 == 0 or gen == 0:
            print(f"  Gen {gen+1}/{args.introduction_time}: "
                  f"C-freq(overall)={metrics['overall_c_freq']:.3f}, "
                  f"M-freq(overall)={metrics['overall_m_freq']:.3f}")

    print(f"\nPhase 1 completed. Clade 2.4 established.")
    print(f"  Final Clade 2.4 frequency: {invasion_timeline[-1]['overall_c_freq']:.1%}")

    # ========== Phase 2: Introduce Clade 2.5 and observe competition ==========
    print(f"\n")
    print(f"Phase 2: Clade 2.5 Invasion ({args.introduction_time} → {args.generations})")

    # Introduce M-genotypes
    hospital_populations = introduce_m_genotypes(
        hospital_populations,
        args.introduction_sites,
        metro_indices,
        M_genotypes,
        args.introduction_dose,
        args.population
    )

    # Phase 2 simulation loop
    for gen in range(args.introduction_time, args.generations):
        # 1. Calculate fitness
        for h in range(args.hospitals):
            fitness_values = np.array([
                calculate_fitness(geno, hospital_types[h], TAU_sim, BETA_sim, args.selection)
                for geno in hospital_populations[h]
            ])

            # 2. Wright-Fisher reproduction
            hospital_populations[h] = wright_fisher_reproduction(
                hospital_populations[h],
                fitness_values,
                args.population
            )

        # 3. Migration
        need_matrix = (gen == args.generations - 1) or (gen % 10 == 0)
        if need_matrix:
            hospital_populations, current_transfer_matrix = migrate_between_hospitals_with_transfer(
                hospital_populations,
                hospital_migration_rates,
                hospital_types,
                args.population,
                coordinates=coordinates,
                decay_length=args.decay_length,
                return_matrix=True
            )
            # If it's the last generation, save to global variable
            if gen == args.generations - 1:
                transfer_matrix = current_transfer_matrix
        else:
            hospital_populations = migrate_between_hospitals_with_transfer(
                hospital_populations,
                hospital_migration_rates,
                hospital_types,
                args.population,
                coordinates=coordinates,
                decay_length=args.decay_length
            )

        # 4. Record data
        metrics = calculate_clade_frequencies(hospital_populations, hospital_types,
                                             M_genotypes, C_genotypes)

        # Calculate prevalence (absolute numbers / total population)
        total_population = args.hospitals * args.population
        n24_total = sum(np.sum([pop == g for g in C_genotypes]) for pop in hospital_populations)
        n25_total = sum(np.sum([pop == g for g in M_genotypes]) for pop in hospital_populations)
        prevalence24 = n24_total / total_population
        prevalence25 = n25_total / total_population

        invasion_timeline.append({
            'generation': gen,
            **metrics,
            'prevalence24': prevalence24,
            'prevalence25': prevalence25
        })

        # Record detailed genotype frequencies
        current_genotype_freqs = np.zeros((args.hospitals, args.genotypes))
        for h in range(args.hospitals):
            for g in range(args.genotypes):
                current_genotype_freqs[h, g] = np.mean([geno == g for geno in hospital_populations[h]])

        time_series['overall_freqs'].append(np.mean(current_genotype_freqs, axis=0))
        time_series['metro_freqs'].append(np.mean(current_genotype_freqs[metro_indices], axis=0))
        time_series['periph_freqs'].append(np.mean(current_genotype_freqs[periph_indices], axis=0))

        # Record frequencies for each hospital
        hospital_freqs_dict = {}
        for h in range(args.hospitals):
            hospital_freqs_dict[h] = current_genotype_freqs[h, :].tolist()
        time_series['hospital_freqs'].append(hospital_freqs_dict)

        # Calculate spatial spread (every 10 generations to reduce computation)
        if gen % 10 == 0:
            m_spread, c_spread = calculate_spatial_spread(
                hospital_populations, coordinates, current_transfer_matrix, M_genotypes, C_genotypes
            )
            time_series['spatial_spread']['timesteps'].append(gen)
            time_series['spatial_spread']['m_spread'].append(m_spread)
            time_series['spatial_spread']['c_spread'].append(c_spread)

        # Progress display
        if (gen + 1) % 50 == 0 or gen == args.introduction_time:
            print(f"  Gen {gen+1}/{args.generations}: "
                  f"C-freq(overall)={metrics['overall_c_freq']:.3f}, "
                  f"M-freq(overall)={metrics['overall_m_freq']:.3f}")

    # ========== Summary ==========
    print(f"\n")
    print(f"Final Results (Gen {args.generations})")
    final_metrics = invasion_timeline[-1]
    print(f"  Clade 2.5 (M-genotypes):")
    print(f"    Metro:      {final_metrics['metro_m_freq']:.1%}")
    print(f"    Peripheral: {final_metrics['periph_m_freq']:.1%}")
    print(f"    Overall:    {final_metrics['overall_m_freq']:.1%}")
    print(f"\n  Clade 2.4 (C-genotypes):")
    print(f"    Metro:      {final_metrics['metro_c_freq']:.1%}")
    print(f"    Peripheral: {final_metrics['periph_c_freq']:.1%}")
    print(f"    Overall:    {final_metrics['overall_c_freq']:.1%}")

    # ========== Calculate final genotype frequencies (for network state map) ==========
    final_genotype_freqs = np.zeros((args.hospitals, args.genotypes))
    for h in range(args.hospitals):
        for g in range(args.genotypes):
            final_genotype_freqs[h, g] = np.mean([geno == g for geno in hospital_populations[h]])

    # ========== Save Results ==========
    print("\n" + "="*70)
    print("Saving Results")
    print("="*70)

    # Convert to DataFrame
    timeline_df = pd.DataFrame(invasion_timeline)

    # Save timeline CSV
    timeline_path = f"{output_run_dir}/data/timeline.csv"
    timeline_df.to_csv(timeline_path, index=False)
    print(f"\nTimeline saved: {timeline_path}")

    # Save genotype timeline (detailed per-genotype frequencies)
    genotype_timeline_path = f"{output_run_dir}/data/genotype_timeline.csv"
    export_genotype_timeline_csv(
        time_series=time_series,
        n_genotypes=args.genotypes,
        output_path=genotype_timeline_path
    )

    # Export transfer matrices (before/after invasion)
    print("\n[Transfer Matrices] Exporting before/after invasion...")
    if transfer_matrix_before is not None:
        before_path = f"{output_run_dir}/data/transfer_matrix_before.csv"
        export_transfer_matrix(transfer_matrix_before, hospital_types, before_path)
        print(f"  Before invasion: {before_path}")

    if transfer_matrix is not None:
        after_path = f"{output_run_dir}/data/transfer_matrix_after.csv"
        export_transfer_matrix(transfer_matrix, hospital_types, after_path)
        print(f"  After invasion: {after_path}")

    # ========== Generate Visualizations ==========
    print("\n" + "="*70)
    print("Generating Visualizations")
    print("="*70)

    use_smooth = not args.no_smooth

    # Invasion curves
    static_path = f"{output_run_dir}/figures/invasion_combined.pdf"
    plot_invasion_curve_combined(timeline_df, args.introduction_time, static_path,
                                smooth=use_smooth, smooth_window=args.smooth_window)

    interactive_path = f"{output_run_dir}/figures/invasion_interactive.html"
    plot_invasion_curve_combined_plotly(timeline_df, args.introduction_time, interactive_path)

    separated_path = f"{output_run_dir}/figures/invasion_separated.pdf"
    plot_invasion_curve(timeline_df, args.introduction_time, separated_path)

    prevalence_path = f"{output_run_dir}/figures/invasion_prevalence.pdf"
    plot_prevalence_dynamics(timeline_df, args.introduction_time, prevalence_path,
                            smooth=use_smooth, smooth_window=args.smooth_window)

    # Spatial spread
    print("\n[Spatial Spread] Generating time series...")
    spatial_spread_path = f"{output_run_dir}/figures/spatial_spread.pdf"
    spatial_spread_data = {
        'timesteps': np.array(time_series['spatial_spread']['timesteps']),
        'm_spread': np.array(time_series['spatial_spread']['m_spread']),
        'c_spread': np.array(time_series['spatial_spread']['c_spread'])
    }
    plot_spatial_spread_over_time(
        spatial_spread_data=spatial_spread_data,
        output_path=spatial_spread_path,
        intervention_start=args.introduction_time
    )

    # Network state
    print("\n[Network State] Generating hospital network state visualization...")
    network_state_csv = f"{output_run_dir}/data/network_state.csv"
    export_hospital_network_state(
        hospital_types=hospital_types,
        genotype_freqs=final_genotype_freqs,
        coordinates=coordinates,
        TAU_VALUES=TAU_sim,
        BETA_VALUES=BETA_sim,
        output_path=network_state_csv
    )
    print(f"  Network state CSV saved: {network_state_csv}")

    network_state_plot = f"{output_run_dir}/figures/hospital_network_state.pdf"
    plot_hospital_network_state(
        csv_path=network_state_csv,
        output_path=network_state_plot
    )
    print(f"  Network state plot saved: {network_state_plot}")

    # Network analysis JSON
    print("\n[Network Analysis] Exporting...")
    if transfer_matrix is not None:
        network_analysis_json = f"{output_run_dir}/data/network_analysis.json"

        # Create network graph
        max_transfer = np.max(transfer_matrix)
        if max_transfer > 0:
            adaptive_threshold = max(max_transfer * 0.01, 1e-6)
            G_for_json = create_network_from_transfer_matrix(
                transfer_matrix=transfer_matrix,
                hospital_types=hospital_types,
                edge_threshold=adaptive_threshold,
                top_k_per_node=10
            )
        else:
            G_for_json = create_network_from_transfer_matrix(
                transfer_matrix=transfer_matrix,
                hospital_types=hospital_types,
                edge_threshold=0,
                top_k_per_node=10
            )

        export_network_analysis_json(
            G=G_for_json,
            hospital_types=hospital_types,
            genotype_freqs=final_genotype_freqs,
            TAU_VALUES=TAU_sim,
            BETA_VALUES=BETA_sim,
            time_series=time_series,
            current_step=args.generations,
            output_path=network_analysis_json
        )
        print(f"  Network analysis JSON saved: {network_analysis_json}")
    else:
        print("  Skipping network analysis JSON (no transfer matrix)")

    # Export run configuration
    print("\n[Run Config] Exporting parameters...")
    config_path = f"{output_run_dir}/run_config.json"
    export_run_config(args, TAU_sim, BETA_sim, 'invasion', config_path)

    print("\n" + "="*70)
    print("Invasion Scenario Completed!")
    print("="*70)
    print(f"\nAll outputs saved to: {output_run_dir}")

    return {
        'timeline': timeline_df,
        'final_metrics': final_metrics,
        'args': args
    }


def main():
    """Main function to run the invasion scenario"""
    args = parse_arguments()
    results = run_invasion_scenario(args)
    return results


if __name__ == '__main__':
    main()
