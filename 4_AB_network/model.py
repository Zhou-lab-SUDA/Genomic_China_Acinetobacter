#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Import configurations and utilities
from config import (
    TAU_VALUES, BETA_VALUES,
    DEFAULT_N_HOSPITALS, DEFAULT_N_METROS, DEFAULT_GENERATIONS,
    DEFAULT_POPULATION, DEFAULT_N_GENOTYPES,
    DEFAULT_METRO_MIGRATION, DEFAULT_PERIPH_MIGRATION,
    DEFAULT_SELECTION_STRENGTH, DEFAULT_SPATIAL_SCALE,
    DEFAULT_DECAY_LENGTH, DEFAULT_OUTPUT_DIR, DEFAULT_SEED,
    DEFAULT_INITIAL_PREVALENCE, DEFAULT_SEED_GENOTYPES, DEFAULT_INITIAL_M_FREQ,
    INVASION_DEFAULT_GENERATIONS, DEFAULT_SEED_HOSPITAL_FRACTION,
    DEFAULT_INTRODUCTION_TIME, DEFAULT_INTRODUCTION_DOSE, DEFAULT_INTRODUCTION_SITES,
    COVID_DEFAULT_GENERATIONS, DEFAULT_INTERVENTION_START, DEFAULT_RECOVERY_START,
    DEFAULT_BASELINE_DECAY, DEFAULT_INTERVENTION_DECAY, DEFAULT_RECOVERY_DECAY,
    DEFAULT_METRO_CLOSURE, DEFAULT_PERIPH_CLOSURE, DEFAULT_SMOOTH_WINDOW,
    classify_genotypes, get_timestamp, generate_random_genotype_values
)

# Import core simulation functions
from core.engine import (
    calculate_fitness,
    wright_fisher_reproduction,
    migrate_between_hospitals_with_transfer,
    generate_hospital_coordinates,
    initialize_simulation,
    calculate_spatial_spread
)

# Import visualization functions
from visualization.basic_plots import (
    plot_genotype_trajectories,
    plot_three_strategies,
    plot_three_strategies_interactive,
    plot_hospital_network,
    plot_clade_comparison_time_series,
    plot_hospital_network_state,
    create_network_from_transfer_matrix,
    export_hospital_network_state,
    plot_genotypes_radar_chart,
    calculate_genotypes_radar_metrics,
    plot_spatial_spread_over_time
)

# Import statistics and export utilities
from utils.statistics import calculate_fst_statistics
from utils.export import (
    export_results_csv,
    export_transfer_matrix,
    export_network_analysis_json,
    export_run_config  # NEW: replaces file1 and records all parameters
)


def run_basic_simulation(args):
    """
    Basic simulation without specific scenarios
    """
    print("\n" + "="*70)
    print("Basic Simulation")
    print("="*70)

    # Create independent run directory
    timestamp = get_timestamp()
    output_run_dir = f"{args.output_dir}/run_{timestamp}_basic_h{args.hospitals}_m{args.metros}_g{args.generations}_s{args.seed}"
    os.makedirs(f"{output_run_dir}/data", exist_ok=True)
    os.makedirs(f"{output_run_dir}/figures", exist_ok=True)
    print(f"\nOutput directory: {output_run_dir}")

    # Set random seed
    np.random.seed(args.seed)
    print(f"Random seed: {args.seed}")

    # Generate M/C values (random by default, fixed if specified)
    if args.fixed_genotypes:
        print("Using fixed genotype values from config...")
        TAU_sim = TAU_VALUES
        BETA_sim = BETA_VALUES
    else:
        print(f"Generating random genotype values based on seed {args.seed}...")
        TAU_sim, BETA_sim = generate_random_genotype_values(
            n_genotypes=args.genotypes,
            seed=args.seed,
            strategy='complementary'
        )
        print(f"Generated TAU_VALUES: {TAU_sim}")
        print(f"Generated BETA_VALUES: {BETA_sim}")

    # Initialize hospital coordinates
    print(f"\nInitializing {args.hospitals} hospitals (including {args.metros} metro hospitals)...")
    coordinates = generate_hospital_coordinates(
        args.hospitals,
        spatial_scale=args.spatial_scale,
        seed=args.seed
    )

    # Initialize populations
    print("Initializing populations...")
    populations, hospital_types = initialize_simulation(
        n_hospitals=args.hospitals,
        n_metros=args.metros,
        population_size=args.population,
        n_genotypes=args.genotypes,
        TAU_array=TAU_sim,
    initial_prevalence=DEFAULT_INITIAL_PREVALENCE,
    seed_genotypes=DEFAULT_SEED_GENOTYPES
    )

    # Set migration rates
    migration_rates = [
        args.metro_migration if htype == 'metro' else args.peripheral_migration
        for htype in hospital_types
    ]

    # Prepare data collection
    timeline = []
    genotype_classes = classify_genotypes(TAU_sim, exclude_neutral=False)
    M_genotypes = genotype_classes['metro']
    C_genotypes = genotype_classes['periph']

    # Prepare detailed time series data for visualizations
    time_series = {
        'overall_freqs': [],
        'metro_freqs': [],
        'periph_freqs': [],
        'hospital_freqs': [],  # Add historical records for each hospital
        'spatial_spread': {'timesteps': [], 'm_spread': [], 'c_spread': []}  # NEW: spatial spread time series
    }
    transfer_matrix = None  # Will be updated in last generation

    print(f"\nStarting simulation for {args.generations} generations...")

    # Main simulation loop
    for generation in range(1, args.generations + 1):
        # 1. Selection and reproduction
        for h in range(args.hospitals):
            fitness = np.array([
                calculate_fitness(
                    g, hospital_types[h], TAU_sim, BETA_sim, args.selection
                ) for g in populations[h]
            ])
            populations[h] = wright_fisher_reproduction(
                populations[h], fitness, args.population
            )

        # 2. Migration (get transfer_matrix in last generation or when calculating spatial spread)
        # Need transfer_matrix for spatial spread calculation (every 10 generations)
        return_matrix = (generation == args.generations) or (generation % 10 == 0 or generation == 1)
        if return_matrix:
            populations, transfer_matrix = migrate_between_hospitals_with_transfer(
                populations, migration_rates, hospital_types, args.population,
                coordinates=(coordinates[0], coordinates[1]),
                decay_length=args.decay_length,
                return_matrix=True
            )
        else:
            populations = migrate_between_hospitals_with_transfer(
                populations, migration_rates, hospital_types, args.population,
                coordinates=(coordinates[0], coordinates[1]),
                decay_length=args.decay_length,
                return_matrix=False
            )

        # 3. Record data
        # Calculate genotype frequencies
        metro_indices = [i for i, t in enumerate(hospital_types) if t == 'metro']
        periph_indices = [i for i, t in enumerate(hospital_types) if t == 'peripheral']

        # Calculate per-genotype frequencies (for all genotypes)
        overall_freqs_t = np.zeros(args.genotypes)
        metro_freqs_t = np.zeros(args.genotypes)
        periph_freqs_t = np.zeros(args.genotypes)

        for g in range(args.genotypes):
            # Overall frequency
            overall_freqs_t[g] = np.mean([
                np.mean([geno == g for geno in populations[i]])
                for i in range(args.hospitals)
            ])
            # Metro frequency
            if metro_indices:
                metro_freqs_t[g] = np.mean([
                    np.mean([geno == g for geno in populations[i]])
                    for i in metro_indices
                ])
            # Peripheral frequency
            if periph_indices:
                periph_freqs_t[g] = np.mean([
                    np.mean([geno == g for geno in populations[i]])
                    for i in periph_indices
                ])

        time_series['overall_freqs'].append(overall_freqs_t)
        time_series['metro_freqs'].append(metro_freqs_t)
        time_series['periph_freqs'].append(periph_freqs_t)

        # Save genotype frequency for each hospital (formatted as dictionary, key as hospital ID)
        hospital_freqs_dict = {}
        for h in range(args.hospitals):
            hospital_freq_h = np.zeros(args.genotypes)
            for g in range(args.genotypes):
                hospital_freq_h[g] = np.mean([geno == g for geno in populations[h]])
            hospital_freqs_dict[h] = hospital_freq_h.tolist()
        time_series['hospital_freqs'].append(hospital_freqs_dict)

        # Calculate spatial spread for M and C genotypes (every 10 generations to reduce computation)
        # Note: transfer_matrix is available because we set return_matrix=True for these generations
        if generation % 10 == 0 or generation == 1:
            m_spread, c_spread = calculate_spatial_spread(
                populations, coordinates, transfer_matrix, M_genotypes, C_genotypes
            )
            time_series['spatial_spread']['timesteps'].append(generation)
            time_series['spatial_spread']['m_spread'].append(m_spread)
            time_series['spatial_spread']['c_spread'].append(c_spread)

        # Record timeline data (every 10 generations for efficiency)
        if generation % 10 == 0 or generation == 1:
            # Calculate M-genotype and C-genotype frequencies
            metro_m_freq = np.mean([
                np.mean([g in M_genotypes for g in populations[i]])
                for i in metro_indices
            ]) if metro_indices else 0.0

            periph_m_freq = np.mean([
                np.mean([g in M_genotypes for g in populations[i]])
                for i in periph_indices
            ]) if periph_indices else 0.0

            overall_m_freq = np.mean([
                np.mean([g in M_genotypes for g in populations[i]])
                for i in range(args.hospitals)
            ])

            metro_c_freq = 1.0 - metro_m_freq
            periph_c_freq = 1.0 - periph_m_freq
            overall_c_freq = 1.0 - overall_m_freq

            timeline.append({
                'generation': generation,
                'metro_m_freq': metro_m_freq,
                'periph_m_freq': periph_m_freq,
                'overall_m_freq': overall_m_freq,
                'metro_c_freq': metro_c_freq,
                'periph_c_freq': periph_c_freq,
                'overall_c_freq': overall_c_freq,
            })

            # Print progress
            if generation % 100 == 0:
                print(f"Generation {generation:4d} | "
                      f"M-freq: Metro {metro_m_freq:.3f}, Periph {periph_m_freq:.3f}, "
                      f"Overall {overall_m_freq:.3f}")

    print("Simulation complete!\n")

    # Convert time_series lists to numpy arrays
    time_series['overall_freqs'] = np.array(time_series['overall_freqs'])
    time_series['metro_freqs'] = np.array(time_series['metro_freqs'])
    time_series['periph_freqs'] = np.array(time_series['periph_freqs'])

    # Calculate final genotype frequencies for each hospital
    final_genotype_freqs = np.zeros((args.hospitals, args.genotypes))
    for h in range(args.hospitals):
        for g in range(args.genotypes):
            final_genotype_freqs[h, g] = np.mean([geno == g for geno in populations[h]])

    # Save data
    df = pd.DataFrame(timeline)

    # Save timeline
    timeline_path = f"{output_run_dir}/data/timeline.csv"
    df.to_csv(timeline_path, index=False)
    print(f"\nTimeline saved to: {timeline_path}")

    # Generate visualizations
    print("\n" + "="*70)
    print("Generating Visualizations")
    print("="*70)

    # Spatial spread over time
    print("\n[0/5] Generating spatial spread time series plot...")
    spatial_filename = f"{output_run_dir}/figures/spatial_spread.pdf"
    spatial_spread_data = {
        'timesteps': np.array(time_series['spatial_spread']['timesteps']),
        'm_spread': np.array(time_series['spatial_spread']['m_spread']),
        'c_spread': np.array(time_series['spatial_spread']['c_spread'])
    }
    plot_spatial_spread_over_time(
        spatial_spread_data=spatial_spread_data,
        output_path=spatial_filename
    )
    print(f"Spatial spread plot saved: {spatial_filename}")

    # Genotype trajectories
    print("\n[1/5] Generating genotype trajectory plot...")
    traj_filename = f"{output_run_dir}/figures/genotype_trajectories.pdf"
    plot_genotype_trajectories(
        time_series=time_series,
        final_freqs=final_genotype_freqs,
        hospital_types=hospital_types,
        TAU_VALUES=TAU_sim,
        BETA_VALUES=BETA_sim,
        output_path=traj_filename
    )
    print(f"Trajectory plot saved: {traj_filename}")

    # Three strategies (static)
    print("\n[2/5] Generating three-strategy analysis plot...")
    strategies_filename = f"{output_run_dir}/figures/adaptation.pdf"
    plot_three_strategies(
        final_freqs=final_genotype_freqs,
        hospital_types=hospital_types,
        TAU_VALUES=TAU_sim,
        BETA_VALUES=BETA_sim,
        time_series=time_series,
        output_path=strategies_filename
    )
    print(f"Three-strategy plot saved: {strategies_filename}")

    # Three strategies (interactive HTML)
    print("\n[3/5] Generating interactive Plotly version...")
    interactive_filename = f"{output_run_dir}/figures/adaptation_interactive.html"
    plot_three_strategies_interactive(
        final_freqs=final_genotype_freqs,
        hospital_types=hospital_types,
        TAU_VALUES=TAU_sim,
        BETA_VALUES=BETA_sim,
        time_series=time_series,
        output_path=interactive_filename
    )
    print(f"Interactive plot saved: {interactive_filename}")

    # Genotypes comparison
    if transfer_matrix is not None:
        print("\n[4/5] Generating genotypes comparison time series...")
        comparison_filename = f"{output_run_dir}/figures/genotypes_comparison.pdf"

        plot_clade_comparison_time_series(
            time_series=time_series,
            hospital_types=hospital_types,
            TAU_VALUES=TAU_VALUES,
            output_path=comparison_filename
        )
        print(f"Comparison plot saved: {comparison_filename}")

        # Export network state
        network_state_csv = f"{output_run_dir}/data/network_state.csv"
        export_hospital_network_state(
            hospital_types=hospital_types,
            genotype_freqs=final_genotype_freqs,
            coordinates=coordinates,
            TAU_VALUES=TAU_sim,
            BETA_VALUES=BETA_sim,
            output_path=network_state_csv
        )
        print(f"Network state CSV saved: {network_state_csv}")

        # Plot network state
        network_state_plot = f"{output_run_dir}/figures/hospital_network_state.pdf"
        plot_hospital_network_state(
            csv_path=network_state_csv,
            output_path=network_state_plot
        )
        print(f"Network state plot saved: {network_state_plot}")
    else:
        print("\n[4/5] Skipping comparison and network state (no transfer matrix)")

    print("="*70)

    # Generate a simple summary
    print("\nSimulation Summary")
    final = timeline[-1]
    print(f"Final generation: {final['generation']}")
    print(f"\nM-genotype frequencies:")
    print(f"  Metro:   {final['metro_m_freq']:.3f}")
    print(f"  Peripheral:   {final['periph_m_freq']:.3f}")
    print(f"  Overall:     {final['overall_m_freq']:.3f}")
    print(f"\nC-genotype frequencies:")
    print(f"  Metro:   {final['metro_c_freq']:.3f}")
    print(f"  Peripheral:   {final['periph_c_freq']:.3f}")
    print(f"  Overall:     {final['overall_c_freq']:.3f}")

    # ========== Calculate statistics and export additional files ==========
    print("\n" + "="*70)
    print("Calculating Statistics and Exporting Additional Files")
    print("="*70)

    # Calculate FST and diversity statistics
    print("\nCalculating FST statistics...")
    stats = calculate_fst_statistics(
        genotype_freqs=final_genotype_freqs,
        hospital_types=hospital_types,
        TAU_VALUES=TAU_sim,
        BETA_VALUES=BETA_sim
    )

    # Export results CSV
    print("\n[Results] Exporting statistics...")
    results_path = f"{output_run_dir}/data/results.csv"
    export_results_csv(stats, args.metro_migration, args.peripheral_migration, results_path)

    # Export transfer matrix (only if exists)
    if transfer_matrix is not None:
        print("\n[Transfer Matrix] Exporting...")
        transfer_path = f"{output_run_dir}/data/transfer_matrix.csv"
        export_transfer_matrix(transfer_matrix, hospital_types, transfer_path)

        # Calculate radar metrics (will be integrated into network_analysis.json)
        print("\n[Radar Metrics] Calculating (will be included in network_analysis.json)...")
        G_for_radar = create_network_from_transfer_matrix(
            transfer_matrix=transfer_matrix,
            hospital_types=hospital_types,
            edge_threshold=0.02,
            top_k_per_node=8
        )

        radar_metrics = calculate_genotypes_radar_metrics(
            genotype_freqs=final_genotype_freqs,
            hospital_types=hospital_types,
            TAU_VALUES=TAU_sim,
            BETA_VALUES=BETA_sim,
            time_series=time_series,
            transfer_matrix=transfer_matrix,
            fst_per_genotype=stats['fst_per_genotype'],
            G=G_for_radar
        )

        # Network analysis JSON (includes radar_metrics)
        print("\n[Network Analysis] Exporting (includes radar metrics)...")
        network_path = f"{output_run_dir}/data/network_analysis.json"

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
            print("Warning: transfer_matrix is all zeros, creating default network...")
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
            radar_metrics=radar_metrics,  # NEW: integrate file7
            output_path=network_path
        )
    else:
        print("\n[Network Analysis] Skipping (no transfer matrix)")

    # Export run configuration
    print("\n[Run Config] Exporting parameters...")
    config_path = f"{output_run_dir}/run_config.json"
    export_run_config(args, TAU_sim, BETA_sim, 'basic', config_path)

    print("="*70)
    print(f"All outputs saved to: {output_run_dir}")
    print("="*70)

    return df


def run_invasion_scenario(args):
    """
    Run Invasion scenario
    Call the logic from scenarios/invasion.py
    """
    print("\nInvasion Scenario")
    print("Calling scenarios/invasion.py...")

    # cmd
    cmd_args = [
        'invasion.py',  # argv[0]
        '--hospitals', str(args.hospitals),
        '--metros', str(args.metros),
        '--generations', str(args.generations),
        '--population', str(args.population),
        '--genotypes', str(args.genotypes),
        '--seed', str(args.seed),
        '--metro-migration', str(args.metro_migration),
        '--peripheral-migration', str(args.peripheral_migration),
        '--selection', str(args.selection),
        '--initial-prevalence', str(args.initial_prevalence),
        '--seed-hospital-fraction', str(args.seed_hospital_fraction),
        '--introduction-time', str(args.introduction_time),
        '--introduction-dose', str(args.introduction_dose),
        '--introduction-sites', str(args.introduction_sites),
        '--spatial-scale', str(args.spatial_scale),
        '--output-dir', args.output_dir,
    ]

    if args.decay_length is not None:
        cmd_args.extend(['--decay-length', str(args.decay_length)])

    if args.no_smooth:
        cmd_args.append('--no-smooth')

    if hasattr(args, 'smooth_window') and args.smooth_window:
        cmd_args.extend(['--smooth-window', str(args.smooth_window)])

    # Run invasion scenario
    from scenarios import invasion

    # Save sys.argv
    original_argv = sys.argv
    try:
        # Set sys.argv to cmd_args
        sys.argv = cmd_args
        # Call the main function of invasion
        invasion.main()
    finally:
        # Restore original sys.argv
        sys.argv = original_argv


def run_covid_scenario(args):
    """
    Run COVID-19 Intervention scenario
    Call the logic from scenarios/covid_intervention.py
    """
    print("\nCOVID-19 Intervention Scenario")
    print("Calling scenarios/covid_intervention.py...")

    # cmd args
    cmd_args = [
        'covid_intervention.py',  # argv[0]
        '--hospitals', str(args.hospitals),
        '--metros', str(args.metros),
        '--generations', str(args.generations),
        '--population', str(args.population),
        '--genotypes', str(args.genotypes),
        '--seed', str(args.seed),
        '--metro-migration', str(args.metro_migration),
        '--peripheral-migration', str(args.peripheral_migration),
        '--selection', str(args.selection),
        '--initial-prevalence', str(args.initial_prevalence),
        '--initial-m-freq', str(args.initial_m_freq),
        '--intervention-start', str(args.intervention_start),
        '--recovery-start', str(args.recovery_start),
        '--metro-closure', str(args.metro_closure),
        '--periph-closure', str(args.periph_closure),
        '--spatial-scale', str(args.spatial_scale),
        '--baseline-decay', str(args.baseline_decay),
        '--intervention-decay', str(args.intervention_decay),
        '--recovery-decay', str(args.recovery_decay),
        '--output-dir', args.output_dir,
    ]

    if hasattr(args, 'no_smooth') and args.no_smooth:
        cmd_args.append('--no-smooth')

    if hasattr(args, 'smooth_window') and args.smooth_window:
        cmd_args.extend(['--smooth-window', str(args.smooth_window)])

    # Run COVID scenario
    from scenarios import covid_intervention

    # Save sys.argv
    original_argv = sys.argv
    try:
        # Set sys.argv to cmd_args
        sys.argv = cmd_args
        # Call the main function
        covid_intervention.main()
    finally:
        # Restore original sys.argv
        sys.argv = original_argv


def create_parser():
    """create the argument parser with subcommands"""
    parser = argparse.ArgumentParser(
        description='Hospital Network Pathogen Transmission Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic simulation:
    python model.py --hospitals 100 --generations 1000

  Invasion Scenario:
    python model.py invasion --generations 2000 --introduction-time 500

  COVID-19 Intervention Scenario:
    python model.py covid --generations 1500 --intervention-start 400 --recovery-start 800

        """
    )

    # Create subparsers for different scenarios
    subparsers = parser.add_subparsers(
        dest='scenario',
        help='Select simulation scenario'
    )

    # ===== Basic Simulation Parameters =====
    parser.add_argument('-n', '--hospitals', type=int, default=DEFAULT_N_HOSPITALS,
                       help=f'Total hospitals (default: {DEFAULT_N_HOSPITALS})')
    parser.add_argument('-m', '--metros', type=int, default=DEFAULT_N_METROS,
                       help=f'Metro hospitals (default: {DEFAULT_N_METROS})')
    parser.add_argument('-g', '--generations', type=int, default=DEFAULT_GENERATIONS,
                       help=f'Generations (default: {DEFAULT_GENERATIONS})')
    parser.add_argument('--population', type=int, default=DEFAULT_POPULATION,
                       help=f'Population per hospital (default: {DEFAULT_POPULATION})')
    parser.add_argument('--genotypes', type=int, default=DEFAULT_N_GENOTYPES,
                       help=f'Number of genotypes (default: {DEFAULT_N_GENOTYPES})')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                       help=f'Random seed (default: {DEFAULT_SEED})')
    parser.add_argument('--fixed-genotypes', action='store_true',
                       help='Use fixed M/C genotype values from config (default: generate random values based on seed)')
    parser.add_argument('--metro-migration', type=float, default=DEFAULT_METRO_MIGRATION,
                       help=f'Metro migration rate (default: {DEFAULT_METRO_MIGRATION})')
    parser.add_argument('--peripheral-migration', type=float, default=DEFAULT_PERIPH_MIGRATION,
                       help=f'Peripheral migration rate (default: {DEFAULT_PERIPH_MIGRATION})')
    parser.add_argument('--selection', type=float, default=DEFAULT_SELECTION_STRENGTH,
                       help=f'Selection strength (default: {DEFAULT_SELECTION_STRENGTH})')
    parser.add_argument('--spatial-scale', type=float, default=DEFAULT_SPATIAL_SCALE,
                       help=f'Spatial scale (km) (default: {DEFAULT_SPATIAL_SCALE})')
    parser.add_argument('--decay-length', type=float, default=DEFAULT_DECAY_LENGTH,
                       help=f'Distance decay length (km) (default: {DEFAULT_DECAY_LENGTH})')
    parser.add_argument('-o', '--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                       help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')
    

    # ===== Invasion subcommand =====
    invasion_parser = subparsers.add_parser(
        'invasion',
        help='Invasion Scenario: Simulate introduction of Clade 2.5 into 2.4 hospital network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Invasion
    invasion_parser.add_argument('-n', '--hospitals', type=int, default=DEFAULT_N_HOSPITALS,
                                help='Total hospitals')
    invasion_parser.add_argument('-m', '--metros', type=int, default=DEFAULT_N_METROS,
                                help='Metro hospitals')
    invasion_parser.add_argument('-g', '--generations', type=int, default=INVASION_DEFAULT_GENERATIONS,
                                help='Generations')
    invasion_parser.add_argument('--population', type=int, default=DEFAULT_POPULATION,
                                help='Population per hospital')
    invasion_parser.add_argument('--genotypes', type=int, default=DEFAULT_N_GENOTYPES,
                                help='Number of genotypes')
    invasion_parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                                help='Random seed')
    invasion_parser.add_argument('--metro-migration', type=float, default=DEFAULT_METRO_MIGRATION,
                                help='Metro migration rate')
    invasion_parser.add_argument('--peripheral-migration', type=float, default=DEFAULT_PERIPH_MIGRATION,
                                help='Peripheral migration rate')
    invasion_parser.add_argument('--selection', type=float, default=DEFAULT_SELECTION_STRENGTH,
                                help='Selection strength')
    invasion_parser.add_argument('--initial-prevalence', type=float, default=DEFAULT_INITIAL_PREVALENCE,
                                help='Initial Clade 2.4 prevalence (0-1)')
    invasion_parser.add_argument('--seed-hospital-fraction', type=float, default=DEFAULT_SEED_HOSPITAL_FRACTION,
                                help='Fraction of hospitals with initial infections (0-1, default: 1.0)')
    invasion_parser.add_argument('--introduction-time', type=int, default=DEFAULT_INTRODUCTION_TIME,
                                help='Introduction time')
    invasion_parser.add_argument('--introduction-dose', type=float, default=DEFAULT_INTRODUCTION_DOSE,
                                help='Introduction dose')
    invasion_parser.add_argument('--introduction-sites', type=int, default=DEFAULT_INTRODUCTION_SITES,
                                help='Number of introduction sites')
    invasion_parser.add_argument('--spatial-scale', type=float, default=DEFAULT_SPATIAL_SCALE,
                                help='Spatial scale (km)')
    invasion_parser.add_argument('--decay-length', type=float, default=DEFAULT_DECAY_LENGTH,
                                help='Distance decay length (km)')
    invasion_parser.add_argument('-o', '--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                                help='Output directory')
    invasion_parser.add_argument('--no-smooth', action='store_true',
                                help='Disable smoothing')
    invasion_parser.add_argument('--smooth-window', type=int, default=DEFAULT_SMOOTH_WINDOW,
                                help='Smoothing window size')

    # ===== COVID Intervention subcommand =====
    covid_parser = subparsers.add_parser(
        'covid',
        help='COVID-19 Intervention: Three-phase differential hospital control',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # COVID Basic
    covid_parser.add_argument('-n', '--hospitals', type=int, default=DEFAULT_N_HOSPITALS,
                             help='Total hospitals')
    covid_parser.add_argument('-m', '--metros', type=int, default=DEFAULT_N_METROS,
                             help='Metro hospitals')
    covid_parser.add_argument('-g', '--generations', type=int, default=COVID_DEFAULT_GENERATIONS,
                             help='Total generations')
    covid_parser.add_argument('--population', type=int, default=DEFAULT_POPULATION,
                             help='Population per hospital')
    covid_parser.add_argument('--genotypes', type=int, default=DEFAULT_N_GENOTYPES,
                             help='Number of genotypes')
    covid_parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                             help='Random seed')
    # Migration rates
    covid_parser.add_argument('--metro-migration', type=float, default=DEFAULT_METRO_MIGRATION,
                             help='Metro hospital baseline migration rate')
    covid_parser.add_argument('--peripheral-migration', type=float, default=DEFAULT_PERIPH_MIGRATION,
                             help='Peripheral hospital baseline migration rate')
    covid_parser.add_argument('--selection', type=float, default=DEFAULT_SELECTION_STRENGTH,
                             help='Selection strength')
    # Initialization parameters
    covid_parser.add_argument('--initial-prevalence', type=float, default=DEFAULT_INITIAL_PREVALENCE,
                             help='Initial overall prevalence (0-1)')
    covid_parser.add_argument('--initial-m-freq', type=float, default=DEFAULT_INITIAL_M_FREQ,
                             help='Initial Clade 2.5 frequency in infected population (0-1)')
    # Three-phase time points
    covid_parser.add_argument('--intervention-start', type=int, default=DEFAULT_INTERVENTION_START,
                             help='Generation to start COVID intervention')
    covid_parser.add_argument('--recovery-start', type=int, default=DEFAULT_RECOVERY_START,
                             help='Generation to start recovery phase')
    # Hospital closure parameters
    covid_parser.add_argument('--metro-closure', type=float, default=DEFAULT_METRO_CLOSURE,
                             help='Metro hospital migration reduction ratio (0.83 = -83%%)')
    covid_parser.add_argument('--periph-closure', type=float, default=DEFAULT_PERIPH_CLOSURE,
                             help='Peripheral hospital migration reduction ratio (0.50 = -50%%)')
    # Spatial parameters (Three-phase)
    covid_parser.add_argument('--spatial-scale', type=float, default=DEFAULT_SPATIAL_SCALE,
                             help='Spatial scale (km)')
    covid_parser.add_argument('--baseline-decay', type=float, default=DEFAULT_BASELINE_DECAY,
                             help='Baseline distance decay length (km)')
    covid_parser.add_argument('--intervention-decay', type=float, default=DEFAULT_INTERVENTION_DECAY,
                             help='COVID intervention decay length (km)')
    covid_parser.add_argument('--recovery-decay', type=float, default=DEFAULT_RECOVERY_DECAY,
                             help='Recovery phase decay length (km)')
    # Output control
    covid_parser.add_argument('-o', '--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                             help='Output directory')
    covid_parser.add_argument('--no-smooth', action='store_true',
                             help='Disable smoothing in plots')
    covid_parser.add_argument('--smooth-window', type=int, default=DEFAULT_SMOOTH_WINDOW,
                             help='Smoothing window size')

    return parser


def main():
    """Main function to run the simulation based on selected scenario"""
    parser = create_parser()
    args = parser.parse_args()

    # Run the selected scenario
    if args.scenario == 'invasion':
        run_invasion_scenario(args)
    elif args.scenario == 'covid':
        run_covid_scenario(args)
    elif args.scenario is None:
        # No subcommand provided, run basic simulation
        run_basic_simulation(args)
    else:
        print(f"Error: Unknown scenario '{args.scenario}'")
        print("Available scenarios: invasion, covid")
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
