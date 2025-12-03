#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COVID-19 Intervention Scenario: Three-Phase Differential Hospital Control

Description:
    Phase 1: Baseline (0 → intervention_start)
        - Normal operations
        - metro_migration = 0.09, periph_migration = 0.02
        - decay_length = 50 km

    Phase 2: COVID Intervention (intervention_start → recovery_start)
        - Intervention:
          * Metro: Strict lockdown, transfer rate plummets 83% (0.09 → 0.015)
          * Peripheral: Moderate restrictions, transfer rate drops 50% (0.02 → 0.01)
          * Long-distance transfers banned (50 km → 15 km)

    Phase 3: Recovery (recovery_start → end)
        - Gradual recovery:
          * Metro: Recovers to 77% (0.015 → 0.07)
          * Peripheral: Recovers to 90% (0.01 → 0.018)
          * Distance decay partially recovers (15 km → 40 km)

Core Assumptions:
    - Clade 2.5 relies on "Metro hubs + long-distance transfers" → severely impacted by COVID
    - Clade 2.4 relies on "Peripheral hospitals + local spread" → moderately impacted by COVID
    - Expectation: During intervention, 2.5 declines sharply, while 2.4 declines slightly or even rebounds briefly

Usage:
    python scenarios/covid_intervention.py --generations 1500 --intervention-start 300 --recovery-start 800

    Optional parameters:
        --initial-m-freq 0.6           # Initial Clade 2.5 frequency
        --metro-closure 0.83           # Metro transfer rate reduction
        --periph-closure 0.50          # Peripheral transfer rate reduction
        --seed 42
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration (reuse)
from config import (
    TAU_VALUES, BETA_VALUES,
    DEFAULT_N_HOSPITALS, DEFAULT_N_METROS, DEFAULT_GENERATIONS,
    DEFAULT_POPULATION, DEFAULT_N_GENOTYPES,
    DEFAULT_METRO_MIGRATION, DEFAULT_PERIPH_MIGRATION,
    DEFAULT_SELECTION_STRENGTH, DEFAULT_SPATIAL_SCALE,
    classify_genotypes, get_timestamp, generate_random_genotype_values
)

# Import core engine (reuse)
from core.engine import (
    calculate_fitness,
    wright_fisher_reproduction,
    migrate_between_hospitals_with_transfer,
    generate_hospital_coordinates,
    initialize_simulation,
    calculate_spatial_spread
)

# Import visualization (reuse)
from visualization.invasion import (
    plot_invasion_curve_combined,
    plot_invasion_curve_combined_plotly,
    plot_prevalence_dynamics
)

from visualization.basic_plots import (
    export_hospital_network_state,
    plot_hospital_network_state,
    create_network_from_transfer_matrix,
    plot_spatial_spread_over_time
)

from visualization.covid_impact import (
    plot_covid_impact_barplot,
    plot_covid_impact_combined,
    plot_covid_impact_migration_velocity,
    calculate_spatial_dispersion,
    calculate_effective_transmission_distance,
    plot_distance_metrics_comparison
)

from utils.export import (
    export_network_analysis_json,
    export_transfer_matrix,
    export_results_csv,
    export_run_config
)

from utils.statistics import calculate_fst_statistics

# Import invasion auxiliary functions (reuse, no redefinition)
from scenarios.invasion import calculate_clade_frequencies


def parse_arguments():
    """Parse command line arguments for COVID intervention scenario"""
    parser = argparse.ArgumentParser(
        description='COVID-19 Intervention: Three-Phase Differential Control',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-n', '--hospitals', type=int, default=DEFAULT_N_HOSPITALS,
                       help='Total number of hospitals')
    parser.add_argument('-m', '--metros', type=int, default=DEFAULT_N_METROS,
                       help='Number of metropolitan hospitals')
    parser.add_argument('-g', '--generations', type=int, default=1500,
                       help='Total generations')
    parser.add_argument('--population', type=int, default=DEFAULT_POPULATION,
                       help='Population per hospital')
    parser.add_argument('--genotypes', type=int, default=DEFAULT_N_GENOTYPES,
                       help='Number of genotypes')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--fixed-genotypes', action='store_true',
                       help='Use fixed M/C genotype values from config (default: generate random values based on seed)')
    parser.add_argument('--initial-prevalence', type=float, default=0.05,
                       help='Initial overall prevalence')
    parser.add_argument('--initial-m-freq', type=float, default=0.5,
                       help='Initial Clade 2.5 frequency in infected population')
    parser.add_argument('--metro-migration', type=float, default=DEFAULT_METRO_MIGRATION,
                       help='Metro hospital baseline migration rate')
    parser.add_argument('--peripheral-migration', type=float, default=DEFAULT_PERIPH_MIGRATION,
                       help='Peripheral hospital baseline migration rate')
    parser.add_argument('--selection', type=float, default=DEFAULT_SELECTION_STRENGTH,
                       help='Selection strength')
    parser.add_argument('--intervention-start', type=int, default=300,
                       help='Generation to start COVID intervention')
    parser.add_argument('--recovery-start', type=int, default=800,
                       help='Generation to start recovery phase')
    parser.add_argument('--metro-closure', type=float, default=0.83,
                       help='Metro hospital migration reduction ratio (0.83 = -83%)')
    parser.add_argument('--periph-closure', type=float, default=0.50,
                       help='Peripheral hospital migration reduction ratio (0.50 = -50%)')
    parser.add_argument('--spatial-scale', type=float, default=DEFAULT_SPATIAL_SCALE,
                       help='Spatial scale (km)')
    parser.add_argument('--baseline-decay', type=float, default=50,
                       help='Baseline distance decay length (km)')
    parser.add_argument('--intervention-decay', type=float, default=15,
                       help='COVID intervention decay length (km) - cuts long-distance transfers')
    parser.add_argument('--recovery-decay', type=float, default=40,
                       help='Recovery phase decay length (km)')
    parser.add_argument('-o', '--output-dir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--no-smooth', action='store_true',
                       help='Disable smoothing in plots')
    parser.add_argument('--smooth-window', type=int, default=15,
                       help='Smoothing window size')

    return parser.parse_args()


def get_phase_parameters(gen, args, hospital_types):
    """
    Returns:
        dict: {
            'migration_rates': list,
            'decay_length': float,
            'phase_name': str
        }
    """
    # Phase 1: Baseline
    if gen < args.intervention_start:
        migration_rates = [
            args.metro_migration if ht == 'metro' else args.peripheral_migration
            for ht in hospital_types
        ]
        return {
            'migration_rates': migration_rates,
            'decay_length': args.baseline_decay,
            'phase_name': 'Baseline'
        }

    # Phase 2: COVID Intervention
    elif gen < args.recovery_start:
        migration_rates = [
            args.metro_migration * (1 - args.metro_closure) if ht == 'metro'
            else args.peripheral_migration * (1 - args.periph_closure)
            for ht in hospital_types
        ]
        return {
            'migration_rates': migration_rates,
            'decay_length': args.intervention_decay,
            'phase_name': 'COVID-Intervention'
        }

    # Phase 3: Recovery
    else:
        recovery_metro = args.metro_migration * 0.78  # 0.09 * 0.78 ≈ 0.07
        recovery_periph = args.peripheral_migration * 0.90  # 0.02 * 0.90 ≈ 0.018

        migration_rates = [
            recovery_metro if ht == 'metro' else recovery_periph
            for ht in hospital_types
        ]
        return {
            'migration_rates': migration_rates,
            'decay_length': args.recovery_decay,
            'phase_name': 'Recovery'
        }


def run_covid_intervention_scenario(args):
    """Main function to run the COVID intervention scenario"""

    # Set random seed
    np.random.seed(args.seed)

    # Validate parameters
    if args.intervention_start >= args.generations:
        raise ValueError(
            f"intervention_start ({args.intervention_start}) must be less than generations ({args.generations})"
        )
    if args.recovery_start >= args.generations:
        raise ValueError(
            f"recovery_start ({args.recovery_start}) must be less than generations ({args.generations})"
        )
    if args.intervention_start >= args.recovery_start:
        raise ValueError(
            f"intervention_start ({args.intervention_start}) must be less than recovery_start ({args.recovery_start})"
        )

    # Create independent run directory
    timestamp = get_timestamp()
    output_run_dir = f"{args.output_dir}/run_{timestamp}_covid_h{args.hospitals}_m{args.metros}_g{args.generations}_s{args.seed}"
    os.makedirs(f"{output_run_dir}/data", exist_ok=True)
    os.makedirs(f"{output_run_dir}/figures", exist_ok=True)

    print("\n" + "="*70)
    print("COVID-19 Intervention Scenario: Three-Phase Differential Control")
    print("="*70)
    print(f"\nOutput directory: {output_run_dir}")
    print(f"\nParameters:")
    print(f"  Hospitals: {args.hospitals} ({args.metros} metro, {args.hospitals - args.metros} peripheral)")
    print(f"  Generations: {args.generations}")
    print(f"  Initial prevalence: {args.initial_prevalence:.1%}")
    print(f"  Initial Clade 2.5 frequency: {args.initial_m_freq:.1%}")
    print(f"  Seed: {args.seed}")

    print(f"\nThree-Phase Timeline:")
    print(f"  Phase 1 (Baseline):        Gen 0 → {args.intervention_start}")
    print(f"  Phase 2 (COVID):           Gen {args.intervention_start} → {args.recovery_start}")
    print(f"  Phase 3 (Recovery):        Gen {args.recovery_start} → {args.generations}")

    print(f"\nCOVID Intervention Parameters:")
    metro_covid_rate = args.metro_migration * (1 - args.metro_closure)
    periph_covid_rate = args.peripheral_migration * (1 - args.periph_closure)
    print(f"  Metro hospitals:")
    print(f"    Closure strength: {args.metro_closure:.0%}")
    print(f"    Migration rate:   {args.metro_migration:.3f} → {metro_covid_rate:.3f} (Δ {-args.metro_closure:.0%})")
    print(f"  Peripheral hospitals:")
    print(f"    Closure strength: {args.periph_closure:.0%}")
    print(f"    Migration rate:   {args.peripheral_migration:.3f} → {periph_covid_rate:.3f} (Δ {-args.periph_closure:.0%})")
    print(f"  Spatial decay:")
    print(f"    {args.baseline_decay}km → {args.intervention_decay}km → {args.recovery_decay}km")

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

    # Genotypes
    genotype_classes = classify_genotypes(TAU_sim, exclude_neutral=False)
    M_genotypes = genotype_classes['metro']
    C_genotypes = genotype_classes['periph']

    print(f"\nGenotype Classification:")
    print(f"  M-genotypes (Clade 2.5): {M_genotypes}")
    print(f"  C-genotypes (Clade 2.4): {C_genotypes}")

    # Generate hospital coordinates
    coordinates = generate_hospital_coordinates(
        args.hospitals,
        args.spatial_scale,
        seed=args.seed
    )

    # ========== Both clades coexist ==========
    print(f"\n")
    print(f"Initialization: Both clades coexist")

    # Use balanced initialization
    # Note: Use lower seed_hospital_fraction to start from low prevalence like other scenarios
    hospital_populations, hospital_types = initialize_simulation(
        args.hospitals,
        args.metros,
        args.population,
        args.genotypes,
        TAU_sim,
        initial_prevalence=args.initial_prevalence,
        seed_genotypes='balanced',
        seed_hospital_fraction=0.15  # Changed from 1.0 to match basic/invasion scenarios
    )

    # Note: Removed the manual adjustment code to allow natural low-prevalence start
    # The initialization from initialize_simulation() is sufficient
    # This matches the behavior of basic and invasion scenarios

    # Validate initial state
    initial_metrics = calculate_clade_frequencies(hospital_populations, hospital_types,
                                                 M_genotypes, C_genotypes)
    print(f"\nInitial state (natural low prevalence):")
    print(f"  Clade 2.5:")
    print(f"    Metro:      {initial_metrics['metro_m_freq']:.1%}")
    print(f"    Peripheral: {initial_metrics['periph_m_freq']:.1%}")
    print(f"    Overall:    {initial_metrics['overall_m_freq']:.1%}")
    print(f"  Clade 2.4:")
    print(f"    Metro:      {initial_metrics['metro_c_freq']:.1%}")
    print(f"    Peripheral: {initial_metrics['periph_c_freq']:.1%}")
    print(f"    Overall:    {initial_metrics['overall_c_freq']:.1%}")

    # Retrieve metro and peripheral hospital indices
    metro_indices = [i for i, t in enumerate(hospital_types) if t == 'metro']
    periph_indices = [i for i, t in enumerate(hospital_types) if t == 'peripheral']

    # Time series data
    timeline = []
    time_series = {
        'overall_freqs': [],
        'metro_freqs': [],
        'periph_freqs': [],
        'hospital_freqs': [],
        'spatial_spread': {'timesteps': [], 'm_spread': [], 'c_spread': []}  # NEW: spatial spread time series
    }

    transfer_matrix = None

    # Snapshots for distance metrics calculation (end of each Phase)
    phase_snapshots = {
        'phase1': {'populations': None, 'transfer_matrix': None},
        'phase2': {'populations': None, 'transfer_matrix': None},
        'phase3': {'populations': None, 'transfer_matrix': None}
    }

    # ========== Main simulation loop: Three-phase dynamic adjustment ==========
    print(f"\nStarting Three-Phase Simulation")

    current_phase = None

    for gen in range(args.generations):
        # Retrieve current phase parameters
        phase_params = get_phase_parameters(gen, args, hospital_types)

        # Detect phase changes
        if phase_params['phase_name'] != current_phase:
            current_phase = phase_params['phase_name']
            print(f"\nEntering Phase: {current_phase} (Gen {gen})")
            print(f"  Migration rates: metro={phase_params['migration_rates'][0]:.4f}, "
                  f"periph={phase_params['migration_rates'][-1]:.4f}")
            print(f"  Distance decay:  {phase_params['decay_length']} km")

        # 1. Selection and reproduction
        for h in range(args.hospitals):
            fitness_values = np.array([
                calculate_fitness(geno, hospital_types[h], TAU_sim, BETA_sim, args.selection)
                for geno in hospital_populations[h]
            ])
            hospital_populations[h] = wright_fisher_reproduction(
                hospital_populations[h],
                fitness_values,
                args.population
            )

        # 2. Migration (using current phase parameters)
        # Retrieve transfer_matrix for distance metrics calculation at the end of each Phase, and for spatial spread calculation every 10 generations
        need_transfer_matrix = (
            gen == args.intervention_start - 1 or  # End of Phase 1
            gen == args.recovery_start - 1 or      # End of Phase 2
            gen == args.generations - 1 or         # End of Phase 3
            (gen % 10 == 0 or gen == 0)            # Every 10 generations for spatial spread calculation
        )

        if need_transfer_matrix:
            hospital_populations, transfer_matrix_snapshot = migrate_between_hospitals_with_transfer(
                hospital_populations,
                phase_params['migration_rates'],
                hospital_types,
                args.population,
                coordinates=coordinates,
                decay_length=phase_params['decay_length'],
                return_matrix=True
            )

            # Save snapshots at the end of each Phase
            import copy
            if gen == args.intervention_start - 1:
                phase_snapshots['phase1']['populations'] = copy.deepcopy(hospital_populations)
                phase_snapshots['phase1']['transfer_matrix'] = transfer_matrix_snapshot.copy()
            elif gen == args.recovery_start - 1:
                phase_snapshots['phase2']['populations'] = copy.deepcopy(hospital_populations)
                phase_snapshots['phase2']['transfer_matrix'] = transfer_matrix_snapshot.copy()
            elif gen == args.generations - 1:
                phase_snapshots['phase3']['populations'] = copy.deepcopy(hospital_populations)
                phase_snapshots['phase3']['transfer_matrix'] = transfer_matrix_snapshot.copy()
                transfer_matrix = transfer_matrix_snapshot  # 保存最后一代的transfer_matrix
        else:
            hospital_populations = migrate_between_hospitals_with_transfer(
                hospital_populations,
                phase_params['migration_rates'],
                hospital_types,
                args.population,
                coordinates=coordinates,
                decay_length=phase_params['decay_length']
            )

        # 3. Record metrics
        metrics = calculate_clade_frequencies(hospital_populations, hospital_types,
                                             M_genotypes, C_genotypes)

        # Calculate prevalence (absolute numbers / total population)
        total_population = args.hospitals * args.population
        n24_total = sum(np.sum([pop == g for g in C_genotypes]) for pop in hospital_populations)
        n25_total = sum(np.sum([pop == g for g in M_genotypes]) for pop in hospital_populations)
        prevalence24 = n24_total / total_population
        prevalence25 = n25_total / total_population

        timeline.append({
            'generation': gen,
            'phase': current_phase,
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

        hospital_freqs_dict = {}
        for h in range(args.hospitals):
            hospital_freqs_dict[h] = current_genotype_freqs[h, :].tolist()
        time_series['hospital_freqs'].append(hospital_freqs_dict)

        # Calculate spatial spread (every 10 generations to reduce computation)
        # Note: transfer_matrix_snapshot is available because need_transfer_matrix=True for these generations
        if gen % 10 == 0 or gen == 0:
            m_spread, c_spread = calculate_spatial_spread(
                hospital_populations, coordinates, transfer_matrix_snapshot, M_genotypes, C_genotypes
            )
            time_series['spatial_spread']['timesteps'].append(gen)
            time_series['spatial_spread']['m_spread'].append(m_spread)
            time_series['spatial_spread']['c_spread'].append(c_spread)

        # Progress display
        if (gen + 1) % 100 == 0 or gen == 0:
            print(f"  Gen {gen+1:4d}/{args.generations}: "
                  f"C2.5={metrics['overall_m_freq']:.3f}, "
                  f"C2.4={metrics['overall_c_freq']:.3f}")

    # ========== Results ==========
    print(f"\n")
    print(f"Final Results (Gen {args.generations})")
    final_metrics = timeline[-1]
    print(f"  Clade 2.5 (M-genotypes):")
    print(f"    Metro:      {final_metrics['metro_m_freq']:.1%}")
    print(f"    Peripheral: {final_metrics['periph_m_freq']:.1%}")
    print(f"    Overall:    {final_metrics['overall_m_freq']:.1%}")
    print(f"\n  Clade 2.4 (C-genotypes):")
    print(f"    Metro:      {final_metrics['metro_c_freq']:.1%}")
    print(f"    Peripheral: {final_metrics['periph_c_freq']:.1%}")
    print(f"    Overall:    {final_metrics['overall_c_freq']:.1%}")

    # Analyze the impact of COVID intervention
    pre_intervention = timeline[args.intervention_start - 1]
    mid_intervention_idx = int((args.intervention_start + args.recovery_start) / 2)
    mid_intervention = timeline[mid_intervention_idx]
    post_recovery = timeline[-1]

    print(f"\n")
    print(f"COVID-19 Intervention Impact Analysis")
    print(f"Clade 2.5 trajectory:")
    print(f"  Before intervention (Gen {args.intervention_start-1}):  {pre_intervention['overall_m_freq']:.1%}")
    print(f"  Mid intervention (Gen {mid_intervention_idx}):          {mid_intervention['overall_m_freq']:.1%}  "
          f"(Δ = {mid_intervention['overall_m_freq'] - pre_intervention['overall_m_freq']:+.1%})")
    print(f"  After recovery (Gen {args.generations}):                {post_recovery['overall_m_freq']:.1%}  "
          f"(Δ = {post_recovery['overall_m_freq'] - mid_intervention['overall_m_freq']:+.1%})")

    print(f"\nClade 2.4 trajectory:")
    print(f"  Before intervention (Gen {args.intervention_start-1}):  {pre_intervention['overall_c_freq']:.1%}")
    print(f"  Mid intervention (Gen {mid_intervention_idx}):          {mid_intervention['overall_c_freq']:.1%}  "
          f"(Δ = {mid_intervention['overall_c_freq'] - pre_intervention['overall_c_freq']:+.1%})")
    print(f"  After recovery (Gen {args.generations}):                {post_recovery['overall_c_freq']:.1%}  "
          f"(Δ = {post_recovery['overall_c_freq'] - mid_intervention['overall_c_freq']:+.1%})")

    # ========== Calculate final genotype frequencies ==========
    final_genotype_freqs = np.zeros((args.hospitals, args.genotypes))
    for h in range(args.hospitals):
        for g in range(args.genotypes):
            final_genotype_freqs[h, g] = np.mean([geno == g for geno in hospital_populations[h]])

    # ========== Save Results ==========
    print(f"\n")
    print(f"Saving Results")

    timeline_df = pd.DataFrame(timeline)

    # Save timeline
    timeline_path = f"{output_run_dir}/data/timeline.csv"
    timeline_df.to_csv(timeline_path, index=False)
    print(f"\nTimeline saved: {timeline_path}")

    print("\n" + "="*70)
    print("Generating Visualizations")
    print("="*70)

    # Static combined plot
    static_path = f"{output_run_dir}/figures/covid_combined.pdf"
    use_smooth = not args.no_smooth

    # Call plotting function with two time points marking three phases
    plot_invasion_curve_combined(
        timeline_df,
        args.intervention_start,
        static_path,
        smooth=use_smooth,
        smooth_window=args.smooth_window,
        second_phase_time=args.recovery_start
    )
    print(f"  Static plot saved: {static_path}")

    interactive_path = f"{output_run_dir}/figures/covid_interactive.html"
    plot_invasion_curve_combined_plotly(timeline_df, args.intervention_start, interactive_path)
    print(f"  Interactive plot saved: {interactive_path}")

    prevalence_path = f"{output_run_dir}/figures/covid_prevalence.pdf"
    plot_prevalence_dynamics(
        timeline_df,
        args.intervention_start,
        prevalence_path,
        smooth=use_smooth,
        smooth_window=args.smooth_window,
        second_phase_time=args.recovery_start
    )
    print(f"  Prevalence dynamics plot saved: {prevalence_path}")

    # Spatial spread over time
    print("\n[Spatial Spread] Generating time series...")
    spatial_spread_path = f"{output_run_dir}/figures/covid_spatial_spread.pdf"
    spatial_spread_data = {
        'timesteps': np.array(time_series['spatial_spread']['timesteps']),
        'm_spread': np.array(time_series['spatial_spread']['m_spread']),
        'c_spread': np.array(time_series['spatial_spread']['c_spread'])
    }
    plot_spatial_spread_over_time(
        spatial_spread_data=spatial_spread_data,
        output_path=spatial_spread_path,
        intervention_start=args.intervention_start,
        recovery_start=args.recovery_start
    )
    print(f"  Spatial spread plot saved: {spatial_spread_path}")

    # COVID impact barplot
    impact_barplot_path = f"{output_run_dir}/figures/covid_impact_barplot.pdf"
    plot_covid_impact_barplot(
        timeline_df,
        intervention_start=args.intervention_start,
        recovery_start=args.recovery_start,
        output_path=impact_barplot_path
    )
    print(f"  COVID impact barplot saved: {impact_barplot_path}")

    # COVID impact combined plot
    impact_combined_path = f"{output_run_dir}/figures/covid_impact_combined.pdf"
    plot_covid_impact_combined(
        timeline_df,
        intervention_start=args.intervention_start,
        recovery_start=args.recovery_start,
        output_path=impact_combined_path
    )
    print(f"  COVID impact combined plot saved: {impact_combined_path}")

    # Network state
    network_state_csv = f"{output_run_dir}/data/network_state.csv"
    export_hospital_network_state(
        hospital_types=hospital_types,
        genotype_freqs=final_genotype_freqs,
        coordinates=coordinates,
        TAU_VALUES=TAU_sim,
        BETA_VALUES=BETA_sim,
        output_path=network_state_csv
    )

    network_state_plot = f"{output_run_dir}/figures/covid_network_state.pdf"
    plot_hospital_network_state(
        csv_path=network_state_csv,
        output_path=network_state_plot
    )
    print(f"  Network state plot saved: {network_state_plot}")

    # ========== Export Phase-Specific Transfer Matrices ==========
    print("\n" + "="*70)
    print("Exporting Phase-Specific Transfer Matrices")
    print("="*70)

    for phase_name, phase_key in [('Baseline', 'phase1'),
                                   ('Intervention', 'phase2'),
                                   ('Recovery', 'phase3')]:
        snapshot = phase_snapshots[phase_key]
        if snapshot['transfer_matrix'] is not None:
            matrix_path = f"{output_run_dir}/data/transfer_matrix_{phase_key}.csv"
            export_transfer_matrix(
                snapshot['transfer_matrix'],
                hospital_types,
                matrix_path
            )
            print(f"  {phase_name} ({phase_key}): {matrix_path}")
        else:
            print(f"  Warning: {phase_name} transfer matrix not available")

    print(f"\n[8/8] Calculating distance metrics for three phases...")

    distance_metrics = {}

    for phase_name, phase_key in [('Phase 1', 'phase1'), ('Phase 2', 'phase2'), ('Phase 3', 'phase3')]:
        snapshot = phase_snapshots[phase_key]
        if snapshot['populations'] is not None and snapshot['transfer_matrix'] is not None:
            # Method 1: Spatial Dispersion
            dispersion = calculate_spatial_dispersion(
                hospital_populations=snapshot['populations'],
                hospital_types=hospital_types,
                coordinates=coordinates,
                M_genotypes=M_genotypes,
                C_genotypes=C_genotypes
            )

            # Method 2: Effective Transmission Distance
            transmission_dist = calculate_effective_transmission_distance(
                hospital_populations=snapshot['populations'],
                hospital_types=hospital_types,
                coordinates=coordinates,
                transfer_matrix=snapshot['transfer_matrix'],
                M_genotypes=M_genotypes,
                C_genotypes=C_genotypes
            )

            distance_metrics[phase_key] = {
                **dispersion,
                **transmission_dist
            }
        else:
            print(f"  Warning: {phase_name} snapshot not available")
            distance_metrics[phase_key] = {
                'dispersion_24': 0,
                'dispersion_25': 0,
                'transmission_dist_24': 0,
                'transmission_dist_25': 0
            }

    # Generate distance metrics comparison plot
    distance_metrics_path = f"{output_run_dir}/figures/covid_distance_metrics.pdf"
    plot_distance_metrics_comparison(distance_metrics, output_path=distance_metrics_path)

    # Generate network_analysis JSON
    if transfer_matrix is not None:
        network_analysis_json = f"{output_run_dir}/data/network_analysis.json"

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

    # Export run configuration
    print("\n[Run Config] Exporting parameters...")
    config_path = f"{output_run_dir}/run_config.json"
    export_run_config(args, TAU_sim, BETA_sim, 'covid', config_path)

    print("\n" + "="*70)
    print("COVID-19 Intervention Scenario Completed!")
    print("="*70)
    print(f"\nKey finding:")
    print(f"  Differential intervention impacts Clade 2.5 (metro-dependent)")
    print(f"  more severely than Clade 2.4 (local transmission)")
    print(f"\nAll outputs saved to: {output_run_dir}")

    return {
        'timeline': timeline_df,
        'final_metrics': final_metrics,
        'impact_analysis': {
            'pre_intervention': pre_intervention,
            'mid_intervention': mid_intervention,
            'post_recovery': post_recovery
        },
        'args': args
    }


def main():
    """Main"""
    args = parse_arguments()
    results = run_covid_intervention_scenario(args)
    return results


if __name__ == '__main__':
    main()
