#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Simulation Engine
simulation + genetics + migration
"""

import numpy as np
from typing import List, Tuple, Union
import sys
import os

# import config parameters
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TAU_VALUES, BETA_VALUES


def calculate_fitness(genotype: int, hospital_type: str, TAU_array, BETA_array, selection_strength: float = 0.1) -> float:
    """
    Calculate fitness based on genotype-hospital type matching.

    Args:
        genotype: Genotype ID (0 to N_GENOTYPES-1)
        hospital_type: 'metro' or 'peripheral'
        TAU_array: Array of τ values (outpatient colonization capacity)
        BETA_array: Array of β values (inpatient infection capacity)
        selection_strength: Selection strength (default 0.1)

    Returns:
        Fitness value in [0.01, 1.0]
    """
    tau = TAU_array[genotype]
    beta = BETA_array[genotype]

    # Calculate match factor directly based on hospital type
    if hospital_type == 'metro':
        match_factor = 0.7 * tau + 0.3 * beta  # Metro hospitals (high O/I) prefer high-τ genotypes
    else:  # peripheral
        match_factor = 0.6 * beta + 0.4 * tau  # Peripheral hospitals (low O/I) prefer high-β genotypes

    # Fitness: baseline ± selection effect
    # match_factor ∈ [0, 1] → (2*match_factor - 1) ∈ [-1, 1]
    # fitness ∈ [0.5 - s, 0.5 + s] = [0.4, 0.6] when s=0.1
    fitness = 0.5 + selection_strength * (2 * match_factor - 1)

    return max(0.01, min(1.0, fitness))


def wright_fisher_reproduction(population: np.ndarray, fitness_values: np.ndarray, population_size: int) -> np.ndarray:
    """
    Wright-Fisher reproduction model

    Parameters:
        population: Current generation population (array, each element is a genotype ID)
        fitness_values: Fitness value for each individual
        population_size: Size of population

    Returns:
        next_generation: Next generation population (same size as input)

    Mechanism:
        1. Calculate reproduction probability for each individual based on fitness proportion
        2. Sample with replacement to generate next generation
        3. Introduce genetic drift (random sampling)
    """
    total_fitness = np.sum(fitness_values)
    probabilities = fitness_values / total_fitness

    next_generation = np.random.choice(
        population,
        size=population_size,
        p=probabilities,
        replace=True
    )
    return next_generation


def migrate_between_hospitals_with_transfer(hospital_populations: List[np.ndarray],
                                          hospital_migration_rates: List[float],
                                          hospital_types: List[str],
                                          population_size: int,
                                          coordinates: Tuple[np.ndarray, np.ndarray] = None,
                                          decay_length: float = None,
                                          return_matrix: bool = False,
                                          periph_to_metro_pref: float = None,
                                          metro_to_periph_resist: float = None,
                                          periph_to_periph_penalty: float = None,
                                          metro_to_metro_pref: float = None,
                                          metro_decay_multiplier: float = None) -> Union[List[np.ndarray], Tuple[List[np.ndarray], np.ndarray]]:
    """
    Inter-hospital migration - with spatial distance decay mechanism and transfer preferences

    Parameters:
        hospital_populations: Population array for each hospital
        hospital_migration_rates: Base migration rate for each hospital
        hospital_types: Type of each hospital ('metro' or 'peripheral')
        population_size: Size of each hospital population
        coordinates: Tuple of (x_coords, y_coords) for spatial distance calculation
        decay_length: Characteristic length for exponential distance decay
        return_matrix: If True, return (new_populations, transfer_matrix)
        periph_to_metro_pref: Small→Large transfer preference multiplier (default from config)
        metro_to_periph_resist: Large→Small transfer resistance multiplier (default from config)
        periph_to_periph_penalty: Small→Small transfer penalty multiplier (default from config)
        metro_to_metro_pref: Large→Large transfer preference multiplier (default from config)
        metro_decay_multiplier: Distance decay multiplier for large hospitals (default from config)

    Returns:
        new_populations: Population array list after migration
        (optional) transfer_matrix: Transfer probability matrix

    Migration mechanism:
        1. Transfer direction preference (configurable):
           - Peripheral→Metro: base_prob × periph_to_metro_pref
           - Metro→Peripheral: base_prob × metro_to_periph_resist
           - Peripheral→Peripheral: base_prob × periph_to_periph_penalty
           - Metro→Metro: base_prob × metro_to_metro_pref

        2. Distance decay (optional, if coordinates provided):
           - decay_factor = exp(-distance / effective_decay_length)
           - Large hospitals use decay_length × metro_decay_multiplier
           - Reduces transfer probability with geographic distance

        3. Migration execution:
           - Use base migration rate directly (no genotype-dependent boost)
           - Random selection of migrants (no genotype bias)
    """
    # Import config defaults if parameters not provided
    from config import (
        DEFAULT_PERIPH_TO_METRO_PREFERENCE,
        DEFAULT_METRO_TO_PERIPH_RESISTANCE,
        DEFAULT_PERIPH_TO_PERIPH_PENALTY,
        DEFAULT_METRO_TO_METRO_PREFERENCE,
        DEFAULT_METRO_DECAY_MULTIPLIER
    )

    # Use config defaults if not specified
    if periph_to_metro_pref is None:
        periph_to_metro_pref = DEFAULT_PERIPH_TO_METRO_PREFERENCE
    if metro_to_periph_resist is None:
        metro_to_periph_resist = DEFAULT_METRO_TO_PERIPH_RESISTANCE
    if periph_to_periph_penalty is None:
        periph_to_periph_penalty = DEFAULT_PERIPH_TO_PERIPH_PENALTY
    if metro_to_metro_pref is None:
        metro_to_metro_pref = DEFAULT_METRO_TO_METRO_PREFERENCE
    if metro_decay_multiplier is None:
        metro_decay_multiplier = DEFAULT_METRO_DECAY_MULTIPLIER
    n_hospitals = len(hospital_populations)
    new_populations = [pop.copy() for pop in hospital_populations]

    # Build transfer probability matrix
    transfer_matrix = np.zeros((n_hospitals, n_hospitals))

    # Extract coordinates if provided
    if coordinates is not None:
        x_coords, y_coords = coordinates
        use_distance_decay = True
    else:
        use_distance_decay = False

    for i in range(n_hospitals):
        for j in range(n_hospitals):
            if i != j:
                base_prob = 0.1

                # Layer 1: Transfer direction preference (hierarchy) - using config parameters
                if hospital_types[i] == 'peripheral' and hospital_types[j] == 'metro':
                    base_prob *= periph_to_metro_pref  # Upward transfer preference
                elif hospital_types[i] == 'metro' and hospital_types[j] == 'peripheral':
                    base_prob *= metro_to_periph_resist  # Downward transfer resistance
                elif hospital_types[i] == 'peripheral' and hospital_types[j] == 'peripheral':
                    base_prob *= periph_to_periph_penalty  # Lateral transfer among small hospitals is rare
                elif hospital_types[i] == 'metro' and hospital_types[j] == 'metro':
                    base_prob *= metro_to_metro_pref  # Metro-to-metro transfer preference (enable hub network)

                # Layer 2: Distance decay (with differentiated decay length)
                if use_distance_decay and decay_length is not None:
                    dx = x_coords[i] - x_coords[j]
                    dy = y_coords[i] - y_coords[j]
                    distance = np.sqrt(dx**2 + dy**2)

                    # Differentiated decay: large hospitals accept longer-distance transfers
                    if hospital_types[j] == 'metro':
                        effective_decay_length = decay_length * metro_decay_multiplier  # Large hospitals have wider catchment area
                    else:
                        effective_decay_length = decay_length

                    decay_factor = np.exp(-distance / effective_decay_length)
                    base_prob *= decay_factor

                transfer_matrix[i, j] = base_prob

    # Normalize transfer probability matrix
    for i in range(n_hospitals):
        row_sum = transfer_matrix[i, :].sum()
        if row_sum > 0:
            transfer_matrix[i, :] /= row_sum

    # Execute migration
    for i in range(n_hospitals):
        # Use base migration rate directly (no genotype-dependent boost)
        effective_migration_rate = hospital_migration_rates[i]
        actual_migrants = int(population_size * effective_migration_rate)

        if actual_migrants > 0:
            actual_migrants = min(actual_migrants, population_size)

            # Random selection of migrants (no genotype bias)
            migrant_indices = np.random.choice(population_size, actual_migrants, replace=False)
            migrants = hospital_populations[i][migrant_indices]

            for migrant_genotype in migrants:
                target_hospital = np.random.choice(n_hospitals, p=transfer_matrix[i, :])

                if target_hospital != i:
                    replace_idx = np.random.randint(population_size)
                    new_populations[target_hospital][replace_idx] = migrant_genotype

    if return_matrix:
        return new_populations, transfer_matrix
    else:
        return new_populations


def generate_hospital_coordinates(n_hospitals: int, spatial_scale: float = 100.0, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 2D spatial coordinates for hospitals.

    Parameters:
        n_hospitals: Number of hospitals
        spatial_scale: Size of the spatial domain (e.g., 100 = 100x100 grid)
        seed: Random seed for reproducibility

    Returns:
        x_coords: X coordinates for each hospital
        y_coords: Y coordinates for each hospital
    """
    if seed is not None:
        np.random.seed(seed)

    x_coords = np.random.uniform(0, spatial_scale, n_hospitals)
    y_coords = np.random.uniform(0, spatial_scale, n_hospitals)

    return x_coords, y_coords


def initialize_simulation(n_hospitals: int,
                         n_metros: int,
                         population_size: int,
                         n_genotypes: int,
                         TAU_array,
                         initial_prevalence=0.05,
                         seed_genotypes='balanced',
                         seed_hospital_fraction=0.15) -> Tuple[List[np.ndarray], List[str]]:
    """
    Ver4: Low prevalence initialization, simulating transmission process

    Parameters:
        n_hospitals: Total number of hospitals
        n_metros: Number of metro hospitals
        population_size: Population size per hospital
        n_genotypes: Number of genotypes
        TAU_array: Array of τ values (outpatient colonization capacity)
        initial_prevalence: Initial infection rate (default 5%)
        seed_genotypes: 'balanced'(both groups), 'τ_only'(high-τ genotypes only), 'β_only'(high-β genotypes only)
        seed_hospital_fraction: Fraction of hospitals with initial infections (default 0.15)

    Returns:
        hospital_populations: Initial population for each hospital
        hospital_types: Type list for each hospital
    """
    hospital_types = []

    for i in range(n_hospitals):
        if i < n_metros:
            hospital_types.append('metro')
        else:
            hospital_types.append('peripheral')

    # Determine which genotypes are seeds (initial infected)
    # Use stricter thresholds to ensure clear separation between high-τ and high-β genotypes
    tau_genotypes = [g for g in range(n_genotypes) if TAU_array[g] > 0.6]
    beta_genotypes = [g for g in range(n_genotypes) if TAU_array[g] < 0.4]

    # Select neutral genotype as background (uninfected state)
    # Choose genotype with τ value closest to 0.5 as "background" (susceptible, not occupied by any strategy group)
    neutral_genotypes = [g for g in range(n_genotypes) if 0.4 <= TAU_array[g] <= 0.6]
    if len(neutral_genotypes) > 0:
        background_genotype = min(neutral_genotypes, key=lambda g: abs(TAU_array[g] - 0.5))
    else:
        # If no neutral genotype, randomly select one
        background_genotype = n_genotypes // 2

    # Initialize population: single-origin seeding scenario
    # Only a fraction of hospitals have initial infections, others are completely susceptible
    n_seed_hospitals = max(1, int(n_hospitals * seed_hospital_fraction))
    seed_hospital_indices = np.random.choice(n_hospitals, size=n_seed_hospitals, replace=False)

    print(f"\nSingle-origin seeding:")
    print(f"  Seed hospitals: {n_seed_hospitals}/{n_hospitals} ({n_seed_hospitals/n_hospitals:.1%})")
    print(f"  Seed hospital IDs: {sorted(seed_hospital_indices.tolist())}")

    hospital_populations = []
    for i in range(n_hospitals):
        # 特殊情况：如果是全医院覆盖(seed_hospital_fraction=1.0)且高prevalence(>0.9)
        # 说明是invasion场景，应该让所有个体都是目标基因型，不留background
        use_full_colonization = (seed_hospital_fraction == 1.0 and initial_prevalence > 0.9)

        if use_full_colonization:
            # Invasion场景：直接用目标genotype填充，不留neutral background
            # 这样避免neutral在大医院的适应度优势
            if seed_genotypes == 'beta_only' and len(beta_genotypes) > 0:
                # 全部初始化为C-genotypes（从beta_genotypes中随机选择）
                population = np.array([np.random.choice(beta_genotypes) for _ in range(population_size)])
            elif seed_genotypes == 'tau_only' and len(tau_genotypes) > 0:
                # 全部初始化为M-genotypes
                population = np.array([np.random.choice(tau_genotypes) for _ in range(population_size)])
            elif seed_genotypes == 'balanced':
                # 平衡模式：50% M, 50% C
                population = np.array([
                    np.random.choice(tau_genotypes) if np.random.random() < 0.5 and len(tau_genotypes) > 0
                    else np.random.choice(beta_genotypes) if len(beta_genotypes) > 0
                    else background_genotype
                    for _ in range(population_size)
                ])
            else:
                # Fallback：使用background
                population = np.full(population_size, background_genotype, dtype=int)
        else:
            # 正常情况：大部分是background genotype (uninfected state)
            population = np.full(population_size, background_genotype, dtype=int)

        # Only seed hospitals have initial infections (仅在非full_colonization模式)
        if i in seed_hospital_indices and not use_full_colonization:
            # Calculate initial infected individual count
            n_infected = int(population_size * initial_prevalence)

            if n_infected > 0:
                # Randomly select individuals to infect
                infected_indices = np.random.choice(population_size, n_infected, replace=False)

                # Assign genotypes according to seed_genotypes mode
                if seed_genotypes == 'balanced':
                    # Both genotype groups present, randomly assign
                    for idx in infected_indices:
                        if np.random.random() < 0.5 and len(tau_genotypes) > 0:
                            population[idx] = np.random.choice(tau_genotypes)
                        elif len(beta_genotypes) > 0:
                            population[idx] = np.random.choice(beta_genotypes)
                        else:
                            population[idx] = np.random.choice(tau_genotypes)

                elif seed_genotypes == 'tau_only':
                    # Only M-genotypes as initial infection
                    if len(tau_genotypes) > 0:
                        for idx in infected_indices:
                            population[idx] = np.random.choice(tau_genotypes)

                elif seed_genotypes == 'beta_only':
                    # Only C-genotypes as initial infection
                    if len(beta_genotypes) > 0:
                        for idx in infected_indices:
                            population[idx] = np.random.choice(beta_genotypes)

        hospital_populations.append(population)

    # Print initialization statistics
    if len(tau_genotypes) > 0 and len(beta_genotypes) > 0:
        total_individuals = n_hospitals * population_size
        m_count = sum(np.sum(pop == g) for pop in hospital_populations for g in tau_genotypes)
        c_count = sum(np.sum(pop == g) for pop in hospital_populations for g in beta_genotypes)
        neutral_count = total_individuals - m_count - c_count

        print(f"\nInitialization completed:")
        print(f"  High-τ genotypes (τ>0.6): {m_count}/{total_individuals} ({m_count/total_individuals:.1%})")
        print(f"  High-β genotypes (τ<0.4): {c_count}/{total_individuals} ({c_count/total_individuals:.1%})")
        print(f"  Neutral genotypes (0.4≤τ≤0.6): {neutral_count}/{total_individuals} ({neutral_count/total_individuals:.1%})")
        print(f"  Background genotype ID: {background_genotype} (τ={TAU_array[background_genotype]:.2f})")

    return hospital_populations, hospital_types


def calculate_spatial_spread(hospital_populations: List[np.ndarray],
                             coordinates: Tuple[np.ndarray, np.ndarray],
                             transfer_matrix: np.ndarray,
                             tau_genotypes: List[int],
                             beta_genotypes: List[int]) -> Tuple[float, float]:
    """
    Calculate effective transmission distance for M and C genotype groups (Method 2 - Recommended).

    This measures the average jump distance through the transfer network,
    weighted by actual transfer probabilities and infected population size.

    Formula: Effective transmission distance = Σ(distance_ij × transfer_ij × n_i) / Σ(transfer_ij × n_i)

    Physical meaning:
        - Reflects actual transmission dynamics through the transfer network
        - Considers directionality of transfers (peripheral→metro ≠ metro→peripheral)
        - Only counts hospital pairs with actual transfer connections
        - Higher values = longer-range transmission strategy
        - Expected: M > C (metro-specialists have longer transmission distances)

    Parameters:
        hospital_populations: List of population arrays for each hospital
        coordinates: Tuple of (x_coords, y_coords) spatial coordinates
        transfer_matrix: Transfer probability matrix [n_hospitals, n_hospitals]
        tau_genotypes: List of M-genotype indices (metro-specialists, Clade 2.5-like)
        beta_genotypes: List of C-genotype indices (peripheral-specialists, Clade 2.4-like)

    Returns:
        (m_spread, c_spread): Effective transmission distance (km) for M and C genotypes
    """
    x_coords, y_coords = coordinates
    n_hospitals = len(hospital_populations)

    # Calculate absolute count of M and C genotypes at each hospital
    n_M = np.zeros(n_hospitals)
    n_C = np.zeros(n_hospitals)

    for h in range(n_hospitals):
        pop = hospital_populations[h]
        n_M[h] = sum(1 for g in pop if g in tau_genotypes)
        n_C[h] = sum(1 for g in pop if g in beta_genotypes)

    # Calculate pairwise distance matrix
    distance_matrix = np.zeros((n_hospitals, n_hospitals))
    for i in range(n_hospitals):
        for j in range(n_hospitals):
            if i != j:
                dx = x_coords[i] - x_coords[j]
                dy = y_coords[i] - y_coords[j]
                distance_matrix[i, j] = np.sqrt(dx**2 + dy**2)

    # Calculate effective transmission distance for M-genotypes
    numerator_M = 0.0
    denominator_M = 0.0
    for i in range(n_hospitals):
        for j in range(n_hospitals):
            if i != j and transfer_matrix[i, j] > 0:
                weight = transfer_matrix[i, j] * n_M[i]
                numerator_M += distance_matrix[i, j] * weight
                denominator_M += weight

    m_spread = numerator_M / denominator_M if denominator_M > 1e-10 else 0.0

    # Calculate effective transmission distance for C-genotypes
    numerator_C = 0.0
    denominator_C = 0.0
    for i in range(n_hospitals):
        for j in range(n_hospitals):
            if i != j and transfer_matrix[i, j] > 0:
                weight = transfer_matrix[i, j] * n_C[i]
                numerator_C += distance_matrix[i, j] * weight
                denominator_C += weight

    c_spread = numerator_C / denominator_C if denominator_C > 1e-10 else 0.0

    return m_spread, c_spread
