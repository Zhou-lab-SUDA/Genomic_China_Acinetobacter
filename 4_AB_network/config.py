#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global Configuration for Hospital Pathogen Transmission Simulation
"""

import numpy as np

# ==================== Genotype Transmission Parameters ====================
# τ (Tau) values: Outpatient colonization capacity (0=low, 1=high)
# Represents strain's ability to colonize mobile outpatients
# Higher τ → enhanced dissemination via ambulatory patients in high-O/I hospitals
TAU_VALUES = np.array([0.5, 0.3, 0.1, 0.7, 0.8, 0.85, 0.9, 0.95])

# β (Beta) values: Inpatient infection capacity (0=low, 1=high)
# Represents strain's ability to cause severe infections requiring hospitalization
# Higher β → enhanced persistence in immobilized inpatients in low-O/I hospitals
# Note: τ and β are complementary (β = 1 - τ) reflecting transmission mode trade-offs
BETA_VALUES = np.array([0.5, 0.7, 0.9, 0.3, 0.2, 0.15, 0.1, 0.05])

# ==================== Simulation Default Parameters ====================
DEFAULT_N_HOSPITALS = 100
DEFAULT_N_METROS = 35
DEFAULT_GENERATIONS = 1000
DEFAULT_POPULATION = 1000
DEFAULT_N_GENOTYPES = 8
DEFAULT_SEED = 42

# ==================== Migration/Transmission Parameters ====================
# These parameters reflect differential patient mobility driven by
# outpatient/inpatient (O/I) ratio differences between city types.
#
# === BIOLOGICAL RATIONALE ===
#
# O/I Ratio Definition:
#   O/I = (outpatient visits) / (inpatient admissions)
#
# High O/I cities (mega-cities):高O/I城市（门诊为主）→ 病人可移动 → 高传播
#   - Outpatient-dominated healthcare (门诊为主)
#   - Patients carry A. baumannii as upper respiratory colonizers
#   - Remain mobile: outpatient visits, inter-city travel, return home
#   - Serve as transmission vectors across cities
#   - Corresponds to Clade 2.5's colonization capacity (high nasal persistence)
#
# Low O/I cities (peripheral cities):低O/I城市（住院为主）→ 病人固定 → 低传播
#   - Inpatient-dominated healthcare (住院为主)
#   - Patients present with acute infections (pneumonia, bacteremia)
#   - Immobilized during hospitalization
#   - Limited inter-city transmission
#   - Corresponds to Clade 2.4's virulence profile (high acute lethality)
#
# === EMPIRICAL BASIS ===
#
# National healthcare statistics (China, 2015-2023):
#   Mega-cities (Beijing, Shanghai, Guangzhou)
#   Peripheral cities (地级市)
#   Ratio of ratios: ~2.0-3.3x (median 2.5x)
#
# === MODEL IMPLEMENTATION ===
#
# We capture O/I → mobility relationship through differential migration rates:
#
#   High O/I (mega-cities)   → High patient mobility → metro_migration = 0.08
#   Low O/I (peripheral)     → Low patient mobility  → periph_migration = 0.02
#
#   Mobility ratio μ = metro_migration / periph_migration = 4.0x
#
# The 4.0x mobility ratio reflects combined effects of:
#   1. Direct O/I difference (~2.5x from statistics)
#   2. Non-linear amplification: mobile colonizers have disproportionate
#      transmission advantage in hub-connected networks
#
# This parametrization is conservative. Empirical patient flow data
# suggests mega-cities may exhibit even higher relative mobility
# (Ref: Zhao et al. 2025, Nat Cities; Li et al. 2021, BMC Health Serv Res)
#
# === UNITS AND INTERPRETATION ===
#
# Migration rate = fraction of population migrating per generation
#   - 1 generation ≈ 1 transmission cycle (model time unit)
#   - Can be interpreted as annual inter-city patient flow if generation = 1 year
#
DEFAULT_METRO_MIGRATION = 0.07      # High-mobility cities (outpatient-dominated)
DEFAULT_PERIPH_MIGRATION = 0.02     # Low-mobility cities (inpatient-dominated)
DEFAULT_SELECTION_STRENGTH = 0.1

# ==================== Spatial Parameters ====================
DEFAULT_SPATIAL_SCALE = 300.0  # km
DEFAULT_DECAY_LENGTH = 30  # None = no spatial decay

# ==================== Transfer Network Parameters ====================
# Transfer direction preferences (multipliers for base transfer probability)
# These parameters control the hierarchical structure of the hospital network

# Large → Large (hub-to-hub transfer): Specialist referrals between major centers
# Realistic range: 1.2-1.5 (higher values = stronger hub network effect)
DEFAULT_METRO_TO_METRO_PREFERENCE = 1

# Large → Small (downward referral): Return/recovery transfers
# Realistic range: 0.7-0.9 (lower values = stronger resistance to downward transfer)
DEFAULT_METRO_TO_PERIPH_RESISTANCE = 1

# Small → Large (upward referral): Primary referral pattern
# Realistic range: 1.2-1.5 (higher values = stronger upward bias)
DEFAULT_PERIPH_TO_METRO_PREFERENCE = 1.2

# Small → Small (lateral transfer): Rarely occurs in real hospital systems
# Realistic range: 0.2-0.5 (lower values = stronger suppression)
DEFAULT_PERIPH_TO_PERIPH_PENALTY = 0.2

# Distance decay differentiation: Large hospitals have wider catchment areas
# Realistic range: 1.3-2.0 (higher values = large hospitals accept more distant transfers)
DEFAULT_METRO_DECAY_MULTIPLIER = 1.3

# ==================== Initialization Parameters ====================
DEFAULT_INITIAL_PREVALENCE = 0.05  # 5%
DEFAULT_SEED_GENOTYPES = 'balanced'  # 'balanced', 'M_only', 'C_only'
DEFAULT_INITIAL_M_FREQ = 0.5  # For COVID scenario

# ==================== Visualization/Output Parameters ====================
FIGURE_DPI = 300
DEFAULT_OUTPUT_DIR = 'outputs'
DEFAULT_SMOOTH_WINDOW = 10

# prefix (simplified naming without fig numbers)
PLOT_FILENAMES = {
    'trajectories': 'genotype_trajectories',
    'heatmaps': 'adaptation',
    'network': 'hospital_network',
    'comparison': 'genotypes_comparison',
    'strategies': 'adaptation_interactive',
    'network_state': 'hospital_network_state',
    'spatial_spread': 'spatial_spread',
    'invasion': 'invasion_curve_combined',
}

# ==================== Output File Naming ====================
# Unified file naming across all scenarios
OUTPUT_FILES = {
    'timeline': 'timeline.csv',
    'results': 'results.csv',                    # Replaces file3
    'transfer_matrix': 'transfer_matrix.csv',    # Replaces file5
    'network_analysis': 'network_analysis.json', # Replaces file4, includes radar_metrics (file7)
    'network_state': 'network_state.csv',
    'run_config': 'run_config.json',             # NEW: Replaces file1, records all parameters
}

# COVID scenario phase-specific files
COVID_PHASE_TRANSFER_MATRICES = {
    'phase1': 'transfer_matrix_phase1.csv',  # Baseline phase
    'phase2': 'transfer_matrix_phase2.csv',  # Intervention phase
    'phase3': 'transfer_matrix_phase3.csv',  # Recovery phase
}

# Invasion scenario before/after files
INVASION_TRANSFER_MATRICES = {
    'before': 'transfer_matrix_before.csv',  # Before introduction
    'after': 'transfer_matrix_after.csv',    # After introduction (final)
}

# ==================== Scenarios ====================

# Sensitivity Analysis
SENSITIVITY_DEFAULT_REPLICATES = 10
SENSITIVITY_DEFAULT_VALUES = {
    'initial_prevalence': [0.01, 0.05, 0.1, 0.2, 0.3],
    'metro_migration': [0.03, 0.06, 0.09, 0.12, 0.15],
    'decay_length': [5, 10, 15, 20, 30, 50, 100],
}

# Invasion scenario defaults
INVASION_DEFAULT_GENERATIONS = 1000
DEFAULT_SEED_HOSPITAL_FRACTION = 1.0
DEFAULT_INTRODUCTION_TIME = 300
DEFAULT_INTRODUCTION_DOSE = 0.05
DEFAULT_INTRODUCTION_SITES = 3

# Backward-compatible aliases used by scenarios/invasion.py
INVASION_DEFAULT_INTRODUCTION_TIME = DEFAULT_INTRODUCTION_TIME
INVASION_DEFAULT_DOSE = DEFAULT_INTRODUCTION_DOSE
INVASION_DEFAULT_SITES = DEFAULT_INTRODUCTION_SITES

# COVID-19 intervention scenario defaults
COVID_DEFAULT_GENERATIONS = 1500
DEFAULT_INTERVENTION_START = 600
DEFAULT_RECOVERY_START = 1000
DEFAULT_BASELINE_DECAY = 50
DEFAULT_INTERVENTION_DECAY = 30
DEFAULT_RECOVERY_DECAY = 40
DEFAULT_METRO_CLOSURE = 0.7
DEFAULT_PERIPH_CLOSURE = 0.5

# ==================== Genotype Classification Thresholds ====================
# Genotypes with τ > METRO_THRESHOLD are considered "metro-oriented"
# Changed from 0.6 to 0.5 to include intermediate genotypes (2024-12-12)
METRO_THRESHOLD = 0.5
# Genotypes with τ < PERIPH_THRESHOLD are considered "periph-oriented"
# Changed from 0.4 to 0.5 to include intermediate genotypes (2024-12-12)
PERIPH_THRESHOLD = 0.5

# ==================== Classification Function ====================
def classify_genotypes(TAU_VALUES, exclude_neutral=True):
    """
    Classify genotypes into metro-oriented, periph-oriented, and neutral groups.

    Parameters:
        TAU_VALUES: Array of τ values (outpatient colonization capacity)
        exclude_neutral: If True, exclude neutral genotypes (0.4 <= τ <= 0.6)

    Returns:
        dict with keys: 'metro', 'periph', 'neutral'
    """
    metro_genotypes = [g for g in range(len(TAU_VALUES)) if TAU_VALUES[g] > METRO_THRESHOLD]
    periph_genotypes = [g for g in range(len(TAU_VALUES)) if TAU_VALUES[g] < PERIPH_THRESHOLD]
    neutral_genotypes = [g for g in range(len(TAU_VALUES))
                        if PERIPH_THRESHOLD <= TAU_VALUES[g] <= METRO_THRESHOLD]

    if exclude_neutral:
        return {
            'metro': metro_genotypes,
            'periph': periph_genotypes,
            'neutral': []
        }
    else:
        return {
            'metro': metro_genotypes,
            'periph': periph_genotypes,
            'neutral': neutral_genotypes
        }


def generate_random_genotype_values(n_genotypes=8, seed=None, strategy='complementary'):
    """
    Generate random τ and β values based on seed.

    IMPORTANT: Genotype 0 (G0) is ALWAYS the neutral background with τ=β=0.5
    The remaining genotypes (G1 to Gn-1) are randomly generated.

    This allows different random seeds to produce different genotype configurations,
    making each simulation unique while maintaining reproducibility.

    Parameters:
        n_genotypes: Number of genotypes (default: 8)
        seed: Random seed for reproducibility
        strategy: How to generate β values
                 'complementary' - β = 1 - τ (inverse relationship)
                 'random' - β values independent from τ
                 'correlated' - β values positively correlated with τ

    Returns:
        TAU_VALUES: Array of τ values (outpatient colonization capacity)
        BETA_VALUES: Array of β values (inpatient infection capacity)
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    # Generate τ values for genotypes 1 to n-1
    # G0 will be fixed at 0.5 (neutral background)
    TAU_VALUES_random = rng.uniform(0.05, 0.95, size=n_genotypes - 1)
    TAU_VALUES_random = np.sort(TAU_VALUES_random)  # Sort to create a gradient

    # Insert G0 = 0.5 at the beginning
    TAU_VALUES = np.insert(TAU_VALUES_random, 0, 0.5)

    # Generate β values based on strategy
    if strategy == 'complementary':
        # Classic complementary relationship: β = 1 - τ
        BETA_VALUES = 1.0 - TAU_VALUES
    elif strategy == 'random':
        # Independent random values
        BETA_VALUES_random = rng.uniform(0.05, 0.95, size=n_genotypes - 1)
        BETA_VALUES_random = np.sort(BETA_VALUES_random)[::-1]  # Sort descending
        BETA_VALUES = np.insert(BETA_VALUES_random, 0, 0.5)
    elif strategy == 'correlated':
        # Positively correlated: both increase together
        BETA_VALUES_random = TAU_VALUES_random + rng.uniform(-0.1, 0.1, size=n_genotypes - 1)
        BETA_VALUES_random = np.clip(BETA_VALUES_random, 0.05, 0.95)
        BETA_VALUES = np.insert(BETA_VALUES_random, 0, 0.5)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return TAU_VALUES, BETA_VALUES


def get_timestamp():
    """Generate timestamp string for output files"""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_mobility_ratio(metro_mig=None, periph_mig=None):
    """
    Calculate the mobility ratio between mega-cities and peripheral cities.

    This ratio reflects the differential O/I ratios between city types and
    quantifies how much more mobile patients are in mega-cities compared to
    peripheral cities.

    Biological interpretation:
        Mobility ratio = (metro patient mobility) / (peripheral patient mobility)

        This reflects:
        - O/I ratio differences (mega-cities have 2-3x higher O/I)
        - Mobility amplification (outpatients are disproportionately mobile)
        - Network effects (hubs concentrate mobile transmission vectors)

    Parameters:
        metro_mig: Metro migration rate (default: DEFAULT_METRO_MIGRATION)
        periph_mig: Peripheral migration rate (default: DEFAULT_PERIPH_MIGRATION)

    Returns:
        float: Mobility ratio (typically 3-6x for realistic scenarios)

    Examples:
        >>> get_mobility_ratio()  # Using defaults
        4.0

        >>> get_mobility_ratio(0.09, 0.02)  # Higher metro mobility
        4.5

        >>> get_mobility_ratio(0.06, 0.03)  # Reduced disparity
        2.0
    """
    if metro_mig is None:
        metro_mig = DEFAULT_METRO_MIGRATION
    if periph_mig is None:
        periph_mig = DEFAULT_PERIPH_MIGRATION

    if periph_mig == 0:
        raise ValueError("Peripheral migration cannot be zero")

    return metro_mig / periph_mig


def validate_mobility_ratio(metro_mig, periph_mig, min_ratio=2.0, max_ratio=8.0):
    """
    Validate that mobility ratio falls within biologically plausible range.

    Based on empirical O/I ratio data, realistic mobility ratios should be
    between 2x (minimal differentiation) and 8x (extreme differentiation).

    Parameters:
        metro_mig: Metro migration rate
        periph_mig: Peripheral migration rate
        min_ratio: Minimum plausible ratio (default: 2.0)
        max_ratio: Maximum plausible ratio (default: 8.0)

    Returns:
        bool: True if ratio is within range

    Raises:
        Warning if ratio is outside empirical range
    """
    ratio = get_mobility_ratio(metro_mig, periph_mig)

    if ratio < min_ratio:
        import warnings
        warnings.warn(
            f"Mobility ratio ({ratio:.2f}x) is below empirical minimum ({min_ratio}x). "
            f"This may lead to insufficient niche differentiation for stable coexistence.",
            UserWarning
        )
        return False
    elif ratio > max_ratio:
        import warnings
        warnings.warn(
            f"Mobility ratio ({ratio:.2f}x) exceeds empirical maximum ({max_ratio}x). "
            f"This may be unrealistic for real-world healthcare systems.",
            UserWarning
        )
        return False

    return True
