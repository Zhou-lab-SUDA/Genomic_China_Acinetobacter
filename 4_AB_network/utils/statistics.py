#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistics Module

From model_main_d.py：
- FST and diversity statistics calculation
- Shannon diversity
- Simpson diversity
- Within-group FST (metro/peripheral)
"""

import numpy as np


def calculate_fst_statistics(genotype_freqs, hospital_types, TAU_VALUES, BETA_VALUES):
    """
    Calculate FST and diversity statistics
    
    Parameters:
        genotype_freqs: ndarray, shape (n_hospitals, n_genotypes)
        hospital_types: list of str ('metro' or 'peripheral')
        TAU_VALUES: ndarray, M values for each genotype
        BETA_VALUES: ndarray, C values for each genotype
    
    Returns:
        dict with keys:
            - fst_per_genotype: FST for each genotype
            - fst_mean_weighted: Weighted average FST
            - shannon_diversity: Shannon diversity per hospital
            - simpson_diversity: Simpson diversity per hospital
            - dominant_genotype: ID of dominant genotype
            - metro_specialist_freq: Frequency of metro specialists
            - periph_specialist_freq: Frequency of peripheral specialists
            - metro_within_fst_simple: Simple average of metro within-FST
            - metro_within_fst_weighted: Weighted average of metro within-FST
            - periph_within_fst_simple: Simple average of periph within-FST
            - periph_within_fst_weighted: Weighted average of periph within-FST
    """
    N_GENOTYPES = len(TAU_VALUES)
    N_HOSPITALS = len(hospital_types)
    
    # Calculate overall frequencies
    total_freqs = np.mean(genotype_freqs, axis=0)
    dominant_genotype = np.argmax(total_freqs)
    
    # Get indices
    metro_indices = [i for i, t in enumerate(hospital_types) if t == 'metro']
    periph_indices = [i for i, t in enumerate(hospital_types) if t == 'peripheral']
    
    metro_freqs = np.mean(genotype_freqs[metro_indices, :], axis=0) if metro_indices else np.zeros(N_GENOTYPES)
    periph_freqs = np.mean(genotype_freqs[periph_indices, :], axis=0) if periph_indices else np.zeros(N_GENOTYPES)
    
    # FST calculation - calculate FST for each genotype
    fst_per_genotype = np.zeros(N_GENOTYPES)
    
    for g in range(N_GENOTYPES):
        hospital_freqs_g = genotype_freqs[:, g]
        p_total = np.mean(hospital_freqs_g)
        p_var = np.var(hospital_freqs_g)
        
        if 0 < p_total < 1:
            fst_per_genotype[g] = p_var / (p_total * (1 - p_total))
        else:
            fst_per_genotype[g] = 0.0
    
    # Weighted average FST
    fst_mean_weighted = np.sum(fst_per_genotype * total_freqs)
    
    # Diversity statistics
    shannon_diversity = []
    simpson_diversity = []
    
    for h in range(N_HOSPITALS):
        freqs = genotype_freqs[h, :]
        freqs = freqs[freqs > 0]
        if len(freqs) > 0:
            shannon = -np.sum(freqs * np.log(freqs))
            simpson = 1 - np.sum(freqs ** 2)
        else:
            shannon = simpson = 0.0
        shannon_diversity.append(shannon)
        simpson_diversity.append(simpson)
    
    # Calculate grouped FST: metro hospitals internal vs peripheral hospitals internal
    # Metro hospitals internal FST
    metro_freqs_subset = genotype_freqs[metro_indices, :] if metro_indices else np.zeros((0, N_GENOTYPES))
    metro_within_fst = np.zeros(N_GENOTYPES)
    
    if len(metro_indices) > 0:
        for g in range(N_GENOTYPES):
            metro_g_freqs = metro_freqs_subset[:, g]
            metro_p_mean = np.mean(metro_g_freqs)
            metro_p_var = np.var(metro_g_freqs)
            if 0 < metro_p_mean < 1:
                metro_within_fst[g] = metro_p_var / (metro_p_mean * (1 - metro_p_mean))
            else:
                metro_within_fst[g] = 0.0
    
    # Peripheral hospitals internal FST
    periph_freqs_subset = genotype_freqs[periph_indices, :] if periph_indices else np.zeros((0, N_GENOTYPES))
    periph_within_fst = np.zeros(N_GENOTYPES)
    
    if len(periph_indices) > 0:
        for g in range(N_GENOTYPES):
            periph_g_freqs = periph_freqs_subset[:, g]
            periph_p_mean = np.mean(periph_g_freqs)
            periph_p_var = np.var(periph_g_freqs)
            if 0 < periph_p_mean < 1:
                periph_within_fst[g] = periph_p_var / (periph_p_mean * (1 - periph_p_mean))
            else:
                periph_within_fst[g] = 0.0
    
    # Simple and weighted average of grouped FST
    metro_within_fst_simple = np.mean(metro_within_fst)
    periph_within_fst_simple = np.mean(periph_within_fst)
    metro_within_fst_weighted = np.sum(metro_within_fst * metro_freqs)
    periph_within_fst_weighted = np.sum(periph_within_fst * periph_freqs)
    
    # Calculate specialist frequencies
    high_M_genotypes = np.where(TAU_VALUES > 0.7)[0]  # Metro specialists
    high_C_genotypes = np.where(BETA_VALUES > 0.7)[0]  # Peripheral specialists
    
    metro_specialist_freq = np.sum(total_freqs[high_M_genotypes]) if len(high_M_genotypes) > 0 else 0.0
    periph_specialist_freq = np.sum(total_freqs[high_C_genotypes]) if len(high_C_genotypes) > 0 else 0.0
    
    return {
        'fst_per_genotype': fst_per_genotype,
        'fst_mean_weighted': fst_mean_weighted,
        'shannon_diversity': shannon_diversity,
        'simpson_diversity': simpson_diversity,
        'dominant_genotype': dominant_genotype,
        'metro_specialist_freq': metro_specialist_freq,
        'periph_specialist_freq': periph_specialist_freq,
        'metro_within_fst_simple': metro_within_fst_simple,
        'metro_within_fst_weighted': metro_within_fst_weighted,
        'periph_within_fst_simple': periph_within_fst_simple,
        'periph_within_fst_weighted': periph_within_fst_weighted
    }
