#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract genotype_timeline.csv from existing network_analysis.json

This script is useful for generating genotype_timeline.csv from old simulation results
that don't have this file yet but have the data in network_analysis.json.

Usage:
    python extract_genotype_timeline.py <path_to_network_analysis.json>

Example:
    python extract_genotype_timeline.py outputs/run_20251212_133145_basic_h100_m35_g1000_s42/data/network_analysis.json
"""

import json
import sys
import os
import numpy as np
import pandas as pd


def extract_genotype_timeline(json_path, output_path=None):
    """
    Extract genotype timeline from network_analysis.json and save to CSV

    Parameters:
        json_path: Path to network_analysis.json
        output_path: Path to save genotype_timeline.csv (default: same directory as json)
    """
    # Read JSON file
    print(f"Reading: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Check if time_series exists
    if 'time_series' not in data:
        print("ERROR: time_series not found in JSON file")
        return False

    time_series = data['time_series']

    # Get arrays
    overall_freqs = np.array(time_series['overall_freqs'])
    metro_freqs = np.array(time_series['metro_freqs'])
    periph_freqs = np.array(time_series['periph_freqs'])

    n_generations = len(overall_freqs)
    n_genotypes = overall_freqs.shape[1]

    print(f"Found time_series data:")
    print(f"  Generations: {n_generations}")
    print(f"  Genotypes: {n_genotypes}")

    # Build records (sample every 10 generations to match timeline.csv)
    records = []
    for gen_idx in range(n_generations):
        generation = gen_idx + 1

        # Only export generation 1, 10, 20, 30, ... to match timeline.csv
        if generation == 1 or generation % 10 == 0:
            record = {'generation': generation}

            # Add metro frequencies
            for g in range(n_genotypes):
                record[f'metro_g{g}'] = float(metro_freqs[gen_idx][g])

            # Add peripheral frequencies
            for g in range(n_genotypes):
                record[f'periph_g{g}'] = float(periph_freqs[gen_idx][g])

            # Add overall frequencies
            for g in range(n_genotypes):
                record[f'overall_g{g}'] = float(overall_freqs[gen_idx][g])

            records.append(record)

    # Create DataFrame
    df = pd.DataFrame(records)

    # Determine output path
    if output_path is None:
        json_dir = os.path.dirname(json_path)
        output_path = os.path.join(json_dir, 'genotype_timeline.csv')

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\nGenotype timeline CSV saved: {output_path}")
    print(f"  Total records: {len(records)}")
    print(f"  Columns: {len(df.columns)}")

    return True


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    json_path = sys.argv[1]

    if not os.path.exists(json_path):
        print(f"ERROR: File not found: {json_path}")
        sys.exit(1)

    output_path = None
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]

    success = extract_genotype_timeline(json_path, output_path)

    if success:
        print("\n✓ Success!")
    else:
        print("\n✗ Failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
