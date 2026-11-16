# Phylogenomics and transmission inference

This directory contains the dated phylogeny, empirical city-level transmission network, spatial-dissemination analyses and sensitivity analyses. `sample_metadata.xlsx` is the shared genome metadata table used by the temporal-signal, SNP-distance, network and clade-prevalence analyses.

## Primary files

- `treetime.annotated.nexus`: dated phylogeny with reconstructed geographic states.
- `China_Transmission_Normalization.tsv`: normalized empirical city-to-city transmission table.
- `PairwiseChinaCities.distance`: pairwise city distances used for spatial summaries.
- `Network.py`: community structure and hierarchical-city summaries.
- `Speed.py`: phylogeny-based dissemination distance and velocity.
- `sample_metadata.xlsx`: shared sample date, location and lineage metadata.

## Analysis modules

| Directory | Analysis | Principal outputs |
| --- | --- | --- |
| [`temporal_signal/`](temporal_signal/) | Lineage-specific root-to-tip regression and date randomisation | Lineage regressions, all 10 x 10 randomisation replicates and null summaries |
| [`temporal_signal/sampling_sensitivity/`](temporal_signal/sampling_sensitivity/) | Temporal-signal estimation under five sampling schemes | Replicate and summary tables for the combined dataset and each lineage |
| [`recombination/`](recombination/) | Chromosome-wide recombination profiles and phylogenetic-block definition | RecHMM, Gubbins and ClonalFrameML profiles, block coordinates and masked SNP counts |
| [`snp_distance_validation/`](snp_distance_validation/) | Pairwise SNP/time and SNP/geography calibration | Pair-level, annual and geographic-bin summaries |
| [`network_sensitivity/`](network_sensitivity/) | Network reconstruction under geographic and temporal resampling | Replicate networks, NMI, ARI and network-metric summaries |
| [`clade_prevalence_sensitivity/`](clade_prevalence_sensitivity/) | Annual lineage composition under four resampling schemes | Annual and overall replicate proportions and intervals |
| [`covid_sampling_adjustment/`](covid_sampling_adjustment/) | Lineage-year sampling adjustment and placebo | Fixed-clock annual fits and contrasts, Supplementary Fig. 7 |

Each subdirectory contains a README describing its design and the provenance of its tables. Code and data are separated within the subdirectory.

## Network definitions

The empirical network is reconstructed from geographic-state changes along the dated phylogeny. Nodes represent sampled cities, directed edges represent inferred intercity transitions, and edge weights record the number of transitions normalized for downstream comparison.

City-level healthcare accessibility and medical-mobility indicators enter the urban association analyses as external covariates. The simulation in [`../3_AB_network/`](../3_AB_network/) constructs its own synthetic network from the parameters in `config.py`.

## Phylogenetic scope

Temporal signal is assessed separately for the combined collection and ESL2.3, ESL2.4 and ESL2.5. The lineage-specific correlations are R = 0.33 for ESL2.4 and R = 0.36 for ESL2.5 (permutation P = 0.001 for both). ESL2.3 yields R = 0.01 and P = 0.974 and is excluded from molecular-dating summaries.

Recombinant regions are masked before phylogenetic analysis. The documented low-recombination block spans NC_017847.1 positions 1,215,000-2,930,999 and retains 23,524 of 35,609 SNPs after masking; see [`recombination/`](recombination/) for software-specific profiles and coordinate conventions.
