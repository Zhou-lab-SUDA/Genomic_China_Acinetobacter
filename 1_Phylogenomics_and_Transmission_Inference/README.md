# Phylogenomics and transmission inference

This directory contains the dated phylogeny, empirical city-level transmission network, spatial-dissemination analyses and sensitivity analyses. `sample_metadata.xlsx` is the shared genome metadata table used by the temporal-signal, SNP-distance, network and clade-prevalence analyses.

## Primary files

- `treetime.annotated.nexus`: dated phylogeny with reconstructed geographic states.
- `China_Transmission_Normalization.tsv`: normalized empirical city-to-city transmission table.
- `PairwiseChinaCities.distance`: pairwise city distances used for spatial summaries.
- `Network.py`: community structure and hierarchical-city summaries.
- `Speed.py`: phylogeny-based dissemination distance and velocity.
- `sample_metadata.xlsx`: shared sample date, location and lineage metadata.

## Validation and sensitivity analyses

| Directory | Question addressed | Principal outputs |
| --- | --- | --- |
| [`temporal_signal/`](temporal_signal/) | Is sampling time associated with root-to-tip distance within each lineage, and does it exceed a permuted-date null? | Lineage regressions, all 10 x 10 randomisation replicates and null summaries |
| [`temporal_signal/sampling_sensitivity/`](temporal_signal/sampling_sensitivity/) | Does temporal signal persist under alternative sampling schemes? | Replicate and summary tables for five schemes |
| [`recombination/`](recombination/) | Where is recombination concentrated and what sequence region supports phylogenetic reconstruction? | Three-software profiles, block coordinates and masked SNP counts |
| [`snp_distance_validation/`](snp_distance_validation/) | Are time- and geography-based conclusions supported by recombination-masked pairwise SNP distance? | Pairwise, annual and binned summaries |
| [`network_sensitivity/`](network_sensitivity/) | Does the inferred transmission-network organization persist under geographic and temporal resampling? | Replicate networks, NMI, ARI and network-metric summaries |
| [`clade_prevalence_sensitivity/`](clade_prevalence_sensitivity/) | Does the temporal lineage-composition pattern persist after metadata resampling? | Annual and overall replicate proportions and intervals |

Each subdirectory contains a README describing its design and the provenance of its tables. Code and data are separated within the subdirectory.

## Network definitions

The empirical network is reconstructed from geographic-state changes along the dated phylogeny. Nodes are sampled cities. Directed edges are inferred intercity transitions, and edge weights are the numbers of inferred transmission events. It is therefore a genomic transmission network, not an observed hospital-transfer network.

City-level healthcare accessibility and medical-mobility indicators are external covariates; they do not define the network edges. The simulation in [`../3_AB_network/`](../3_AB_network/) uses a separate synthetic network and should not be interpreted as a reconstruction of this empirical network.

## Phylogenetic scope

Temporal signal is assessed separately for the combined collection and ESL2.3, ESL2.4 and ESL2.5. ESL2.4 and ESL2.5 support the lineage-specific temporal inference. Time-dependent estimates for ESL2.3 are not interpreted because its root-to-tip analysis showed no detectable temporal signal.

Recombinant regions are masked before phylogenetic analysis. The documented low-recombination block spans NC_017847.1 positions 1,215,000-2,930,999 and retains 23,524 of 35,609 SNPs after masking; see [`recombination/`](recombination/) for software-specific profiles and coordinate conventions.

The two existing overview images are retained at the points where they are discussed in the [repository README](../README.md): `Figure_module_scan_connection_and_city_level.png` for network structure and `F2.png` for dissemination speed and distance. Each additional image instruction is located in the README of its own analysis directory.
