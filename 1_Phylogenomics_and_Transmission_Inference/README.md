# Phylogenomics and transmission inference

This directory contains the dated phylogeny, city-level transmission network,
spatial-dissemination analyses and sensitivity analyses.

`sample_metadata.xlsx` is the shared genome metadata table used by the
temporal-signal, SNP-distance, network and clade-prevalence analyses. Keeping a
single copy prevents the same sample annotations from diverging among scripts.

## Network definitions

The empirical network is reconstructed from geographic state changes along the
dated phylogeny. Nodes are sampled cities. Directed edges are inferred
intercity transitions, and edge weights are the numbers of inferred
transmission events. The network is therefore a genomic transmission network;
it is not an observed hospital-transfer network.

City-level healthcare accessibility and medical-mobility indicators are used
as external covariates in the urban analyses. They do not define the empirical
network edges. The simulation in `../3_AB_network` uses a separate synthetic
network and should not be interpreted as a reconstruction of the empirical
network.

## Contents

- `Network.py`: community structure of the empirical city network.
- `Speed.py`: phylogeny-based dissemination distance and velocity.
- `temporal_signal/`: lineage-specific root-to-tip analyses, date
  randomisation and sampling sensitivity.
- `snp_distance_validation/`: comparison of molecular dates with
  recombination-masked pairwise SNP distances.
- `network_sensitivity/`: network reconstruction under alternative sampling
  schemes.
- `recombination/`: chromosome-wide recombination profiles and block
  definitions.
- `clade_prevalence_sensitivity/`: annual clade proportions under metadata
  resampling.
