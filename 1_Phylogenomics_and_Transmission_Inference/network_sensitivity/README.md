# Transmission-network sampling sensitivity

The empirical city network was reconstructed after random, population-weighted,
year-balanced, city-balanced and combined city/year-balanced sampling. Networks
were compared on a common 52-city backbone using edge counts, city-level
transmission counts, community assignments, normalized mutual information
(NMI) and adjusted Rand index (ARI).

NMI measures similarity of the overall community structure. ARI is more
sensitive to reassignment of individual cities. A reduction in ARI with a
relatively stable NMI therefore indicates local module changes without loss of
the larger network organization.

The four degree-correlation summaries use
`data/network_file_resampling/`. The five-scheme Nexus reconstruction uses the
tables directly under `data/`. Use the first path for degree-correlation plots
and the second for Nexus-based network and community comparisons.

The five-scheme reconstruction starts from `data/treetime_annotated.nexus`,
the city-distance and date tables, and the shared `../sample_metadata.xlsx`.
Replicate-level and summary TSV files are derived outputs of the scripts in
`code/`.

<p align="left">
  <img src="network_sampling_sensitivity.png" width="900" alt="Transmission-network sensitivity under five sampling strategies">
</p>
