# SNP-distance validation

These analyses compare divergence time with pairwise SNP distance calculated
from the recombination-masked alignment. `code/snp_time_validation.py` produces
the pair-level time comparison and year calibration,
`code/hybrid_snp_geographic_spread.py` summarizes geographic-distance bins and
thresholds, and `code/period_km_per_snp.py` calculates annual and
period-specific kilometres per SNP.

Calendar time is used for the dissemination-velocity trajectory and direct
comparison between epidemiological periods. The SNP-based tables provide a
parallel distance scale for the same spatial comparisons.

The upstream scripts require the full SNP matrix, sample index, dated tree,
sampling dates, city-distance matrix and genome metadata. Figure-level derived
tables are included under `data/`.

> **Figure: SNP-distance validation**
> Add `snp_distance_validation.png` here, summarizing divergence time versus masked SNP distance, geographic spread versus SNP distance and the period-specific km-per-SNP comparison.
