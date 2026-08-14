# Temporal signal

Root-to-tip distances were evaluated for the combined Chinese ESL2 collection
and separately for ESL2.3, ESL2.4 and ESL2.5. ESL2.4 and ESL2.5 contain the
lineage-specific temporal signal used for molecular dating. ESL2.3 showed no
detectable temporal signal, so time-dependent estimates for that lineage were
not interpreted.

## Date randomisation

Sampling dates were permuted among tips ten times. Within each permutation,
the root-to-tip correlation was calculated in ten independent 50% subsamples.
The median of those ten correlations represents one date permutation. The
2.5th and 97.5th empirical percentiles across the ten permutation-level
medians form the null interval shown in the summary table. The observed point
estimate is the full-dataset correlation; its sampling interval is estimated
separately from ten 50% subsamples of the unpermuted data.

`data/date_randomisation_replicates.tsv` contains all 10 x 10 results;
`data/date_randomisation_randomisation_summaries.tsv` contains the
permutation-level medians and intervals. Run `code/date_randomisation.py` to
recalculate the tables.

`sampling_sensitivity/` evaluates the same temporal-signal statistic under
random, population-weighted, year-balanced, city-balanced and combined
city/year-balanced sampling.
