# Network connectivity and genomic sampling depth

One row in [cities.tsv](data/cities.tsv) represents one city from the original prevalence snapshot joined to Supplementary Table 2 degree. Sampling depth is the combined ESL2.4 and ESL2.5 count. The prevalence columns preserve the all-genome denominator from Fig. 3. The stratified Spearman analyses use 48 cities with a degree value. The paired prevalence comparisons use 49 cities and city-equal weights. Depth groups are <20, 20–49 and ≥50. Spearman P values are two-sided and CI limits use a city bootstrap. Paired sign-flip P values are BH-adjusted across the three groups.

```sh
python code/analysis.py --out /tmp/ab-depth --plot
```

Compare the regenerated tables with [correlations](data/S2f_Sampling_correlations.tsv) and [paired group summaries](data/mean_lineage_prevalence_results.tsv). Supplementary Fig. 3g uses the separate [fractional-logit analysis](adjusted_models/).

The between-depth tests compare the city-equal means separately within each lineage with 500,000 permutations and BH correction across six comparisons. They answer a different question from the paired lineage comparison in each depth group.
