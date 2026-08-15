# City-level associations

The univariate analysis tests Spearman associations between ESL2.5 relative
abundance and 22 city variables. Nominal permutation P values and Storey FDR
q-values are retained in the result tables. The smallest Storey q-value is
0.0768, for air pollution.

The FDR adjustment applies to 22 separate marginal tests. RDA and SEM evaluate
joint structure among predefined groups of standardized urban, healthcare and
mobility variables and are reported as multivariate analyses in the manuscript.

The permutation results in `data/city_ESL25_spearman_permutation.tsv` are the
input to the multiple-testing scripts. The other TSV files retain the plotted
values and Storey q-values.

> **Figure: city-variable multiple testing**
> Add `city_association_multiple_testing.png` here, displaying nominal permutation P values and Storey FDR q-values for all 22 city variables.
