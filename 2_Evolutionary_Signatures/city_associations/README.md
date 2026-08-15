# City-level associations

The univariate analysis tests Spearman associations between ESL2.5 relative
abundance and 22 city variables. Nominal permutation P values and Storey FDR
q-values are retained in the result tables. No individual variable remains
significant at q < 0.05 after this correction.

The univariate FDR analysis and the multivariate RDA/SEM analyses answer
different questions. FDR adjustment concerns 22 separate marginal tests. RDA
and SEM evaluate joint structure among predefined groups of standardized urban,
healthcare and mobility variables; they do not make the marginal associations
individually significant. The RDA/SEM scripts and final model-input table are
not part of this module.

The permutation results in `data/city_ESL25_spearman_permutation.tsv` are the
input to the multiple-testing scripts. The other TSV files retain the plotted
values and Storey q-values.

> **Figure placeholder — city-variable multiple testing**
> Add `city_association_multiple_testing.png` here, displaying nominal permutation P values and Storey FDR q-values for all 22 city variables.
