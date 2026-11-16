# Historical urban-characteristic sensitivity

Supplementary Fig. 8 compares time-matched 2010–2017 urban characteristics with the contemporary Table S2 snapshot for a fixed cohort of 18 cities and 1,477 genomes. The response is fixed 2010–2017 ESL2.5 divided by all included genomes. Historical values use observed period means with contributor-year counts in S10b. The separate historical and contemporary PCAs use seven harmonized indicators, sample-SD standardization and a common outcome-independent sign convention. The two PC1s have Spearman ρ = 0.740 (permutation P = 0.00069). Historical correlations across 13 indicators are assessed with two-sided permutations and BH adjustment. The seven paired vintage correlations use a separate adjustment family per vintage.

```sh
python code/analysis.py --out /tmp/ab-historical --plot
```

S10a and S10e identify the matched cohort and PCA inputs. S10c, S10d, S10f and S10g contain archived correlations, loadings and diagnostics. Coverage and changes in statistical definitions constrain historical inference. Some contemporary and historical healthcare indicators differ in the occupations they cover. The comparison is descriptive and cannot establish why lineage replacement occurred.
