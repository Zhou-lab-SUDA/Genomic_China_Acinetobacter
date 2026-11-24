# Fractional-logit sensitivity analysis

Supplementary Fig. 3g uses the same 47 complete-case cities in five fractional-logit models. Each city contributes one equally weighted proportion. Effects are odds ratios per one sample SD of network degree. The two sampling specifications use log1p count or depth categories. The fuller models include log GDP per capita, log population and urbanisation rate. Wald intervals and two-sided P values use HC3 covariance. The script also writes leave-one-city-out and collinearity diagnostics.

```sh
python code/analysis.py --out /tmp/ab-degree-models --plot
```

The [archived results](data/S2e_Adjusted_degree_associations.tsv) contain the five comparisons and additional sensitivity fits. The [city input](../data/cities.tsv) defines the fixed cohort.
