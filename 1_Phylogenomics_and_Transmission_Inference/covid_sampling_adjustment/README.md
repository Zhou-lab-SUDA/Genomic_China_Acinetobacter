# Pandemic-period dissemination and sampling adjustment

Supplementary Fig. 7 and the fitted lineage series used in Fig. 2c and Fig. 5j draw on 26 lineage-year observations from 2011–2023. The OLS model includes lineage, period, their interaction, log1p annual genomes and annual sampled cities. It uses HC3 covariance with a residual-df Student t reference. The placebo restricts the data to 2011–2017 and uses 2014 as the first pseudo-post year. Seven years provide limited precision.

Annual fitted means use each observed sampling profile. Period estimates use the analysis-wide mean sampling covariates. Fixed lineage-specific root-to-tip rates scale estimates and contrasts to km/year conditional on a 1,716,000 bp block. Relative-change Monte Carlo intervals in the archived table use the positive pre/post draws. See the [discrepancy report](../../REVISION_SYNC_DISCREPANCIES.md) for alternative interval conventions and figure labels.

The two lineage estimates fell by 36.4% (P = 0.219) and 54.7% (P = 0.016), respectively. The scaled difference in change was −16.34 km/year (95% CI −41.00 to 8.32, P = 0.182). The observational design and possible right truncation limit interpretation. The [spatial transmission simulation](../../3_AB_network/) is a separate analysis.

Run from this module with the [revision dependencies](../../requirements-revision.txt):

```sh
python code/analysis.py --out /tmp/ab-covid
python code/plot.py --results /tmp/ab-covid --figure /tmp/ab-covid/supp_fig_7.svg
python code/plot_main_panels.py --results /tmp/ab-covid --out /tmp/ab-covid
```

S9a contains observed annual km/SNP, clock-scaled observations, sampling counts and submitted fitted means. S9b–e are archived speed and contrast results. S9f contains the clock inputs.
