# Clinical specimen ascertainment

The aggregate tables support Supplementary Fig. 3d–f. Denominators for the invasive-specimen comparison include records assigned to either Invasive or Non-invasive in Supplementary Table 1. Uncertain and unknown labels are reported in the coverage table. Bloodstream restriction uses explicit blood specimen labels. Specimen type and documented clinical outcomes are separate concepts.

The invasive proportions were 13/59, 17/158 and 294/1,805. Bloodstream-only composition is descriptive and uses ESL2.5 divided by the combined ESL2.4 and ESL2.5 count. City-level bloodstream plots use the [archived connectivity snapshot](data/city_degree_snapshot.tsv) associated with the supplied figure analysis. The [network sampling module](../network_sampling_bias/) uses the Supplementary Table 2 degree field.

```sh
python code/plot.py --figure /tmp/supp_fig_3def.svg
python code/analysis.py --workbook /path/to/Supplementary_Tables.xlsx --city-table data/city_degree_snapshot.tsv --output-dir /tmp/ab-specimens
```

The statistical rerun requires the manuscript workbook with its S1 Metadata sheet. The module distributes city and group aggregates. Isolate-level contingency tests are descriptive given within-city dependence.
