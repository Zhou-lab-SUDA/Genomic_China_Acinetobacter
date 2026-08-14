# Capsular-polysaccharide quantification

Capsular material was extracted from cultures normalized to approximately
10^9 CFU using 1% Zwittergent 3-14 in 100 mM citric acid and ethanol
precipitation. Uronic acid was measured after sodium-tetraborate/sulfuric-acid
and carbazole reactions at 530 nm. K-type-matched material was used for the
standard curves.

`data/capsular_polysaccharide_measurements.tsv` contains two measurements for
each of 15 isolates. The isolate mean is the independent unit. The calculation
script reports lineage and KL-type summaries and two-sided Mann-Whitney U
comparisons without treating the two assay measurements as independent
isolates.

The input table is the `UA` worksheet from `Source_Data_Tablesv2.xlsx`, with
column names standardized and the two measurements retained separately.
