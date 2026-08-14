# Mouse nasal colonisation

The source table contains longitudinal nasal bacterial loads for 15 mice per
lineage at days 1, 2, 3, 5, 7 and 14. Three representative isolates were used
for each lineage. Bacterial loads are recorded as log10 CFU.

The mouse is the independent unit. `code/analyse_nasal_colonisation.py`
summarizes each lineage and time point and performs two-sided Mann-Whitney U
tests between lineages at each day, followed by Benjamini-Hochberg adjustment
across the six time-point comparisons. These per-day tests do not model the
within-mouse longitudinal correlation and should be interpreted as time-point-
specific comparisons.

The long-format input was extracted from worksheet `S7 Colonisation` in
`Supplementary_Tablesv3.xlsx`. Entries recorded as `NA` remain missing and are
excluded from the corresponding time-point summary and test; no bacterial-load
values are imputed.
