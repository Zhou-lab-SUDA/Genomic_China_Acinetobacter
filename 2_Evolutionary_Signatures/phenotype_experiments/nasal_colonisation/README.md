# Mouse nasal colonisation

The source table contains longitudinal nasal bacterial loads for 15 mice per
lineage at days 1, 2, 3, 5, 7 and 14. Three representative isolates were used
for each lineage. Bacterial loads are recorded as log10 CFU.

The mouse is the independent unit. `code/analyse_nasal_colonisation.py`
summarizes each lineage and time point and performs two-sided Mann-Whitney U
tests between lineages at each day, followed by Benjamini-Hochberg adjustment
across the six time-point comparisons. The reported statistics are therefore
day-specific lineage comparisons; individual mouse trajectories remain
available in the long-format table.

The long-format input was extracted from worksheet `S7 Colonisation` in
`Supplementary_Tablesv3.xlsx`. Entries recorded as `NA` remain missing and are
excluded from the corresponding time-point summary and test. All statistics use
the observed bacterial-load values.

> **Figure: mouse nasal colonisation**
> Add `nasal_colonisation.png` here, showing the longitudinal log10 CFU trajectories or time-point summaries for ESL2.4 and ESL2.5.
