# A549-cell adhesion

Six isolates were tested: three ESL2.4 isolates and three ESL2.5 isolates. Each
isolate was measured at MOI 10 and MOI 20 in three independent experiments.
Rows in `data/selected_strains_MOI10_MOI20_logCFU.tsv` are independent
experiment-level measurements; technical plating counts are not treated as
independent biological replicates.

This assay measures adhesion to A549 epithelial cells. It is not a direct
measure of nasal colonisation, transmission or virulence.

`data/source_adhesion_measurements.xlsx` contains the complete source plate
counts for five isolates per lineage; `data/adhesion_measurements.tsv` is its
tidy representation. The focused comparison uses three isolates per lineage
and three independent experiments per isolate at each MOI, retained in
`data/selected_strains_MOI10_MOI20_adhesion.tsv`.

Statistical comparisons use the mean of each of the six isolates, rather than
treating replicate plate counts as independent observations.
`code/analyse_adhesion.py` writes isolate summaries, lineage summaries and
two-sided exact Mann-Whitney U tests for the focused comparison.

> **Figure placeholder — A549-cell adhesion**
> Add `a549_adhesion.png` here, showing MOI 10 and MOI 20 adhesion for the focused isolates and using isolate means as the comparison unit.
