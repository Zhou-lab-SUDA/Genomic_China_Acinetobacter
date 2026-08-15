# Gene-expression qPCR

The input contains technical triplicate Ct measurements for 16S rRNA and four
targets in three ESL2.4 and three ESL2.5 isolates. The target written as
`pliA` in the source workbook is retained in `target_as_reported` and mapped to
`pilA` in `target_standardized`.

Technical replicates are averaged within each isolate. Expression is summarized
as delta Ct relative to the isolate's 16S rRNA value and as 2^-delta Ct. The
isolate is the independent unit. Two-sided Mann-Whitney U tests compare the
three isolates per lineage for each target, with Benjamini-Hochberg adjustment
across the four targets.

The input table was extracted from worksheet `qPCR` in
`Source_Data_Tablesv2.xlsx`; all three technical Ct values are retained.

> **Figure: gene-expression qPCR**
> Add `gene_expression_qpcr.png` here, showing isolate-level normalized expression for the four targets.
