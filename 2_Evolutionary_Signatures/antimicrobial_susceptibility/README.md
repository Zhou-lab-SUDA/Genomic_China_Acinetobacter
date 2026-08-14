# Antimicrobial susceptibility

Isolates are classified as susceptible, intermediate or resistant using the
clinical interpretations reported by the hospital microbiology laboratory and
CLSI M100 (31st edition, 2021) where Acinetobacter breakpoints are available.
Resistance proportions are compared between ESL2.4 and ESL2.5 using two-sided
Fisher exact tests with Benjamini-Hochberg correction across antimicrobial
agents.

The categorical analysis is used because a twofold MIC difference does not by
itself imply a clinically different susceptibility category.

`data/SIR_raw_table.tsv` is the source table. `SIR_cleaned_long.tsv` contains
deduplicated isolate-level calls; the remaining TSV files are drug-level
percentages and statistical comparisons derived by the scripts in `code/`.
