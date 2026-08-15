# Antimicrobial susceptibility

Isolates are classified as susceptible, intermediate or resistant using the
clinical interpretations reported by the hospital microbiology laboratory and
CLSI M100 (31st edition, 2021) where Acinetobacter breakpoints are available.
Resistance proportions are compared between ESL2.4 and ESL2.5 using two-sided
Fisher exact tests with Benjamini-Hochberg correction across antimicrobial
agents.

The primary comparison uses clinical S/I/R categories. For each antimicrobial,
the script constructs a resistant-versus-non-resistant 2 x 2 table, applies a
two-sided Fisher exact test and adjusts the resulting P values across drugs.

`data/SIR_raw_table.tsv` is the source table. `SIR_cleaned_long.tsv` contains
deduplicated isolate-level calls; the remaining TSV files are drug-level
percentages and statistical comparisons derived by the scripts in `code/`.

> **Figure: antimicrobial susceptibility**
> Add `antimicrobial_susceptibility.png` here, showing ESL2.4 and ESL2.5 S/I/R proportions by antimicrobial and the corresponding statistical comparisons.
