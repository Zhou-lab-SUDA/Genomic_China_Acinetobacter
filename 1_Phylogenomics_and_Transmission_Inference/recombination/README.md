# Recombination profile

The chromosome is summarized in 399 consecutive 10-kb windows. The standardized
tables contain separate RecHMM, Gubbins, ClonalFrameML and mutation-event
series, using zero-based half-open coordinates. Recombination is heterogeneous
across the chromosome. The phylogenetic SNP matrix is produced by masking the
intervals identified in these profiles.

The low-recombination block used for phylogenetic reconstruction corresponds to
NC_017847.1 positions 1,215,000-2,930,999. Recombination masking retained
23,524 of 35,609 SNPs in this block. The repository provides the block
coordinates and downstream event tables; `LRB.fasta` is the upstream sequence
input used to build the phylogenetic SNP matrix.

The source workbook contains blank mutation-event cells for 10-kb window
indices 22 and 23. The standardized tables encode both cells as `NA`, preserving
the distinction between an unreported value and a measured count of zero.

`data/source_recombination_mutation_events.xlsx` is the workbook supplied as
`41467_2025_58106_MOESM7_ESM.xlsx`; the TSV files are standardized derivatives
created by `code/prepare_recombination_mutation_data.py`.

> **Figure: recombination landscape**
> Add `recombination_landscape.png` here, showing the RecHMM, Gubbins and ClonalFrameML profiles, mutation events and the low-recombination block.
