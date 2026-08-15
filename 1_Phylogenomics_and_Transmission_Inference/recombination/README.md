# Recombination profile

The chromosome is summarized in 399 consecutive 10-kb windows. The standardized
tables contain separate RecHMM, Gubbins, ClonalFrameML and mutation-event
series, using zero-based half-open coordinates. Recombination is heterogeneous
across the chromosome; the phylogenetic analyses therefore use
recombination-masked SNPs rather than describing the alignment as
recombination-free.

The low-recombination block used for phylogenetic reconstruction corresponds to
NC_017847.1 positions 1,215,000-2,930,999. Recombination masking retained
23,524 of 35,609 SNPs in this block. The block sequence cited as `LRB.fasta`
is not distributed in this repository.

The source workbook contains two blank mutation-event cells, for 10-kb window
indices 22 and 23. They remain missing rather than being converted to zero. The
workbook does not specify the statistical test or P values used for significance
stars in the associated figure.

`data/source_recombination_mutation_events.xlsx` is the workbook supplied as
`41467_2025_58106_MOESM7_ESM.xlsx`; the TSV files are standardized derivatives
created by `code/prepare_recombination_mutation_data.py`.

> **Figure placeholder — recombination landscape**
> Add `recombination_landscape.png` here, showing the RecHMM, Gubbins and ClonalFrameML profiles, mutation events and the low-recombination block.
