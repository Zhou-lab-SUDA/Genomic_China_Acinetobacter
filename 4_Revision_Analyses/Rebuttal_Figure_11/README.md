# Rebuttal Figure 11

Recombination landscape across 399 consecutive 10-kb chromosome windows. The
published supplementary workbook is preserved as
`data/source_recombination_mutation_events.xlsx` and standardized into:

- `recombination_mutation_events_10kb.tsv`: one row per 10-kb window, with
  explicit zero-based, half-open coordinates and separate RecHMM, Gubbins,
  ClonalFrameML and mutation-event fields.
- `recombination_mutation_events_10kb_long.tsv`: tidy one-row-per-window-and-
  series format.
- `recombination_block_segments.tsv`: seven contiguous segments. HRB1 occurs
  at both chromosome ends and is therefore retained as two segments.
- `recombination_mutation_block_summary.tsv`: descriptive statistics by block
  and event series.
- `source_data_quality.tsv`: completeness and range checks.

Run `python code/prepare_recombination_mutation_data.py` to rebuild the tables,
then `python code/plot_recombination_landscape.py` to reproduce the three
recombination tracks and the RecHMM block-distribution panel.

The workbook also contains mutation-event counts, but the submitted RF11 PDF
does not plot them. They are retained in the standardized tables for source
completeness and are not included in the RF11 plotting script. These counts do
not classify mutation types such as synonymous or nonsynonymous variants.

Two mutation-event cells are blank in the source workbook (10-kb window
indices 22 and 23). They remain missing; no zero-value imputation was applied.
The workbook does not provide the test specification or P values underlying
the significance stars in the submitted PDF, so the reproducible component
plot does not recreate those stars.
