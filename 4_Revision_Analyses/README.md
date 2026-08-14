# Rebuttal figure source-data package

This directory contains frozen source tables, analysis/plotting scripts,
provenance notes and explicit placeholders for final figures. Figure files are
intentionally omitted so that the final submitted versions can be added
manually.

## Directory convention

- `Rebuttal_Figure_N/FIGURE_PLACEHOLDER.md`: location and description of the
  final figure(s) to be added manually.
- `Rebuttal_Figure_N/data/`: tabular source data and frozen intermediate results.
- `Rebuttal_Figure_N/code/`: analysis or plotting scripts relevant to the figure.
- `Rebuttal_Figure_N/README.md`: figure-specific methods, inputs and caveats.
- `DATA_PROVENANCE.tsv`: mapping from packaged files to their original source and derivation.
- `TABLE_VALIDATION.tsv`: machine-generated parse and schema checks for packaged tables.

All packaged scripts use relative paths where the complete inputs are present.
Scripts retained only as upstream analytical provenance are identified in the
figure README. Temporary development comments and machine-specific runtime
paths are not included. Scientific provenance, random seeds and analytical
limitations are retained.

## Python environment

The scripts require Python 3.10 or later. Common dependencies are `pandas`,
`numpy`, `matplotlib` and `openpyxl`. Some SVG-producing scripts use only the
Python standard library.

Run a figure script from any working directory, for example:

```powershell
python "Rebuttal_Figure_1/code/plot_sir_resistance_rate.py"
```

Reproduced outputs have names ending in `_reproduced`. Generated figure files
should be reviewed before being added to the repository.

## Incomplete source packages

- RF2 survival analysis: the fixed survival panel and its data are not included
  in this snapshot because documentation of the analysis-cohort definition is
  pending. The date-randomisation analysis assigned to RF2 is complete.
- RF11 now includes the published event-count workbook and standardized
  tables. The workbook does not include the statistical-test specification or
  P values used for the significance stars in the submitted PDF.
- RF13: the directory was empty when this package was prepared.

See `PLACEHOLDERS.md` for repository-level missing inputs and accession fields.
