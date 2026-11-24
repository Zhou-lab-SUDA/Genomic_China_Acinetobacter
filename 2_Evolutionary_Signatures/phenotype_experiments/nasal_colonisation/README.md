# Mouse nasal colonisation

The source table contains nasal bacterial loads at days 1, 2, 3, 5, 7 and 14
after inoculation. Three representative isolates were used per lineage, with
five mice planned per isolate at each time point. Nasal tissue collection
followed euthanasia, so each day comprises a separate group of mice.
Bacterial loads are recorded as log10 CFU.

The mouse is the independent unit. `code/analyse_nasal_colonisation.py`
summarizes each lineage and time point and performs two-sided Mann-Whitney U
tests between lineages at each day, followed by Benjamini-Hochberg adjustment
across the six time-point comparisons. The reported statistics are therefore
day-specific lineage comparisons. The long-format table records individual
measurements within each isolate and sampling day. Mouse identifiers must be
interpreted within each sampling day.

The long-format input was extracted from worksheet `S7 Colonisation` in
`Supplementary_Tablesv3.xlsx`. Entries recorded as `NA` remain missing and are
excluded from the corresponding time-point summary and test. All statistics use
the observed bacterial-load values.

The nasal-colonisation design and bacterial-load results are shown with nasal sIgA in [`../animal_experiments.png`](../animal_experiments.png).

## Isolate-level view

Supplementary Fig. 6a shows nasal colonisation by isolate. The input contains 180 slots and 175 measured log10 CFU values across six isolates. Each point is a measurement at one time point. Lines connect isolate means across days, with SEM bars. Tissue collection is terminal and the display treats sampling days as separate groups.

```sh
python code/plot_isolates.py --input data/nasal_colonisation_measurements.tsv --output-dir /tmp/ab-nasal
```
