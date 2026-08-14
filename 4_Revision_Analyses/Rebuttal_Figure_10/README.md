# Rebuttal Figure 10

Sampling-bias sensitivity of the inferred city transmission network. As
requested, this package contains the analytical data and component scripts but
does not attempt to recreate the final multi-panel page layout.

The principal frozen tables are genome-level replicate metrics,
strategy-averaged edge weights, city-level degree replicates, NMI/ARI and other
network summaries, and the displayed network edge/node tables. Raw analytical
inputs (annotated nexus, dates, city distances, normalized network and metadata
workbook) are also included. The four degree-scatter panels use the separate
`data/network_file_resampling/` snapshot; this distinction is retained because
the later five-strategy nexus snapshot is not the source of those four panels.

`selected_network_metrics_ci.tsv` is treated as a frozen figure-ready table.
The component plotting script reads it directly rather than recomputing it from
a later replicate-table version.

Component scripts have local defaults. The most direct panel commands are:

```powershell
python code/plot_degree_spearman_scatter.py
python code/plot_selected_network_metrics_ci.py
python code/plot_china_skeleton_nexus_downsampling.py
```
