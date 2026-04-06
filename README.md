# *Healthcare Infrastructure Shapes Evolutionary Trade-offs and Geographic Dissemination of Multidrug-Resistant Acinetobacter baumannii*

## Overview

This repository contains the code, intermediate resources, and source data accompanying the paper:

**Healthcare Infrastructure Shapes Evolutionary Trade-offs and Geographic Dissemination of Multidrug-Resistant *Acinetobacter baumannii***

The repository is organized around the major analytical components of the study, following the structure of the manuscript. In brief, the study integrates large-scale phylogenomics, transmission-network reconstruction, urban metadata analysis, comparative genomics, experimental validation, and mechanistic simulation to investigate how healthcare-system architecture shapes the evolution and dissemination of multidrug-resistant *A. baumannii*.

The main folders correspond to the core components of the paper, while the final folder contains source data used for figure generation.

## Repository structure

## 1. Phylogenomics and Transmission Inference

This folder contains scripts and processed inputs for reconstructing the evolutionary history and transmission dynamics of *Acinetobacter baumannii* across China.

Major analytical components include:

- time-calibrated phylogeny reconstruction  
- phylogeographic transmission inference  
- transmission-module identification  
- dissemination distance and velocity estimation  


### Time-resolved phylogenetic reconstruction

Time-scaled phylogenies were reconstructed using **BEAST2 (v2.6.6)** and **TreeTime (v0.11.4)**.  
Sampling dates and geographic metadata were obtained from the accompanying metadata table.

TreeTime outputs an annotated Nexus file (`treetime.annotated_2.nexus`) containing:

- node dates (tMRCA estimates)  
- inferred geographic states  
- posterior probabilities of ancestral locations  

This file can be directly visualized in iTOL for inspection of temporal and spatial evolutionary patterns.


### Transmission inference

Transmission dynamics were inferred using the discrete trait reconstruction framework implemented in TreeTime.  

In brief, each branch in the phylogeny was evaluated for changes in geographic state between parent and descendant nodes.  
Branches associated with state transitions were interpreted as putative transmission events between locations.

By aggregating these events across the phylogeny, we constructed a city-level transmission network, in which:

- nodes represent cities  
- edges represent inferred transmission links  
- edge weights reflect normalized transmission intensity  

The resulting network is provided as:

- `China_Transmission_Normalization.tsv`

This file can be directly imported into network analysis platforms (e.g., Gephi) for visualization and further exploration.


### Network modularity and transmission structure

Based on the transmission network, we identified higher-order transmission structure using the script:

- `Network.py`

This script applies the **Louvain community detection algorithm** (equivalent to that implemented in Gephi) to:

- identify transmission modules across resolution parameters  
- quantify intra- and inter-module transmission intensity  
- assess hierarchical structure of dissemination  

In addition, cities were stratified into hierarchical categories  
(e.g., mega cities, provincial capitals, non-provincial cities),  
and their relative contributions to transmission were quantified.

These analyses constitute the majority of **Extended Data Figure 3**.


### Dissemination distance and velocity estimation

Spatial dissemination dynamics were estimated following a phylogeny-based framework inspired by recent work on bacterial transmission dynamics (Nature, 2024).

For each phylogenetic branch associated with a geographic transition,  
transmission velocity was conservatively defined as:

- geographic distance between inferred locations  
- divided by the temporal interval between nodes  

Geographic distances between cities were obtained via the Baidu Map API.

The script:

- `Speed.py`

implements a phylogeny-informed estimation framework that:

- reconstructs time-resolved dissemination distance  
- estimates annual transmission velocity trajectories  
- quantifies temporal shifts in dissemination efficiency  

This analysis produces two key outputs:

1. cumulative dissemination distance as a function of evolutionary time (20-year scale)  
2. comparative distribution of transmission velocities before and after 2019  

These results form the core of **Figure 2 (China panel)**.


---

## 2. Urban Drivers, Comparative Genomics, and Evolutionary Signatures

This folder contains analyses linking transmission dynamics to healthcare infrastructure and genomic adaptation.

Major components include:

- association between clade prevalence and urban connectivity  
- multivariate modelling of city-level drivers  
- pseudogene identification and enrichment analysis  
- mutational hotspot detection  


### Network connectivity and clade prevalence

Using the transmission network (`China_Transmission_Normalization.tsv`),  
we quantified the relationship between city connectivity and clade prevalence using:

- `Degree.py`

This analysis evaluates:

- correlation between node degree and detection rate of ESL2.4 and ESL2.5  
- association between clade prevalence and healthcare mobility indices  

Healthcare mobility data were derived from large-scale analyses of patient movement patterns in China (Nature, 2024).

These analyses correspond to **Figure 3a–c**.


### Urban attributes and multivariate modelling

City-level metadata were compiled from national statistical yearbooks and healthcare reports,  
and are provided in:

- `City_attribution.tsv`  
- `True_World_OI.tsv`

A key variable in this study is the **outpatient-to-inpatient ratio (O/I ratio)**,  
which serves as a proxy for healthcare system centrality and patient mobility.

We used:

- `OI_dev.py` to evaluate relationships between O/I ratio and economic indicators (e.g., GDP)  
- `City_Attr.py` to assess correlations between ESL2.5 prevalence and urban attributes  

Further, we applied:

- redundancy analysis (RDA)  
- structural equation modelling (SEM)  

to quantify the relative contributions of socioeconomic and healthcare variables to clade prevalence.

These analyses constitute **Figure 3d–g**.


### Pseudogene identification and enrichment analysis

Pan-genome reconstruction was performed using **PEPPAN (vX.X.X)**.

Putative gene disruptions (pseudogenes) were classified based on structural criteria, including:

- truncation  
- frameshift mutations  
- premature stop codons  

Pseudogenes enriched in ESL2.5 were defined using a frequency-based criterion:

- ≥50% presence in ESL2.5  
- <50% presence in non-ESL2.5  

Results are provided in:

- `Pseudogenes.tsv`

Functional enrichment (COG and KEGG) was performed based on these gene sets.


### Mutational hotspot identification

Genomes were aligned to the MDR-TJ reference using EToKi (minimap2 backend).

SNP density was calculated using sliding windows across the genome,  
and hotspots were defined as regions exceeding the 95th percentile of SNP density.

Results are provided in:

- `Mutations_hotspot.tsv`

These analyses form **Figure 4a–c**.


---

## 3. Niche-Specific Transmission Model (`AB_network`)

This folder contains a simulation framework modelling genotype competition under heterogeneous healthcare-network structures.

The model implements the conceptual framework described in the manuscript,  
in which distinct evolutionary strategies are favored in different healthcare environments.

Major components include:

- scale-free healthcare network construction  
- genotype competition under heterogeneous outpatient/inpatient conditions  
- perturbation analysis under mobility restriction scenarios (e.g., COVID-19)  
- exploration of niche-specific coexistence dynamics  

An interactive version of the simulation is available at:

https://naclist.github.io/naclist-portfolio/sources/ab_china_transmission_simulation/

## Requirements

The repository primarily uses **Python 3** for data parsing, summarization, and simulation. Some analyses also rely on external phylogenetic and statistical tools.

### Python packages
Commonly used Python dependencies include:

- `pandas`
- `numpy`
- `ete3`
- `scipy`
- `matplotlib`
- `networkx`

Depending on the specific module, additional packages may be required.

### External tools
Some analyses in this study were performed using external tools that are not bundled in this repository, including but not limited to:

- **EToKi**
- **IQ-TREE2**
- **TreeTime**
- **BEAST2**
- **Gephi**
- **R** (for selected downstream statistical analyses such as RDA and SEM)

Please ensure that these tools are installed and accessible in your environment if full reproduction of the analysis is intended.

## Citation

If you use this repository, its code, or processed source files in your work, please cite the accompanying paper:

**Li S, Wu Y, Zhou Y, Zhong L, Jiang Y, Wang Y, Li J, Lin H, Li H, Xia S, Du H, Zhang R, Lou Y, Wang S, *et al.***  
*Healthcare Infrastructure Shapes Evolutionary Trade-offs and Geographic Dissemination of Multidrug-Resistant Acinetobacter baumannii*.

If appropriate, please also cite the software tools used in the analysis pipeline, including EToKi, IQ-TREE2, TreeTime, BEAST2, and other relevant packages described in the manuscript.

## Notes

- This repository is intended to accompany the published study and therefore emphasizes **analysis transparency** and **figure reproducibility**, rather than providing a single-click end-to-end pipeline.
- Some scripts represent intermediate processing utilities developed during analysis and may require adaptation to local file structures or naming conventions.
- Large raw sequencing datasets are not stored directly in this repository; users should refer to the manuscript for accession numbers and public data availability statements.
- Folder names and script names are retained where possible to preserve consistency with the original analysis workflow.
- Users interested in reproducing a specific figure or result are encouraged to begin with the corresponding source-data tables and then trace the relevant upstream scripts.
