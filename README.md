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

Time-scaled phylogenies were reconstructed using **[BEAST2 (v2.6.6)](https://www.beast2.org/)** and **[TreeTime (v0.11.4)](https://github.com/neherlab/treetime)**.  
Sampling dates and geographic metadata were obtained from the accompanying metadata table.

TreeTime outputs an annotated Nexus file ([`treetime.annotated.nexus`](https://github.com/Zhou-lab-SUDA/Genomic_China_Acinetobacter/blob/main/1_Phylogenomics_and_Transmission_Inference/treetime.annotated.nexus)) containing:

- node dates (tMRCA estimates)  
- inferred geographic states  
- posterior probabilities of ancestral locations  

This file can be directly visualized in [iTOL](https://itol.embl.de/) for inspection of temporal and spatial evolutionary patterns.


### Transmission inference

Transmission dynamics were inferred using the discrete trait reconstruction framework implemented in TreeTime mugration module.  

In brief, each branch in the phylogeny was evaluated for changes in geographic state between parent and descendant nodes.  
Branches associated with state transitions were interpreted as putative transmission events between locations.

By aggregating these events across the phylogeny, we constructed a city-level transmission network, in which:

- nodes represent cities  
- edges represent inferred transmission links  
- edge weights reflect normalized transmission intensity  

The resulting network is provided as:

- [`China_Transmission_Normalization.tsv`](https://github.com/Zhou-lab-SUDA/Genomic_China_Acinetobacter/blob/main/1_Phylogenomics_and_Transmission_Inference/China_Transmission_Normalization.tsv)

This file can be directly imported into network analysis platforms (e.g., [Gephi](https://gephi.org/)) for visualization and further exploration.


### Network modularity and transmission structure

Based on the transmission network, we identified higher-order transmission structure using the script:

- [`Network.py`]()

This script applies the **Louvain community detection algorithm** (equivalent to that implemented in Gephi) to:

- identify transmission modules across resolution parameters  
- quantify intra- and inter-module transmission intensity  
- assess hierarchical structure of dissemination  

In addition, cities were stratified into hierarchical categories  
(mega cities, provincial capitals, non-provincial cities, [Source from China government](https://www.ndrc.gov.cn/xwdt/ztzl/xxczhjs/ghzc/201605/t20160509_971910.html)),  
and their relative contributions to transmission were quantified.

These analyses constitute the majority of **Extended Data Figure 3**.

<p align="center">
  <img src="[images/figure1.png](https://github.com/Zhou-lab-SUDA/Genomic_China_Acinetobacter/blob/main/1_Phylogenomics_and_Transmission_Inference/Figure_module_scan_connection_and_city_level.png)" width="600" alt="EDF3">
</p>

### Dissemination distance and velocity estimation

Spatial dissemination dynamics were estimated following a phylogeny-based framework inspired by recent work from Belman et al. on bacterial transmission dynamics ([Nature, 2024](https://doi.org/10.1038/s41586-024-07626-3)).

For each phylogenetic branch associated with a geographic transition,  
transmission velocity was conservatively defined as:

- geographic distance between inferred locations  
- divided by the temporal interval between nodes  

Geographic distances between cities were obtained via [the Baidu Map API](https://lbs.baidu.com/), a publicly accessible data source available upon application, and can be found in [PairwiseChinaCities.distance](https://github.com/Zhou-lab-SUDA/Genomic_China_Acinetobacter/blob/main/1_Phylogenomics_and_Transmission_Inference/PairwiseChinaCities.distance).

We employed the script:

- [`Speed.py`]()

to implement a phylogeny-informed estimation framework that:

- reconstructs time-resolved dissemination distance  
- estimates annual transmission velocity trajectories  
- quantifies temporal shifts in dissemination efficiency  

This analysis produces two key outputs:

1. cumulative dissemination distance as a function of evolutionary time (20-year scale)  
2. comparative distribution of transmission velocities before and after 2019  

These results form the core of **Figure 2 (China panel)**.

<p align="center">
  <img src="images/figure2.png" width="600" alt="F2">
</p>

---

## 2. Urban Drivers, Comparative Genomics, and Evolutionary Signatures

This folder contains analyses linking transmission dynamics to healthcare infrastructure and genomic adaptation.

Major components include:

- association between clade prevalence and urban connectivity  
- multivariate modelling of city-level drivers  
- pseudogene identification and enrichment analysis  
- mutational hotspot detection  


### Network connectivity and clade prevalence

Using the transmission network ([`China_Transmission_Normalization.tsv`](https://github.com/Zhou-lab-SUDA/Genomic_China_Acinetobacter/blob/main/1_Phylogenomics_and_Transmission_Inference/China_Transmission_Normalization.tsv)
),  
we quantified the relationship between city connectivity and clade prevalence using:

- [`Degree.py`]()

This analysis evaluates:

- correlation between node degree and detection rate of ESL2.4 and ESL2.5  
- association between clade prevalence and healthcare mobility indices  

Healthcare mobility data were derived from large-scale analyses of patient movement patterns in China by Zhao et al.([Nature cities, 2024](https://doi.org/10.1038/s44284-024-00185-8)).

These analyses correspond to **Figure 3a-c**.

<p align="center">
  <img src="images/figure3.png" width="600" alt="F31">
</p>

### Urban attributes and multivariate modelling

We downloaded and manually curated a set of data most relevant to the local healthcare environment from [the China Statistical Yearbook](https://www.stats.gov.cn/zs/tjwh/tjkw/tjzl/202302/t20230220_1913734.html) and [the China Health Statistics Yearbook](https://www.nhc.gov.cn/mohwsbwstjxxzx/tjtjnj/202501/8193a8edda0f49df80eb5a8ef5e2547c.shtml), which are available in the file [`City_attribution.tsv`](https://github.com/Zhou-lab-SUDA/Genomic_China_Acinetobacter/blob/main/2_Evolutionary_Signatures/ChinaCityInfo.xlsx). Because more than half of the municipal health departments lack published statistics, we aggregated the number of inpatients and outpatient visits only at the provincial level ([`True_World_OI.tsv`](https://github.com/Zhou-lab-SUDA/Genomic_China_Acinetobacter/blob/main/2_Evolutionary_Signatures/TrueValueOI.xlsx)).

The outpatient/inpatient ratio (**O/I ratio**) is a key concept in our study. The **O/I ratio** indicates the extent to which a region acts as a destination for cross-regional healthcare-seeking mobility. Using OI_dev.py, we show that the **O/I ratio** is strongly correlated with per capita GDP, as expected. Owing to China’s universal health insurance policy and the concentration of high-level medical resources, most patients tend to seek care at the best local hospitals or nearby regional medical centers rather than nearby general clinics, even for non‑severe conditions. 

Consequently, first‑tier cities and provincial capitals bear a much higher outpatient burden, a considerable proportion of which consists of cross‑regional patients.
As noted earlier, lineage 2.5 was more likely to be detected in highly connected cities. We therefore used [`City_Attr.py`]() to evaluate the Spearman correlation between the detection of 2.5 and various city attributes. Furthermore, we applied redundancy analysis (RDA) and structural equation modeling (SEM) to assess the effects of different factors on the detection rate of 2.5 across cities.

These analyses constitute **Figure 3d-g**.

<p align="center">
  <img src="images/figure4.png" width="600" alt="F32">
</p>

### Pseudogene identification and enrichment analysis

Pan-genome reconstruction was performed using **PEPPAN (v1.0.5)** via following commands:

```
PEPPAN -P GCF_000187205.gff --min_cds 80 --nucl *.gff
```

Putative gene disruptions (pseudogenes) were classified based on structural criteria, including:

- truncation  
- frameshift mutations  
- premature stop codons  

Pseudogenes enriched in ESL2.5 were defined using a frequency-based criterion:

- ≥50% presence in ESL2.5  
- <50% presence in non-ESL2.5  

Results are provided in:

- [`Pseudogenes.tsv`](https://github.com/Zhou-lab-SUDA/Genomic_China_Acinetobacter/blob/main/2_Evolutionary_Signatures/ESL2.5_Pseudogenes.txt)

Functional enrichment (COG and KEGG) was performed based on these gene sets.


### Mutational hotspot identification

Genomes were aligned to the MDR-TJ reference using EToKi (minimap2 backend).

SNP density per Kb was calculated using sliding windows across the genome, and hotspots were defined as regions exceeding the 95th percentile of SNP density.

Results are provided in:

- [`Mutations_hotspot.tsv`](https://github.com/Zhou-lab-SUDA/Genomic_China_Acinetobacter/blob/main/2_Evolutionary_Signatures/ESL2.5_Mutations.txt)

These analyses form **Figure 4a-c**.

<p align="center">
  <img src="images/figure5.png" width="600" alt="F4">
</p>

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

## Reproduction

We provide a reproducible workflow (`Main.py`), which integrates `Network.py`, `Speed.py`, `Degree.py`, and `City_Attr.py` as referenced earlier. Running this script reproduces most of the analytical results and figure visualizations presented in the paper.

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
