# Code and source to accompany the paper  
# *Healthcare Infrastructure Shapes Evolutionary Trade-offs and Geographic Dissemination of Multidrug-Resistant Acinetobacter baumannii*

## Overview

This repository contains the code, intermediate resources, and source data accompanying the paper:

**Healthcare Infrastructure Shapes Evolutionary Trade-offs and Geographic Dissemination of Multidrug-Resistant *Acinetobacter baumannii***

The repository is organized around the major analytical components of the study, following the structure of the manuscript. In brief, the study integrates large-scale phylogenomics, transmission-network reconstruction, urban metadata analysis, comparative genomics, experimental validation, and mechanistic simulation to investigate how healthcare-system architecture shapes the evolution and dissemination of multidrug-resistant *A. baumannii*.

The main folders correspond to the core components of the paper, while the final folder contains source data used for figure generation.

## Repository structure

1. **Phylogenomics and Transmission Inference**  
   This folder contains scripts and processed inputs used for genomic data integration, lineage assignment, time-resolved phylogenetic reconstruction, ancestral geographic-state inference, and city-level transmission network analysis.  
   Major analyses include:
   - time-calibrated phylogeny construction
   - geographic transmission inference
   - transmission-module identification
   - dissemination distance and velocity estimation

2. **Urban Drivers, Comparative Genomics, and Evolutionary Signatures**  
   This folder contains scripts and supporting files used for downstream analyses linking pathogen dissemination patterns to healthcare infrastructure and genomic adaptation.  
   Major analyses include:
   - correlation of clade prevalence with urban and healthcare indicators
   - multivariate analysis of city-level drivers
   - pseudogene / gene disruption parsing
   - mutational hotspot summarization
   - comparative analyses of *A. baumannii* and *Klebsiella pneumoniae*

3. **Figure Source Data**  
   This folder contains the processed tables used to generate the main figures and selected extended data figures in the manuscript. These files are intended to facilitate figure reproduction and result inspection.

4. **Niche-Specific Transmission Model (`AB_network`)**  
   This folder contains the simulation framework used to model genotype competition under heterogeneous healthcare-network structures. It implements the conceptual transmission model described in the manuscript, in which distinct ecotypes are favored in metropolitan versus peripheral healthcare settings.  
   Major components include:
   - scale-free healthcare network construction
   - genotype competition under heterogeneous outpatient/inpatient conditions
   - perturbation analysis under COVID-like mobility restriction scenarios
   - exploration of niche-specific coexistence dynamics

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
