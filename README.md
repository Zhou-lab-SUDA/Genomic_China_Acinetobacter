# Healthcare Infrastructure Shapes Evolutionary Trade-offs and Geographic Dissemination of Multidrug-Resistant *Acinetobacter baumannii*

This repository contains the analysis code, processed data and model implementation accompanying the study **Healthcare Infrastructure Shapes Evolutionary Trade-offs and Geographic Dissemination of Multidrug-Resistant *Acinetobacter baumannii***.

The study combines phylogenomics, phylogeography, sampling-sensitivity analyses, urban healthcare indicators, lineage-associated genomic variation, phenotype experiments and a spatial population-genetic model to examine the dissemination and ecological differentiation of the major ESL2 lineages in China.

## Repository map

| Section | Scope | Main entry point |
| --- | --- | --- |
| 1. Phylogenomics and transmission inference | Dated phylogeny, empirical city network, dissemination distance and velocity, temporal-signal tests, recombination masking and sampling sensitivity | [`1_Phylogenomics_and_Transmission_Inference/`](1_Phylogenomics_and_Transmission_Inference/) |
| 2. Evolutionary signatures and phenotypes | City-level associations, lineage-associated SNP-density peaks, antimicrobial susceptibility and experimental phenotypes | [`2_Evolutionary_Signatures/`](2_Evolutionary_Signatures/) |
| 3. Spatial transmission model | Synthetic city-network model of genotype competition, invasion and mobility intervention | [`3_AB_network/`](3_AB_network/) |

Each analysis directory separates `data/` from `code/` where possible. Its local README identifies the input tables, derived outputs, statistical unit and interpretation limits.

## 1. Phylogenomics and transmission inference

The [phylogenomics module](1_Phylogenomics_and_Transmission_Inference/) contains the shared genome metadata, dated tree, normalized city-to-city transmission table and the analyses used to assess their robustness.

| Analysis | Contents |
| --- | --- |
| [Temporal signal](1_Phylogenomics_and_Transmission_Inference/temporal_signal/) | Root-to-tip regression for the combined collection and individual ESL2 lineages; 10 date permutations, each evaluated through 10 independent subsamples |
| [Temporal-signal sampling sensitivity](1_Phylogenomics_and_Transmission_Inference/temporal_signal/sampling_sensitivity/) | Random, population-weighted, year-balanced, city-balanced and combined city/year-balanced sampling |
| [Recombination](1_Phylogenomics_and_Transmission_Inference/recombination/) | RecHMM, Gubbins and ClonalFrameML profiles; low-recombination block coordinates; SNP counts before and after masking |
| [SNP-distance validation](1_Phylogenomics_and_Transmission_Inference/snp_distance_validation/) | Pairwise recombination-masked SNP distance versus divergence time and geographic spread |
| [Network sensitivity](1_Phylogenomics_and_Transmission_Inference/network_sensitivity/) | Five sampling strategies, replicate networks, normalized mutual information and adjusted Rand index |
| [Clade-prevalence sensitivity](1_Phylogenomics_and_Transmission_Inference/clade_prevalence_sensitivity/) | Annual and overall lineage proportions under four metadata-resampling schemes |

The empirical transmission network is inferred from changes in reconstructed geographic state along the dated phylogeny. Its nodes are sampled cities and its edges are inferred intercity transitions; it is not a directly observed hospital-transfer network. [`Network.py`](1_Phylogenomics_and_Transmission_Inference/Network.py) evaluates network community structure, while [`Speed.py`](1_Phylogenomics_and_Transmission_Inference/Speed.py) summarizes phylogeny-based dissemination distance and velocity.

## 2. Evolutionary signatures and phenotype experiments

The [evolutionary-signatures module](2_Evolutionary_Signatures/) links city characteristics and genomic measurements to observed lineage distributions, and provides the experimental measurements used to compare ESL2.4 and ESL2.5.

### Urban and genomic analyses

- [City-level associations](2_Evolutionary_Signatures/city_associations/) retain permutation-based Spearman tests for 22 city variables and Storey FDR estimates. The marginal tests are documented separately from multivariate RDA and SEM.
- [Lineage-associated SNP density](2_Evolutionary_Signatures/snp_density/) provides annotated 10-kb profiles for all SNPs and the nonsynonymous subset. These are reported as SNP-density peaks, with residual recombination considered for the exceptional high-synonymous peak.
- [`Degree.py`](2_Evolutionary_Signatures/Degree.py) evaluates the association between empirical city-network connectivity and lineage prevalence.
- The top-level tables in this module contain the city covariates, outpatient/inpatient indicators, pseudogene calls and lineage-associated mutation annotations used by the corresponding analyses.

### Phenotype experiments

The [phenotype-experiment section](2_Evolutionary_Signatures/phenotype_experiments/) is organized by assay:

| Experiment | Available material |
| --- | --- |
| [Antimicrobial susceptibility](2_Evolutionary_Signatures/phenotype_experiments/antimicrobial_susceptibility/) | Isolate-level S/I/R calls, drug-level comparisons and FDR-adjusted statistics |
| [Capsular polysaccharide](2_Evolutionary_Signatures/phenotype_experiments/capsular_polysaccharide/) | Uronic-acid measurements, isolate summaries and lineage/KL-type comparisons |
| [Gene-expression qPCR](2_Evolutionary_Signatures/phenotype_experiments/gene_expression_qpcr/) | Technical Ct replicates, isolate-level normalization and target-wise comparisons |
| [A549-cell adhesion](2_Evolutionary_Signatures/phenotype_experiments/a549_adhesion/) | Source plate counts, focused isolate comparison and isolate-level statistics |
| [Mouse nasal colonisation](2_Evolutionary_Signatures/phenotype_experiments/nasal_colonisation/) | Longitudinal mouse-level bacterial loads and time-point-specific comparisons |
| [Systemic-infection survival](2_Evolutionary_Signatures/phenotype_experiments/systemic_infection_survival/) | Mouse-level event data, Kaplan-Meier estimates, log-rank analysis and subset-selection record |
| [Transmission electron microscopy](2_Evolutionary_Signatures/phenotype_experiments/transmission_electron_microscopy/) | Experimental method; quantitative micrograph measurements are not available |
| [Nasal secretory IgA](2_Evolutionary_Signatures/phenotype_experiments/secretory_iga/) | Experimental method; the quantitative source table is not available |

The independent experimental unit is defined in each assay README. In particular, A549 comparisons use the isolate mean, qPCR technical replicates are averaged within isolates, and longitudinal colonisation measurements retain the mouse as the unit. These experiments establish measured associations in representative isolates; they do not by themselves establish that capsule type or a particular gene disruption causes a lineage-level phenotype.

## 3. Spatial transmission model

[`3_AB_network/`](3_AB_network/) contains a synthetic population-genetic simulation of competing transmission strategies across a heterogeneous city network. It supports basic, invasion and three-phase mobility-intervention scenarios.

The simulated network is conceptually motivated by the contrast between metropolitan and peripheral healthcare environments but is not fitted to the empirical genomic network or to direct patient-transfer data. See the [model README](3_AB_network/README.md) for commands, parameters and output formats.

## Reproducing an analysis

There is no single top-level driver because the repository combines phylogenetic software, statistical summaries, experimental assays and simulation. For a specific result:

1. Open the relevant module README from the map above.
2. Check the stated source table, statistical unit and assumptions.
3. Run the script from its analysis directory so that its relative `data/` paths resolve correctly.
4. Compare the regenerated table with the included derived output before recreating a figure.

Large raw sequencing files and the low-recombination-block FASTA are not stored in this repository. Processed inputs needed for the documented downstream analyses are included where redistribution is possible. Figure image files are included only for analyses already represented in the repository; additional panels can be regenerated from the corresponding source tables and scripts.

## Software requirements

Most tabular analyses use Python 3.10 or later. Common packages are:

- NumPy, pandas and SciPy
- Matplotlib and seaborn
- openpyxl
- NetworkX, python-louvain and ETE Toolkit
- Plotly for interactive model output

The phenotype module provides its [Python dependency declaration](2_Evolutionary_Signatures/phenotype_experiments/#python-dependencies), and the simulation has a separate [`requirements.txt`](3_AB_network/requirements.txt). Individual scripts may use only a subset of these packages.

Upstream phylogenomic reconstruction additionally used EToKi, IQ-TREE 2, RecHMM, Gubbins, ClonalFrameML, TempEst, TreeTime and BEAST 2. Selected multivariate analyses use R. These external tools are not bundled here.

## Data interpretation

- Temporal signal is evaluated by lineage. Time-dependent estimates are not interpreted for ESL2.3 because no detectable signal was found.
- Recombination is masked; the input alignment is not described as recombination-free.
- Network and annual clade-proportion sensitivity analyses address unequal sampling among cities and years, but percentages remain conditional on the available genomes.
- Nominal city-level associations and their FDR-adjusted results are both retained. The univariate tests and multivariate RDA/SEM address different questions.
- The systemic-infection subset analysis is outcome-dependent and is documented as exploratory in its local README and selection table.
- Empirical phylogeographic inference and the synthetic transmission model are separate analytical objects.

## Citation

If you use this repository, its code or processed source data, please cite the accompanying paper:

Li S, Wu Y, Zhou Y, Zhong L, Jiang Y, Wang Y, Li J, Lin H, Li H, Xia S, Du H, Zhang R, Lou Y, Wang S, *et al.* **Healthcare Infrastructure Shapes Evolutionary Trade-offs and Geographic Dissemination of Multidrug-Resistant *Acinetobacter baumannii*.**

Please also cite the original software and external data sources relevant to the analysis being reused.

## License

See [`LICENSE`](LICENSE).
