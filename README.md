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

### Empirical transmission structure

The existing network figure shows the resolution scan, within- versus cross-module connection weights, the distribution of transmission events between city levels and transmission weight by city hierarchy.

<p align="left">
  <img src="1_Phylogenomics_and_Transmission_Inference/Figure_module_scan_connection_and_city_level.png" width="650" alt="Transmission-network modularity and city hierarchy">
</p>

### Dissemination distance and velocity

The existing dissemination figure compares lineage-specific spread speeds between epidemiological periods and summarizes geographic distance over evolutionary time.

<p align="left">
  <img src="1_Phylogenomics_and_Transmission_Inference/F2.png" width="650" alt="Phylogeny-based dissemination speed and distance">
</p>

### Additional phylogenomic figure positions

The following image positions are intentionally left open. Add each final image at the listed path; the adjacent analysis directory already contains its data and plotting code.

> **Figure placeholder — lineage-specific temporal signal**
> Add `1_Phylogenomics_and_Transmission_Inference/temporal_signal/root_to_tip_temporal_signal.png`. The figure should contain the combined, ESL2.3, ESL2.4 and ESL2.5 root-to-tip regression panels.

> **Figure placeholder — date randomisation**
> Add `1_Phylogenomics_and_Transmission_Inference/temporal_signal/date_randomisation.png`. The figure should compare the observed correlation and its sampling interval with the distribution of the ten permutation-level medians for each lineage.

> **Figure placeholder — temporal-signal sampling sensitivity**
> Add `1_Phylogenomics_and_Transmission_Inference/temporal_signal/sampling_sensitivity/temporal_signal_sampling_sensitivity.png`. The figure should show lineage-specific temporal-signal estimates across the five sampling strategies.

> **Figure placeholder — recombination landscape**
> Add `1_Phylogenomics_and_Transmission_Inference/recombination/recombination_landscape.png`. The figure should show chromosome-wide RecHMM, Gubbins and ClonalFrameML profiles, mutation events, and the low-recombination block used for phylogenetic reconstruction.

> **Figure placeholder — SNP-distance validation**
> Add `1_Phylogenomics_and_Transmission_Inference/snp_distance_validation/snp_distance_validation.png`. The figure should summarize divergence time versus recombination-masked SNP distance, geographic spread versus SNP distance and the period-specific km-per-SNP comparison.

> **Figure placeholder — transmission-network sampling sensitivity**
> Add `1_Phylogenomics_and_Transmission_Inference/network_sensitivity/network_sampling_sensitivity.png`. The figure should compare network metrics and community stability across the five sampling strategies, including NMI and ARI.

> **Figure placeholder — clade-prevalence sampling sensitivity**
> Add `1_Phylogenomics_and_Transmission_Inference/clade_prevalence_sensitivity/clade_prevalence_sampling_sensitivity.png`. The figure should compare observed annual lineage proportions with intervals obtained under the four resampling schemes.

## 2. Evolutionary signatures and phenotype experiments

The [evolutionary-signatures module](2_Evolutionary_Signatures/) links city characteristics and genomic measurements to observed lineage distributions, and provides the experimental measurements used to compare ESL2.4 and ESL2.5.

### Urban and genomic analyses

- [City-level associations](2_Evolutionary_Signatures/city_associations/) retain permutation-based Spearman tests for 22 city variables and Storey FDR estimates. The marginal tests are documented separately from multivariate RDA and SEM.
- [Lineage-associated SNP density](2_Evolutionary_Signatures/snp_density/) provides annotated 10-kb profiles for all SNPs and the nonsynonymous subset. These are reported as SNP-density peaks, with residual recombination considered for the exceptional high-synonymous peak.
- [`Degree.py`](2_Evolutionary_Signatures/Degree.py) evaluates the association between empirical city-network connectivity and lineage prevalence.
- The top-level tables in this module contain the city covariates, outpatient/inpatient indicators, pseudogene calls and lineage-associated mutation annotations used by the corresponding analyses.

The existing city-connectivity figure contains the associations between inferred transmission events and ESL2.4/ESL2.5 prevalence, together with lineage prevalence across patient-mobility strata.

<p align="left">
  <img src="2_Evolutionary_Signatures/F31.png" width="700" alt="City connectivity, patient mobility and lineage prevalence">
</p>

The existing urban-attribute figure contains the outpatient/inpatient comparison, ranked city-variable correlations, RDA and SEM.

<p align="left">
  <img src="2_Evolutionary_Signatures/F32.png" width="700" alt="Urban attributes, RDA and structural equation model">
</p>

The existing genomic-signature figure contains lineage-associated gene disruptions, functional enrichment and chromosome-wide mutation-density profiles.

<p align="left">
  <img src="2_Evolutionary_Signatures/F4.png" width="700" alt="Gene disruptions, functional enrichment and mutation density">
</p>

> **Figure placeholder — city-variable multiple testing**
> Add `2_Evolutionary_Signatures/city_associations/city_association_multiple_testing.png`. The figure should display nominal permutation P values together with Storey FDR q-values for all 22 city variables.

> **Figure placeholder — annotated SNP-density peaks**
> Add `2_Evolutionary_Signatures/snp_density/snp_density_peaks.png`. The figure should compare all-SNP and nonsynonymous 10-kb density profiles and annotate the principal peaks, including the high-synonymous peak considered potentially affected by residual recombination.

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

> **Figure placeholders — phenotype experiments**
> Final panels should be added under their corresponding assay directories. The required panel contents and suggested filenames are listed in the [phenotype figure-placement table](2_Evolutionary_Signatures/phenotype_experiments/#figure-placement).

## 3. Spatial transmission model

[`3_AB_network/`](3_AB_network/) contains a synthetic population-genetic simulation of competing transmission strategies across a heterogeneous city network. It supports basic, invasion and three-phase mobility-intervention scenarios.

The simulated network is conceptually motivated by the contrast between metropolitan and peripheral healthcare environments but is not fitted to the empirical genomic network or to direct patient-transfer data. See the [model README](3_AB_network/README.md) for commands, parameters and output formats.

> **Figure placeholder — spatial transmission model**
> Add `3_AB_network/model_summary.png`. The figure should show the synthetic network, lineage-frequency trajectories and the basic/invasion/mobility-intervention scenarios. Existing sensitivity PDFs remain available under [`3_AB_network/example_res/sensitivity_results/`](3_AB_network/example_res/sensitivity_results/).

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
