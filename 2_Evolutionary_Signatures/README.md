# Evolutionary signatures and phenotype experiments

This directory contains city-level association analyses, lineage-associated genomic variation and phenotype measurements used to compare ESL2.4 and ESL2.5.

## Directory structure

| Component | Contents |
| --- | --- |
| [`city_associations/`](city_associations/) | Permutation-based Spearman tests for 22 city variables, nominal P values, Storey q-values and plotting tables |
| [`snp_density/`](snp_density/) | Recombination-masked 10-kb SNP-density profiles for all annotated SNPs and the nonsynonymous subset |
| [`phenotype_experiments/`](phenotype_experiments/) | Source measurements, derived tables and calculation scripts for experimental phenotype comparisons |
| `Degree.py` | Association between empirical city-network connectivity and lineage prevalence |
| `ChinaCityInfo.xlsx` | Curated city attributes used in urban analyses |
| `TrueValueOI.xlsx` | Provincial outpatient/inpatient indicators |
| `ESL2.5_Pseudogenes.txt` | ESL2.5-associated putative gene disruptions |
| `ESL2.5_Mutations.txt` and `KpST11_Mutations.txt` | Mutation annotations used in comparative genomic summaries |

## City-level analyses

The univariate analysis tests associations between ESL2.5 relative abundance and 22 city variables using Spearman statistics and permutation P values. `city_associations/` contains the complete marginal-test table, Benjamini-Hochberg adjustment and Storey q-values; the smallest Storey q-value is 0.0768 for air pollution. RDA and SEM provide a separate multivariate analysis of predefined urban, healthcare and mobility variable groups.

`Degree.py` combines lineage prevalence with the empirical city network stored under [`../1_Phylogenomics_and_Transmission_Inference/`](../1_Phylogenomics_and_Transmission_Inference/). Network edges come from phylogeographic transitions; healthcare and mobility indicators enter as city-level covariates.

## Genomic variation

The SNP-density analysis summarizes recombination-masked variants in consecutive 10-kb windows. Separate all-variant and protein-changing profiles show the effect-type composition of each lineage-associated peak. The highest ESL2.5 window is 3,897,000-3,897,999 bp, with 65 variants (54 synonymous and 11 protein-changing); adjacent windows and gene annotations are reported in `snp_density/data/top_peak_annotations.tsv`.

Putative pseudogenes are classified from structural disruptions such as truncations, frameshifts and premature stop codons. `ESL2.5_Pseudogenes.txt` records the lineage frequencies and annotations used for the COG and KEGG summaries. The associated [`gene_expression_qpcr/`](phenotype_experiments/gene_expression_qpcr/) analysis summarizes technical-triplicate Ct values for PilA, CsuE, AbaI and PilT relative to 16S rRNA, using the isolate as the independent unit.

## Phenotype experiments

[`phenotype_experiments/`](phenotype_experiments/) organizes the experimental comparisons into four result groups:

1. antimicrobial susceptibility;
2. capsule architecture and capsular-polysaccharide production, combining transmission electron microscopy with uronic-acid measurements;
3. mouse experiments, combining nasal colonisation and nasal sIgA;
4. A549-cell adhesion.

For quantitative assays, each directory identifies the source table, independent experimental unit, derived outputs and analysis script. The microscopy and sIgA directories document sample preparation, acquisition and readout procedures. Gene-expression qPCR is presented with gene disruptions in the genomic-variation section above.

## Dependencies

The tabular analyses require Python 3.10 or later. The complete package declaration and installation command for the experimental scripts are provided in the [`phenotype_experiments` README](phenotype_experiments/#python-dependencies).
