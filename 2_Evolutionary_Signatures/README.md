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

The univariate analysis tests associations between ESL2.5 relative abundance and 22 city variables. The repository retains both nominal permutation P values and Storey FDR q-values; no individual variable remains significant at q < 0.05. These marginal tests are distinct from RDA and SEM, which evaluate joint structure among predefined groups of standardized urban, healthcare and mobility variables. The RDA/SEM scripts and final model-input table are not included in `city_associations/`.

The empirical network used by `Degree.py` is reconstructed from phylogeographic transitions and is stored under [`../1_Phylogenomics_and_Transmission_Inference/`](../1_Phylogenomics_and_Transmission_Inference/). Healthcare and mobility indicators are covariates and do not define its edges.

## Genomic variation

The SNP-density analysis uses recombination-masked variants summarized in consecutive 10-kb windows. The regions are described as lineage-associated SNP-density peaks rather than definitive mutational hotspots. Separate all-SNP and nonsynonymous tables allow the contribution of mutation annotation type to be inspected. The peak at 3,890,000-3,899,999 bp has an unusually high synonymous fraction and may reflect residual recombination or horizontal transfer.

Putative pseudogenes are classified from structural disruptions such as truncations, frameshifts and premature stop codons. Their lineage association does not establish that an individual disruption causes an experimental phenotype.

## Phenotype experiments

[`phenotype_experiments/`](phenotype_experiments/) contains eight assay-specific directories:

- antimicrobial susceptibility;
- capsular-polysaccharide quantification;
- gene-expression qPCR;
- A549-cell adhesion;
- mouse nasal colonisation;
- mouse systemic-infection survival;
- transmission electron microscopy;
- nasal secretory IgA.

For quantitative assays, each directory identifies the source table, independent experimental unit, derived outputs and analysis script. The microscopy and sIgA directories retain the available experimental methods; quantitative source measurements were not reconstructed when the underlying tables were unavailable.

The experiments characterize representative isolates and support measured associations between lineage and phenotype. They do not by themselves establish that capsule type or a particular gene disruption causes a lineage-level difference in adhesion, colonisation, transmission or virulence.

## Dependencies

The tabular analyses require Python 3.10 or later. The complete package declaration and installation command for the experimental scripts are provided in the [`phenotype_experiments` README](phenotype_experiments/#python-dependencies).

## Figure placement

The following existing overview images are retained at this level:

- `F31.png`: transmission events, lineage prevalence and patient-mobility strata.
- `F32.png`: outpatient/inpatient burden, urban correlates, RDA and SEM.
- `F4.png`: lineage-associated gene disruptions, functional enrichment and mutation density.

Additional final images should be added at these paths:

| Path | Required content |
| --- | --- |
| `city_associations/city_association_multiple_testing.png` | Nominal permutation P values and Storey FDR q-values for 22 city variables |
| `snp_density/snp_density_peaks.png` | All-SNP and nonsynonymous 10-kb density profiles with peak annotations |

Assay-specific image positions are listed in the [`phenotype_experiments` README](phenotype_experiments/#figure-placement).
