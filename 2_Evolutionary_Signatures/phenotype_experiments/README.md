# Phenotype experiments

This section groups the experimental measurements used to compare ESL2.4 and
ESL2.5. Each quantitative analysis is kept in a separate directory with its
input table, derived results, calculation script and definition of the
independent experimental unit.

- `antimicrobial_susceptibility/`: isolate-level clinical S/I/R calls.
- `capsular_polysaccharide/`: uronic-acid measurements of capsular material.
- `gene_expression_qpcr/`: target-gene Ct values normalized to 16S rRNA.
- `a549_adhesion/`: epithelial-cell adhesion at MOI 10 and 20.
- `nasal_colonisation/`: longitudinal nasal bacterial loads in mice.
- `systemic_infection_survival/`: seven-day mouse-survival analysis.
- `transmission_electron_microscopy/`: negative-stain imaging method.
- `secretory_iga/`: nasal-mucus sIgA measurement method.

The assays characterize representative isolates and do not by themselves
establish that capsule type or a particular gene disruption causes a lineage-
level difference in colonisation, transmission or virulence.

## Python dependencies

The analysis scripts require Python 3.10 or later and the following packages:

- NumPy 1.26 or later
- pandas 2.1 or later
- Matplotlib 3.8 or later
- SciPy 1.11 or later
- openpyxl 3.1 or later

They can be installed with:

```bash
python -m pip install "numpy>=1.26" "pandas>=2.1" "matplotlib>=3.8" "scipy>=1.11" "openpyxl>=3.1"
```

## Figure placement

Final assay images are not committed at these positions yet. Add them with the suggested filenames below; the quantitative assay directories contain the corresponding data and analysis scripts.

| Assay | Suggested image path | Required panel content |
| --- | --- | --- |
| Antimicrobial susceptibility | `antimicrobial_susceptibility/antimicrobial_susceptibility.png` | ESL2.4 and ESL2.5 S/I/R proportions by antimicrobial, with the statistical comparison represented in the caption or panel annotation |
| Capsular polysaccharide | `capsular_polysaccharide/capsular_polysaccharide.png` | Isolate-level capsular-polysaccharide measurements summarized by lineage and, where shown, KL type |
| Gene-expression qPCR | `gene_expression_qpcr/gene_expression_qpcr.png` | Isolate-level normalized expression for the four targets, with technical replicates not plotted as independent isolates |
| A549-cell adhesion | `a549_adhesion/a549_adhesion.png` | MOI 10 and MOI 20 adhesion for the focused isolates, using isolate means as the comparison unit |
| Mouse nasal colonisation | `nasal_colonisation/nasal_colonisation.png` | Longitudinal log10 CFU trajectories or time-point summaries for ESL2.4 and ESL2.5 |
| Systemic-infection survival | `systemic_infection_survival/systemic_infection_survival.png` | Seven-day Kaplan-Meier curves for the stored 18-versus-18 subset, with censoring and log-rank result |
| Transmission electron microscopy | `transmission_electron_microscopy/transmission_electron_microscopy.png` | Representative micrographs and scale bars; no quantitative thickness panel unless source measurements are supplied |
| Nasal secretory IgA | `secretory_iga/secretory_iga.png` | Lineage comparison of sIgA concentrations once the quantitative source table is supplied |

After an image is added, insert it immediately below the corresponding assay description using, for example:

```html
<p align="left">
  <img src="a549_adhesion/a549_adhesion.png" width="650" alt="A549-cell adhesion at MOI 10 and 20">
</p>
```
