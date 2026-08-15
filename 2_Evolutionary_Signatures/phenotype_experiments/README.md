# Phenotype experiments

Phenotype measurements comparing ESL2.4 and ESL2.5 are organized by assay.
Each quantitative analysis has its own input table, derived results,
calculation script and definition of the independent experimental unit.

- `antimicrobial_susceptibility/`: isolate-level clinical S/I/R calls.
- `capsular_polysaccharide/`: uronic-acid measurements of capsular material.
- `gene_expression_qpcr/`: target-gene Ct values normalized to 16S rRNA.
- `a549_adhesion/`: epithelial-cell adhesion at MOI 10 and 20.
- `nasal_colonisation/`: longitudinal nasal bacterial loads in mice.
- `systemic_infection_survival/`: seven-day mouse-survival analysis.
- `transmission_electron_microscopy/`: negative-stain imaging method.
- `secretory_iga/`: nasal-mucus sIgA measurement method.

The assays compare representative ESL2.4 and ESL2.5 isolates across resistance,
capsule production, gene expression, epithelial adhesion, nasal colonisation
and systemic infection.

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
