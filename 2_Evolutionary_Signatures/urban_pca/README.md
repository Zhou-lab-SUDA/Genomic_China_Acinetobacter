# Urban principal component analysis

Fig. 3d–e and Extended Data Fig. 6 use PCA of 15 transformed variables in 60 complete-case cities. Input names identify the supplied transformations and the script centres and divides by the sample SD before SVD. PC1 and PC2 explain 55.9% and 16.1% of variance in the urban inputs. ESL2.5 prevalence is joined after fitting for display. Supplementary correlations such as total GDP and population are projections on the fitted axes. A positive PC2 resource loading refers to higher values of the corresponding resource variable under the documented sign convention.

```sh
python code/analysis.py --out /tmp/ab-urban-pca --plot
```

The results include the standardized matrix, scores, eigenvector loadings and active-variable correlations. [Fig3de_plot_values.tsv](data/Fig3de_plot_values.tsv) contains the display values for Fig. 3d–e. The computed PCA results and the projection vectors used for Extended Data Fig. 6 are archived separately. Fig. 3f uses provincial GDP and outpatient-to-inpatient ratios from [TrueValueOI.xlsx](../TrueValueOI.xlsx).
