#!/usr/bin/env python

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt

file_path = "degree.txt"
sep = "," if file_path.endswith(".csv") else "\t"
use_lowess = True
lowess_frac = 1
window = 3
x_max = 5


color24, color25 = "#1f77b4", "#d62728"  # esl2.4 蓝, esl2.5 红
alpha_fill = 0.25


df = pd.read_csv(file_path, sep=sep)
df.columns = [c.strip() for c in df.columns]
x = df["Degree"].astype(float)
y24 = df["esl2.4"].astype(float)
y25 = df["esl2.5"].astype(float)



def corr_row(name, func, _x, _y):
    r, p = func(_x, _y)
    return {"target": name.split()[0], "method": name.split()[1], "coef": round(r, 3), "p_value": round(p, 3)}


stats = []
for y, lab in [(y24, "esl2.4"), (y25, "esl2.5")]:
    stats += [corr_row(f"{lab} Pearson", pearsonr, x, y),
              corr_row(f"{lab} Spearman", spearmanr, x, y),
              corr_row(f"{lab} Kendall", kendalltau, x, y)]

print("\n=== Correlation summary ===")
print(pd.DataFrame(stats).to_string(index=False))


def lin_fit_ci(_x, _y):
    X = sm.add_constant(_x)
    model = sm.OLS(_y, X).fit()
    frame = model.get_prediction(X).summary_frame(alpha=0.05)  # mean_ci_lower / upper
    return model, frame


fig, ax = plt.subplots(figsize=(4, 4))


def add_series(_x, _y, color, label):
    ax.scatter(_x, _y, s=45, color=color, alpha=0.8, edgecolors="none", label=f"{label} points")

    if use_lowess:
        z = lowess(_y, _x, frac=lowess_frac, return_sorted=True)
        ax.plot(z[:, 0], z[:, 1], lw=2, color=color, label=f"{label} LOWESS")
    else:
        y_ma = pd.Series(_y).rolling(window=window, center=True).mean()
        ax.plot(_x, y_ma, lw=2, color=color, label=f"{label} MA({window})")

    _, frame = lin_fit_ci(_x, _y)
    order = np.argsort(_x)
    xs = _x.iloc[order]
    y_pred = frame["mean"].iloc[order]
    ci_lo = frame["mean_ci_lower"].iloc[order]
    ci_up = frame["mean_ci_upper"].iloc[order]
    ax.plot(xs, y_pred, ls="--", lw=2, color=color, label=f"{label} linear")
    ax.fill_between(xs, ci_lo, ci_up, color=color, alpha=alpha_fill)


add_series(x, y24, color24, "esl2.4")
add_series(x, y25, color25, "esl2.5")

ax.set_xlabel("Degree")
ax.set_ylabel("Value")
ax.set_title("Degree vs esl2.4 & esl2.5")
ax.set_xlim(0, x_max)
ax.legend(ncol=2, fontsize=9)
plt.tight_layout()
plt.show()
