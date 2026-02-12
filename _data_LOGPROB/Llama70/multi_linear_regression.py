import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from scipy.stats import norm
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt


human_files = {
    'MB': 'human_datasets/human_MB.csv',
    'ME': 'human_datasets/human_ME.csv',
    'MI': 'human_datasets/human_MI.csv',
    'MM': 'human_datasets/human_MM.csv',
    'BA': 'human_datasets/human_BA.csv'
}

synthetic_files = {
    'MB': 'synthetic_datasets/synthetic_MB.csv',
    'ME': 'synthetic_datasets/synthetic_ME.csv',
    'MI': 'synthetic_datasets/synthetic_MI.csv',
    'MM': 'synthetic_datasets/synthetic_MM.csv',
    'BA': 'synthetic_datasets/synthetic_BA.csv'
}

datasets = ['MB','ME','MI','MM','BA']
kinds =  [human_files, synthetic_files]

def fix_commas(df):
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.replace(",", ".", regex=False)
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return df

def corr_with_pvalues(df):

    cols = df.columns
    n = len(cols)

    corr_mat = pd.DataFrame(index=cols, columns=cols, dtype=float)
    p_mat = pd.DataFrame(index=cols, columns=cols, dtype=float)

    for i in range(n):
        for j in range(n):

            x = df[cols[i]]
            y = df[cols[j]]

            r, p = spearmanr(x, y)

            corr_mat.iloc[i,j] = r
            p_mat.iloc[i,j] = p

    return corr_mat, p_mat

def starify(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""

def build_annot(corr, pval):

    annot = corr.copy().astype(str)

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            r = corr.iloc[i,j]
            p = pval.iloc[i,j]

            annot.iloc[i,j] = f"{r:.2f}{starify(p)}"

    return annot

for kind in kinds:
    for dataset in datasets:
        df = pd.read_csv(kind[dataset]).dropna()
        name_df = ((kind[dataset]).split("/"))[-1][:-4]

        if "synthetic" in kind[dataset]:
            df = df.drop(columns=["metaphor", "annotator"])
        else:
            df = df.drop(columns=["metaphor"])

        df = fix_commas(df)
        corr, pvals = corr_with_pvalues(df)

        print("Correlazioni:")
        print(corr)
        print("\nP-values:")
        print(pvals)

        annot = build_annot(corr, pvals)
        plt.figure(figsize=(8,6))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(
            corr,
            mask=mask,
            annot=annot,
            fmt="",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=.5
        )

        plt.title("Correlation matrix (* p<.05, ** p<.01, *** p<.001)")
        plt.tight_layout()

        plt.savefig("_results/heatmaps/corr_heatmap_starred_" + name_df + ".png", dpi=300)
        plt.close()