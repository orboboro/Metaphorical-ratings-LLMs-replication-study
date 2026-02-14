import pandas as pd
import numpy as np
import os
from scipy.stats import spearmanr, pearsonr
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

# ---------------------------
# UTILITY
# ---------------------------

def fix_commas(df):
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.replace(",", ".", regex=False)
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return df

def spearman_corr_long(df):
    """Restituisce un dataframe tidy con tutte le coppie di colonne e correlazione spearman + p"""
    cols = df.columns
    records = []
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            x = df[cols[i]]
            y = df[cols[j]]
            r, p = spearmanr(x, y)
            records.append({
                "dimension1": cols[i],
                "dimension2": cols[j],
                "spearman_r": r,
                "p_value": p
            })
    return pd.DataFrame(records)

# ---------------------------
# PROCESSO PRINCIPALE
# ---------------------------

all_corrs = []

for dataset in datasets:
    human_df = pd.read_csv(human_files[dataset]).dropna()
    synth_df = pd.read_csv(synthetic_files[dataset]).dropna()

    # rimuovi colonne non numeriche
    human_drop_cols = ["metaphor"]
    synth_drop_cols = ["metaphor", "annotator"]
    human_df = human_df.drop(columns=[c for c in human_drop_cols if c in human_df.columns])
    synth_df = synth_df.drop(columns=[c for c in synth_drop_cols if c in synth_df.columns])

    human_df = fix_commas(human_df)
    synth_df = fix_commas(synth_df)

    # Matrici di correlazione in formato tidy
    human_corr_long = spearman_corr_long(human_df)
    human_corr_long["dataset"] = dataset
    human_corr_long["rater_type"] = "human"

    synth_corr_long = spearman_corr_long(synth_df)
    synth_corr_long["dataset"] = dataset
    synth_corr_long["rater_type"] = "synthetic"

    # concatena
    all_corrs.append(human_corr_long)
    all_corrs.append(synth_corr_long)

# Unisci tutti i dataset
all_corrs_df = pd.concat(all_corrs, ignore_index=True)

# Salva CSV
os.makedirs("_results", exist_ok=True)
all_corrs_df.to_csv("_results/all_spearman_correlations.csv", index=False)

print("Tutte le correlazioni salvate in '_results/all_spearman_correlations.csv'")
