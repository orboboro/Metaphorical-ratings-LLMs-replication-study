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

# ---------------------------
# MANTEL TEST
# ---------------------------

def mantel_test(corr_human, corr_synth, n_perm=1000, seed=42):
    np.random.seed(seed)
    # estrai off-diagonal
    idx = ~np.eye(corr_human.shape[0], dtype=bool)
    human_off = corr_human.values[idx]
    synth_off = corr_synth.values[idx]

    # correlazione osservata
    r_obs, _ = pearsonr(human_off, synth_off)

    # permutazioni
    perm_r = []
    for _ in range(n_perm):
        synth_perm = np.random.permutation(synth_off)
        r, _ = pearsonr(human_off, synth_perm)
        perm_r.append(r)

    perm_r = np.array(perm_r)
    # p-value: proporzione di permutazioni >= osservato
    p_val = (np.sum(np.abs(perm_r) >= np.abs(r_obs)) + 1) / (n_perm + 1)
    return r_obs, p_val

# ---------------------------
# PROCESSO PRINCIPALE
# ---------------------------

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

    # matrice di correlazione
    corr_human, _ = corr_with_pvalues(human_df)
    corr_synth, _ = corr_with_pvalues(synth_df)

    # Mantel test
    r_mantel, p_mantel = mantel_test(corr_human, corr_synth, n_perm=1000)

    print(f"Dataset {dataset}: Mantel r = {r_mantel:.3f}, p = {p_mantel:.4f}")

    # Heatmap con annotazioni classiche
    annot = build_annot(corr_human, corr_human*0 + 1)  # p dummy solo per annot
    plt.figure(figsize=(8,6))
    mask = np.triu(np.ones_like(corr_human, dtype=bool), k=1)
    sns.heatmap(corr_human, mask=mask, annot=annot, fmt="", cmap="coolwarm",
                vmin=-1, vmax=1, square=True, linewidths=.5)
    plt.title(f"{dataset} Human Correlations\nMantel test with Pearson vs Synthetic: {r_mantel:.2f}, p={p_mantel:.3f}")
    plt.tight_layout()
    os.makedirs("_results/heatmaps", exist_ok=True)
    plt.savefig(f"_results/heatmaps/corr_heatmap_mantel_{dataset}.png", dpi=300)
    plt.close()
