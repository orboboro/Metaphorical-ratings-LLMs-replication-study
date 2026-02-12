import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

human_path = "human_datasets/"
synthetic_path = "synthetic_datasets/"
out_path = "_results/boxplots"

os.makedirs(out_path, exist_ok=True)

# -----------------------
# FILE LISTE ESPLICITE
# -----------------------

human_files = [
    "human_BA.csv",
    "human_MB.csv",
    "human_ME.csv",
    "human_MI.csv",
    "human_MM.csv"
]

synthetic_files = [
    "synthetic_BA.csv",
    "synthetic_MB.csv",
    "synthetic_ME.csv",
    "synthetic_MI.csv",
    "synthetic_MM.csv"
]

# -----------------------
# LETTURA HUMAN
# -----------------------

def read_human_csv(path):
    df = pd.read_csv(path)

    for col in df.columns:
        if col != "metaphor":
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.columns = [c.replace("_human", "") for c in df.columns]
    return df

# -----------------------
# LETTURA SYNTHETIC
# -----------------------

def read_synth_csv(path):
    df = pd.read_csv(path)

    if "annotator" in df.columns:
        df = df.drop(columns=["annotator"])

    for col in df.columns:
        if col != "metaphor":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.columns = [c.replace("_synthetic", "") for c in df.columns]
    return df

# -----------------------
# CARICAMENTO
# -----------------------

human_dfs = []
for fname in human_files:
    full = os.path.join(human_path, fname)
    human_dfs.append(read_human_csv(full))

synthetic_dfs = []
for fname in synthetic_files:
    full = os.path.join(synthetic_path, fname)
    synthetic_dfs.append(read_synth_csv(full))

human_all = pd.concat(human_dfs, ignore_index=True)
synthetic_all = pd.concat(synthetic_dfs, ignore_index=True)

# -----------------------
# MERGE
# -----------------------

merged = pd.merge(
    human_all,
    synthetic_all,
    on="metaphor",
    how="outer",
    suffixes=("_human", "_synthetic")
)

# -----------------------
# DIMENSIONI
# -----------------------

dimensions = sorted({
    c.replace("_human","")
    for c in merged.columns
    if c.endswith("_human")
})

print("Dimensioni:", dimensions)

# -----------------------
# FUNZIONE F-TEST
# -----------------------

def f_test(x, y):
    x = np.array(x)
    y = np.array(y)
    # Evita divisione per zero se varianza nulla
    if np.var(y, ddof=1) == 0:
        return np.nan, np.nan
    f_stat = np.var(x, ddof=1)/np.var(y, ddof=1)
    dfn = x.size-1
    dfd = y.size-1
    p_val = 2 * min(scipy.stats.f.cdf(f_stat, dfn, dfd), 1-scipy.stats.f.cdf(f_stat, dfn, dfd))
    return f_stat, p_val

# -----------------------
# STD E F-TEST
# -----------------------

rows = []

for dim in dimensions:
    hcol = dim + "_human"
    scol = dim + "_synthetic"

    h_vals = merged[hcol].dropna() if hcol in merged.columns else pd.Series(dtype=float)
    s_vals = merged[scol].dropna() if scol in merged.columns else pd.Series(dtype=float)

    n_h = len(h_vals)
    n_s = len(s_vals)

    std_h = h_vals.std() if n_h > 1 else np.nan
    std_s = s_vals.std() if n_s > 1 else np.nan

    f_stat, p_val = (np.nan, np.nan)
    if n_h > 1 and n_s > 1:
        f_stat, p_val = f_test(h_vals, s_vals)

    rows.append([
        dim,
        n_h,
        n_s,
        std_h,
        std_s,
        f_stat,
        p_val
    ])

std_df = pd.DataFrame(
    rows,
    columns=[
        "dimension",
        "n_human",
        "n_synthetic",
        "std_human",
        "std_synthetic",
        "F_stat",
        "p_value"
    ]
)
std_df.to_csv("_results/std_and_f_test_summary.csv", index=False)

print(std_df)

# -----------------------
# BOXPLOT
# -----------------------

for dim in dimensions:

    hcol = dim + "_human"
    scol = dim + "_synthetic"

    data = []
    labels = []

    if hcol in merged.columns:
        vals = pd.to_numeric(merged[hcol], errors="coerce").dropna()
        if len(vals) > 0:
            data.append(vals)
            labels.append("Human")

    if scol in merged.columns:
        vals = pd.to_numeric(merged[scol], errors="coerce").dropna()
        if len(vals) > 0:
            data.append(vals)
            labels.append("Synthetic")

    if len(data) == 0:
        continue

    plt.figure()
    plt.boxplot(data, labels=labels)
    plt.title(dim)
    plt.ylabel("Rating")
    plt.savefig(f"{out_path}/{dim}_boxplot.png", bbox_inches="tight")
    plt.close()

print("Boxplot salvati in", out_path)
