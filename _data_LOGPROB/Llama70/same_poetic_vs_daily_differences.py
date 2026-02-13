import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from pathlib import Path

HUMAN_DIR = Path("human_datasets")
SYN_DIR = Path("synthetic_datasets")

############################
# FUNZIONI UTILI
############################

def load_human_csv(path):
    df = pd.read_csv(path)
    # sostituisce virgola decimale con punto
    for col in df.columns:
        if col != "metaphor":
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_synth_csv(path):
    return pd.read_csv(path)


def get_available_cols(df, cols):
    return [c for c in cols if c in df.columns]


############################
# CARICAMENTO HUMAN
############################

human_BA = load_human_csv(HUMAN_DIR / "human_BA.csv")
human_MB = load_human_csv(HUMAN_DIR / "human_MB.csv")
human_ME = load_human_csv(HUMAN_DIR / "human_ME.csv")
human_MM = load_human_csv(HUMAN_DIR / "human_MM.csv")

human_other = pd.concat([human_MB, human_ME, human_MM], ignore_index=True)

target_cols_human = [
    "FAMILIARITY_human",
    "MEANINGFULNESS_human",
    "DIFFICULTY_human"
]

############################
# CARICAMENTO SYNTHETIC
############################

syn_BA = load_synth_csv(SYN_DIR / "synthetic_BA.csv")
syn_MB = load_synth_csv(SYN_DIR / "synthetic_MB.csv")
syn_ME = load_synth_csv(SYN_DIR / "synthetic_ME.csv")
syn_MM = load_synth_csv(SYN_DIR / "synthetic_MM.csv")

syn_other = pd.concat([syn_MB, syn_ME, syn_MM], ignore_index=True)

target_cols_syn = [
    "FAMILIARITY_synthetic",
    "MEANINGFULNESS_synthetic",
    "DIFFICULTY_synthetic"
]

############################
# ANALISI
############################

results = []

def analyze_group(dfA, dfB, cols, label):
    for col in cols:
        if col not in dfA.columns or col not in dfB.columns:
            continue

        A_vals = dfA[col].dropna()
        B_vals = dfB[col].dropna()

        mean_A = A_vals.mean()
        mean_B = B_vals.mean()
        diff = mean_A - mean_B

        t, p = ttest_ind(A_vals, B_vals, equal_var=False)

        results.append({
            "group": label,
            "metric": col,
            "mean_BA": mean_A,
            "mean_MB_ME_MM": mean_B,
            "difference": diff,
            "ttest_t": t,
            "ttest_p": p,
            "significant_0.05": p < 0.05
        })


analyze_group(human_BA, human_other, target_cols_human, "human")
analyze_group(syn_BA, syn_other, target_cols_syn, "synthetic")

############################
# SALVATAGGIO
############################

res_df = pd.DataFrame(results)
res_df.to_csv("metaphor_rating_comparison.csv", index=False)

print("\nRisultati salvati in metaphor_rating_comparison.csv")
print(res_df)
