# compute_spearman_metaphors.py
import pandas as pd
import numpy as np
import os
import re
from scipy.stats import spearmanr

# Definizione dei percorsi per i dati

human_path = "data/human_datasets/"
synthetic_path = "data/synthetic_datasets/"
raw_path = "data/original_datasets/"

human_data = {
    "MB": human_path + "human_MB.csv",
    "ME": human_path + "human_ME.csv",
    "MI": human_path + "human_MI.csv",
    "MM": human_path + "human_MM.csv",
}

synthetic_data = {
    "MB": synthetic_path + "synthetic_MB_meta-llama-Llama-3.3-70B-Instruct_.csv",
    "ME": synthetic_path + "synthetic_ME_meta-llama-Llama-3.3-70B-Instruct_.csv",
    "MI": synthetic_path + "synthetic_MI_meta-llama-Llama-3.3-70B-Instruct_.csv",
    "MM": synthetic_path + "synthetic_MM_meta-llama-Llama-3.3-70B-Instruct_.csv",
}

raw_data = {
    "MB": raw_path + "raw_MB.csv",
    "ME": raw_path + "raw_ME.csv",
    "MI": raw_path + "raw_MI.csv",
    "MM": raw_path + "raw_MM.csv",
}

# Mappatura dimensioni per dataset (nome colonna humana : nome colonna sintetica)

dataset_dimensions = {
    "MB": {
        "FAMILIARITY_human": "FAMILIARITY",
        "MEANINGFULNESS_human": "MEANINGFULNESS",
        "BODY_RELATEDNESS_human": "BODY_RELATEDNESS",
    },
    "ME": {
        "FAMILIARITY_human": "FAMILIARITY",
        "MEANINGFULNESS_human": "MEANINGFULNESS",
        "DIFFICULTY_human": "DIFFICULTY",
    },
    "MI": {
        "PHISICALITY_human": "PHISICALITY",
        "IMAGEABILITY_human": "IMAGEABILITY",
    },
    "MM": {
        "FAMILIARITY_human": "FAMILIARITY",
        "MEANINGFULNESS_human": "MEANINGFULNESS",
    },
}

def normalize_text(s):
    if pd.isna(s):
        return s
    return re.sub(r'\s+', ' ', str(s).strip())

def to_numeric_italian(x):
    
    if pd.isna(x):
        return np.nan
    
    s = str(x).strip('"').strip("'")
    s = s.replace(',', '.')
    s = s.replace(' ', '')

    return float(s)

def build_used_sets_from_raw(raw_df, dataset_name):

    used = {}
    raw = raw_df.fillna('')
    # Lowercase columns for safer matching
    cols = [c for c in raw.columns]
    # initialize based on dataset
    if dataset_name == "MB":
        used = {"FAMILIARITY": set(), "MEANINGFULNESS": set(), "BODY_RELATEDNESS": set()}
        for _, row in raw.iterrows():
            m = normalize_text(row.get("Metaphor"))
            if row.get("Bambini et al. (2013)") == "Y":
                used["FAMILIARITY"].add(m); used["MEANINGFULNESS"].add(m)
            if row.get("Canal et al. (2022)") == "Y":
                used["FAMILIARITY"].add(m)
            if row.get("Bambini et al. (2024)") == "Y":
                used["FAMILIARITY"].add(m)
            if row.get("Lago et al. (2024)") == "Y":
                used["FAMILIARITY"].add(m)

    elif dataset_name == "ME":

        used = {"FAMILIARITY": set(), "MEANINGFULNESS": set(), "DIFFICULTY": set()}
        for _, row in raw.iterrows():
            m = normalize_text(row.get("Metaphor"))
            if row.get("Bambini et al. (2013)") == "Y":
                used["FAMILIARITY"].add(m); used["MEANINGFULNESS"].add(m); used["DIFFICULTY"].add(m)
            if row.get("Canal et al. (2022)") == "Y":
                used["FAMILIARITY"].add(m)
            if row.get("Bambini et al. (2024)") == "Y":
                used["FAMILIARITY"].add(m); used["DIFFICULTY"].add(m)
            if row.get("Lago et al. (2024)", "") == "Y":
                used["FAMILIARITY"].add(m)

    elif dataset_name == "MI":

        used = {"PHISICALITY": set(), "IMAGEABILITY": set()}
        for _, row in raw.iterrows():
            m = normalize_text(row.get("Metaphor"))
            if row.get("Canal et al. (2022)", "") == "Y":
                used["PHISICALITY"].add(m)
            if row.get("Bambini et al. (2024)", "") == "Y":
                used["IMAGEABILITY"].add(m)

    elif dataset_name == "MM":

        used = {"FAMILIARITY": set(), "MEANINGFULNESS": set()}
        for _, row in raw.iterrows():
            m = normalize_text(row.get("Metaphor"))
            if row.get("Bambini et al. (2013)", "") == "Y":
                used["FAMILIARITY"].add(m); used["MEANINGFULNESS"].add(m)
            if row.get("Canal et al. (2022)", "") == "Y":
                used["FAMILIARITY"].add(m)
            if row.get("Bambini et al. (2024)", "") == "Y":
                used["FAMILIARITY"].add(m)
            if row.get("Lago et al. (2024)", "") == "Y":
                used["FAMILIARITY"].add(m)

    return used

def compute_per_dataset(dataset_name, human_path, synth_path, raw_path=None):

    human_df = pd.read_csv(human_path)
    synth_df = pd.read_csv(synth_path)
    raw_df = pd.read_csv(raw_path) if raw_path else None

    for df in [human_df, synth_df]:
        if "Metaphor" not in df.columns and "metaphor" in df.columns:
            df.rename(columns={"metaphor": "Metaphor"}, inplace=True)

    human_df["Metaphor_norm"] = human_df["Metaphor"].apply(normalize_text)
    synth_df["Metaphor_norm"] = synth_df["metaphor"].apply(normalize_text) if "metaphor" in synth_df.columns else synth_df["metaphor"].apply(normalize_text) if "metaphor" in synth_df.columns else synth_df["metaphor".capitalize()].apply(normalize_text)

    # Build used sets for exclusions per-dimension
    used_sets = build_used_sets_from_raw(raw_df, dataset_name)

    results = []

    dims = dataset_dimensions.get(dataset_name, {})
    for human_col, dim_prefix in dims.items():
        # create human series mapping Metaphor -> numeric
        if human_col not in human_df.columns:
            # maybe human columns are named without _human suffix; try to find column that contains prefix
            matches = [c for c in human_df.columns if c.upper().startswith(dim_prefix)]
            if matches:
                human_col = matches[0]
            else:
                print(f"Colonna umana {human_col} non trovata in {human_path}; salto {dataset_name} {dim_prefix}.")
                continue

        human_series = human_df[["Metaphor_norm", human_col]].copy()
        human_series[human_col] = human_series[human_col].apply(to_numeric_italian)

        # Build synthetic pivot: one row per metaphor, columns = annotator ids
        # ensure annotator column exists
        if "annotator" not in synth_df.columns:
            # try lowercase
            if "Annotator" in synth_df.columns:
                synth_df.rename(columns={"Annotator": "annotator"}, inplace=True)
        # infer synthetic column name
        synth_col = f"{dim_prefix}_synthetic"
        if synth_col not in synth_df.columns:
            # try variants
            candidates = [c for c in synth_df.columns if c.upper().startswith(dim_prefix)]
            if candidates:
                synth_col = candidates[0]
            else:
                print(f"Colonna sintetica per {dim_prefix} non trovata in {synth_path}; salto.")
                continue

        # normalize synthetic numeric
        synth_df[synth_col + "_num"] = synth_df[synth_col].apply(to_numeric_italian)
        # normalize annotator
        if "annotator" not in synth_df.columns:
            # assume each row is a single annotator file (unlikely) -> treat all as annotator 1
            synth_df["annotator"] = 1
        # ensure Metaphor_norm exists in synth_df
        if "Metaphor_norm" not in synth_df.columns:
            if "metaphor" in synth_df.columns:
                synth_df["Metaphor_norm"] = synth_df["metaphor"].apply(normalize_text)
            elif "Metaphor" in synth_df.columns:
                synth_df["Metaphor_norm"] = synth_df["Metaphor"].apply(normalize_text)
            else:
                print(f"Nessuna colonna 'metaphor' trovata in {synth_path}; salto.")
                continue

        # pivot to get annotator columns
        pivot = synth_df.pivot_table(index="Metaphor_norm", columns="annotator", values=synth_col + "_num", aggfunc='mean')
        # sort annotator columns numeric order if possible
        pivot = pivot.reindex(sorted(pivot.columns, key=lambda x: int(x) if str(x).isdigit() else x), axis=1)

        # compute mean across annotators
        pivot["mean_synthetic"] = pivot.mean(axis=1, skipna=True)

        # Merge human values and synthetic values (annotator1 and mean)
        merged = human_series.merge(pivot.reset_index(), how='inner', left_on="Metaphor_norm", right_on="Metaphor_norm")
        # exclude used metaphors for this dimension if we have used sets
        if used_sets and dim_prefix in used_sets:
            used = used_sets[dim_prefix]
            if used:
                merged = merged[~merged["Metaphor_norm"].isin(used)]

        # keep only rows where human value not null
        merged = merged[~merged[human_col].isna()].copy()
        # ensure at least 3 pairs for Spearman (scipy will warn otherwise)
        # annotator 1 may be in column 1 of pivot (depends on labeling); try to pick annotator '1' if present
        annot1_col = None
        if 1 in pivot.columns:
            annot1_col = 1
        else:
            # try first annotator column
            if len(pivot.columns) > 0:
                annot1_col = pivot.columns[0]

        if annot1_col is None or annot1_col not in merged.columns:
            # No annotator-specific column found -> we can't compute annotator1 correlation; only mean
            annot1_available = False
        else:
            annot1_available = True

        # Compute correlations
        # human vs annotator1
        if annot1_available:
            x = merged[human_col].astype(float)
            y = merged[annot1_col].astype(float)
            # drop nans in pair
            mask = ~(x.isna() | y.isna())
            if mask.sum() >= 2:
                r1, p1 = spearmanr(x[mask], y[mask])
            else:
                r1, p1 = np.nan, np.nan
        else:
            r1, p1 = np.nan, np.nan

        # human vs mean
        if "mean_synthetic" in merged.columns:
            x = merged[human_col].astype(float)
            y = merged["mean_synthetic"].astype(float)
            mask = ~(x.isna() | y.isna())
            if mask.sum() >= 2:
                rmean, pmean = spearmanr(x[mask], y[mask])
            else:
                rmean, pmean = np.nan, np.nan
        else:
            rmean, pmean = np.nan, np.nan

        n_pairs_annot1 = int(mask.sum()) if annot1_available else 0
        # Note: recompute n for mean too
        if "mean_synthetic" in merged.columns:
            n_pairs_mean = int((~(merged[human_col].isna() | merged["mean_synthetic"].isna())).sum())
        else:
            n_pairs_mean = 0

        results.append({
            "dataset": dataset_name,
            "dimension": dim_prefix,
            "human_col": human_col,
            "synth_col": synth_col,
            "n_annot1": n_pairs_annot1,
            "spearman_r_annot1": r1,
            "pval_annot1": p1,
            "n_mean": n_pairs_mean,
            "spearman_r_mean": rmean,
            "pval_mean": pmean,
        })

    return results

def main():
    all_results = []
    for ds in human_data.keys():
        human_path = human_data.get(ds)
        synth_path = synthetic_data.get(ds)
        raw_path = raw_data.get(ds) if ds in raw_data else None
        r = compute_per_dataset(ds, human_path, synth_path, raw_path)
        if r:
            all_results.extend(r)

    if not all_results:
        print("Nessun risultato calcolato (verifica i nomi dei file).")
        return

    results_df = pd.DataFrame(all_results)
    # salva CSV dei risultati
    results_df.to_csv("spearman_results.csv", index=False)
    # stampa tabella sintetica
    print("\nRisultati calcolati (salvati in spearman_results.csv):\n")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()
