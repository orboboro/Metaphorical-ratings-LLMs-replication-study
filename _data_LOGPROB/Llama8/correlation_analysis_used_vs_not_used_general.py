import os
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path

# =====================
# PATHS E FILE
# =====================

human_path = "human_datasets/"
synthetic_path = "synthetic_datasets/"
raw_path = "original_datasets/"

human_files = {
    'MB': 'human_MB.csv',
    'ME': 'human_ME.csv',
    'MI': 'human_MI.csv',
    'MM': 'human_MM.csv',
}

synthetic_files = {
    'MB': 'synthetic_MB.csv',
    'ME': 'synthetic_ME.csv',
    'MI': 'synthetic_MI.csv',
    'MM': 'synthetic_MM.csv',
}

raw_files = {
    'MB': 'raw_MB.csv',
    'ME': 'raw_ME.csv',
    'MI': 'raw_MI.csv',
    'MM': 'raw_MM.csv',
}

dimensions_map = {
    'MB': ['FAMILIARITY', 'MEANINGFULNESS', 'BODY_RELATEDNESS'],
    'ME': ['FAMILIARITY', 'MEANINGFULNESS', 'DIFFICULTY'],
    'MI': ['PHISICALITY', 'IMAGEABILITY'],
    'MM': ['FAMILIARITY', 'MEANINGFULNESS'],
}

# =====================
# NORMALIZZAZIONE ME
# =====================

def normalize_me(series):
    return 1 + (series - 1) * (6 / 4)

# =====================
# METAFRE USED (IDENTICO AL CODICE ORIGINALE)
# =====================

used_metaphors = {dim: set() for dims in dimensions_map.values() for dim in dims}

for name, fname in raw_files.items():
    path = os.path.join(raw_path, fname)
    raw_df = pd.read_csv(path)

    for _, row in raw_df.iterrows():

        if row.get("Bambini et al. (2013)") == "Y":
            used_metaphors["FAMILIARITY"].add(row["Metaphor"])
            used_metaphors["MEANINGFULNESS"].add(row["Metaphor"])
            used_metaphors["DIFFICULTY"].add(row["Metaphor"])

        if row.get("Canal et al. (2022)") == "Y":
            used_metaphors["FAMILIARITY"].add(row["Metaphor"])
            used_metaphors["PHISICALITY"].add(row["Metaphor"])

        if row.get("Lago et al. (2024)") == "Y":
            used_metaphors["FAMILIARITY"].add(row["Metaphor"])

# insieme globale di tutte le metafore used (indipendente dalla dimensione)
used_metaphors_global = set().union(*used_metaphors.values())

# =====================
# RACCOLTA DATI (SENZA DISTINZIONE PER DIMENSIONE)
# =====================

all_rows = []

for ds_name in ['MB', 'ME', 'MI', 'MM']:

    human_df = pd.read_csv(os.path.join(human_path, human_files[ds_name]), decimal=',')
    synth_df = pd.read_csv(os.path.join(synthetic_path, synthetic_files[ds_name]), decimal=',')

    # normalizzazione solo per ME
    if ds_name == 'ME':
        for col in human_df.columns:
            if col.endswith('_human'):
                human_df[col] = normalize_me(pd.to_numeric(human_df[col], errors='coerce'))
        for col in synth_df.columns:
            if col.endswith('_synthetic'):
                synth_df[col] = normalize_me(pd.to_numeric(synth_df[col], errors='coerce'))

    for dim in dimensions_map[ds_name]:
        human_col = f"{dim}_human"
        synth_col = f"{dim}_synthetic"

        if human_col not in human_df.columns or synth_col not in synth_df.columns:
            continue

        h = human_df[['metaphor', human_col]].copy()
        s = synth_df[['metaphor', synth_col]].copy()

        h.rename(columns={human_col: 'human'}, inplace=True)
        s.rename(columns={synth_col: 'synthetic'}, inplace=True)

        h['metaphor'] = h['metaphor'].astype(str).str.strip()
        s['metaphor'] = s['metaphor'].astype(str).str.strip()

        h['human'] = pd.to_numeric(h['human'], errors='coerce')
        s['synthetic'] = pd.to_numeric(s['synthetic'], errors='coerce')

        merged = h.merge(s, on='metaphor', how='left')

        for _, r in merged.iterrows():
            all_rows.append({
                'metaphor': r['metaphor'],
                'human': r['human'],
                'synthetic': r['synthetic'],
                'used': r['metaphor'] in used_metaphors_global
            })

df_all = pd.DataFrame(all_rows)

# =====================
# CORRELAZIONI USED VS NOT USED (GLOBALI)
# =====================

results = []

for used_flag, group in df_all.groupby('used'):
    label = 'used' if used_flag else 'not_used'
    sub = group[['human', 'synthetic']].dropna()

    corr, p = spearmanr(sub['human'], sub['synthetic'])
    n = len(sub)

    results.append({
        'group': label,
        'n_item': n,
        'corr': corr,
        'p_value': p
    })

results_df = pd.DataFrame(results).set_index('group')

# =====================
# DIFFERENZA PERCENTUALE
# =====================

if 'used' in results_df.index and 'not_used' in results_df.index:
    delta = results_df.loc['not_used', 'corr'] - results_df.loc['used', 'corr']
    pct_change = (delta / abs(results_df.loc['used', 'corr'])) * 100 if results_df.loc['used', 'corr'] != 0 else float('nan')
else:
    pct_change = float('nan')

summary_df = pd.DataFrame([{
    'used_corr': results_df.loc['used', 'corr'] if 'used' in results_df.index else float('nan'),
    'used_p_value': results_df.loc['used', 'p_value'] if 'used' in results_df.index else float('nan'),
    'not_used_corr': results_df.loc['not_used', 'corr'] if 'not_used' in results_df.index else float('nan'),
    'not_used_p_value': results_df.loc['not_used', 'p_value'] if 'not_used' in results_df.index else float('nan'),
    'pct_change': pct_change
}])

# =====================
# OUTPUT
# =====================

print("\n=== Correlazioni globali (tutte le dimensioni insieme) ===")
print(results_df)

print("\n=== Differenza used vs not_used ===")
print(summary_df)

out_dir = "_results"
Path(out_dir).mkdir(exist_ok=True)

summary_df.to_csv(out_dir + "/results_used_vs_not_used_general.csv", index=False)
