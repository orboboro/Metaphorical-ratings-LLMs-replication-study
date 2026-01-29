import os
import pandas as pd
from scipy.stats import spearmanr, pearsonr

from pathlib import Path

human_path = "human_datasets/"
synthetic_path = "synthetic_datasets/"

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

dimensions_map = {
    'MB': ['FAMILIARITY', 'MEANINGFULNESS', 'BODY_RELATEDNESS'],
    'ME': ['FAMILIARITY', 'MEANINGFULNESS', 'DIFFICULTY'],
    'MI': ['PHISICALITY', 'IMAGEABILITY'],
    'MM': ['FAMILIARITY', 'MEANINGFULNESS'],
}

# Funzione per normalizzare i giudizi dati tra 1 e 5 come se fossero tra 1 e 7
def normalize_me(series):
    return 1 + (series - 1) * (6 / 4)

# Lista che conterrà TUTTE le coppie human–synthetic
all_rows = []

# Caricare i dataset umani e sintetici
for ds_name in ['MB', 'ME', 'MI', 'MM']:

    hfile = os.path.join(human_path, human_files[ds_name])
    human_df = pd.read_csv(hfile, decimal=',')

    sfile = os.path.join(synthetic_path, synthetic_files[ds_name])
    synth_df = pd.read_csv(sfile, decimal=',')

    # Normalizzazione ME
    if ds_name == 'ME':
        for col in human_df.columns:
            if col.endswith('_human'):
                human_df[col] = normalize_me(pd.to_numeric(human_df[col], errors='coerce'))
        for col in synth_df.columns:
            if col.endswith('_synthetic'):
                synth_df[col] = normalize_me(pd.to_numeric(synth_df[col], errors='coerce'))

    dims = dimensions_map[ds_name]

    for dim in dims:
        human_col = f"{dim}_human"
        synth_col = f"{dim}_synthetic"

        human_vals = human_df[['metaphor', human_col]].copy()
        human_vals.rename(columns={human_col: 'human'}, inplace=True)
        human_vals['metaphor'] = human_vals['metaphor'].astype(str).str.strip()
        human_vals['human'] = pd.to_numeric(human_vals['human'], errors='coerce')

        synth_vals = synth_df[['metaphor', synth_col]].copy()
        synth_vals.rename(columns={synth_col: 'synthetic'}, inplace=True)
        synth_vals['metaphor'] = synth_vals['metaphor'].astype(str).str.strip()
        synth_vals['synthetic'] = pd.to_numeric(synth_vals['synthetic'], errors='coerce')

        merged = human_vals.merge(synth_vals, on='metaphor', how='left')

        for _, r in merged.iterrows():
            all_rows.append({
                'dataset': ds_name,
                'dimension': dim,
                'metaphor': r['metaphor'],
                'human': r['human'],
                'synthetic': r['synthetic']
            })

# DataFrame globale
df_all = pd.DataFrame(all_rows)

# Calcolo correlazione Spearman globale
sub = df_all[['human', 'synthetic']].dropna()
s_corr, s_p_value = spearmanr(sub['human'], sub['synthetic'])
p_corr, p_p_value = pearsonr(sub['human'], sub['synthetic'])
n_items = len(sub)

results_global = pd.DataFrame([{
    'n_item': n_items,
    'spearman_corr': s_corr,
    's_p_value': s_p_value,
    'pearson_corr': p_corr,
    'p_p_value': p_p_value
}])

print('\n=== Correlazione generale (senza distinguere per dimensione) ===')
print(results_global)

# Salvataggio risultati
out_dir = '_results'
if not Path(out_dir).exists():
    Path(out_dir).mkdir()

results_global.to_csv(out_dir + '/results_general.csv', index=False)
