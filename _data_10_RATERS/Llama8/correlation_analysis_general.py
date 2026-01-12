import os
import pandas as pd
from scipy.stats import spearmanr
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

def normalize_me(series):
    return 1 + (series - 1) * (6 / 4)

def normalize_metaphor(series):
    return series.astype(str).str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)

all_rows = []

# Carica tutti i dataset e aggrega
for ds_name in ['MB', 'ME', 'MI', 'MM']:

    human_df = pd.read_csv(os.path.join(human_path, human_files[ds_name]), decimal=',')
    synth_df = pd.read_csv(os.path.join(synthetic_path, synthetic_files[ds_name]), decimal=',')

    human_df['metaphor'] = normalize_metaphor(human_df['metaphor'])
    synth_df['metaphor'] = normalize_metaphor(synth_df['metaphor'])

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
        human_vals['human'] = pd.to_numeric(human_vals['human'], errors='coerce')

        synth_vals = synth_df[['annotator', 'metaphor', synth_col]].copy()
        synth_vals[synth_col] = pd.to_numeric(synth_vals[synth_col], errors='coerce')

        # media dei sintetici per metafora
        synth_mean = synth_vals.groupby('metaphor')[synth_col].mean().reset_index()
        synth_mean.rename(columns={synth_col:'synthetic'}, inplace=True)

        merged = human_vals.merge(synth_mean, on='metaphor', how='inner', validate='one_to_one')
        merged.dropna(subset=['human', 'synthetic'], inplace=True)

        for _, r in merged.iterrows():
            all_rows.append({
                'dataset': ds_name,
                'metaphor': r['metaphor'],
                'human': r['human'],
                'synthetic': r['synthetic']
            })

# dataframe globale
df_all = pd.DataFrame(all_rows)

# correlazione generale
sub = df_all[['human','synthetic']].dropna()
corr, p_value = spearmanr(sub['human'], sub['synthetic'])
n_items = len(sub)

results_df = pd.DataFrame([{
    'n_item': n_items,
    'spearman_corr': corr,
    'p_value': p_value
}])

print('\n=== Correlazione generale tra giudizi umani e sintetici ===')
print(results_df)

# salvataggio CSV
out_dir = '_results'
Path(out_dir).mkdir(exist_ok=True)
results_df.to_csv(os.path.join(out_dir, 'results_general.csv'), index=False)
