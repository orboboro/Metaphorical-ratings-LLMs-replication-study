import os
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path

human_path = "human_datasets/"
synthetic_path = "synthetic_datasets/"

human_files = {
    'MB': 'human_MB.csv',
    'MI': 'human_MI.csv',
}

synthetic_files = {
    'MB': 'synthetic_MB.csv',
    'MI': 'synthetic_MI.csv',
}

dimensions_map = {
    'MB': ['FAMILIARITY', 'MEANINGFULNESS', 'BODY_RELATEDNESS'],
    'MI': ['PHISICALITY', 'IMAGEABILITY'],
}

# Funzione per normalizzare i giudizi dati tra 1 e 5 come se fossero tra 1 e 7
def normalize_me(series):
    return 1 + (series - 1) * (6 / 4)

results = []

for ds_name in ['MB', 'MI']:

    hfile = os.path.join(human_path, human_files[ds_name])
    sfile = os.path.join(synthetic_path, synthetic_files[ds_name])

    human_df = pd.read_csv(hfile, decimal=',')
    synth_df = pd.read_csv(sfile, decimal=',')

    dims = dimensions_map[ds_name]

    # dimensione di split
    if ds_name == 'MB':
        split_dim = 'BODY_RELATEDNESS'
    else:  # MI
        split_dim = 'PHISICALITY'

    split_col = f"{split_dim}_human"

    # tabella per identificare le metafore >4 e <4
    split_df = human_df[['metaphor', split_col]].copy()
    split_df[split_col] = pd.to_numeric(split_df[split_col], errors='coerce')

    high_metaphors = set(split_df.loc[split_df[split_col] > 4.5, 'metaphor'])
    low_metaphors = set(split_df.loc[split_df[split_col] < 3.5, 'metaphor'])

    rows_high = []
    rows_low = []

    human_col = f"{split_dim}_human"
    synth_col = f"{split_dim}_synthetic"

    human_vals = human_df[['metaphor', human_col]].copy()
    human_vals.rename(columns={human_col: 'human'}, inplace=True)
    human_vals['human'] = pd.to_numeric(human_vals['human'], errors='coerce')

    synth_vals = synth_df[['metaphor', synth_col]].copy()
    synth_vals.rename(columns={synth_col: 'synthetic'}, inplace=True)
    synth_vals['synthetic'] = pd.to_numeric(synth_vals['synthetic'], errors='coerce')

    merged = human_vals.merge(
        synth_vals,
        on='metaphor',
        how='inner',
        validate='one_to_one'
    ).dropna(subset=['human', 'synthetic'])

    for _, r in merged.iterrows():
        if r['metaphor'] in high_metaphors:
            rows_high.append(r)
        elif r['metaphor'] in low_metaphors:
            rows_low.append(r)

    # correlazioni
    for label, rows in [('high', rows_high), ('low', rows_low)]:
        df = pd.DataFrame(rows)
        n = len(df)

        if n >= 2:
            corr, p_value = spearmanr(df['human'], df['synthetic'])
        else:
            corr, p_value = float('nan'), float('nan')

        results.append({
            'study': ds_name,
            'group': f'{split_dim}_{label}',
            'n_item': n,
            'spearman_corr': corr,
            'p_value': p_value
        })

results_df = pd.DataFrame(results)

print('\n=== Correlazioni per MB e MI more_phisical vs less_phisical ===')
print(results_df)

out_dir = '_results'
if not Path(out_dir).exists():
    Path(out_dir).mkdir()

results_df.to_csv(
    out_dir + '/results_embodied.csv',
    index=False
)