import os
import pandas as pd
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

# normalizzazione ME (1–5 → 1–7)
def normalize_me(series):
    return 1 + (series - 1) * (6 / 4)

def normalize_metaphor(series):
    return (
        series.astype(str)
              .str.strip()
              .str.lower()
              .str.replace(r'\s+', ' ', regex=True)
    )

rows = []

for ds_name in ['MB', 'ME', 'MI', 'MM']:

    hfile = os.path.join(human_path, human_files[ds_name])
    sfile = os.path.join(synthetic_path, synthetic_files[ds_name])

    human_df = pd.read_csv(hfile, decimal=',')
    synth_df = pd.read_csv(sfile, decimal=',')

    # normalizza metafore
    human_df['metaphor'] = normalize_metaphor(human_df['metaphor'])
    synth_df['metaphor'] = normalize_metaphor(synth_df['metaphor'])

    # normalizzazione ME
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

        human_vals = human_df[['metaphor', human_col]].copy()
        human_vals.rename(columns={human_col: 'human'}, inplace=True)
        human_vals['human'] = pd.to_numeric(human_vals['human'], errors='coerce')

        synth_vals = synth_df[['metaphor', synth_col]].copy()
        synth_vals.rename(columns={synth_col: 'synthetic'}, inplace=True)
        synth_vals['synthetic'] = pd.to_numeric(synth_vals['synthetic'], errors='coerce')

        merged = (
            human_vals
            .merge(
                synth_vals,
                on='metaphor',
                how='inner',
                validate='one_to_one'
            )
            .dropna(subset=['human', 'synthetic'])
        )

        for _, r in merged.iterrows():
            rows.append({
                'dimension': dim,
                'metaphor': r['metaphor'],
                'human': r['human'],
                'synthetic': r['synthetic'],
                'abs_diff': abs(r['human'] - r['synthetic'])
            })

# dataframe globale
df = pd.DataFrame(rows)

results = []

for dim, sub_df in df.groupby('dimension'):

    sub_df = sub_df.sort_values('abs_diff')

    # top 3: differenza minima
    best = sub_df.head(3)
    # bottom 3: differenza massima
    worst = sub_df.tail(3)

    for _, r in best.iterrows():
        results.append({
            'dimension': dim,
            'rank_type': 'top_3_min_diff',
            'metaphor': r['metaphor'],
            'human': r['human'],
            'synthetic': r['synthetic'],
            'abs_diff': r['abs_diff']
        })

    for _, r in worst.iterrows():
        results.append({
            'dimension': dim,
            'rank_type': 'bottom_3_max_diff',
            'metaphor': r['metaphor'],
            'human': r['human'],
            'synthetic': r['synthetic'],
            'abs_diff': r['abs_diff']
        })

results_df = pd.DataFrame(results)

print("\n=== Top 3 e Bottom 3 metafore per dimensione (globale) ===")
print(results_df)

out_dir = "_results"
Path(out_dir).mkdir(exist_ok=True)

results_df.to_csv(
    os.path.join(out_dir, "top_bottom_3_metaphors_by_dimension.csv"),
    index=False
)
