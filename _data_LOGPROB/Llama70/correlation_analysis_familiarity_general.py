import os
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path

human_path = "human_datasets/"
synthetic_path = "synthetic_datasets/"

human_files = {
    'MB': 'human_MB.csv',
    'ME': 'human_ME.csv',
    'MM': 'human_MM.csv',
    'BA': 'human_BA.csv'
}

synthetic_files = {
    'MB': 'synthetic_MB.csv',
    'ME': 'synthetic_ME.csv',
    'MM': 'synthetic_MM.csv',
    'BA': 'synthetic_BA.csv'
}

dimensions_map = {
    'MB': ['FAMILIARITY', 'MEANINGFULNESS', 'BODY_RELATEDNESS'],
    'ME': ['FAMILIARITY', 'MEANINGFULNESS', 'DIFFICULTY'],
    'MM': ['FAMILIARITY', 'MEANINGFULNESS'],
    'BA': ['FAMILIARITY', 'MEANINGFULNESS', 'DIFFICULTY']
}

# -------------------------
# accumulatori globali
# -------------------------

high_human_all = []
high_synth_all = []

low_human_all = []
low_synth_all = []

# -------------------------
# loop dataset
# -------------------------

for ds_name in ['MB', 'MM', 'ME', 'BA']:

    human_df = pd.read_csv(
        os.path.join(human_path, human_files[ds_name]),
        decimal=','
    )
    synth_df = pd.read_csv(
        os.path.join(synthetic_path, synthetic_files[ds_name]),
        decimal=','
    )

    dims = dimensions_map[ds_name]

    # split su familiarità
    split_col = "FAMILIARITY_human"
    human_df[split_col] = pd.to_numeric(human_df[split_col], errors='coerce')

    high_metaphors = set(
        human_df.loc[human_df[split_col] > 3, 'metaphor']
    )
    low_metaphors = set(
        human_df.loc[human_df[split_col] < 3, 'metaphor']
    )

    # -------------------------
    # per ogni dimensione
    # -------------------------

    for dim in dims:

        human_col = f"{dim}_human"
        synth_col = f"{dim}_synthetic"

        human_df[human_col] = pd.to_numeric(
            human_df[human_col], errors='coerce'
        )
        synth_df[synth_col] = pd.to_numeric(
            synth_df[synth_col], errors='coerce'
        )

        merged = (
            human_df[['metaphor', human_col]]
            .merge(
                synth_df[['metaphor', synth_col]],
                on='metaphor',
                how='inner',
                validate='one_to_one'
            )
            .rename(columns={
                human_col: 'human',
                synth_col: 'synthetic'
            })
            .dropna()
        )

        high_subset = merged[merged['metaphor'].isin(high_metaphors)]
        low_subset = merged[merged['metaphor'].isin(low_metaphors)]

        high_human_all.extend(high_subset['human'].tolist())
        high_synth_all.extend(high_subset['synthetic'].tolist())

        low_human_all.extend(low_subset['human'].tolist())
        low_synth_all.extend(low_subset['synthetic'].tolist())

# -------------------------
# correlazioni finali globali
# -------------------------

def compute_corr(h, s):
    if len(h) >= 2:
        return spearmanr(h, s)
    return float('nan'), float('nan')

corr_high, p_high = compute_corr(high_human_all, high_synth_all)
corr_low, p_low = compute_corr(low_human_all, low_synth_all)

results_df = pd.DataFrame([
    {
        'group': 'high_familiarity',
        'n_pairs': len(high_human_all),
        'spearman_corr': corr_high,
        'p_value': p_high
    },
    {
        'group': 'low_familiarity',
        'n_pairs': len(low_human_all),
        'spearman_corr': corr_low,
        'p_value': p_low
    }
])

print('\n=== Correlazioni globali (tutte le dimensioni aggregate) ===')
print(results_df)

# -------------------------
# save
# -------------------------

out_dir = '_results'
Path(out_dir).mkdir(exist_ok=True)

results_df.to_csv(
    f'{out_dir}/results_familiarity_general.csv',
    index=False
)
