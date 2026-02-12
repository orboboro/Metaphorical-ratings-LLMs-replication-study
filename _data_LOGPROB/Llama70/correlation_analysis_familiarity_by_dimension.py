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

agg_data = {}

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

    split_dim = 'FAMILIARITY'
    split_col = f"{split_dim}_human"
    median = 2.93

    human_df[split_col] = pd.to_numeric(human_df[split_col], errors='coerce')

    high_metaphors = set(
        human_df.loc[human_df[split_col] > float(median), 'metaphor']
    )
    low_metaphors = set(
        human_df.loc[human_df[split_col] < float(median), 'metaphor']
    )

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

        # inizializza struttura aggregata
        agg_data.setdefault(dim, {
            'high_human': [],
            'high_synth': [],
            'low_human': [],
            'low_synth': []
        })

        for label, metaphor_set in [
            ('high', high_metaphors),
            ('low', low_metaphors)
        ]:

            subset = merged[merged['metaphor'].isin(metaphor_set)]

            if len(subset) > 0:
                agg_data[dim][f'{label}_human'].extend(subset['human'].tolist())
                agg_data[dim][f'{label}_synth'].extend(subset['synthetic'].tolist())


# -------------------------
# calcolo finale aggregato
# -------------------------

results = []

for dim, data in agg_data.items():

    for label in ['high', 'low']:

        h = data[f'{label}_human']
        s = data[f'{label}_synth']
        n = len(h)

        if n >= 2:
            corr, p_value = spearmanr(h, s)
        else:
            corr, p_value = float('nan'), float('nan')

        results.append({
            'split_dimension': 'FAMILIARITY',
            'group': label,
            'target_dimension': dim,
            'n_item': n,
            'spearman_corr': corr,
            'p_value': p_value
        })

results_df = pd.DataFrame(results)

print('\n=== Correlazioni aggregate per dimensione ===')
print(results_df)

out_dir = '_results'
Path(out_dir).mkdir(exist_ok=True)

results_df.to_csv(
    f'{out_dir}/results_familiarity_by_dimension.csv',
    index=False
)
