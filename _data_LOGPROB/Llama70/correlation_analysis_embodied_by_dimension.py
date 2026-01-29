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

results = []

for ds_name in ['MB', 'MI']:

    human_df = pd.read_csv(
        os.path.join(human_path, human_files[ds_name]),
        decimal=','
    )
    synth_df = pd.read_csv(
        os.path.join(synthetic_path, synthetic_files[ds_name]),
        decimal=','
    )

    dims = dimensions_map[ds_name]

    # dimensione di split
    split_dim = 'BODY_RELATEDNESS' if ds_name == 'MB' else 'PHISICALITY'
    split_col = f"{split_dim}_human"

    # conversione numerica
    human_df[split_col] = pd.to_numeric(human_df[split_col], errors='coerce')

    high_metaphors = set(
        human_df.loc[human_df[split_col] > 5, 'metaphor']
    )
    low_metaphors = set(
        human_df.loc[human_df[split_col] < 3, 'metaphor']
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

        for label, metaphor_set in [
            ('high', high_metaphors),
            ('low', low_metaphors)
        ]:

            subset = merged[merged['metaphor'].isin(metaphor_set)]
            n = len(subset)

            if n >= 2:
                corr, p_value = spearmanr(
                    subset['human'], subset['synthetic']
                )
            else:
                corr, p_value = float('nan'), float('nan')

            results.append({
                'study': ds_name,
                'split_dimension': split_dim,
                'group': label,
                'target_dimension': dim,
                'n_item': n,
                'spearman_corr': corr,
                'p_value': p_value
            })

results_df = pd.DataFrame(results)

print('\n=== Correlazioni per dimensione e gruppo ===')
print(results_df)

out_dir = '_results'
Path(out_dir).mkdir(exist_ok=True)

results_df.to_csv(
    f'{out_dir}/results_embodied_by_dimension.csv',
    index=False
)
