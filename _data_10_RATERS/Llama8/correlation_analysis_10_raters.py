import os
import pandas as pd
from scipy.stats import spearmanr
from collections import defaultdict
from pathlib import Path

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

# Funzione per normalizzare i giudizi dati tra 1 e 5 come se fosssero tra 1 e 7

def normalize_me(series):
    return 1 + (series - 1) * (6 / 4)

# crea un dizionario vuoto in cui quando creo una chiave questa in automatico ha come valore una lista vuota
rows_by_dimension = defaultdict(list)

# Caricare i dataset umani e sintetici e costruire tabelle per dimensione
# Per ogni dimensione creeremo una lista di righe con: metaphor, human_value, synthetic_value

for ds_name in ['MB','ME','MI','MM']:
    
    hfile = os.path.join(human_path, human_files[ds_name])
    human_df = pd.read_csv(hfile, decimal=',') # molti campioni usano virgola come separatore decimale. pandas supporta decimal=','
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

        # costruisco tabella sintetica: per metafora prendo il valore dell'annotator==1 e la media su tutti gli annotatori
        # converto valori in numerici (ignorando errori -> NaN)
        human_vals = human_df[['metaphor', human_col]].copy()
        human_vals.rename(columns={human_col: 'human'}, inplace=True)
        human_vals['metaphor'] = human_vals['metaphor'].astype(str).str.strip()
        human_vals['human'] = pd.to_numeric(human_vals['human'], errors='coerce')

        synth_vals = synth_df[['annotator','metaphor', synth_col]].copy()
        synth_vals.rename(columns={synth_col: 'synthetic'}, inplace=True)
        synth_vals['metaphor'] = synth_vals['metaphor'].astype(str).str.strip()
        synth_vals[synth_col] = pd.to_numeric(synth_vals['synthetic'], errors='coerce')

        # mean across annotators
        synth_mean = synth_vals.groupby('metaphor')[synth_col].mean().reset_index().rename(columns={synth_col:'synthetic_mean'})
        # merge
        merged = human_vals.merge(synth_mean, on='metaphor', how='left')

        # salvare righe per questa dimensione
        for _, r in merged.iterrows():
            rows_by_dimension[dim].append({
                'dataset': ds_name,
                'metaphor': r['metaphor'],
                'human': r['human'],
                'synthetic': r.get('synthetic_mean'),
            })

LLAMA_DIR = Path(__file__).resolve().parent
MEMORY_DIR = LLAMA_DIR.parent
GENERAL_DIR = MEMORY_DIR.parent
TARGET_FILE = GENERAL_DIR / "_data_LOGPROB" / "Llama70" / "_results" / "results_everyday_global.csv"
every_day_df = pd.read_csv(TARGET_FILE, decimal=',')
every_day_df = every_day_df.set_index('dimension')
results = []

def safe(val):
        try:
            return float(val)
        except Exception:
            return float('nan')

for dim, rows in rows_by_dimension.items():
    df_dim = pd.DataFrame(rows)
    sub = df_dim[['human','synthetic']].dropna()
    corr_10_raters, p = spearmanr(sub['human'], sub['synthetic'])
    n = len(sub)

    corr_every_day = (every_day_df.loc[dim])['corr']
    delta = safe(corr_10_raters - safe(corr_every_day))
    pct = (delta / abs(safe(corr_every_day))) * 100

    results.append({
        'dimension' : dim,
        'n_item': n,
        'corr_poetic': corr_10_raters,
        'p_value': p,
        'pct change' : pct
    })

res_df = pd.DataFrame(results)

out_dir = '_results'
if not Path(out_dir).exists():
    Path(out_dir).mkdir()

print('\n=== Correlazioni con memoria: ===', '\n=== "pct_change" indica in percentuale quanto la correlazione nella strategia con 10 raters incrementa o decresce rispetto a quella con le log probabilities')
print(res_df)
res_df.to_csv(out_dir + '/results_10_raters.csv', index=False)
