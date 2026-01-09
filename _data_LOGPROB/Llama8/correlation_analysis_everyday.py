"""
Calcola le correlazioni di Spearman tra valutazioni umane e sintetiche (run 1 e media su annotatori)
per ogni dimensione dei 4 dataset (MB, ME, MI, MM), separando metafore già usate in studi precedenti
da quelle non usate.

Input: aspettati le cartelle:
  data/human_datasets/
  data/synthetic_datasets/
  data/original_datasets/

Output: stampa a video una tabella riepilogativa e salva `results_correlations.csv`.

Nota: mantengo il codice semplice come richiesto; gestisco NaN eliminandoli dalle coppie usate
per il calcolo di Spearman.
"""

import os
import pandas as pd
from scipy.stats import spearmanr
from collections import defaultdict
from pathlib import Path

# Percorsi (come indicato dall'utente)
human_path = "human_datasets/"
synthetic_path = "synthetic_datasets/"
raw_path = "original_datasets/"

# File mapping (nomi come forniti dall'utente)
human_files = {
    'MB': 'human_MB.csv',
    'ME': 'human_ME.csv',
    'MI': 'human_MI.csv',
    'MM': 'human_MM.csv',
}

synthetic_files = {
    'MB': 'synthetic_MB_meta-llama-Meta-Llama-3-8B-Instruct_.csv',
    'ME': 'synthetic_ME_meta-llama-Meta-Llama-3-8B-Instruct_.csv',
    'MI': 'synthetic_MI_meta-llama-Meta-Llama-3-8B-Instruct_.csv',
    'MM': 'synthetic_MM_meta-llama-Meta-Llama-3-8B-Instruct_.csv',
}

raw_files = {
    'MB': 'raw_MB.csv',
    'ME': 'raw_ME.csv',
    'MI': 'raw_MI.csv',
    'MM': 'raw_MM.csv',
}

# Dimensioni per dataset (nomi coerenti con quelli nei file forniti dall'utente)
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

# Rintracciare le metafore gia' usate in altri studi

used_metaphors = {dim: set() for dims in dimensions_map.values() for dim in dims}

for name, fname in raw_files.items():
    path = os.path.join(raw_path, fname)
    raw_df = pd.read_csv(path)
    for idx, row in raw_df.iterrows():

        if row.get("Bambini et al. (2013)") == "Y":
            used_metaphors["FAMILIARITY"].add(row["Metaphor"]) 
            used_metaphors["MEANINGFULNESS"].add(row["Metaphor"]) 
            used_metaphors["DIFFICULTY"].add(row["Metaphor"]) 

        if row.get("Canal et al. (2022)") == "Y":
            used_metaphors["FAMILIARITY"].add(row["Metaphor"]) 
            used_metaphors["PHISICALITY"].add(row["Metaphor"])

        if row.get("Lago et al. (2024)") == "Y":
            used_metaphors["FAMILIARITY"].add(row["Metaphor"]) 

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

        synth_vals = synth_df[['metaphor', synth_col]].copy()
        synth_vals.rename(columns={synth_col: 'synthetic'}, inplace=True)
        synth_vals['metaphor'] = synth_vals['metaphor'].astype(str).str.strip()
        synth_vals['synthetic'] = pd.to_numeric(synth_vals['synthetic'], errors='coerce')

        merged = human_vals.merge(synth_vals, on='metaphor', how='left')

        # salvare righe per questa dimensione
    
        for _, r in merged.iterrows():
            rows_by_dimension[dim].append({
                'dataset': ds_name,
                'metaphor': r['metaphor'],
                'human': r['human'],
                'synthetic': r.get('synthetic')
            })

# Per ogni dimensione calcolare Spearman globalmente, poi separare metafore usate vs non usate e calcolare Spearman separatamente
results = []
for dim, rows in rows_by_dimension.items():
    df_dim = pd.DataFrame(rows)
    # rimuovere righe con human NaN
    df_dim = df_dim.dropna(subset=['human']).copy()
    # aggiungere colonna used
    df_dim['used'] = df_dim['metaphor'].apply(lambda m: m in used_metaphors.get(dim, set())) # il secondo parametro è il valore da restituiore se nel dizionario non c'è la chiave (dim) specificata

    for used_flag, group_df in df_dim.groupby('used'):
        label = 'used' if used_flag else 'not_used'

        sub = group_df[['human','synthetic']].dropna()
        corr, p = spearmanr(sub['human'], sub['synthetic'])
        n = len(sub)

        # misura del cambiamento (corr non-used minus corr used): qui calcolo semplice delta e percentuale relativa
        results.append({
            'dimension': dim,
            'group': label,
            'n_item': n,
            'corr': corr,
            'p_value': p,
        })

# organizzo risultati in DataFrame e calcolo differenze tra used e not_used per ogni dimensione
res_df = pd.DataFrame(results)
summary_rows = []
for dim in res_df['dimension'].unique():
    sub = res_df[res_df['dimension']==dim].set_index('group') # la colonna 'group' (che contiene stringhe: 'used' o ' not_used") diventa l’indice invece dei numeri predefiniti
    used_row = sub.loc['used'] if 'used' in sub.index else None
    not_used_row = sub.loc['not_used'] if 'not_used' in sub.index else None

    def safe(val):
        try:
            return float(val)
        except Exception:
            return float('nan')

    if used_row is not None and not_used_row is not None:
        # calcolo della differenza in percentuale della correlazione coi giudizi umani delle metafore used vs not_used
        delta = safe(not_used_row['corr']) - safe(used_row['corr'])
        pct = (delta / abs(safe(used_row['corr']))) * 100 if pd.notna(used_row['corr']) and used_row['corr']!=0 else float('nan')
    else:
        delta = pct = float('nan')

    summary_rows.append({
        'dimension': dim,
        'used_corr': used_row['corr'] if used_row is not None else float('nan'),
        'used_p_value' : used_row['p_value'] if used_row is not None else float('nan'),
        'not_used_corr': not_used_row['corr'] if not_used_row is not None else float('nan'),
        'not_used_p_value' : not_used_row['p_value'] if used_row is not None else float('nan'),
        'pct_change': pct,
    })

summary_df = pd.DataFrame(summary_rows)

print('\n=== Sintesi delta tra metafore used vs not_used ===')
print(summary_df)

out_dir = 'results_everyday'
if not Path(out_dir).exists():
    Path(out_dir).mkdir()

summary_df.to_csv(out_dir + '/results_used_VS_not_used.csv', index=False)
