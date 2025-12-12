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

# Percorsi (come indicato dall'utente)
human_path = "data/human_datasets/"
synthetic_path = "data/synthetic_datasets/"
raw_path = "data/original_datasets/"

# File mapping (nomi come forniti dall'utente)
human_files = {
    'MB': 'human_MB.csv',
    'ME': 'human_ME.csv',
    'MI': 'human_MI.csv',
    'MM': 'human_MM.csv',
}

synthetic_files = {
    'MB': 'synthetic_MB_meta-llama-Llama-3.3-70B-Instruct_.csv',
    'ME': 'synthetic_ME_meta-llama-Llama-3.3-70B-Instruct_.csv',
    'MI': 'synthetic_MI_meta-llama-Llama-3.3-70B-Instruct_.csv',
    'MM': 'synthetic_MM_meta-llama-Llama-3.3-70B-Instruct_.csv',
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

# --- 1) Rintracciare le metafore gia' usate in altri studi (codice fornito dall'utente adattato) ---
used_metaphors = {dim: set() for dims in dimensions_map.values() for dim in dims}
# Assumo che i raw CSV contengano colonne con esattamente questi nomi come indicato dallo user
for name, fname in raw_files.items():
    path = os.path.join(raw_path, fname)
    if not os.path.exists(path):
        # Se i file raw non esistono, continuiamo: used_metaphors resteranno vuoti
        continue
    raw_df = pd.read_csv(path)
    for idx, row in raw_df.iterrows():
        # Le colonne esaminate sono prese dal codice dell'utente
        try:
            if row.get("Bambini et al. (2013)") == "Y":
                used_metaphors["FAMILIARITY"].add(row["Metaphor"]) 
                used_metaphors["MEANINGFULNESS"].add(row["Metaphor"]) 
                used_metaphors["DIFFICULTY"].add(row["Metaphor"]) 

            if row.get("Canal et al. (2022)") == "Y":
                used_metaphors["FAMILIARITY"].add(row["Metaphor"]) 
                used_metaphors["PHISICALITY"].add(row["Metaphor"]) 

            if row.get("Bambini et al. (2024)") == "Y":
                used_metaphors["FAMILIARITY"].add(row["Metaphor"]) 
                used_metaphors["IMAGEABILITY"].add(row["Metaphor"]) 
                used_metaphors["DIFFICULTY"].add(row["Metaphor"]) 

            if row.get("Lago et al. (2024)") == "Y":
                used_metaphors["FAMILIARITY"].add(row["Metaphor"]) 
        except Exception:
            # manteniamo il codice semplice: ignoro righe malformate
            pass

# --- 2) Caricare i dataset umani e sintetici e costruire tabelle per dimensione ---
# Per ogni dimensione creeremo una lista di righe con: metaphor, human_value, synthetic_run1, synthetic_mean
from collections import defaultdict
rows_by_dimension = defaultdict(list)

for ds_name in ['MB','ME','MI','MM']:
    # leggi human
    hfile = os.path.join(human_path, human_files[ds_name])
    if not os.path.exists(hfile):
        # skip se mancante
        continue
    # molti campioni usano virgola come separatore decimale. pandas supporta decimal=','
    human_df = pd.read_csv(hfile, decimal=',')
    # normalize metaphor column name: alcune tabelle hanno 'metaphor' minuscolo come nell'esempio
    human_df.rename(columns=lambda c: c.strip(), inplace=True)

    # leggi synthetic
    sfile = os.path.join(synthetic_path, synthetic_files[ds_name])
    if not os.path.exists(sfile):
        continue
    synth_df = pd.read_csv(sfile, decimal=',')
    synth_df.rename(columns=lambda c: c.strip(), inplace=True)

    dims = dimensions_map[ds_name]
    for dim in dims:
        human_col = f"{dim}_human"
        synth_col = f"{dim}_synthetic"
        if human_col not in human_df.columns or synth_col not in synth_df.columns:
            # se la dimensione non è presente nei file, salto
            continue

        # costruisco tabella sintetica: per metafora prendo il valore dell'annotator==1 e la media su tutti gli annotatori
        # assicurarmi che 'annotator' esista
        if 'annotator' not in synth_df.columns:
            # se non c'è la colonna annotator, assumo che ogni riga sia un annotatore diverso
            synth_df['annotator'] = 1

        # converto valori in numerici (ignorando errori -> NaN)
        human_vals = human_df[['metaphor', human_col]].copy()
        human_vals.rename(columns={human_col: 'human'}, inplace=True)
        # pulizia banale: rimuovere spazi e quotes
        human_vals['metaphor'] = human_vals['metaphor'].astype(str).str.strip()
        human_vals['human'] = pd.to_numeric(human_vals['human'], errors='coerce')

        synth_vals = synth_df[['annotator','metaphor', synth_col]].copy()
        synth_vals['metaphor'] = synth_vals['metaphor'].astype(str).str.strip()
        synth_vals[synth_col] = pd.to_numeric(synth_vals[synth_col], errors='coerce')

        # mean across annotators
        synth_mean = synth_vals.groupby('metaphor')[synth_col].mean().reset_index().rename(columns={synth_col:'synthetic_mean'})
        # annotator 1 values (run1)
        run1 = synth_vals[synth_vals['annotator']==1][['metaphor', synth_col]].rename(columns={synth_col:'synthetic_run1'})
        # merge
        merged = human_vals.merge(run1, on='metaphor', how='left').merge(synth_mean, on='metaphor', how='left')

        # salvare righe per questa dimensione
        for _, r in merged.iterrows():
            rows_by_dimension[dim].append({
                'dataset': ds_name,
                'metaphor': r['metaphor'],
                'human': r['human'],
                'synthetic_run1': r.get('synthetic_run1'),
                'synthetic_mean': r.get('synthetic_mean'),
            })

# --- 3) Per ogni dimensione, separare metafore usate vs non usate e calcolare Spearman ---
results = []
for dim, rows in rows_by_dimension.items():
    df_dim = pd.DataFrame(rows)
    # rimuovere righe con human NaN (spec richiesto)
    df_dim = df_dim[~df_dim['human'].isna()].copy()

    # aggiungere flag used
    df_dim['used'] = df_dim['metaphor'].apply(lambda m: m in used_metaphors.get(dim, set()))

    for used_flag, group_df in df_dim.groupby('used'):
        label = 'used' if used_flag else 'not_used'

        # run1
        sub = group_df[['human','synthetic_run1']].dropna()
        if len(sub) >= 2:
            corr1, p1 = spearmanr(sub['human'], sub['synthetic_run1'])
            n1 = len(sub)
        else:
            corr1, p1, n1 = float('nan'), float('nan'), len(sub)

        # mean
        subm = group_df[['human','synthetic_mean']].dropna()
        if len(subm) >= 2:
            corrm, pm = spearmanr(subm['human'], subm['synthetic_mean'])
            nm = len(subm)
        else:
            corrm, pm, nm = float('nan'), float('nan'), len(subm)

        # misura del cambiamento (corr non-used minus corr used): qui calcolo semplice delta e percentuale relativa
        results.append({
            'dimension': dim,
            'group': label,
            'n_run1': n1,
            'corr_run1': corr1,
            'p_run1': p1,
            'n_mean': nm,
            'corr_mean': corrm,
            'p_mean': pm,
        })

# organizzo risultati in DataFrame e calcolo differenze tra used e not_used per ogni dimensione
res_df = pd.DataFrame(results)
summary_rows = []
for dim in res_df['dimension'].unique():
    sub = res_df[res_df['dimension']==dim].set_index('group')
    used_row = sub.loc['used'] if 'used' in sub.index else None
    not_used_row = sub.loc['not_used'] if 'not_used' in sub.index else None

    def safe(val):
        try:
            return float(val)
        except Exception:
            return float('nan')

    if used_row is not None and not_used_row is not None:
        # delta absolute and percent change for run1 and mean
        delta_run1 = safe(not_used_row['corr_run1']) - safe(used_row['corr_run1'])
        pct_run1 = (delta_run1 / abs(safe(used_row['corr_run1']))) * 100 if pd.notna(used_row['corr_run1']) and used_row['corr_run1']!=0 else float('nan')

        delta_mean = safe(not_used_row['corr_mean']) - safe(used_row['corr_mean'])
        pct_mean = (delta_mean / abs(safe(used_row['corr_mean']))) * 100 if pd.notna(used_row['corr_mean']) and used_row['corr_mean']!=0 else float('nan')
    else:
        delta_run1 = pct_run1 = delta_mean = pct_mean = float('nan')

    summary_rows.append({
        'dimension': dim,
        'used_corr_run1': used_row['corr_run1'] if used_row is not None else float('nan'),
        'not_used_corr_run1': not_used_row['corr_run1'] if not_used_row is not None else float('nan'),
        'delta_corr_run1': delta_run1,
        'pct_change_run1': pct_run1,
        'used_corr_mean': used_row['corr_mean'] if used_row is not None else float('nan'),
        'not_used_corr_mean': not_used_row['corr_mean'] if not_used_row is not None else float('nan'),
        'delta_corr_mean': delta_mean,
        'pct_change_mean': pct_mean,
    })

summary_df = pd.DataFrame(summary_rows)

# stampa e salvataggio
print('\n=== Risultati dettagliati per gruppo (used / not_used) ===')
print(res_df)
print('\n=== Sintesi delta tra metafore used vs not_used ===')
print(summary_df)

out_path = 'results_correlations.csv'
summary_df.to_csv(out_path, index=False)
print(f"\nSintesi salvata in: {out_path}")
