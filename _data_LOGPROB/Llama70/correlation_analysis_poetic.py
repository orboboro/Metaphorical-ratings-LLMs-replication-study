import os
import pandas as pd
from scipy.stats import spearmanr
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


human_path = "human_datasets/"
synthetic_path = "synthetic_datasets/"

human_file = 'human_BA.csv'
synthetic_file = 'synthetic_BA.csv'

dimensions= ['FAMILIARITY', 'DIFFICULTY', 'MEANINGFULNESS']

out_dir = '_results'
if not Path(out_dir).exists():
    Path(out_dir).mkdir()

# crea un dizionario vuoto in cui quando creo una chiave questa in automatico ha come valore una lista vuota
rows_by_dimension = defaultdict(list)

hfile = os.path.join(human_path, human_file)
human_df = pd.read_csv(hfile, decimal=',') # molti campioni usano virgola come separatore decimale. pandas supporta decimal=','
sfile = os.path.join(synthetic_path, synthetic_file)
synth_df = pd.read_csv(sfile, decimal=',')

for dim in dimensions:
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
            'metaphor': r['metaphor'],
            'human': r['human'],
            'synthetic': r.get('synthetic')
        })

# Per ogni dimensione calcolare Spearman e percentage change rispetto a quello delle everyday metaphors

every_day_df = pd.read_csv(out_dir + '/results_everyday_global.csv', decimal=',')
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

    corr_poetic, p = spearmanr(sub['human'], sub['synthetic'])
    n = len(sub)

    corr_every_day = (every_day_df.loc[dim])['corr']
    delta = safe(corr_poetic - safe(corr_every_day))
    pct = (delta / abs(safe(corr_every_day))) * 100

    results.append({
        'dimension' : dim,
        'n_item': n,
        'corr_poetic': corr_poetic,
        'p_value': p,
        'pct change' : pct
    })

    # =========================
    # GRAFICO CORRELAZIONE
    # =========================
    if len(sub) > 1:
        x = sub['human'].values
        y = sub['synthetic'].values

        plt.figure()

        # scatter
        plt.scatter(x, y)

        # retta di regressione lineare
        m, b = np.polyfit(x, y, 1)
        plt.plot(x, m*x + b)

        plt.xlabel("Human score")
        plt.ylabel("Synthetic score")
        plt.title(f"{dim} — Spearman={corr_poetic:.3f}  n={n}")

        plt.tight_layout()
        plt.savefig(f"{out_dir}/corr_{dim.lower()}.png", dpi=300)
        plt.close()
