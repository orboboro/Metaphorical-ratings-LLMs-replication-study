import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Dati
data = {
    "familiarity": ["high","low","high","low","high","low","high","low"],
    "dimension": ["Familiarity","Familiarity","Meaningfulness","Meaningfulness",
                  "Body relatedness","Body relatedness","Difficulty","Difficulty"],
    "spearman_corr": [0.30565530792406964,0.4413728075075479,0.6686950705331726,
                      0.6187238887485114,0.7261261988212666,0.8909090909090911,
                      0.47017558681165794,0.4528364754821528],
    "p_value": [0.00010436675290270726,1.1251697069119902e-08,1.4435942361771815e-21,
                1.5556308483036647e-17,3.5728492248434736e-08,0.0002334581703787316,
                0.0003340102550208981,1.720644158773435e-08]
}

df = pd.DataFrame(data)

# Barre per ogni dimensione
dimensions = df['dimension'].unique()
x = np.arange(len(dimensions))  # posizioni delle dimensioni
width = 0.35  # larghezza barre

# Creazione figura
fig, ax = plt.subplots(figsize=(10,6))

# Barre high e low
high_corr = df[df['familiarity']=='high'].set_index('dimension')['spearman_corr'].reindex(dimensions)
low_corr = df[df['familiarity']=='low'].set_index('dimension')['spearman_corr'].reindex(dimensions)

# Disegna le barre
bars1 = ax.bar(x - width/2, high_corr, width, label='High Familiarity', color='skyblue')
bars2 = ax.bar(x + width/2, low_corr, width, label='Low Familiarity', color='salmon')

# Aggiungi asterischi sopra le barre se p_value > 0.05 o > 0.1
for i, dim in enumerate(dimensions):
    # High
    p = df[(df['familiarity']=='high') & (df['dimension']==dim)]['p_value'].values[0]
    if p > 0.1:
        ax.text(i - width/2, high_corr[i]+0.03, '**', ha='center', va='bottom', fontsize=12)
    elif p > 0.05:
        ax.text(i - width/2, high_corr[i]+0.03, '*', ha='center', va='bottom', fontsize=12)
    # Low
    p = df[(df['familiarity']=='low') & (df['dimension']==dim)]['p_value'].values[0]
    if p > 0.1:
        ax.text(i + width/2, low_corr[i]+0.03, '**', ha='center', va='bottom', fontsize=12)
    elif p > 0.05:
        ax.text(i + width/2, low_corr[i]+0.03, '*', ha='center', va='bottom', fontsize=12)

# Etichette e titolo
ax.set_ylabel('Spearman Correlation')
ax.set_xticks(x)
ax.set_xticklabels(dimensions, rotation=30, ha='right')
ax.set_title('Spearman Correlations by Familiarity Level')
ax.legend()
ax.set_ylim(0, 1.1)  # lascia spazio per asterischi

plt.tight_layout()
plt.savefig("spearman_correlations.png", dpi=300)
plt.show()
