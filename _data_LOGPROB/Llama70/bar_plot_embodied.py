import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Dati
data = {
    "split_dimension": ["BODY_RELATEDNESS","BODY_RELATEDNESS","BODY_RELATEDNESS","BODY_RELATEDNESS",
                        "BODY_RELATEDNESS","BODY_RELATEDNESS","PHISICALITY","PHISICALITY",
                        "PHISICALITY","PHISICALITY"],
    "group": ["high","low","high","low","high","low","high","low","high","low"],
    "target_dimension": ["FAMILIARITY","FAMILIARITY","MEANINGFULNESS","MEANINGFULNESS",
                         "BODY_RELATEDNESS","BODY_RELATEDNESS","PHISICALITY","PHISICALITY",
                         "IMAGEABILITY","IMAGEABILITY"],
    "n_item": [32,22,32,32,32,32,21,21,21,21],
    "spearman_corr": [0.033,0.707,0.004,0.652,0.127,0.378,0.441,0.427,0.440,0.149],
    "p_value": [0.8577,0.00024,0.9833,0.000052,0.4902,0.0331,0.0453,0.0536,0.0457,0.5199]
}

df = pd.DataFrame(data)

# Colori
color_high = "#1f77b4"
color_low = "#ff7f0e"  # stesso colore tra split_dimension

# Facet per split_dimension
split_dims = df['split_dimension'].unique()
fig, axes = plt.subplots(1, len(split_dims), figsize=(12,5), sharey=True)

if len(split_dims) == 1:
    axes = [axes]

for ax, split in zip(axes, split_dims):
    df_split = df[df['split_dimension'] == split]
    target_dims = df_split['target_dimension'].unique()
    x = np.arange(len(target_dims))
    width = 0.35

    high_vals = df_split[df_split['group']=='high']['spearman_corr'].values
    low_vals = df_split[df_split['group']=='low']['spearman_corr'].values
    high_p = df_split[df_split['group']=='high']['p_value'].values
    low_p = df_split[df_split['group']=='low']['p_value'].values

    rects1 = ax.bar(x - width/2, high_vals, width, label='High', color=color_high)
    rects2 = ax.bar(x + width/2, low_vals, width, label='Low', color=color_low)

    # aggiungi asterischi sopra le barre
    for i, pv in enumerate(high_p):
        if pv > 0.1:
            ax.text(x[i] - width/2, high_vals[i]+0.05, '**', ha='center', va='bottom', fontsize=12)
        elif pv > 0.05:
            ax.text(x[i] - width/2, high_vals[i]+0.05, '*', ha='center', va='bottom', fontsize=12)

    for i, pv in enumerate(low_p):
        if pv > 0.1:
            ax.text(x[i] + width/2, low_vals[i]+0.05, '**', ha='center', va='bottom', fontsize=12)
        elif pv > 0.05:
            ax.text(x[i] + width/2, low_vals[i]+0.05, '*', ha='center', va='bottom', fontsize=12)

    ax.set_title(split.replace("_"," ").title())
    ax.set_xticks(x)
    ax.set_xticklabels(target_dims, rotation=45, ha='right')
    ax.set_ylim(0,1)
    ax.set_ylabel("Spearman Correlation")
    ax.legend()

plt.tight_layout()
plt.savefig("barplot_correlations.png", dpi=300)
plt.show()
