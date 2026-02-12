import pandas as pd
import os
from scipy.stats import ttest_ind

human_path = "human_datasets/"
synthetic_path = "synthetic_datasets/"

def read_human_csv(path):
    df = pd.read_csv(path)
    for col in df.columns:
        if col != "metaphor":
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .astype(float)
            )
    return df

human_BA = read_human_csv(os.path.join(human_path, "human_BA.csv"))
human_poetic_fam = human_BA["FAMILIARITY_human"].dropna()
human_daily_files = ["human_MB.csv", "human_ME.csv", "human_MM.csv"]
human_daily_vals = []

for f in human_daily_files:
    df = read_human_csv(os.path.join(human_path, f))
    if "FAMILIARITY_human" in df.columns:
        human_daily_vals.append(df["FAMILIARITY_human"])

human_daily_fam = pd.concat(human_daily_vals).dropna()

synthetic_BA = pd.read_csv(os.path.join(synthetic_path, "synthetic_BA.csv"))
llm_poetic_fam = synthetic_BA["FAMILIARITY_synthetic"].dropna()
synthetic_daily_files = ["synthetic_MB.csv", "synthetic_ME.csv", "synthetic_MM.csv"]

llm_daily_vals = []

for f in synthetic_daily_files:
    df = pd.read_csv(os.path.join(synthetic_path, f))
    if "FAMILIARITY_synthetic" in df.columns:
        llm_daily_vals.append(df["FAMILIARITY_synthetic"])

llm_daily_fam = pd.concat(llm_daily_vals).dropna()


t_human = ttest_ind(human_poetic_fam, human_daily_fam, equal_var=False)
t_llm = ttest_ind(llm_poetic_fam, llm_daily_fam, equal_var=False)


results = pd.DataFrame([
    {
        "group": "human",
        "poetic_mean": human_poetic_fam.mean(),
        "daily_mean": human_daily_fam.mean(),
        "difference_daily_minus_poetic": human_daily_fam.mean() - human_poetic_fam.mean(),
        "poetic_n": len(human_poetic_fam),
        "daily_n": len(human_daily_fam),
        "t_stat": t_human.statistic,
        "p_value": t_human.pvalue
    },
    {
        "group": "LLM",
        "poetic_mean": llm_poetic_fam.mean(),
        "daily_mean": llm_daily_fam.mean(),
        "difference_daily_minus_poetic": llm_daily_fam.mean() - llm_poetic_fam.mean(),
        "poetic_n": len(llm_poetic_fam),
        "daily_n": len(llm_daily_fam),
        "t_stat": t_llm.statistic,
        "p_value": t_llm.pvalue
    }
])

results.to_csv("familiarity_ttest_results.csv", index=False)

print("\n=== RISULTATI ===")
print(results)
print("\nTabella salvata in: familiarity_ttest_results.csv")
