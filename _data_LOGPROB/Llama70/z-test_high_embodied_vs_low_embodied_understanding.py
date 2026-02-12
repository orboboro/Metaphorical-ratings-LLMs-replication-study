import math
from scipy.stats import norm
import pandas as pd

df = pd.read_csv("_results/results_embodied.csv")

r1 = df.loc[df["group"] == "BODY_RELATEDNESS_high", "spearman_corr"].iloc[0]
r2 = df.loc[df["group"] == "BODY_RELATEDNESS_low", "spearman_corr"].iloc[0]

n1 = df.loc[df["group"] == "BODY_RELATEDNESS_high", "n_item"].iloc[0]
n2 = df.loc[df["group"] == "BODY_RELATEDNESS_low", "n_item"].iloc[0]

def fisher_z(r):
    return 0.5 * math.log((1 + r) / (1 - r))

z1 = fisher_z(r1)
z2 = fisher_z(r2)

# errore standard
se = math.sqrt(1/(n1 - 3) + 1/(n2 - 3))

# statistica z
z_stat = (z1 - z2) / se

# p-value due code
p_value = 2 * (1 - norm.cdf(abs(z_stat)))

print("z statistic:", z_stat)
print("p-value:", p_value)

summary_df = pd.DataFrame([{
    'z_statistic': z_stat,
    'p_value': p_value
}])

summary_df.to_csv("_results/z_test_less_embodied_less_performance.csv", index=False)