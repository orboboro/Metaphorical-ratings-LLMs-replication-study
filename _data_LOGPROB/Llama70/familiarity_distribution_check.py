import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, shapiro, normaltest

human_path = "human_datasets/"
synthetic_path = "synthetic_datasets/"

human_files = [
    "human_BA.csv",
    "human_MB.csv",
    "human_ME.csv",
    "human_MM.csv"
]

synthetic_files = [
    "synthetic_BA.csv",
    "synthetic_MB.csv",
    "synthetic_ME.csv",
    "synthetic_MM.csv"
]

# -----------------------
# RACCOLTA FAMILIARITY HUMAN
# -----------------------

human_vals = []

for fname in human_files:
    df = pd.read_csv(os.path.join(human_path, fname))
    if "FAMILIARITY_human" in df.columns:
        vals = (
            df["FAMILIARITY_human"]
            .astype(str)
            .str.replace(",", ".", regex=False)
        )
        vals = pd.to_numeric(vals, errors="coerce").dropna()
        human_vals.extend(vals.tolist())

human_vals = np.array(human_vals)

# -----------------------
# RACCOLTA FAMILIARITY SYNTHETIC
# -----------------------

synthetic_vals = []

for fname in synthetic_files:
    df = pd.read_csv(os.path.join(synthetic_path, fname))
    if "FAMILIARITY_synthetic" in df.columns:
        vals = pd.to_numeric(df["FAMILIARITY_synthetic"], errors="coerce").dropna()
        synthetic_vals.extend(vals.tolist())

synthetic_vals = np.array(synthetic_vals)

# -----------------------
# STATISTICHE BASE
# -----------------------

print("\n=== HUMAN FAMILIARITY ===")
print("N:", len(human_vals))
print("Mean:", human_vals.mean())
print("Median:", np.median(human_vals))

print("\n=== SYNTHETIC FAMILIARITY ===")
print("N:", len(synthetic_vals))
print("Mean:", synthetic_vals.mean())
print("Median:", np.median(synthetic_vals))

# -----------------------
# TEST NORMALITÀ
# -----------------------

def normality_report(values, label):
    print(f"\n--- Normality tests: {label} ---")

    s, p = shapiro(values)
    print("Shapiro-Wilk p =", p)

    if p > 0.05:
        print("→ non si rifiuta normalità")
    else:
        print("→ distribuzione NON normale")

normality_report(human_vals, "Human")
normality_report(synthetic_vals, "Synthetic")

# -----------------------
# KDE con gaussian_kde
# -----------------------

xmin = min(human_vals.min(), synthetic_vals.min())
xmax = max(human_vals.max(), synthetic_vals.max())

xs = np.linspace(xmin, xmax, 400)

kde_h = gaussian_kde(human_vals)
kde_s = gaussian_kde(synthetic_vals)

# -----------------------
# GRAFICO
# -----------------------

plt.figure()

plt.plot(xs, kde_h(xs), label="Human")
plt.plot(xs, kde_s(xs), label="Synthetic")

plt.axvline(human_vals.mean(), linestyle="--", alpha=0.7)
plt.axvline(synthetic_vals.mean(), linestyle=":", alpha=0.7)

plt.title("Familiarity Distribution — Human vs Synthetic")
plt.xlabel("Familiarity")
plt.ylabel("Density")
plt.legend()

os.makedirs("_results", exist_ok=True)
plt.savefig("_results/familiarity_distribution.png", bbox_inches="tight")
plt.show()
